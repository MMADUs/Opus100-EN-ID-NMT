# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path
from typing import List

import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from tqdm import tqdm

from dataset import get_dataloaders, causal_mask
from model import build_model
from config import to_device, get_default_device
from utils import time_formatter


def greedy_decode(
    model, encoder_input, encoder_mask, tokenizer_tgt, seq_len, device=get_default_device()
) -> List[str]:
    """
    Batched greedy decoding for validation.

    Args:
        model (Transformer): Trained model
        encoder_input (torch.Tensor): (B, seq_len)
        encoder_mask (torch.Tensor): (B, 1, 1, seq_len)
        tokenizer_tgt (Tokenizer): Tokenizer for target language
        seq_len (int): Maximum sequence length
        device (torch.device): Device

    Returns:
        List[str]: Decoded target sentences for the batch
    """
    B = encoder_input.size(0)
    sos = tokenizer_tgt.token_to_id("[SOS]")
    eos = tokenizer_tgt.token_to_id("[EOS]")
    pad = tokenizer_tgt.token_to_id("[PAD]")

    # encode source
    encoder_output = model.encode(encoder_input, encoder_mask)

    # initialize decoder input
    decoder_input = torch.full((B, 1), sos, dtype=torch.long, device=device)
    finished = [False] * B

    for _ in range(seq_len):
        decoder_mask = causal_mask(decoder_input.size(1)).to(device).type_as(encoder_mask)

        # decode
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        logits = model.project(out[:, -1])  # (B, vocab_size)
        next_tokens = torch.argmax(logits, dim=-1)  # (B,)

        decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)

        # mark finished sequences
        for i in range(B):
            if next_tokens[i].item() == eos:
                finished[i] = True

        if all(finished):
            break

    # convert tokens to strings
    results = []
    for seq in decoder_input:
        seq_ids = seq.tolist()
        cleaned_ids = [t for t in seq_ids if t not in (sos, eos, pad)]
        results.append(tokenizer_tgt.decode(cleaned_ids))

    return results


def preload_state(
    conf, tokenizer_src, tokenizer_tgt
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """
    Load model and optimizer state from a checkpoint.

    This function loads a previously saved model checkpoint including the model weights,
    optimizer state, and the epoch at which training was interrupted. This enables
    resuming training from a previous point.

    Args:
        conf (EasyDict): Model configuration
        tokenizer_src (Tokenizer): Source language tokenizer
        tokenizer_tgt (Tokenizer): Target language tokenizer

    Returns:
        tuple[torch.nn.Module, torch.optim.Optimizer, int]: A tuple containing:
            - model: The loaded Transformer model on device
            - optimizer: The loaded Adam optimizer with previous state
            - epoch: The epoch number at which training was interrupted
    """
    checkpoint = torch.load(conf.model_output)

    # load model
    model = build_model(conf, tokenizer_src, tokenizer_tgt)
    model.load_state_dict(checkpoint["model"]["model"])
    model = to_device(model)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, eps=1e-9)
    optimizer.load_state_dict(checkpoint["optimizer"]["optimizer"])

    # last epoch
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def train_model(conf, callback, preload: bool = False) -> dict:
    """
    The training loop.

    This function implements the complete training loop with validation. It trains the
    Transformer model using mixed precision (autocast) for efficiency, validates on
    validation set, and tracks metrics like loss, CER, WER, and BLEU score. The function
    also handles checkpoint saving and early stopping via callbacks.

    Args:
        conf (EasyDict): Model configuration
        callback (TrainingCallback): Callback for early stopping and checkpoint saving
        preload (bool): Whether to load from a previous checkpoint, default is False

    Returns:
        dict: Training history with keys:
            - "train_loss" (list[float]): Training loss per epoch
            - "val_loss" (list[float]): Validation loss per epoch
            - "val_CER" (list[float]): Character Error Rate per epoch
            - "val_WER" (list[float]): Word Error Rate per epoch
            - "val_BLEU" (list[float]): BLEU score per epoch
    """
    train_dl, val_dl, _tdl, tokenizer_src, tokenizer_tgt = get_dataloaders(conf)

    init_epoch = 0

    model_path = Path(conf.model_output)

    if not preload or not model_path.exists():
        # model
        model = build_model(conf, tokenizer_src, tokenizer_tgt)
        model = to_device(model)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, eps=1e-9)
    else:
        model, optimizer, last_epoch = preload_state(conf, tokenizer_src, tokenizer_tgt)
        init_epoch = last_epoch + 1

    # criterion
    # cross entropy loss comes together with log softmax inside
    loss_fn = to_device(
        nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
    )

    # gradient scaler
    scaler = GradScaler(device="cuda")

    # init callbacks
    callback.init()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_CER": [],
        "val_WER": [],
        "val_BLEU": [],
    }

    start_time = time.time()

    for epoch in range(init_epoch, conf.num_epochs):
        epoch_start = time.time()
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0

        batch_iter = tqdm(train_dl, desc=f"epoch {epoch+1}")

        for batch in batch_iter:
            # model data input
            encoder_input = to_device(batch["encoder_input"])  # (B, seq_len)
            decoder_input = to_device(batch["decoder_input"])  # (B, seq_len)

            # model mask input
            encoder_mask = to_device(batch["encoder_mask"])  # (B, 1, 1, seq_len)
            decoder_mask = to_device(batch["decoder_mask"])  # (B, 1, seq_len, seq_len)

            # label to compare with output
            label = to_device(batch["label"])

            # reset optimizer
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                # forward
                proj_output = model(
                    encoder_input, decoder_input, encoder_mask, decoder_mask
                )

                # compute loss
                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
                )

            # scale gradient then backward
            scaler.scale(loss).backward()

            # step optimizer
            scaler.step(optimizer)

            # update scaler
            scaler.update()

            train_loss += loss.item()
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

        model.eval()
        val_loss = 0.0
        predicted_texts = []
        expected_texts = []

        with torch.no_grad():
            for batch in val_dl:
                # model data input
                encoder_input = to_device(batch["encoder_input"])  # (B, seq_len)
                decoder_input = to_device(batch["decoder_input"])  # (B, seq_len)

                # model mask input
                encoder_mask = to_device(batch["encoder_mask"])  # (B, 1, 1, seq_len)
                decoder_mask = to_device(
                    batch["decoder_mask"]
                )  # (B, 1, seq_len, seq_len)

                # label to compare with output
                label = to_device(batch["label"])

                with autocast(device_type="cuda"):
                    # forward
                    proj_output = model(
                        encoder_input, decoder_input, encoder_mask, decoder_mask
                    )

                    # compute loss
                    loss = loss_fn(
                        proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                        label.view(-1),
                    )

                # predict sentence to measure validation performance
                pred_texts = greedy_decode(
                    model, encoder_input, encoder_mask, tokenizer_tgt, conf.seq_len
                )
                predicted_texts.extend(pred_texts)
                expected_texts.extend(batch["tgt_text"])

                val_loss += loss.item()

        # avg
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)

        # language model metrics
        cer_metric = CharErrorRate()  # char level correctness
        wer_metric = WordErrorRate()  # word level correctness
        bleu_metric = BLEUScore()  # fluency & n-gram similarity

        cer = cer_metric(predicted_texts, expected_texts).item()
        wer = wer_metric(predicted_texts, expected_texts).item()

        bleu_refs = [[ref] for ref in expected_texts]
        bleu = bleu_metric(predicted_texts, bleu_refs).item()

        # history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_CER"].append(cer)
        history["val_WER"].append(wer)
        history["val_BLEU"].append(bleu)

        # logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1} - {time_formatter(epoch_time)} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"CER={cer} | WER={wer} | BLEU={bleu}"
        )

        # callbacks
        model_dict = {
            "model": model.state_dict(),
        }
        optimizer_dict = {
            "optimizer": optimizer,
        }

        early_stop = callback.step(val_loss, epoch, model_dict, optimizer_dict)
        if early_stop:
            break

        print("\n")

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history
