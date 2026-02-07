# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch

from pathlib import Path
from config import en_id_model as mtconf, to_device, get_default_device
from tokenizers import Tokenizer

from model import build_model
from dataset import causal_mask


class TranslationContext:
    """
    A context class for performing machine translation inference.

    This class manages the translation pipeline, including tokenizer loading,
    model initialization, and translation execution. It handles encoding of
    source text, encoder-decoder inference, and decoding of predicted text.
    """

    def __init__(self):
        # get tokenizers
        self.tokenizer_src = Tokenizer.from_file(
            str(Path(mtconf.tokenizer_file.format(mtconf.lang_src)))
        )
        self.tokenizer_tgt = Tokenizer.from_file(
            str(Path(mtconf.tokenizer_file.format(mtconf.lang_tgt)))
        )

        # load model
        self.model = build_model(mtconf, self.tokenizer_src, self.tokenizer_tgt)

        checkpoint = torch.load(mtconf.model_output)

        self.model.load_state_dict(checkpoint["model"]["model"])
        self.model = to_device(self.model)
        self.model.eval()

    def translate(self, sentence: str) -> str:
        device = get_default_device()

        # special tokens
        pad = self.tokenizer_tgt.token_to_id("[PAD]")
        sos = self.tokenizer_tgt.token_to_id("[SOS]")
        eos = self.tokenizer_tgt.token_to_id("[EOS]")

        with torch.no_grad():
            # build input source
            src_ids = self.tokenizer_src.encode(sentence).ids
            source = torch.tensor(
                [self.tokenizer_src.token_to_id("[SOS]")]
                + src_ids
                + [self.tokenizer_src.token_to_id("[EOS]")]
                + [self.tokenizer_src.token_to_id("[PAD]")]
                * (mtconf.seq_len - len(src_ids) - 2),
                dtype=torch.long,
                device=device,
            )

            # make encoder mask
            source_mask = (
                (source != self.tokenizer_src.token_to_id("[PAD]"))
                .unsqueeze(0)
                .unsqueeze(0)
            )

            # encode input (making this as the context to understand)
            memory = self.model.encode(source, source_mask)

            # start of sentence
            decoder_input = torch.tensor([[sos]], device=device)

            # generate text gradually
            while decoder_input.size(1) < mtconf.seq_len:
                seq_len = decoder_input.size(1)

                # mask to ignore padding and stop looking at future words
                dec_mask = causal_mask(seq_len).to(device) & (
                    decoder_input != pad
                ).unsqueeze(1)

                # predict the next word
                out = self.model.decode(memory, source_mask, decoder_input, dec_mask)
                logits = self.model.project(out[:, -1])
                next_word = torch.argmax(logits, dim=1)

                # append predicted word
                decoder_input = torch.cat(
                    [decoder_input, next_word.unsqueeze(0)], dim=1
                )

                # if end of sentence, stop generating word
                if next_word.item() == eos:
                    break

            # list of predicted tokens
            seq_ids = decoder_input[0].tolist()

            # remove sos, pad, and eos token
            cleaned_ids = []
            for token_id in seq_ids:
                if token_id not in (sos, eos, pad):
                    cleaned_ids.append(token_id)

            # decode to text
            return self.tokenizer_tgt.decode(cleaned_ids)
