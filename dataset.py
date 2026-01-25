# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_default_device


def get_all_sentences(ds, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_or_train_tokenizer(conf, ds, lang: str, train: bool = False) -> Tokenizer:
    """
    Get or train a WordLevel tokenizer.

    If the tokenizer doesn't exist at the specified path or if train=True,
    a new tokenizer is trained on the dataset and saved. Otherwise, an
    existing tokenizer is loaded from disk.

    Args:
        conf (EasyDict): Configuration object containing tokenizer_file path
        ds: Dataset from HuggingFace load_dataset()
        lang (str): Language code
        train (bool): Force retrain tokenizer even if it exists, default False

    Returns:
        Tokenizer: Trained or loaded tokenizer instance
    """
    # tokenizer path
    tokenizer_path = Path(conf.tokenizer_file.format(lang))

    if not Path.exists(tokenizer_path) or train:
        print(f"tokenizing: {lang}")
        # train tokenizer if not exist
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # save tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"tokenizer exist, getting from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_tokenizers(
    conf, ds_train, force_retrain_tokenizer: bool = False
) -> Tuple[Tokenizer, Tokenizer]:
    """
    Get or train tokenizers for both source and target languages.

    Args:
        conf (EasyDict): Configuration object with lang_src and lang_tgt
        ds_train: Dataset from HuggingFace load_dataset()
        force_retrain_tokenizer (bool): Force retrain both tokenizers, default False

    Returns:
        Tuple[Tokenizer, Tokenizer]: Source and target language tokenizers
    """
    tokenizer_src = get_or_train_tokenizer(
        conf, ds_train, conf.lang_src, force_retrain_tokenizer
    )
    tokenizer_tgt = get_or_train_tokenizer(
        conf, ds_train, conf.lang_tgt, force_retrain_tokenizer
    )
    return tokenizer_src, tokenizer_tgt


def get_max_length_sentence(
    conf, ds_train, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer
) -> None:
    """
    Calculate and print the maximum sentence lengths in source and target languages.
    Used for determining optimal sequence length (seq_len) for padding/truncation.

    Args:
        conf (EasyDict): Configuration object with lang_src and lang_tgt
        ds_train: Dataset from HuggingFace load_dataset()
        tokenizer_src (Tokenizer): Source language tokenizer
        tokenizer_tgt (Tokenizer): Target language tokenizer
    """
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_train:
        src_ids = tokenizer_src.encode(item["translation"][conf.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][conf.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of source sentence: {max_len_src}")
    print(f"max length of target sentence: {max_len_tgt}")


def causal_mask(size):
    """
    Create a causal mask for the decoder (upper triangular matrix).

    This mask prevents the model from attending to future tokens during
    self-attention in the decoder. It ensures that predictions for position i
    can only depend on known outputs from positions < i.

    Args:
        size (int): Sequence length (mask will be size x size)

    Returns:
        torch.Tensor: Boolean causal mask of shape (1, size, size)
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    """
    PyTorch Custom Dataset for preprocessing.

    This dataset handles sentence pairs in two languages, tokenizes them,
    adds special tokens, creates attention masks, and returns
    properly formatted batches.

    Args:
        ds: Dataset with "translation" field containing source and target texts
        tokenizer_src (Tokenizer): Source language tokenizer
        tokenizer_tgt (Tokenizer): Target language tokenizer
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        seq_len (int): Fixed sequence length for all samples
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # SOS (start of sentence)
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        # EOS (end of sentence)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        # PAD (padding token)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # transform text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # add SOS, EOS, & PAD
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # we will add <s> and </s>
        # we will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the number of padding tokens is not negative
        # if it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        item = {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

        return item


def prune_pairs(conf, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, ds_raw=None):
    """
    Filter dataset to remove sentence pairs longer than seq_len.

    This function tokenizes each sentence pair and removes pairs where either
    the source or target sentence exceeds the configured sequence length.

    Args:
        conf (EasyDict): Configuration object with lang_src, lang_tgt, seq_len
        tokenizer_src (Tokenizer): Source language tokenizer
        tokenizer_tgt (Tokenizer): Target language tokenizer
        ds_raw: HuggingFace dataset to filter

    Returns:
        Dataset: Filtered dataset with only valid sentence pairs
    """

    # load_dataset is used to get val & test, while ds_raw is the splitted train
    def map_token_len(example):
        example["src_len"] = len(
            tokenizer_src.encode(example["translation"][conf.lang_src]).ids
        )
        example["tgt_len"] = len(
            tokenizer_tgt.encode(example["translation"][conf.lang_tgt]).ids
        )
        return example

    ds_raw = ds_raw.map(map_token_len)

    ds_raw = ds_raw.filter(
        lambda x: x["src_len"] + 2 <= conf.seq_len  # SOS + EOS
        and x["tgt_len"] + 1 <= conf.seq_len  # SOS (decoder input)
    )

    return ds_raw


def build_ds(
    conf, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, ds_raw
) -> BilingualDataset:
    """
    Build the PyTorch Dataset from a raw HuggingFace dataset.

    This function prunes sentence pairs that exceed seq_len and wraps
    the filtered dataset in a PyTorch-compatible BilingualDataset.

    Args:
        conf (EasyDict): Configuration object with lang_src, lang_tgt, seq_len
        tokenizer_src (Tokenizer): Source language tokenizer
        tokenizer_tgt (Tokenizer): Target language tokenizer
        ds_raw: Raw HuggingFace dataset to process

    Returns:
        BilingualDataset: Processed dataset ready for DataLoader
    """
    # prune pairs that are longer than seq_len
    ds = prune_pairs(conf, tokenizer_src, tokenizer_tgt, ds_raw)

    # build the torch dataset
    dataset = BilingualDataset(
        ds, tokenizer_src, tokenizer_tgt, conf.lang_src, conf.lang_tgt, conf.seq_len
    )

    return dataset


class DeviceDataLoader:
    """
    Wrapper around PyTorch DataLoader that automatically moves batches to device.
    handles device transfer (CPU/GPU) automatically
    in the iteration loop, allowing cleaner training code.

    Attributes:
        dl (DataLoader): Underlying PyTorch DataLoader
        device (torch.device): Target device for batch transfer
    """

    def __init__(self, dl: DataLoader, device=get_default_device()):
        self.dl = dl
        self.device = device

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return batch  # leave other types untouched

    def __iter__(self):
        for batch in self.dl:
            yield self._to_device(batch)

    def __len__(self):
        return len(self.dl)


def load_hf_dataset(conf, split: str):
    return load_dataset(
        f"{conf.corpus}", f"{conf.lang_src}-{conf.lang_tgt}", split=split
    )


def get_dataloaders(
    conf, force_retrain_tokenizer: bool = False
) -> Tuple[DeviceDataLoader, DeviceDataLoader, DeviceDataLoader, Tokenizer, Tokenizer]:
    """
    Load and prepare all data loaders (train, validation, test) with tokenizers.

    This is the main entry point for data preparation. It:
    1. Loads the training dataset (with optional subsampling)
    2. Trains or loads tokenizers for both languages
    3. Analyzes maximum sentence lengths
    4. Creates BilingualDatasets for each split
    5. Wraps DataLoaders with device awareness

    Args:
        conf (EasyDict): Configuration object containing:
            - corpus (str): Huggingface corpus dataset name
            - lang_src (str): Source language code
            - lang_tgt (str): Target language code
            - train_set_ratio (float): Fraction of training data to use (0.0-1.0)
            - batch_size (int): Batch size for training
            - seq_len (int): Fixed sequence length
        force_retrain_tokenizer (bool): Force retrain tokenizers even if they exist, default False

    Returns:
        Tuple[DeviceDataLoader, DeviceDataLoader, DeviceDataLoader, Tokenizer, Tokenizer]:
            - train_dataloader (DeviceDataLoader): Training dataset loader
            - val_dataloader (DeviceDataLoader): Validation dataset loader
            - test_dataloader (DeviceDataLoader): Test dataset loader
            - tokenizer_src (Tokenizer): Source language tokenizer
            - tokenizer_tgt (Tokenizer): Target language tokenizer
    """
    # load corpus splits from HuggingFace
    ds_train = load_hf_dataset(conf, split="train")
    ds_val = load_hf_dataset(conf, split="validation")
    ds_test = load_hf_dataset(conf, split="test")

    # configurable subset
    if conf.train_set_ratio < 1.0:
        ds_train = ds_train.train_test_split(train_size=conf.train_set_ratio, seed=42)[
            "train"
        ]

    print(f"total sentence pair for training = {len(ds_train)}")

    # get tokenizers
    tokenizer_src, tokenizer_tgt = get_tokenizers(
        conf, ds_train, force_retrain_tokenizer
    )

    # show the corpus seq_len
    get_max_length_sentence(conf, ds_train, tokenizer_src, tokenizer_tgt)

    # build datasets
    train_ds = build_ds(conf, tokenizer_src, tokenizer_tgt, ds_raw=ds_train)
    val_ds = build_ds(conf, tokenizer_src, tokenizer_tgt, ds_raw=ds_val)
    test_ds = build_ds(conf, tokenizer_src, tokenizer_tgt, ds_raw=ds_test)

    print(f"total sentence pair for training after filtering = {len(train_ds)}")

    # data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_test_batch_size = max(1, conf.batch_size // 2)

    val_dl = DataLoader(
        val_ds, batch_size=val_test_batch_size, num_workers=4, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=val_test_batch_size, num_workers=4, pin_memory=True
    )

    return (
        DeviceDataLoader(train_dl),
        DeviceDataLoader(val_dl),
        DeviceDataLoader(test_dl),
        tokenizer_src,
        tokenizer_tgt,
    )
