# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Converts token IDs to embedding vectors scaled by sqrt(d_model)."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to embeddings."""

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, d_model)
        p_e = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sine to even indices
        p_e[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        p_e[:, 1::2] = torch.cos(position * div_term)

        # add extra dim for batch in p_e
        p_e = p_e.unsqueeze(0)  # (1, seq_len, d_model)

        # register positional encoding as a buffer
        self.register_buffer("p_e", p_e)

    def forward(self, x):
        x = x + self.p_e[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Transformer Model using PyTorch's nn.Transformer.

    Args:
        src_vocab_size (int): Size of the source vocabulary
        tgt_vocab_size (int): Size of the target vocabulary
        src_seq_len (int): Maximum source sequence length
        tgt_seq_len (int): Maximum target sequence length
        d_model (int): Dimension of the model embeddings, default 512
        N (int): Number of encoder/decoder layers, default 6
        h (int): Number of attention heads, default 8
        dropout (float): Dropout rate for regularization, default 0.1
        d_ff (int): Dimension of feed-forward inner layer, default 2048
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ):
        super().__init__()

        # embedding layers
        self.src_embed = InputEmbedding(d_model, src_vocab_size)
        self.tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

        # positional encoding layers
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=h,
            num_encoder_layers=N,
            num_decoder_layers=N,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        # projection layer
        self.project = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # embed and add positional encoding
        src = self.src_embed(src)
        src = self.src_pos(src)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        # pass through transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # final projection
        return self.project(output)


def initialize_parameters(transformer):
    """Initialize model parameters using Xavier uniform initialization."""
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def build_model(conf, tokenizer_src, tokenizer_tgt) -> Transformer:
    """Build and initialize the Transformer model."""
    model = Transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=conf.seq_len,
        tgt_seq_len=conf.seq_len,
        d_model=conf.d_model,
        N=conf.num_layers,
        h=conf.num_heads,
        dropout=conf.dropout,
        d_ff=conf.ffn_dim,
    )

    # init weights
    initialize_parameters(model)

    return model
