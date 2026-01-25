# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Converts token IDs to embedding vectors scaled by sqrt(d_model).

    This layer performs vocabulary embedding and scales the result by
    the square root of the model dimension to prevent the embeddings
    from becoming too small during training.

    Args:
        d_model (int): The dimension of the embedding vectors
        vocab_size (int): The size of the vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # scale the embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to embeddings.

    This layer encodes the absolute position of tokens in a sequence using sine
    and cosine functions of different frequencies. The positional encoding is added
    to the embeddings to give the model information about token positions, since
    the self-attention mechanism alone does not capture sequential order.

    Args:
        d_model (int): The dimension of the embedding vectors
        seq_len (int): The maximum sequence length to encode
        dropout (float): Dropout rate applied to the output
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # provide dropout
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, d_model)
        p_e = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)

        # create a vector of shape (d_model)
        # div_term = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sine to even indices
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        p_e[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        p_e[:, 1::2] = torch.cos(position * div_term)

        # add extra dim for batch in p_e
        p_e = p_e.unsqueeze(0)  # (1, seq_len, d_model)

        # register positional encoding as a buffer
        self.register_buffer("p_e", p_e)

    def forward(self, x):
        # positional encoding = embedding + sinusoidal positional encoding
        x = x + (
            self.p_e[:, : x.shape[1], :].requires_grad_(False)
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This layer implements the multi-head scaled dot-product attention mechanism.
    It splits the d_model dimension into multiple "heads",
    allowing the model to attend to different representation subspaces simultaneously.
    Each head performs attention independently, then results are concatenated and
    projected back to d_model dimensions.

    Args:
        d_model (int): The dimension of the embedding vectors
        h (int): The number of attention heads
        dropout (float): Dropout rate applied to attention weights
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # embedding vector size
        self.h = h  # number of heads

        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        # attention layers
        self.d_k = d_model // h  # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq (query)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk (key)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv (value)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo (output)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # attention formula from the paper
        # attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # decoder mask
        # to prevent peeking at future words during prediction
        # (mask future positions so attention cannot see them)
        if mask is not None:
            # mark with very low value (indicating -inf) to the position where mask == 0
            # this way softmax will make the value 0
            attention_scores = attention_scores.float()
            attention_scores.masked_fill_(mask == 0, float("-inf"))
            attention_scores = attention_scores.to(query.dtype)

        # apply softmax
        attention_scores = attention_scores.softmax(dim=-1)

        # dropout layer
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # attention score output
        attention = attention_scores @ value

        # attention scores can be used for visualization
        return attention, attention_scores

    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # calculate attention
        # multihead(Q,K,V) = Concat(Attention(Q W_Q^(i), K W_K^(i), V W_V^(i)) for i=1..h) * W_O
        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # combine all heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    This layer implements a two-layer feed-forward network applied to each position
    separately and identically. It consists of two linear layer with a ReLU activation in between.
    The network expands the dimension from d_model to d_ff, then projects back to d_model.

    Args:
        d_model (int): The dimension of the embedding vectors
        d_ff (int): The inner dimension (typically 4 * d_model)
        dropout (float): Dropout rate applied after the ReLU activation
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        # linear layers
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.ffn(x)


class LayerNormalization(nn.Module):
    """
    Layer Normalization (LayerNorm).

    This layer normalizes the input across the feature dimension (d_model)
    for each position in the sequence independently. It improves training stability
    and allows for higher learning rates by standardizing activations.

    Args:
        d_model (int): The dimension of the embedding vectors
        eps (float): A small value (epsilon) to prevent division by zero, default is 1e-6
    """

    def __init__(self, d_model: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps

        # both alpha and bias is a learnable parameter
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # (batch, seq_len, hidden_size)
        # keep dimensions for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # eps (epsilon) used to prevent dividing by zero or when std is very small
        # y = α * (x - mean) / (std + ε) + β
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    """
    Residual Connection (Skip Connection) with Layer Normalization.

    This layer implements the residual connection pattern: output = x + f(norm(x)).
    It allows gradients to flow directly through the network and helps prevent
    vanishing gradient problems in deep networks. Layer normalization is applied
    before the sublayer (Pre-Norm architecture).

    Args:
        d_model (int): The dimension of the embedding vectors
        dropout (float): Dropout rate applied to the sublayer output
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        # residual blocks
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        norm = self.norm(x)
        x_hat = sublayer(norm)
        return x + self.dropout(x_hat)


class EncoderBlock(nn.Module):
    """
    Single Encoder Block (layer).

    This block is the fundamental building unit of the encoder stack. It consists of
    two sub-layers: (1) multi-head self-attention mechanism, and (2) position-wise
    feed-forward network. Each sub-layer is wrapped with residual connections and
    layer normalization (Pre-Norm architecture).

    Args:
        d_model (int): The dimension of the embedding vectors
        self_attention_block (MultiHeadAttention): Multi-head attention module
        feed_forward_block (FeedForwardBlock): Feed-forward network module
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()

        # encoder blocks
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # 2 residual connection for self attention + feed forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        # attention -> add & norm
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # feed forward -> add & norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Transformer Encoder Stack.

    The encoder consists of a stack of N identical encoder blocks.
    The output of the encoder is normalized, and passed to the decoder's cross-attention mechanism
    to guide the generation of the target sequence.

    Args:
        d_model (int): The dimension of the embedding vectors
        layers (nn.ModuleList): List of N encoder blocks
    """

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()

        # layers of encoder blocks
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        # multi layer of encoders
        for layer in self.layers:
            x = layer(x, mask)
        # normalize output
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Single Decoder Block (layer).

    This block is the fundamental building unit of the decoder stack. It consists of
    three sub-layers: (1) masked multi-head self-attention, (2) multi-head cross-attention
    over encoder output, and (3) position-wise feed-forward network. Each sub-layer is
    wrapped with residual connections and layer normalization (Pre-Norm architecture).

    Args:
        d_model (int): The dimension of the embedding vectors
        self_attention_block (MultiHeadAttention): Masked self-attention module
        cross_attention_block (MultiHeadAttention): Cross-attention module
        feed_forward_block (FeedForwardBlock): Feed-forward network module
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # 3 residual connection for self attention + cross attention + feed forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # masked attention -> add & norm
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )

        # combine the encoder output to attention -> add & norm
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )

        # feed forward -> add & norm
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder Stack.

    The decoder consists of a stack of N identical decoder blocks.
    The output of the decoder is normalized, and projected to the target vocabulary to predict the
    next token at each position.

    Args:
        d_model (int): The dimension of the embedding vectors
        layers (nn.ModuleList): List of N decoder blocks
    """

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()

        # layers of decoder blocks
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # multi layer of decoders
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # normalize
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Output Projection Layer.

    This layer projects the decoder output from the model dimension (d_model)
    to the target vocabulary size.
    The output logits can be used with a softmax function to obtain probability
    distributions for next token prediction.

    Args:
        d_model (int): The dimension of the input embeddings from the decoder
        vocab_size (int): The size of the target vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # linear layer to project to vocabulary size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    """
    Transformer Model.

    This is the full Transformer architecture combining an encoder and decoder.

    Args:
        src_vocab_size (int): Size of the source vocabulary
        tgt_vocab_size (int): Size of the target vocabulary
        src_seq_len (int): Maximum source sequence length
        tgt_seq_len (int): Maximum target sequence length
        d_model (int): Dimension of the model embeddings, default 512
        N (int): Number of encoder/decoder blocks (layers), default 6
        h (int): Number of attention heads, default 8
        dropout (float): Dropout rate for regularization, default 0.1
        d_ff (int): Dimension of feed-forward inner layer, default 2048

    Attributes:
        src_embed (InputEmbedding): Source vocabulary embedding layer
        tgt_embed (InputEmbedding): Target vocabulary embedding layer
        src_pos (PositionalEncoding): Source positional encoding layer
        tgt_pos (PositionalEncoding): Target positional encoding layer
        encoder (Encoder): Stack of N encoder blocks
        decoder (Decoder): Stack of N decoder blocks
        project (ProjectionLayer): Final projection to target vocabulary
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

        # build encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(
                d_model, encoder_self_attention_block, feed_forward_block, dropout
            )
            encoder_blocks.append(encoder_block)

        # build decoder blocks
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(
                d_model,
                decoder_self_attention_block,
                decoder_cross_attention_block,
                feed_forward_block,
                dropout,
            )
            decoder_blocks.append(decoder_block)

        # transformer encoder decoder
        self.encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        self.decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        # projection layer
        self.project = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output,
        src_mask,
        tgt,
        tgt_mask,
    ):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # training forward pass, inference will be different
        # encode
        encoder_output = self.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, d_model)

        # decode
        decoder_output = self.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  # (B, seq_len, d_model)

        # final projection
        return self.project(decoder_output)  # (batch, seq_len, vocab_size)


def initialize_parameters(transformer):
    # init params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def build_model(conf, tokenizer_src, tokenizer_tgt) -> Transformer:
    # transformer model
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
