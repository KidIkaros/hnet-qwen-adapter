"""The interior-byte decoder — the missing half of a real byte LM (MEGABYTE §2.2 / BLT §3.3).

For each chunk j, the decoder predicts the chunk's bytes left-to-right, conditioned on the trunk's
output at position j-1 (the hidden state that already predicts the chunk's first byte — the
causally SAFE context: it saw only bytes before the chunk).  Conditioning is MEGABYTE-style
additive: the context vector is linear-mapped to per-position slots and added to the previous
byte's embedding.  Chunk 0 gets a learned start context.

Causal-safety rule that shaped the design: the decoder must NOT see chunk j's patch-input vector or
any trunk output at positions >= j — those contain the chunk's own bytes and would leak the target.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS = 256                                # the start-of-chunk byte token id


class ByteDecoder(nn.Module):
    """(B, P) chunk bytes -> (B, P, 256) next-byte logits, conditioned on one trunk context vector
    per chunk.  Position p's input = byte_emb[b_{p-1}] + cond_p (b_{-1} = SOS)."""

    def __init__(self, d_trunk, max_len=16, d_dec=256, n_layers=4, n_heads=4):
        super().__init__()
        self.max_len = max_len
        self.byte_emb = nn.Embedding(257, d_dec)
        self.cond = nn.Linear(d_trunk, max_len * d_dec)
        self.ctx0 = nn.Parameter(torch.zeros(d_trunk))       # learned context for chunk 0
        layer = nn.TransformerEncoderLayer(d_dec, n_heads, 4 * d_dec, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_dec, 256)
        self.register_buffer("causal", torch.triu(torch.full((max_len, max_len), -float("inf")),
                                                  diagonal=1))

    def forward(self, ctx, bytes_in, pad_mask=None):
        """ctx (B, D), bytes_in (B, P) shifted byte ids (pos 0 = SOS), pad_mask (B, P) True=pad."""
        B, P = bytes_in.shape
        h = self.byte_emb(bytes_in) + self.cond(ctx)[:, :P * self.byte_emb.embedding_dim] \
            .reshape(B, P, -1)
        h = self.enc(h, mask=self.causal[:P, :P], src_key_padding_mask=pad_mask)
        return self.head(h)                                   # (B, P, 256)


def chunk_batch(data, spans, max_len=16):
    """(B,P) inputs/targets/mask for a doc's chunks: inputs = [SOS] + bytes[:-1] padded, targets =
    chunk bytes padded, pad_mask True on padding."""
    B = len(spans)
    x = torch.full((B, max_len), SOS, dtype=torch.long)
    y = torch.zeros(B, max_len, dtype=torch.long)
    pad = torch.ones(B, max_len, dtype=torch.bool)
    for m, (s, chunk) in enumerate(spans):
        L = min(len(chunk), max_len)
        x[m, 1:L] = torch.tensor(list(chunk[:L - 1]), dtype=torch.long)
        y[m, :L] = torch.tensor(list(chunk[:L]), dtype=torch.long)
        pad[m, :L] = False
    return x, y, pad
