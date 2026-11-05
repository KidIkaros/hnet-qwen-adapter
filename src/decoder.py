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
        self.register_buffer("causal", torch.triu(torch.full((2 * max_len, 2 * max_len),
                                                             -float("inf")), diagonal=1))

    def forward(self, ctx, bytes_in, pad_mask=None):
        """ctx (B, D) or (B, K, D) with one context per max_len segment; bytes_in (B, L) shifted
        byte ids (pos 0 = SOS), pad_mask (B, L) True=pad."""
        B, L = bytes_in.shape
        d = self.byte_emb.embedding_dim
        h = self.byte_emb(bytes_in)
        cond = self.cond(ctx)                                 # (B, ..., max_len*d)
        for seg in range(0, L, self.max_len):
            c = cond[:, seg // self.max_len] if ctx.dim() == 3 else cond
            w = min(self.max_len, L - seg)
            h[:, seg:seg + w] += c[:, :w * d].reshape(B, w, d)
        h = self.enc(h, mask=self.causal[:L, :L], src_key_padding_mask=pad_mask)
        return self.head(h)                                   # (B, L, 256)


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


def pair_batch(data, spans, max_len=16):
    """MEGABYTE's cross-patch attention: decode each chunk with the PREVIOUS chunk's bytes as
    attended context.  Returns x/y/pad/ctx_idx/loss_mask: x (B, 2P) = [SOS]+prev_chunk+this_chunk
    shifted; loss_mask marks the interior positions of THIS chunk (the only scored ones).  The
    context vector applied per half: ctx[j-1] for the prefix half, ctx[j] for this chunk."""
    B = len(spans)
    x = torch.full((B, 2 * max_len), SOS, dtype=torch.long)
    y = torch.zeros(B, 2 * max_len, dtype=torch.long)
    pad = torch.ones(B, 2 * max_len, dtype=torch.bool)
    loss_mask = torch.zeros(B, 2 * max_len, dtype=torch.bool)
    for m, (s, chunk) in enumerate(spans):
        prev = spans[m - 1][1] if m > 0 else b""
        Lp = min(len(prev), max_len)
        Lc = min(len(chunk), max_len)
        seq = list(prev[:Lp]) + list(chunk[:Lc])
        x[m, 1:len(seq)] = torch.tensor(seq[:-1], dtype=torch.long)
        y[m, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        pad[m, :len(seq)] = False
        for p in range(Lp + 1, Lp + Lc):           # this chunk's positions p>=1 -> interior
            loss_mask[m, p] = True
    return x, y, pad, loss_mask
