"""v1: the learned patch projection — the ZHA recipe applied to our problem.

ZHA trains ONLY the indexer (~0.0002% of params) while the trunk stays frozen; here the indexer's
analog is a single D x D map applied to the constructed patch sum:  patch = (sum of byte-embedding
rows) @ W.  Initialized at the identity it IS the v0 sum baseline; the training run measures how much
of the residual gap a learned linear patch construction recovers.

The loss matches the eval exactly: byte-marginalized NLL of the next chunk's first byte.
"""
import numpy as np
import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    """patch = base @ (I + A B^T): a low-rank residual on the constructed sum.  A inits to zero so the
    map starts EXACTLY at the v0 baseline; rank r keeps the parameter count tiny (the ZHA pattern:
    ~0.0002% of the model trained, the trunk frozen)."""

    def __init__(self, D, rank=32):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(D, rank))
        self.B = nn.Parameter(torch.empty(rank, D))
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)

    def forward(self, base):                                 # (M, D) -> (M, D)
        return base + (base @ self.B.T) @ self.A.T


def pad_chunk_ids(chunk_bytes, bmap, pad_id):
    """(M, Lmax) int64 token ids for a list of byte-chunks: each byte -> its single-byte token id,
    unmappable bytes and padding -> pad_id (an appended zeros row contributes nothing)."""
    M = len(chunk_bytes)
    L = max(len(c) for c in chunk_bytes)
    out = np.full((M, L), pad_id, dtype=np.int64)
    for m, c in enumerate(chunk_bytes):
        for i, b in enumerate(c):
            if b in bmap:
                out[m, i] = bmap[b]
    return torch.tensor(out)


def byte_logprobs(logits, first):
    """(M, V) logits -> (M, 256) byte-level log-probs — the differentiable form of ByteHead's
    marginalization (prob-space index_add under the shared softmax denominator == a grouped
    logsumexp).  `first` is the (V,) int64 map (token -> first byte, -1 for none); padded logits
    columns beyond len(first) are excluded."""
    M, V = logits.shape
    if first.numel() < V:
        first = torch.cat([first, torch.full((V - first.numel(),), -1, dtype=first.dtype)])
    first = first.to(logits.device)
    src = (first >= 0).nonzero().squeeze(1)
    dst = first[src]
    p = logits.softmax(-1)
    out = torch.zeros(M, 256, device=logits.device, dtype=p.dtype)
    out.index_add_(1, dst, p[:, src])
    return out.clamp_min(torch.finfo(p.dtype).tiny).log()      # ~-inf for empty buckets, no nan grads


def doc_tensors(data, spans, bmap, pad_id):
    """Precomputed per-doc constants: padded chunk ids, base sums come from the caller's emb_ext,
    and the (position m, target byte) prediction pairs."""
    ids = pad_chunk_ids([b for _, b in spans], bmap, pad_id)
    ms, bs = [], []
    for m in range(len(spans) - 1):
        b = int(data[spans[m + 1][0]])
        if b in bmap:
            ms.append(m); bs.append(b)
    return ids, torch.tensor(ms, dtype=torch.long), torch.tensor(bs, dtype=torch.long)
