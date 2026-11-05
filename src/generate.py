"""Autoregressive byte generation for the composed system — the scorer becomes a sampler.

Per chunk: the frozen trunk (fed projected patch vectors) proposes the boundary byte via the
byte-marginalized head; the ByteDecoder then generates the chunk's interior bytes conditioned on
the trunk context and the previous chunk's bytes (the pair window it was trained with).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _sample(logits, temperature, top_k, rng):
    if top_k and top_k < logits.numel():
        k = torch.topk(logits, top_k).values[-1]
        logits = logits.clamp_min(k)
    p = (logits / temperature).softmax(-1)
    return int(rng.choice(256, p=p.double().cpu().numpy() / p.double().sum().item()))


def generate(model, dec, proj, head, emb_ext, bmap, trunk_ctx, ctx_pairs,
             seed=b"", n_chunks=12, P=6, hist=None, temperature=0.8, top_k=64,
             pad_id=None, seed_int=0):
    """seed -> bytes.  Periodic-P chunking; each chunk = boundary byte (trunk) + P-1 interior
    bytes (decoder).  hist = bytes of local context the decoder sees (defaults to P, the pair
    window it trained with)."""
    hist = P if hist is None else hist
    rng = np.random.default_rng(seed_int)
    out = bytearray(seed)
    chunks = [bytes(seed[i:i + P]) for i in range(0, len(seed) - len(seed) % P, P)]
    patches = []
    for c in chunks:
        ids = [bmap[b] for b in c if b in bmap]
        patches.append(emb_ext[torch.tensor(ids or [pad_id], device="cuda")].sum(0))
    sos = dec.byte_emb.num_embeddings - 1

    for _ in range(n_chunks):
        base = torch.stack(patches) if patches else torch.zeros(0, emb_ext.shape[1], device="cuda")
        ctx, hs, lg = trunk_ctx(proj(base))                    # (M,D),(M,V); last pos predicts next
        b0 = _sample(head.byte_probs(lg[-1:].float())[0], temperature, top_k, rng)
        chunk = bytearray([b0])
        # (1,2,D) pair for the NEW chunk: [prev chunk's ctx = ctx[-1], this chunk's ctx = hs[-1]]
        ctx2 = torch.stack([ctx[-1], hs[-1]]).unsqueeze(0)     # ctx[-1] is the zeros slot at M=0
        for _k in range(P - 1):
            # the pair-window layout the decoder trained with: [SOS][prev chunk][this chunk so far]
            prev = bytes(out[-P:])
            x = torch.tensor([[sos] + list(prev) + list(chunk)], device="cuda")
            pad = torch.zeros(1, x.shape[1], dtype=torch.bool, device="cuda")
            with torch.no_grad():
                logits = dec(ctx2, x, pad)
            chunk.append(_sample(logits[0, -1].float(), temperature, top_k, rng))
        out += chunk
        ids = [bmap[b] for b in bytes(chunk) if b in bmap]
        patches.append(emb_ext[torch.tensor(ids or [pad_id], device="cuda")].sum(0))
    return bytes(out)
