"""Composition root: the ONE place that wires HF detail -> the H-Net entity.

Every kernel was re-implementing "load trunk -> slice -> fit adapters ->
eval" — the dependency-rule violation.  This module owns the wiring; the
kernels become one-line use cases.

    trunk = load_trunk("Qwen/Qwen3.5-2B-Base")
    hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
    fit_adapters(hn, trunk, meta, fit_docs)
    bpb = doc_bpb(hn, byte_ids, trunk["head"], periodic(T, P))
"""
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import ByteHead
from .construct import ridge_to_bpe
from .corpus import byte_token_map
from .hnet import HNetQwen


def load_trunk(model_name="Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16,
               freeze=True):
    """HF detail -> the loaded checkpoint + byte plumbing (drivers edge)."""
    src = snapshot_download(model_name)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).cuda().eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    text = model.model if hasattr(model, "model") else model.language_model
    if not hasattr(text, "layers"):
        text = text.language_model
    return {"model": model, "tok": tok, "bmap": byte_token_map(tok),
            "head": ByteHead(tok), "text": text}


def build_hnet(trunk, enc_layers=6, dec_layers=8):
    """Re-slice the checkpoint into the H-Net (entity construction)."""
    hn = HNetQwen(trunk["model"], trunk["tok"], trunk["bmap"],
                  enc_layers=enc_layers, dec_layers=dec_layers).cuda()
    lin = [i for i, l in enumerate(hn.layers)
           if l.block_type == "linear_attention"]
    attn = [i for i, l in enumerate(hn.layers)
            if l.block_type == "full_attention"]
    return hn, {"dec_start": lin[-dec_layers], "main_start": attn[0],
                "lin_idx": lin, "attn_idx": attn}


def harvest(trunk, doc):
    """One no-grad trunk pass -> (byte embeddings, all hidden states)."""
    text, bmap = trunk["text"], trunk["bmap"]
    tids = torch.tensor([bmap.get(int(b), 0) for b in doc], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    emb = text.embed_tokens(tids)
    return emb, o.hidden_states


def _enc_out(hn, emb):
    """Encoder slice output for a doc's byte embeddings."""
    hs = emb[None]
    for layer in hn.enc:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    return hs[0]


def _chunk_ends(n_pos, P):
    return [b - 1 for b in range(P, (n_pos // P) * P + 1, P)]


def fit_adapters(hn, trunk, meta, fit_docs, P=6, max_rows=200_000,
                 stride=1):
    """The closed-form recipe — harvest pairs on fit_docs, ridge-fit all
    five adapters in place: w_in/w_out/w_main/w_enc + dechunk weight."""
    text, bmap = trunk["text"], trunk["bmap"]
    dec_start, main_start = meta["dec_start"], meta["main_start"]
    Xin, Yin, Yout, Ymain, Xmain, Xenc = [], [], [], [], [], []
    cache = []
    for d in fit_docs:
        emb, hsl = harvest(trunk, d)
        cache.append((emb, len(d)))
        Xin.append(emb[::stride].float())
        Yin.append(hsl[dec_start][0][::stride].float())
        Yout.append(text.norm(hsl[-1][0])[::stride].float())
        Ymain.append(hsl[main_start][0].float())           # chunk-end pairing
        enc = _enc_out(hn, emb)
        Xenc.append(enc[::stride].float())
        Xmain.append(enc[torch.tensor(_chunk_ends(len(d), P),
                                      device="cuda")].float())
        del hsl, enc
        torch.cuda.empty_cache()
    Xin, Yin, Yout, Ymain, Xmain, Xenc = (
        torch.cat(x) for x in (Xin, Yin, Yout, Ymain, Xmain, Xenc))
    hn.w_in = ridge_to_bpe(Xin[:max_rows].to(torch.bfloat16),
                           Yin[:max_rows].to(torch.bfloat16))
    # decoder-output pairs on real dec-inputs (stride-aligned with Yout)
    Xout = []
    for i, (emb, n) in enumerate(cache[:4]):
        d_hsl = harvest(trunk, fit_docs[i])[1]
        hs = d_hsl[dec_start][0][None].to(torch.bfloat16)
        for layer in hn.dec:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        Xout.append(text.norm(hs[0])[::stride].float())
        del d_hsl, hs
        torch.cuda.empty_cache()
    Xout = torch.cat(Xout)
    hn.w_out = ridge_to_bpe(Xout[:max_rows].to(torch.bfloat16),
                            Yout[:len(Xout)].to(torch.bfloat16))
    # w_main pairs: chunk-end positions of the main-input stream
    Ymain_p, off = [], 0
    for _, n in cache:
        m = n // P
        Ymain_p.append(Ymain[off + P - 1:off + m * P:P]); off += n
    Ymain_p = torch.cat(Ymain_p)
    hn.w_main = ridge_to_bpe(Xmain[:max_rows].to(torch.bfloat16),
                             Ymain_p[:max_rows].to(torch.bfloat16))
    enc_resid = (Yin[:len(Xenc)] - Xin[:len(Xenc)]).to(torch.bfloat16)
    hn.w_enc = ridge_to_bpe(Xenc[:max_rows].to(torch.bfloat16),
                            enc_resid[:max_rows])
    # dechunk: prev-chunk glob -> (w_in(emb) - emb) residual per byte
    G, Rres = [], []
    for emb, n in cache[:8]:
        enc = _enc_out(hn, emb)
        m = n // P
        patch = hn.w_main(enc[torch.tensor(_chunk_ends(n, P),
                                           device="cuda")][None])
        glob = hn.run_stack(hn.main, patch)[0]
        resid = hn.w_in(emb) - emb
        for ci in range(1, m):
            for p_ in range(ci * P, min((ci + 1) * P, n), stride):
                G.append(glob[ci - 1]); Rres.append(resid[p_])
        del enc, patch, glob
        torch.cuda.empty_cache()
    hn.dechunk.weight.data.copy_(ridge_to_bpe(
        torch.stack(G).to(torch.bfloat16),
        torch.stack(Rres).to(torch.bfloat16))(
        torch.eye(hn.hidden, device="cuda", dtype=torch.bfloat16)).T)
    return len(G)


def doc_bpb(hn, byte_ids, head, boundaries, ctx="retrieve", n_ctx=1,
            super_k=1):
    """Held-out byte bpb on one doc — the shared scorer."""
    T = len(byte_ids)
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=boundaries, ctx=ctx,
                      head_first=head.first, n_ctx=n_ctx, super_k=super_k)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return float(nll.sum()) / np.log(2) / (T - 1)


def periodic(T, P, device="cuda"):
    b = torch.zeros(T, device=device); b[::P] = 1.0
    return b


def spacelike(data, T, device="cuda"):
    """Boundary at word-space transitions — SpaceByte's heuristic."""
    b = torch.zeros(T, device=device); prev = True
    for i, x in enumerate(data):
        s = not (48 <= x <= 57 or 65 <= x <= 90 or 97 <= x <= 122
                 or 128 <= x <= 191)
        if s and not prev:
            b[i] = 1.0
        prev = s
    b[0] = 1.0
    return b
