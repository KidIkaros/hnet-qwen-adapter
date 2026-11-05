"""Next-steps kernel: retrieval-depth sweep, enc/dec fraction sweep, generation.

    colab run --gpu A100 kernel/hnet_next.py

A) n_ctx {1,2,4} in decoder_attn_pass — does deeper retrieval help?
B) enc/dec fractions (4,14) vs (6,8) vs (8,6) — is the slice right?
C) generation sanity — decode 200 bytes, text or mojibake?
"""
import os, subprocess, sys

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import glob                                                      # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.adapter import ByteHead                                  # noqa: E402
from src.hnet import HNetQwen                                     # noqa: E402
from src.construct import ridge_to_bpe                            # noqa: E402

src = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
head = ByteHead(tok)
text = model.model if hasattr(model, "model") else model.language_model
if not hasattr(text, "layers"):
    text = text.language_model

all_docs = sorted(glob.glob("data/decontam/*.bin"))
fit_docs = [open(f, "rb").read() for f in all_docs[:6]]
eval_docs = [open(f, "rb").read()[:8192] for f in all_docs[6:10]]
P = 6


def harvest(d, text, bmap):
    """One full-trunk pass -> per-layer hidden states + emb for a doc."""
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    return text.embed_tokens(tids), o.hidden_states


def fit(hn, cache, bmap):
    """Fit w_in/w_out/w_main/w_enc/dechunk on cached harvests."""
    lin_idx = [i for i, l in enumerate(hn.layers)
               if l.block_type == "linear_attention"]
    attn_idx = [i for i, l in enumerate(hn.layers)
                if l.block_type == "full_attention"]
    dec_start, main_start = lin_idx[-len(hn.dec):][0], attn_idx[0]
    Xin = torch.cat([c[0].float() for c in cache])
    Yin = torch.cat([c[1][dec_start][0].float() for c in cache])
    Yout = torch.cat([text.norm(c[1][-1][0]).float() for c in cache])
    Ymain = torch.cat([c[1][main_start][0].float() for c in cache])
    hn.w_in = ridge_to_bpe(Xin[:200_000].to(torch.bfloat16),
                           Yin[:200_000].to(torch.bfloat16))
    # decoder outputs on real dec inputs
    dec_out = []
    for emb, hsl in cache:
        hs = hsl[dec_start][0][None].to(torch.bfloat16)
        for layer in hn.dec:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        dec_out.append(text.norm(hs[0]).float())
    Xout = torch.cat(dec_out)
    hn.w_out = ridge_to_bpe(Xout[:200_000].to(torch.bfloat16),
                            Yout[:len(Xout)].to(torch.bfloat16))
    # w_main: enc-out at chunk-end positions -> first-attn input
    Xmain, Ymain_p, Xenc = [], [], []
    G, Rres = [], []
    for emb, hsl in cache:
        hs = emb[None]
        for layer in hn.enc:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        enc = hs[0]
        n = emb.shape[0] // P
        bounds = [b - 1 for b in range(P, n * P + 1, P)]
        Xmain.append(enc[torch.tensor(bounds, device="cuda")].float())
        Xenc.append(enc.float())
    Xmain = torch.cat(Xmain); Xenc = torch.cat(Xenc)
    off = 0
    for emb, _ in cache:
        n = emb.shape[0] // P
        Ymain_p.append(Ymain[off + P - 1:off + n * P:P]); off += emb.shape[0]
    Ymain_p = torch.cat(Ymain_p)
    hn.w_main = ridge_to_bpe(Xmain[:200_000].to(torch.bfloat16),
                             Ymain_p[:200_000].to(torch.bfloat16))
    enc_resid = (Yin[:len(Xenc)] - Xin[:len(Xenc)]).to(torch.bfloat16)
    hn.w_enc = ridge_to_bpe(Xenc[:200_000].to(torch.bfloat16), enc_resid[:200_000])
    # dechunk: glob_prev -> (w_in(emb) - emb) residual
    for emb, hsl in cache:
        hs = emb[None]
        for layer in hn.enc:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        enc = hs[0]
        n = emb.shape[0] // P
        patch = hn.w_main(enc[torch.tensor([b - 1 for b in range(P, n * P + 1, P)],
                                          device="cuda")][None])
        glob = hn.run_stack(hn.main, patch)[0]
        resid = hn.w_in(emb) - emb
        for ci in range(1, n):
            for p_ in range(ci * P, min((ci + 1) * P, emb.shape[0])):
                G.append(glob[ci - 1]); Rres.append(resid[p_])
    hn.dechunk.weight.data.copy_(ridge_to_bpe(
        torch.stack(G).to(torch.bfloat16), torch.stack(Rres).to(torch.bfloat16))(
        torch.eye(hn.hidden, device="cuda", dtype=torch.bfloat16)).T)


cache = [harvest(d, text, bmap) for d in fit_docs[:4]]
print(f"[next] harvest cached for {len(cache)} docs", flush=True)

eval_ids = [torch.tensor(list(d[:8192]), device="cuda") for d in eval_docs]


def eval_bpb(hn, byte_ids, bnd, wm, ctx, n_ctx=1):
    hn.w_main = wm
    T = len(byte_ids)
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=bnd, ctx=ctx, head_first=head.first,
                      n_ctx=n_ctx)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return float(nll.sum()) / np.log(2) / (T - 1)


def spacelike(data_, T_):
    b = torch.zeros(T_, device="cuda"); prev = True
    for i, x in enumerate(data_):
        s = not (48 <= x <= 57 or 65 <= x <= 90 or 97 <= x <= 122 or 128 <= x <= 191)
        if s and not prev:
            b[i] = 1.0
        prev = s
    b[0] = 1.0
    return b


# ============ A) retrieval-depth sweep on the (4,14) slice ============
print("[A] n_ctx sweep — (4,14) slice, retrieve", flush=True)
hn = HNetQwen(model, tok, bmap, enc_layers=4, dec_layers=14).cuda()
fit(hn, cache, bmap)
for nc in (1, 2, 4):
    row = []
    for d in eval_docs[:2]:
        T = len(d); byte_ids = torch.tensor(list(d), device="cuda")
        b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
        row.append(eval_bpb(hn, byte_ids, b6, hn.w_main, "retrieve", n_ctx=nc))
    print(f"  n_ctx={nc}: {np.mean(row):.4f} ({['%.3f' % r for r in row]})",
          flush=True)

# ============ B) enc/dec fraction sweep ============
print("[B] enc/dec fraction sweep — retr +w_main", flush=True)
for enc_l, dec_l in ((4, 14), (6, 8), (8, 6)):
    hn = HNetQwen(model, tok, bmap, enc_layers=enc_l, dec_layers=dec_l).cuda()
    fit(hn, cache, bmap)
    row = []
    for d in eval_docs[:2]:
        T = len(d); byte_ids = torch.tensor(list(d), device="cuda")
        b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
        row.append(eval_bpb(hn, byte_ids, b6, hn.w_main, "retrieve"))
    print(f"  enc={enc_l} dec={dec_l}: {np.mean(row):.4f} "
          f"({['%.3f' % r for r in row]})", flush=True)

# ============ C) generation sanity ============
print("[C] generation — greedy byte decode, (4,14) slice", flush=True)
hn = HNetQwen(model, tok, bmap, enc_layers=4, dec_layers=14).cuda()
fit(hn, cache, bmap)
prompt = b"The H-Net of Qwen routes bytes through "
gen = list(prompt)
for step in range(120):
    byte_ids = torch.tensor(gen, device="cuda")
    T = len(byte_ids)
    b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=b6, ctx="retrieve",
                      head_first=head.first)
    gen.append(int(bp[0, -1].argmax()))
print(f"  prompt: {prompt!r}", flush=True)
print(f"  output: {bytes(gen[len(prompt):])!r}", flush=True)
