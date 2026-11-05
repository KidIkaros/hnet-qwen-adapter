"""Adapter data-scaling: do the closed-form maps improve with more fit docs?

    colab run --gpu A100 kernel/hnet_dscale.py

Same ridge recipe, three fit-set sizes {4, 12, 24} decontam docs, always
eval'd on held-out docs[30:36].  Best config: (6,8) slice, retrieve n_ctx=1,
+w_main.  Subsamples rows to keep memory bounded — the ridge caps at 200k
rows anyway, so coverage not volume is what scales.
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
P = 6

all_docs = sorted(glob.glob("data/decontam/*.bin"))
eval_docs = [open(f, "rb").read()[:8192] for f in all_docs[30:36]]
fit_pool = [open(f, "rb").read() for f in all_docs[:24]]
print(f"[dscale] fit pool {len(fit_pool)} | held-out {len(eval_docs)}", flush=True)

hn = HNetQwen(model, tok, bmap, enc_layers=6, dec_layers=8).cuda()
lin_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "linear_attention"]
attn_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "full_attention"]
dec_start, main_start = lin_idx[-8], attn_idx[0]


def harvest(d):
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    emb = text.embed_tokens(tids)
    hsl = o.hidden_states
    # subsampled pair rows — coverage over volume
    rows = (emb[::2].float(), hsl[dec_start][0][::2].float(),
            text.norm(hsl[-1][0])[::2].float(), hsl[main_start][0].float())
    del o, hsl
    torch.cuda.empty_cache()
    return emb, rows


def fit(n_docs):
    """Fit adapters on first n_docs of the pool (sequential for leakage-free
    scaling), with per-doc second pass for enc/dec/dechunk pairs."""
    Xin, Yin, Yout, Ymain = [], [], [], []
    cache = []
    for d in fit_pool[:n_docs]:
        emb, rows = harvest(d)
        Xin.append(rows[0]); Yin.append(rows[1])
        Yout.append(rows[2]); Ymain.append(rows[3])
        cache.append(emb)
    Xin, Yin, Yout, Ymain = (torch.cat(x) for x in (Xin, Yin, Yout, Ymain))
    hn.w_in = ridge_to_bpe(Xin[:200_000].to(torch.bfloat16),
                           Yin[:200_000].to(torch.bfloat16))
    # second pass: enc outputs, dec outputs on real inputs, dechunk pairs
    dec_out, Xmain, Xenc, Ymain_p, G, Rres = [], [], [], [], [], []
    off = 0
    for emb in cache:
        n = emb.shape[0] // P
        Ymain_p.append(Ymain[off + P - 1:off + n * P:P]); off += emb.shape[0]
        hs = emb[None]
        for layer in hn.enc:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        enc = hs[0]
        bounds = [b - 1 for b in range(P, n * P + 1, P)]
        Xmain.append(enc[torch.tensor(bounds, device="cuda")].float())
        Xenc.append(enc[::2].float())
        del hs, enc
        torch.cuda.empty_cache()
    Xmain, Xenc, Ymain_p = (torch.cat(x) for x in (Xmain, Xenc, Ymain_p))
    # decoder out on real dec-input (subsampled)
    for d in fit_pool[:min(n_docs, 4)]:
        tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
        with torch.no_grad():
            o = text(input_ids=tids[None], output_hidden_states=True)
        hs = o.hidden_states[dec_start][0][None].to(torch.bfloat16)
        for layer in hn.dec:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
        dec_out.append(text.norm(hs[0])[::2].float())
        del o
        torch.cuda.empty_cache()
    Xout = torch.cat(dec_out)
    hn.w_out = ridge_to_bpe(Xout[:200_000].to(torch.bfloat16),
                            Yout[:len(Xout)].to(torch.bfloat16))
    hn.w_main = ridge_to_bpe(Xmain[:200_000].to(torch.bfloat16),
                             Ymain_p[:200_000].to(torch.bfloat16))
    enc_resid = (Yin[:len(Xenc)] - Xin[:len(Xenc)]).to(torch.bfloat16)
    hn.w_enc = ridge_to_bpe(Xenc[:200_000].to(torch.bfloat16), enc_resid[:200_000])
    for emb in cache[:min(n_docs, 8)]:
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
            for p_ in range(ci * P, min((ci + 1) * P, emb.shape[0]), 2):
                G.append(glob[ci - 1]); Rres.append(resid[p_])
        del patch, glob, enc, hs
        torch.cuda.empty_cache()
    hn.dechunk.weight.data.copy_(ridge_to_bpe(
        torch.stack(G).to(torch.bfloat16), torch.stack(Rres).to(torch.bfloat16))(
        torch.eye(hn.hidden, device="cuda", dtype=torch.bfloat16)).T)
    return len(G)


def eval_bpb(byte_ids):
    T = len(byte_ids)
    b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=b6, ctx="retrieve", n_ctx=1,
                      head_first=head.first)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return float(nll.sum()) / np.log(2) / (T - 1)


for n in (4, 12, 24):
    gp = fit(n)
    row = [eval_bpb(torch.tensor(list(d), device="cuda")) for d in eval_docs]
    print(f"  fit_docs={n:<3} pairs={gp:<6} held-out bpb {np.mean(row):.4f} "
          f"({['%.3f' % r for r in row]})", flush=True)
    del hn.w_in, hn.w_out, hn.w_main, hn.w_enc
    hn.w_in = hn.w_out = hn.w_main = hn.w_enc = None
