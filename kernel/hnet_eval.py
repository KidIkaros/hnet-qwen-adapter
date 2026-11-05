"""Honest eval of the constructed H-Net-of-Qwen: fit on some docs, eval held-out.

    colab run --gpu A100 kernel/hnet_eval.py

Fits w_in/w_out/w_main/dechunk on decontam docs [0:6], evals byte bpb on
held-out docs [6:10] + val.md.  Grid: w_main on/off x prev/retrieve x
boundary rules.  This is the generalization test — do the closed-form
adapters transfer to unseen text?
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
hn = HNetQwen(model, tok, bmap, dec_layers=14).cuda()
head = ByteHead(tok)
text = model.model if hasattr(model, "model") else model.language_model
if not hasattr(text, "layers"):
    text = text.language_model

all_docs = sorted(glob.glob("data/decontam/*.bin"))
fit_docs = [open(f, "rb").read() for f in all_docs[:6]]
eval_docs = [open(f, "rb").read()[:8192] for f in all_docs[6:10]]
eval_docs.append(open("data/val.md", "rb").read()[:8192])
print(f"[eval] fit {len(fit_docs)} | held-out {len(eval_docs)} docs", flush=True)

lin_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "linear_attention"]
attn_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "full_attention"]
dec_start, main_start = lin_idx[-14], attn_idx[0]
P = 6

Xin, Yin, Xout, Yout, Xmain, Ymain, G, Rres = [], [], [], [], [], [], [], []
for di, d in enumerate(fit_docs[:4]):
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    hsl = o.hidden_states
    emb = text.embed_tokens(tids)
    Xin.append(emb.float()); Yin.append(hsl[dec_start][0].float())
    Yout.append(text.norm(hsl[-1][0]).float())
    Ymain.append(hsl[main_start][0].float())
    del o, hsl
    torch.cuda.empty_cache()
    hs = emb[None]
    for layer in hn.enc:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    enc = hs[0]
    n = len(d) // P
    bounds = [b - 1 for b in range(P, n * P + 1, P)]
    Xmain.append(enc[torch.tensor(bounds, device="cuda")].float())
    del hs, enc
    torch.cuda.empty_cache()

Xin, Yin, Yout, Xmain, Ymain = (torch.cat(x) for x in (Xin, Yin, Yout, Xmain, Ymain))
# decoder-output pairs: run dec on real dec inputs
dec_out = []
for d in fit_docs[:4]:
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    hs = o.hidden_states[dec_start][0][None].to(torch.bfloat16)
    for layer in hn.dec:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    dec_out.append(text.norm(hs[0]).float())
    del o
    torch.cuda.empty_cache()
Xout = torch.cat(dec_out)

hn.w_in = ridge_to_bpe(Xin[:200_000].to(torch.bfloat16), Yin[:200_000].to(torch.bfloat16))
hn.w_out = ridge_to_bpe(Xout[:200_000].to(torch.bfloat16), Yout[:len(Xout)].to(torch.bfloat16))
# fix Ymain pairing: recompute per-doc
Ymain_p = []
off = 0
for d in fit_docs[:4]:
    n = len(d) // P
    Ymain_p.append(Ymain[off + P - 1:off + n * P:P]); off += len(d)
Ymain_p = torch.cat(Ymain_p)
hn.w_main = ridge_to_bpe(Xmain[:200_000].to(torch.bfloat16), Ymain_p[:200_000].to(torch.bfloat16))

for d in fit_docs[:4]:
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    hs = text.embed_tokens(tids[None])
    for layer in hn.enc:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    enc = hs[0]
    n = len(d) // P
    patch = hn.w_main(enc[torch.tensor([b - 1 for b in range(P, n * P + 1, P)],
                                      device="cuda")][None])
    glob = hn.run_stack(hn.main, patch)[0]
    resid = hn.w_in(text.embed_tokens(tids)) - text.embed_tokens(tids)
    for ci in range(1, n):
        for p_ in range(ci * P, min((ci + 1) * P, len(d))):
            G.append(glob[ci - 1]); Rres.append(resid[p_])
    del patch, glob, enc, hs
    torch.cuda.empty_cache()
hn.dechunk.weight.data.copy_(ridge_to_bpe(
    torch.stack(G).to(torch.bfloat16), torch.stack(Rres).to(torch.bfloat16))(
    torch.eye(hn.hidden, device="cuda", dtype=torch.bfloat16)).T)
print(f"[eval] adapters fit | dechunk pairs {len(G)}", flush=True)


def eval_doc(byte_ids, bnd, wm, ctx):
    hn.w_main = wm
    T = len(byte_ids)
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=bnd, ctx=ctx, head_first=head.first)
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


for di, d in enumerate(eval_docs):
    T = min(len(d), 8192)
    byte_ids = torch.tensor(list(d[:T]), device="cuda")
    b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
    bs = spacelike(d[:T], T)
    for wm, tag_m in ((None, "no_main"), (hn.w_main, "+w_main")):
        for ctx, tag_c in (("prev", "prev"), ("retrieve", "retr")):
            p6 = eval_doc(byte_ids, b6, wm, ctx)
            sl = eval_doc(byte_ids, bs, wm, ctx)
            print(f"  doc{di} {tag_m:<8} {tag_c:<5} periodic {p6:.3f} | "
                  f"spacelike {sl:.3f}", flush=True)
    del byte_ids
    torch.cuda.empty_cache()
