"""Trainer: end-to-end byte-NLL fine-tune of the re-sliced H-Net-of-Qwen.

    colab run --gpu A100 kernel/hnet_train.py   [env: MODEL, N_TRAIN, EPOCHS]

2B-Base default; enc/dec slices + dechunk + boundary + w_sup trainable,
emb/main/lm_head frozen.  Corpus: weighted dolma3.5 mix (factory).
Eval protocol: enwik8-test (leaderboard) + held decontam docs.
"""
import os, subprocess, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from src.factory import (DOLMA_MIX, build_hnet, corpus_stream,    # noqa: E402
                         doc_bpb, enwik8, fit_adapters, load_trunk,
                         periodic)

P = 6
trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
lin_idx = meta["lin_idx"]
print(f"[train] {len(hn.layers)} layers | DeltaNet {len(lin_idx)} | "
      f"enc 6 dec 8 (start {meta['dec_start']})", flush=True)

# ---- data: dolma mix train | decontam fit/held | enwik8 test eval
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_docs = [d[:4096] for d in all_docs[:4]]
held_docs = [d[:4096] for d in all_docs[38:44]]
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
train_docs = corpus_stream(DOLMA_MIX, N_TRAIN)
print(f"[data] {len(train_docs)} train (dolma mix) | {len(held_docs)} held | "
      f"{len(fit_docs)} adapter-fit", flush=True)
ew8 = enwik8()
ew8_test = torch.tensor(list(ew8["test"][:200_000]), device="cuda")
print(f"[data] enwik8 test slice {len(ew8_test)} bytes", flush=True)

g = fit_adapters(hn, trunk, meta, fit_docs, P=P)
print(f"[fit] adapters fit | {g} dechunk pairs", flush=True)

# ---- trainable: slices + glue (grad checkpointing on the big slices)
for layer in list(hn.enc) + list(hn.dec):
    layer.gradient_checkpointing = True
trainable = list(hn.enc.parameters()) + list(hn.dec.parameters()) + \
    list(hn.dechunk.parameters()) + list(hn.boundary.parameters()) + \
    list(hn.w_sup.parameters())
for p in trainable:
    p.requires_grad_(True)
n_train = sum(p.numel() for p in trainable)
print(f"[train] {n_train/1e6:.1f}M trainable", flush=True)
LR = float(os.environ.get("LR", "1e-4"))
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)


def doc_loss(byte_ids):
    T = len(byte_ids)
    bp, _, _ = hn(byte_ids, boundaries=periodic(T, P), ctx="retrieve",
                  n_ctx=1, super_k=4, head_first=trunk["head"].first)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    # boundary aux: mean boundary prob -> target 1/P (H-Net compression
    # ratio loss — without it, learned routing collapses to ~1.8 spans)
    aux = (hn.last_p.mean() - 1.0 / P) ** 2
    return nll.mean() + 1.0 * aux


def eval_bpb(byte_ids):
    with torch.no_grad():
        return float(doc_loss(byte_ids)) / np.log(2)


EPOCHS = int(os.environ.get("EPOCHS", "1"))
step = 0
for ep in range(EPOCHS):
    for d in train_docs:
        byte_ids = torch.tensor(list(d), device="cuda")
        loss = doc_loss(byte_ids)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        if step % 25 == 0 or step == len(train_docs):
            hb = eval_bpb(torch.tensor(list(held_docs[step % len(held_docs)]),
                                     device="cuda"))
            ew = eval_bpb(ew8_test[:50000])
            print(f"  ep{ep} step {step:<5} loss {float(loss):.3f} "
                  f"|gn| {float(gn):.2f} | held {hb:.3f} | enwik8 {ew:.3f}",
                  flush=True)
        del byte_ids, loss
        torch.cuda.empty_cache()

# ---- final eval: enwik8 test + held decontam
ew_final = eval_bpb(ew8_test)
hb_final = [eval_bpb(torch.tensor(list(hd), device="cuda"))
            for hd in held_docs]
print(f"[eval] enwik8-test bpb {ew_final:.4f}", flush=True)
print(f"[eval] held decontam {np.mean(hb_final):.4f} "
      f"({['%.3f' % v for v in hb_final]})", flush=True)
print("[train] complete", flush=True)
