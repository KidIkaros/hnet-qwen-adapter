"""Honest eval of the constructed H-Net-of-Qwen: fit on some docs, eval held-out.

    colab run --gpu A100 kernel/hnet_eval.py

Use case only — all composition/fitting/eval wiring lives in src/factory.py.
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
from src.factory import (build_hnet, doc_bpb, fit_adapters,        # noqa: E402
                         load_trunk, periodic, spacelike)

P = 6
trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=14)

all_docs = sorted(glob.glob("data/decontam/*.bin"))
fit_docs = [open(f, "rb").read() for f in all_docs[:6]]
eval_docs = [open(f, "rb").read()[:8192] for f in all_docs[6:10]]
print(f"[eval] fit {len(fit_docs)} | held-out {len(eval_docs)} docs",
      flush=True)
g = fit_adapters(hn, trunk, meta, fit_docs[:4], P=P)
print(f"[eval] adapters fit | dechunk pairs {g}", flush=True)

for di, d in enumerate(eval_docs):
    T = min(len(d), 8192)
    byte_ids = torch.tensor(list(d[:T]), device="cuda")
    b6, bs = periodic(T, P), spacelike(d[:T], T)
    for wm, tag_m in ((None, "no_main"), (hn.w_main, "+w_main")):
        hn.w_main = wm
        for ctx, tag_c in (("prev", "prev"), ("retrieve", "retr"),
                           ("own", "own")):
            p6 = doc_bpb(hn, byte_ids, trunk["head"], b6, ctx)
            sl = doc_bpb(hn, byte_ids, trunk["head"], bs, ctx)
            print(f"  doc{di} {tag_m:<8} {tag_c:<5} periodic {p6:.3f} | "
                  f"spacelike {sl:.3f}", flush=True)
    del byte_ids
    torch.cuda.empty_cache()
