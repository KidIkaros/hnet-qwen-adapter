"""Jev score collector: for each toolace item, score EVERY declared tool by
model logprob of "[name(" — saves raw score vectors + gold + tool lists for
OFFLINE decision-layer work (temperature fit, conformal sets, verdicts).

    colab exec -s hrun6 -f kernel/jev_score_collect.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (90), SKIP (2500)
"""
import os, sys, glob, subprocess
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
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import json, time                                                # noqa: E402
from src.probe_common import (RESULTS, build_model, cand_score,  # noqa: E402
                              load_ckpt, stream_toolace)

trunk, hn, meta = build_model()
load_ckpt(hn)

N_ITEMS = int(os.environ.get("N_ITEMS", "90"))
SKIP = int(os.environ.get("SKIP", "2500"))

items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

rows = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    sc = [cand_score(hn, trunk, pid, list(f"[{t}(".encode()))
          for t in it["tools"]]
    rows.append({"tools": it["tools"], "gold": it["gold"], "scores": sc})
    if k % 10 == 0:
        print(f"  scored {k}", flush=True)

json.dump(rows, open(f"{RESULTS}/jev_scores.json", "w"))
print(f"[collect] {len(rows)} items -> results/jev_scores.json", flush=True)
print("[collect] complete", flush=True)
