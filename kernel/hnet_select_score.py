"""Constrained tool selection: score each declared tool by the model's own
byte logprob of emitting "[name(" after ASSISTANT: — argmax over the
DECLARED set only (hallucination impossible by construction).

    colab exec -s hrun6 -f kernel/hnet_select_score.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (15), SKIP (2500)
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

N_ITEMS = int(os.environ.get("N_ITEMS", "15"))
SKIP = int(os.environ.get("SKIP", "2500"))

items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

right_n, top3_n = 0, 0
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    sc = sorted(((t, cand_score(hn, trunk, pid, list(f"[{t}(".encode())))
                 for t in it["tools"]), key=lambda s: -s[1])
    rank = next((i for i, (t, _) in enumerate(sc) if t == it["gold"]), -1)
    right_n += rank == 0
    top3_n += 0 <= rank < 3
    print(f"  item {k:2d} gold={it['gold'][:24]:24s} rank={rank} "
          f"top={sc[0][0][:24]} ({sc[0][1]:.2f})", flush=True)

n = len(items)
res = {"n": n, "right": right_n, "top3": top3_n}
json.dump(res, open(f"{RESULTS}/select_score_{int(time.time())}.json", "w"))
print(f"[select] gold@1 {right_n}/{n} | gold@3 {top3_n}/{n}", flush=True)
print("[select] complete", flush=True)
