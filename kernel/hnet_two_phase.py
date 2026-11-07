"""Two-phase tool call: phase 1 scores declared tools by model logprob of
"[name(", phase 2 greedily emits the rest of the call. End-to-end probe:
does the model pick the right tool AND emit a complete call?

    colab exec -s hrun6 -f kernel/hnet_two_phase.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (40), SKIP (2500), GEN_LEN (400)
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

import torch                                                     # noqa: E402
import json, time                                               # noqa: E402
from src.probe_common import (CAP, RESULTS, build_model,         # noqa: E402
                              cand_score, forward_bp, load_ckpt,
                              stream_toolace)

trunk, hn, meta = build_model()
load_ckpt(hn)

N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
GEN_LEN = int(os.environ.get("GEN_LEN", "400"))
SKIP = int(os.environ.get("SKIP", "2500"))


@torch.no_grad()
def greedy_from(ids, max_new):
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return ids


items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

right_n = top3_n = emit_ok = 0
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    sc = sorted(((t, cand_score(hn, trunk, pid, list(f"[{t}(".encode())))
                 for t in it["tools"]), key=lambda s: -s[1])
    pred = sc[0][0]
    rank = next((i for i, (t, _) in enumerate(sc) if t == it["gold"]), -1)
    right_n += rank == 0; top3_n += 0 <= rank < 3
    ids = greedy_from(pid + list(f"[{pred}(".encode()), GEN_LEN)
    gen = bytes(ids[len(pid):]).decode("utf-8", "replace")
    closed = "]" in gen
    emit_ok += closed
    print(f"  item {k:2d} gold={it['gold'][:20]:20s} pred={pred[:20]:20s} "
          f"rank={rank} call={gen[:70]!r}", flush=True)

n = len(items)
res = {"n": n, "right": right_n, "top3": top3_n, "emit_closed": emit_ok}
json.dump(res, open(f"{RESULTS}/two_phase_{int(time.time())}.json", "w"))
print(f"[2phase] pick@1 {right_n}/{n} | @3 {top3_n}/{n} | "
      f"call-closed {emit_ok}/{n}", flush=True)
print("[2phase] complete", flush=True)
