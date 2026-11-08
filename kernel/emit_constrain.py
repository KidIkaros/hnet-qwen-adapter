"""Constrained-emit baseline (control for the WSP loop edit).

Decode-time suppression with NO weight changes:
  1. run-mask: if last 3 bytes identical, mask that byte (next-best logit)
  2. arg-dup stop: when emitted tail completes a repeated `name="`, truncate
     at the duplicate, append `]` if the call is unclosed, stop.

Same 40-item slice as loop_edit for direct comparison:
  base 21/40 loops, closed 29 | edit 17/40, closed 29-30

    colab exec -s hrun7 -f kernel/emit_constrain.py
    env: VERIFY_SKIP (2500), N_ITEMS (40), GEN_LEN (300)
"""
import os, sys, subprocess, json, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"FAILED {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, dup_class,     # noqa: E402
                              forward_bp, load_ckpt, stream_toolace)

VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

items = stream_toolace(VSKIP, N_ITEMS)
prompts = [f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode()
           for it in items]


@torch.no_grad()
def constrained_emit(pid):
    ids = list(pid)
    n0 = len(ids)
    for _ in range(GEN_LEN):
        bp = forward_bp(hn, trunk, ids)
        logits = bp[0, -1].clone()
        # run-mask: no >3 identical consecutive bytes
        if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
            logits[ids[-1]] = -1e9
        ids.append(int(logits.argmax()))
        tail = bytes(ids[n0:])
        onset, _ = dup_class(tail)
        if onset is not None and onset < len(tail):
            ids = ids[:n0 + onset]                     # truncate at dup
            if b"(" in tail[:onset] and b"]" not in tail[:onset]:
                ids.extend(b"]")                       # close the call
            break
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return bytes(ids[n0:])


loops = closed = 0
for k, pid in enumerate(prompts):
    t = constrained_emit(pid)
    closed += b"]" in t
    loop = dup_class(t)[0] is not None
    loops += loop
    if k < 14:
        print(f"  [con] item {k:2d} {'LOOP' if loop else 'ok  '} "
              f"{t[:55]!r}", flush=True)

res = {"mode": "constrained", "n": len(prompts), "loops": loops,
       "closed": closed}
json.dump(res, open(f"{RESULTS}/emit_constrain_{int(time.time())}.json",
                    "w"))
from src.edit_ledger import record                                # noqa: E402
record({"method": "decode-constraint",
        "target": "greedy_emit: run-mask + arg-dup stop",
        "teach_gain": f"loops {loops}/{len(prompts)}",
        "neighbor_cost": f"closed {closed}",
        "n_eval": len(prompts),
        "receipt": f"emit_constrain_{int(time.time())}.json"})
print(f"[result] loops {loops}/{len(prompts)} closed {closed} | "
      f"vs base 21/40 cl29, edit 17/40 cl29-30", flush=True)
print("[constrain] complete", flush=True)
