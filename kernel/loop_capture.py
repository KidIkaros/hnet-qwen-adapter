"""Loop-direction capture for WSP loop suppression.

Emits calls on the pinned exam via greedy byte decode. For every decode step
captures each trunk layer's MLP input at the last position. Positions are
labeled LOOP (after duplication onset inside a looping emit) or CLEAN (all
steps of non-looping emits). Accumulates per-layer mean states -> npz, the
raw material for finding the direction the trunk writes repetition with.

    colab exec -s hrun7 -f kernel/loop_capture.py
    env: N_ITEMS (60), SKIP (1224), GEN_LEN (300)
"""
import os, sys, subprocess, json, time, re

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

import numpy as np                                                  # noqa: E402
import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, forward_bp,    # noqa: E402
                              load_ckpt, stream_toolace)

N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
SKIP = int(os.environ.get("SKIP", "1224"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()
L = len(hn.layers)
for layer in hn.layers:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        HID = layer.mlp.gate_proj.in_features
        break
else:
    raise SystemExit("no mlp.gate_proj found on trunk layers")

# capture last-position MLP input per layer
CUR = {"x": [None] * L}
hooks = []
for i, layer in enumerate(hn.layers):
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        continue

    def mk(i):
        def f(m, inp):
            CUR["x"][i] = inp[0][0, -1].detach().float().cpu()
        return f
    hooks.append(mlp.register_forward_pre_hook(mk(i)))
tapped = sum(x is not None for x in hooks)
print(f"[capture] tapped {tapped}/{L} layer MLPs", flush=True)

items = stream_toolace(SKIP, N_ITEMS)
ARG = re.compile(rb'(\w+)="')


def dup_onset(text: bytes):
    """Byte offset where duplication starts, else None."""
    seen, pos = set(), None
    for m in ARG.finditer(text):
        if m.group(1) in seen:
            pos = m.start()
            break
        seen.add(m.group(1))
    m2 = re.search(rb"\]\s*\[", text)
    if m2 and (pos is None or m2.start() < pos):
        pos = m2.start()
    if pos is None and len(text) > 240 and b"]" not in text:
        pos = 240
    return pos


sum_loop = [np.zeros(HID, dtype=np.float64) for _ in range(L)]
sum_clean = [np.zeros(HID, dtype=np.float64) for _ in range(L)]
n_loop = n_clean = 0
rows = []

for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())
    # collect last-pos MLP input per step
    steps = []
    ids = list(pid)
    for _ in range(GEN_LEN):
        CUR["x"] = [None] * L
        bp = forward_bp(hn, trunk, ids)
        steps.append([x.clone() if x is not None else None
                      for x in CUR["x"]])
        ids.append(int(bp[0, -1].argmax()))
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    text = bytes(ids[len(pid):])
    onset = dup_onset(text)
    loop = onset is not None
    for t, snap in enumerate(steps):
        tgt = sum_loop if (loop and t >= onset) else sum_clean
        if loop and t >= onset:
            n_loop += 1
        elif not loop:
            n_clean += 1
        else:
            continue  # pre-onset steps of loop items: neither class
        for li, x in enumerate(snap):
            if x is not None:
                tgt[li] += x.numpy()[:HID]
    rows.append({"k": k, "gold": it["gold"], "loop": loop,
                 "onset": onset, "emit": text.decode("utf-8", "replace")[:300]})
    print(f"  item {k:2d} {'LOOP@' + str(onset) if loop else 'clean':10s} "
          f"{text[:60]!r}", flush=True)

np.savez(f"{RESULTS}/loop_sums_{int(time.time())}.npz",
         **{f"sum_loop_{i}": s for i, s in enumerate(sum_loop)},
         **{f"sum_clean_{i}": s for i, s in enumerate(sum_clean)},
         n_loop=n_loop, n_clean=n_clean)
json.dump(rows, open(f"{RESULTS}/loop_rows_{int(time.time())}.json", "w"))
print(f"[capture] loop {sum(r['loop'] for r in rows)}/{len(rows)} | "
      f"loop-steps {n_loop} clean-steps {n_clean}", flush=True)
print("[capture] complete", flush=True)
