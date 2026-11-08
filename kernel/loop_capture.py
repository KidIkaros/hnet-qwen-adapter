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
from src.probe_common import (RESULTS, build_model, dup_class,     # noqa: E402
                              forward_bp, load_ckpt, stream_toolace)

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

# capture last-position MLP output per layer (write-side direction)
CUR = {"x": [None] * L}
hooks = []
for i, layer in enumerate(hn.layers):
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        continue

    def mk(i):
        def f(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            CUR["x"][i] = o[0, -1].detach().float().cpu()
        return f
    hooks.append(mlp.register_forward_hook(mk(i)))
tapped = len(hooks)
print(f"[capture] tapped {tapped}/{L} layer MLPs (output)", flush=True)

items = stream_toolace(SKIP, N_ITEMS)


sums = {c: [np.zeros(HID, dtype=np.float64) for _ in range(L)]
        for c in ("argdup", "runaway", "pre", "clean")}
cnt = dict.fromkeys(sums, 0)
rows = []

for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())
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
    onset, pheno = dup_class(text)
    for t, snap in enumerate(steps):
        if pheno is None:
            cls = "clean"
        elif t >= onset:
            cls = pheno
        else:
            cls = "pre"
        cnt[cls] += 1
        for li, x in enumerate(snap):
            if x is not None:
                sums[cls][li] += x.numpy()[:HID]
    rows.append({"k": k, "gold": it["gold"], "pheno": pheno,
                 "onset": onset, "emit": text.decode("utf-8", "replace")[:300]})
    print(f"  item {k:2d} {f'{pheno}@{onset}' if pheno else 'clean':14s} "
          f"{text[:60]!r}", flush=True)

np.savez(f"{RESULTS}/loop_sums_v2_{int(time.time())}.npz",
         **{f"{c}_{i}": s for c, arr in sums.items()
            for i, s in enumerate(arr)},
         **{f"n_{c}": cnt[c] for c in cnt})
json.dump(rows, open(f"{RESULTS}/loop_rows_v2_{int(time.time())}.json",
                     "w"))
print(f"[capture] {dict((c, cnt[c]) for c in cnt)} steps | "
      f"items {len(rows)}", flush=True)
print("[capture] complete", flush=True)
