"""WSP loop-edit iteration 1: direction subtraction on MLP down_proj columns.

Diagnosis input: loop_sums_*.npz from loop_capture (fit on SKIP=1224 items).
  d_L = mean(MLP-in | loop) - mean(MLP-in | clean), per layer.
  Edit: for each chosen layer, take the K down_proj output columns most
  cosine-aligned with d_L and subtract their projection onto d_L:
      W[:,j] <- W[:,j] - (W[:,j] @ d_hat) * d_hat
Verify on a DIFFERENT slice (SKIP=2500): emit before vs after, same items.
Metrics: loop count, closed-call rate, sample emits. No retraining.

    colab exec -s hrun7 -f kernel/loop_edit.py
    env: VERIFY_SKIP (2500), N_ITEMS (40), TOP_LAYERS (2), TOPK (64), GEN_LEN
"""
import os, sys, subprocess, json, time, re, glob

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

VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))
TOP_LAYERS = int(os.environ.get("TOP_LAYERS", "2"))
TOPK = int(os.environ.get("TOPK", "64"))

npz = sorted(glob.glob(f"{RESULTS}/loop_sums_v2_*.npz"))[-1]
Z = np.load(npz)
L = sum(1 for k in Z.files if k.startswith("argdup_"))
PHENOS = ("argdup", "runaway")
print(f"[dir] {npz} | " + " ".join(
    f"{c}={int(Z[f'n_{c}'])}" for c in (*PHENOS, "pre", "clean")), flush=True)


def direction(cls, base="pre"):
    """mean(cls) - mean(base), normalized, per layer."""
    out, norms = [], []
    for i in range(L):
        d = Z[f"{cls}_{i}"] / max(int(Z[f"n_{cls}"]), 1) - \
            Z[f"{base}_{i}"] / max(int(Z[f"n_{base}"]), 1)
        norms.append(float(np.linalg.norm(d)))
        out.append(d / max(np.linalg.norm(d), 1e-9))
    return out, norms


trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

# score each phenotype direction by best down_proj column alignment
best = None
for cls in PHENOS:
    dirs, norms = direction(cls)
    order = np.argsort(-np.array(norms))
    li = int(order[0])
    W = hn.layers[li].mlp.down_proj.weight
    d = torch.tensor(dirs[li], device=W.device, dtype=W.dtype)
    cos = (W.T @ d) / (W.T.norm(dim=1) + 1e-9)
    mc = float(cos.abs().max())
    print(f"[dir] {cls}: top layer L{li} norm {norms[li]:.1f} "
          f"max|cos| {mc:.3f} | top8 "
          + " ".join(f"L{i}={norms[i]:.1f}" for i in order[:8]), flush=True)
    if best is None or mc > best[0]:
        best = (mc, cls, dirs, norms)

mc, CLS, dirs, norms = best
print(f"[dir] using {CLS} direction (max|cos| {mc:.3f})", flush=True)
order = np.argsort(-np.array(norms))
chosen = [int(i) for i in order[:TOP_LAYERS]]
print(f"[dir] editing layers {chosen}", flush=True)

ARG = re.compile(rb'(\w+)="')


def dup_onset(text: bytes):
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


@torch.no_grad()
def emit(pid):
    ids = list(pid)
    for _ in range(GEN_LEN):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return bytes(ids[len(pid):])


items = stream_toolace(VSKIP, N_ITEMS)
prompts = [f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode()
           for it in items]


def run(tag):
    loops = closed = 0
    emits = []
    for k, pid in enumerate(prompts):
        t = emit(list(pid))
        closed += b"]" in t
        loop = dup_onset(t) is not None
        loops += loop
        emits.append(t)
        if k < 12:
            print(f"  [{tag}] item {k:2d} "
                  f"{'LOOP' if loop else 'ok  '} {t[:55]!r}", flush=True)
    print(f"[{tag}] loops {loops}/{len(prompts)} closed {closed}",
          flush=True)
    return loops, closed, emits


loops0, closed0, e0 = run("base")

# ---- the edit ----
with torch.no_grad():
    for li in chosen:
        mlp = hn.layers[li].mlp
        W = mlp.down_proj.weight            # [hidden, inter]
        d = torch.tensor(dirs[li], device=W.device, dtype=W.dtype)
        cos = (W.T @ d) / (W.T.norm(dim=1) + 1e-9)   # per-column cosine
        cols = cos.abs().argsort(descending=True)[:TOPK]
        proj = W[:, cols].T @ d                        # [K]
        W[:, cols] -= d[:, None] * proj[None, :]       # [hidden,K]
        print(f"[edit] L{li} down_proj: {TOPK} cols, max|cos| "
              f"{cos.abs().max():.3f}", flush=True)

loops1, closed1, e1 = run("edit")

res = {"npz": npz, "layers": chosen, "topk": TOPK,
       "norms": norms, "base_loops": loops0, "edit_loops": loops1,
       "base_closed": closed0, "edit_closed": closed1,
       "n": len(prompts)}
json.dump(res, open(f"{RESULTS}/loop_edit_{int(time.time())}.json", "w"))
print(f"[result] loops {loops0}->{loops1} closed {closed0}->{closed1} "
      f"(n={len(prompts)})", flush=True)
print("[loop_edit] complete", flush=True)
