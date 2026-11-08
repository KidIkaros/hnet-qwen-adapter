"""Pre-onset probe: can internal states predict loop onset early enough?

Captures per-step MLP-output states at the top separation layers
(L14/L21/L22 from loop_sums_v2) during greedy decode, labels every step
t by "a loop onset occurs within the next W steps" (causal: uses only
states at/before t), then trains a logistic probe split BY ITEM.

AUC > ~0.8 with W>=20 means awareness exists latently and early enough
to act on — the glial-gate substrate. Post-hoc-only signal shows as
AUC that collapses when W shrinks.

    colab exec -s hrun7 -f kernel/onset_probe.py
    env: N_ITEMS (60), VAL_SKIP (1300), VAL_ITEMS (40), W (24)
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

import numpy as np                                                  # noqa: E402
import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, dup_class,     # noqa: E402
                              forward_bp, load_ckpt, stream_toolace)

N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
SKIP = int(os.environ.get("SKIP", "1224"))
VSKIP = int(os.environ.get("VAL_SKIP", "1300"))
VITEMS = int(os.environ.get("VAL_ITEMS", "40"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))
W = int(os.environ.get("W", "24"))
LAYERS = [14, 21, 22]

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

CUR = {"x": {li: None for li in LAYERS}}
hooks = []
for li in LAYERS:
    mlp = hn.layers[li].mlp
    def mk(li):
        def f(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            CUR["x"][li] = o[0, -1].detach().float().cpu()
        return f
    hooks.append(mlp.register_forward_hook(mk(li)))
print(f"[probe] tapped layers {LAYERS}", flush=True)


def capture_slice(skip, n):
    """Per-item: steps x (layers*hid) fp16 states + onset step."""
    out = []
    for k, it in enumerate(stream_toolace(skip, n)):
        pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())
        ids = list(pid)
        steps = []
        for _ in range(GEN_LEN):
            CUR["x"] = {li: None for li in LAYERS}
            bp = forward_bp(hn, trunk, ids)
            steps.append(np.concatenate(
                [CUR["x"][li].numpy() for li in LAYERS]).astype(np.float16))
            ids.append(int(bp[0, -1].argmax()))
            if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
                break
        tail = bytes(ids[len(pid):])
        onset_b, pheno = dup_class(tail)
        onset = onset_b            # 1 step == 1 byte; byte offset IS step idx
        out.append({"steps": np.stack(steps), "onset": onset,
                    "pheno": pheno, "emit": tail[:200].decode("utf-8", "replace")})
        print(f"  [{skip}] item {k:2d} {str(pheno):8s}@{onset} "
              f"steps={len(steps)}", flush=True)
    return out


def build_xy(rows):
    """(X, y): step t is positive if an onset lands in (t, t+W]."""
    X, Y, G = [], [], []
    for gi, r in enumerate(rows):
        st = r["steps"]
        n = len(st)
        onset = r["onset"] or 10 ** 9
        for t in range(n):
            X.append(st[t])
            Y.append(1.0 if onset - W <= t < onset else 0.0)
            G.append(gi)
    return np.stack(X).astype(np.float32), np.array(Y), np.array(G)


t0 = time.time()
train_rows = capture_slice(SKIP, N_ITEMS)
val_rows = capture_slice(VSKIP, VITEMS)
np.savez(f"{RESULTS}/onset_states_{int(time.time())}.npz",
         allow_pickle=True,
         train=np.array(train_rows, dtype=object),
         val=np.array(val_rows, dtype=object))
print(f"[probe] captured {len(train_rows)}+{len(val_rows)} items "
      f"in {time.time()-t0:.0f}s", flush=True)

Xtr, ytr, gtr = build_xy(train_rows)
Xva, yva, gva = build_xy(val_rows)

# item-level split already holds (train vs val disjoint); probe = LR on GPU
dev = "cuda"
xt = torch.tensor(Xtr, device=dev)
yt = torch.tensor(ytr, device=dev)
xv = torch.tensor(Xva, device=dev)
w = torch.zeros(xt.shape[1], device=dev, requires_grad=True)
b = torch.zeros(1, device=dev, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=3e-3)
pos_w = (yt.numel() - yt.sum()) / yt.sum().clamp(min=1)
for ep in range(300):
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        xt @ w + b, yt, pos_weight=pos_w)
    loss.backward()
    opt.step()
with torch.no_grad():
    pv = torch.sigmoid(xv @ w + b).cpu().numpy()

# AUC
def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) \
        if n1 and n0 else float("nan")

print(f"[probe] AUC(W={W}) = {auc(yva, pv):.3f} | "
      f"pos-rate train {ytr.mean():.3f} val {yva.mean():.3f}",
      flush=True)
for q in (0.9, 0.95, 0.99):
    th = np.quantile(pv, q)
    sel = pv >= th
    if sel.sum():
        print(f"  top-{1-q:.0%}: prec {yva[sel].mean():.3f} "
              f"cov {sel.mean():.3f}", flush=True)

res = {"auc": auc(yva, pv), "W": W, "layers": LAYERS,
       "n_train_items": len(train_rows), "n_val_items": len(val_rows)}
json.dump(res, open(f"{RESULTS}/onset_probe_{int(time.time())}.json", "w"))
print("[probe] complete", flush=True)
