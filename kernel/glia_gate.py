"""Glial gate: learned slow-state field over captured loop-pressure states.

Consumes the per-step L14/L21/L22 MLP-output states from onset_probe's
capture and trains the glial field from the DESIGN:
    h_t = decay * h_{t-1} + tanh(W_in x_t)
    p_t = sigmoid(w_out . h_t + b)
label: a loop onset occurs in (t, t+W]. Item-split train/val.

This is awareness-to-action's first half: a learned field that predicts
loop onset from its own slow state (vs the linear probe's 0.858).

    colab exec -s hrun7 -f kernel/glia_gate.py
    env: W (24), DECAY (0.9), HID (64), EPOCHS (300)
"""
import os, sys, subprocess, glob, json, time

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
from src.probe_common import RESULTS                               # noqa: E402

W = int(os.environ.get("W", "24"))
DECAY = float(os.environ.get("DECAY", "0.9"))
HID = int(os.environ.get("HID", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "300"))

npzs = sorted(glob.glob(f"{RESULTS}/onset_states_*.npz"))
if not npzs:
    raise SystemExit("no onset_states_*.npz — run onset_probe.py first")
Z = np.load(npzs[-1], allow_pickle=True)
train_rows, val_rows = list(Z["train"]), list(Z["val"])
print(f"[glia] {npzs[-1]} | {len(train_rows)}+{len(val_rows)} items",
      flush=True)


def sequences(rows):
    seqs, labels = [], []
    for r in rows:
        st = torch.tensor(np.asarray(r["steps"], dtype=np.float32))
        onset = 10 ** 9 if r["onset"] is None else int(r["onset"])
        y = torch.tensor([1.0 if onset - W <= t < onset else 0.0
                          for t in range(st.shape[0])])
        seqs.append(st)
        labels.append(y)
    return seqs, labels


class GlialField(torch.nn.Module):
    def __init__(self, dim, hid, decay):
        super().__init__()
        self.decay, self.w_in = decay, torch.nn.Linear(dim, hid)
        self.w_out = torch.nn.Linear(dim + hid, 1)

    def forward(self, x):                    # x [T, dim] -> p [T]
        h = torch.zeros(self.w_in.out_features, device=x.device)
        out = []
        for t in range(x.shape[0]):
            h = self.decay * h + torch.tanh(self.w_in(x[t]))
            out.append(torch.sigmoid(
                self.w_out(torch.cat([x[t], h]))))
        return torch.cat(out)


dev = "cuda"
tr_seqs, tr_y = sequences(train_rows)
va_seqs, va_y = sequences(val_rows)
dim = tr_seqs[0].shape[1]
field = GlialField(dim, HID, DECAY).to(dev)
opt = torch.optim.Adam(field.parameters(), lr=2e-3)
pos_w = torch.tensor(
    (sum(y.numel() for y in tr_y) - sum(y.sum() for y in tr_y))
    / sum(y.sum() for y in tr_y).clamp(min=1), device=dev)

for ep in range(EPOCHS):
    opt.zero_grad()
    loss = 0.0
    for x, y in zip(tr_seqs, tr_y):
        p = field(x.to(dev))
        yy = y.to(dev)
        loss = loss - (pos_w * yy * torch.log(p.clamp(min=1e-7))
                       + (1 - yy) * torch.log((1 - p).clamp(min=1e-7))
                       ).mean()
    (loss / len(tr_seqs)).backward()
    opt.step()
    if ep % 100 == 0:
        print(f"  ep {ep} loss {float(loss/len(tr_seqs)):.4f}", flush=True)

with torch.no_grad():
    pv = torch.cat([field(x.to(dev)).cpu() for x in va_seqs]).numpy()
    yv = torch.cat(va_y).numpy()


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) \
        if n1 and n0 else float("nan")


a = auc(yv, pv)
print(f"[glia] AUC(W={W}) = {a:.3f} | linear probe baseline 0.858",
      flush=True)
for q in (0.9, 0.95, 0.99):
    th = np.quantile(pv, q)
    sel = pv >= th
    if sel.sum():
        print(f"  top-{1-q:.0%}: prec {yv[sel].mean():.3f} "
              f"cov {sel.mean():.3f}", flush=True)

res = {"auc": a, "W": W, "decay": DECAY, "hid": HID,
       "probe_baseline": 0.858}
json.dump(res, open(f"{RESULTS}/glia_gate_{int(time.time())}.json", "w"))
print("[glia] complete", flush=True)
