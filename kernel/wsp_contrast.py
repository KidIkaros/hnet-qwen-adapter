"""WSP contrast collection: generate on toolace items, bucket outputs into
right / hallucinated / no-call populations, harvest gen-span mean activations
across enc/main/dec, and rank layers by population separation — the diagnosis
step for a WSP suppression edit.

    colab exec -s hrun6 -f kernel/wsp_contrast.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (30), GEN_LEN (512), SKIP (2500)
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
import numpy as np                                               # noqa: E402
import re, json                                                  # noqa: E402
from src.probe_common import (CAP, RESULTS, build_model,         # noqa: E402
                              forward_bp, load_ckpt,
                              stream_toolace)

trunk, hn, meta = build_model()
load_ckpt(hn)

N_ITEMS = int(os.environ.get("N_ITEMS", "30"))
GEN_LEN = int(os.environ.get("GEN_LEN", "512"))
SKIP = int(os.environ.get("SKIP", "2500"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")

LAYER_NAMES = ([f"enc_{i}" for i, _ in enumerate(hn.enc)]
               + [f"main_{i}" for i, _ in enumerate(hn.main)]
               + [f"dec_{i}" for i, _ in enumerate(hn.dec)])
ACTS = {n: [] for n in LAYER_NAMES}
GEN_START = [0]                       # positions >= this are "gen"


def hook(name):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor) and t.dim() == 3:
            ACTS[name].append(t[0, GEN_START[0]:].mean(0).float().cpu())
    return h


handles = ([l.register_forward_hook(hook(f"enc_{i}"))
            for i, l in enumerate(hn.enc)]
           + [l.register_forward_hook(hook(f"main_{i}"))
              for i, l in enumerate(hn.main)]
           + [l.register_forward_hook(hook(f"dec_{i}"))
              for i, l in enumerate(hn.dec)])


@torch.no_grad()
def generate(prompt_ids, max_new):
    ids = list(prompt_ids)
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-6:] == list(b"\n\nUSER:") or ids[-2:] == list(b"\n\n"):
            break
    return ids


items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

# ---- pass 1: generate + label
pop = {"right": [], "hall": [], "none": []}
labels = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    ids = generate(pid, GEN_LEN)
    gen = bytes(ids[len(pid):]).decode("utf-8", "replace")
    calls = CALL_RE.findall(gen)
    if it["gold"] in calls:
        lab = "right"
    elif any(c not in it["tools"] for c in calls):
        lab = "hall"
    else:
        lab = "none"
    pop[lab].append((pid, ids))
    labels.append(lab)
    print(f"  item {k:2d} {lab:5s} calls={calls[:2]}", flush=True)
print(f"[labels] {len(pop['right'])} right / {len(pop['hall'])} hall / "
      f"{len(pop['none'])} none", flush=True)

# ---- pass 2: capture gen-span mean activations per population
means = {p_: {n: [] for n in LAYER_NAMES} for p_ in pop}
for p_, pairs in pop.items():
    for pid, ids in pairs:
        for n in LAYER_NAMES:
            ACTS[n].clear()
        GEN_START[0] = max(len(pid), len(ids) - CAP)
        forward_bp(hn, trunk, ids)
        for n in LAYER_NAMES:
            if ACTS[n]:
                means[p_][n].append(torch.stack(ACTS[n]).mean(0))

# ---- contrast directions + separation ranking
sep, dirs = {}, {}
if means["right"] and means["hall"]:
    for n in LAYER_NAMES:
        if means["right"][n] and means["hall"][n]:
            mr = torch.stack(means["right"][n]).mean(0)
            mh = torch.stack(means["hall"][n]).mean(0)
            diff = mh - mr
            sep[n] = float(diff.norm())
            dirs[n] = (diff / (diff.norm() + 1e-8)).numpy()
    rank = sorted(sep, key=sep.get, reverse=True)[:8]
    print("[diag] top layers by |mean_hall - mean_right|:", flush=True)
    for n in rank:
        print(f"  {n:12s} {sep[n]:.2f}", flush=True)
else:
    print("[diag] need both populations", flush=True)

np.savez(f"{RESULTS}/wsp_contrast.npz", **dirs)
json.dump({"labels": labels, "sep": sep,
           "n_right": len(pop["right"]), "n_hall": len(pop["hall"]),
           "n_none": len(pop["none"])},
          open(f"{RESULTS}/wsp_contrast.json", "w"), indent=1)
print(f"[wsp] -> {RESULTS}/wsp_contrast.npz + .json", flush=True)
print("[wsp] complete", flush=True)
