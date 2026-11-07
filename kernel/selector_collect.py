"""Selector-net data collection: for each toolace exam item, capture the
trunk's hidden state at the ASSISTANT: position (the pre-emission semantic
state) at candidate layers, plus the item's declared tool set + gold index.

    colab exec -s hrun6 -f kernel/selector_collect.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (60), SKIP (2500)

Output: results/selector_data.npz — h_<layer>[N, D], plus selector_meta.json.
The selector net is fit OFFLINE from this file, then compiled via qwentensor.
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
import json                                                      # noqa: E402
from src.probe_common import (RESULTS, build_model, forward_bp,  # noqa: E402
                              load_ckpt, stream_toolace)

trunk, hn, meta = build_model()
load_ckpt(hn)

N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
SKIP = int(os.environ.get("SKIP", "2500"))
LAYERS = ([("enc", len(hn.enc) - 1)]
          + [("main", i) for i in range(len(hn.main))]
          + [("dec", len(hn.dec) - 1)])
ACTS = {}


def hook(name):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor) and t.dim() == 3:
            ACTS[name] = t[0, -1].float().cpu()     # last position
    return h


for tag, i in LAYERS:
    getattr(hn, tag)[i].register_forward_hook(hook(f"{tag}_{i}"))

items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

H = {f"{t}_{i}": [] for t, i in LAYERS}
golds, tools_list = [], []
with torch.no_grad():
    for k, it in enumerate(items):
        pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
        ACTS.clear()
        forward_bp(hn, trunk, pid[-4096:])
        for t, i in LAYERS:
            n = f"{t}_{i}"
            H[n].append(ACTS[n] if n in ACTS else torch.zeros(2048))
        golds.append(it["gold"])
        tools_list.append(it["tools"])
        if k % 10 == 0:
            print(f"  captured {k}", flush=True)

np.savez(f"{RESULTS}/selector_data.npz",
         **{n: torch.stack(v).numpy() for n, v in H.items()})
json.dump({"tools": tools_list, "gold": golds},
          open(f"{RESULTS}/selector_meta.json", "w"))
print(f"[collect] -> selector_data.npz + selector_meta.json "
      f"({len(items)} items, {len(H)} layers)", flush=True)
print("[collect] complete", flush=True)
