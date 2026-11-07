"""Selector-net data collection: for each toolace exam item, capture the
trunk's hidden state at the ASSISTANT: position (the pre-emission semantic
state) at candidate layers, plus the item's declared tool set + gold index.

    colab exec -s hrun6 -f kernel/selector_collect.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (60), SKIP (2500)

Output: results/selector_data.npz — h_<layer>[N, D], tools (json), gold[N].
The selector net is fit OFFLINE from this file (closed form / tiny head),
then compiled via qwentensor.
"""
import os, sys, glob, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "/content/qt/hnet-qwen-adapter"
RESULTS = f"{REPO}/results"
P = 6


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import re, json, time                                           # noqa: E402
from datasets import load_dataset                                # noqa: E402

if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", -q if False else "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.factory import (build_hnet, fit_adapters, load_trunk,   # noqa: E402
                         periodic)

trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_adapters(hn, trunk, meta, [d[:4096] for d in all_docs[:4]], P=P)

PREFIX = os.environ.get("CKPT_PREFIX", "hnet_resume_best")
glue = enc = dec = None
for d in [RESULTS, "/content/ckpt"]:
    try:
        glue = torch.load(f"{d}/{PREFIX}_glue.pt", map_location="cuda")
        enc = torch.load(f"{d}/{PREFIX}_enc.pt", map_location="cuda")
        dec = torch.load(f"{d}/{PREFIX}_dec.pt", map_location="cuda")
        break
    except FileNotFoundError:
        continue
hn.w_sup.load_state_dict(glue["w_sup"])
for i, l in enumerate(hn.enc):
    l.load_state_dict(enc[f"enc_{i}"])
for i, l in enumerate(hn.dec):
    l.load_state_dict(dec[f"dec_{i}"])
print(f"[ckpt] loaded {PREFIX}_* from {d}", flush=True)
hn.train()

N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
SKIP = int(os.environ.get("SKIP", "2500"))
CAP = 4096
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(r'"name"\s*:\s*"([\w]+)"')
# candidate readout layers: last enc, each main, last dec
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

ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv if c.get("from") == "assistant"), None)
    gold = CALL_RE.findall(a) if a else []
    tools = TOOL_RE.findall(r["system"])
    if u and gold and tools:
        items.append({"sys": r["system"], "user": u,
                      "gold": gold[0], "tools": tools})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items", flush=True)

H = {f"{t}_{i}": [] for t, i in LAYERS}
golds, tools_list = [], []
with torch.no_grad():
    for k, it in enumerate(items):
        pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
        x = torch.tensor(pid[-CAP:], device="cuda")
        ACTS.clear()
        hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
           n_ctx=1, super_k=4, head_first=trunk["head"].first)
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
