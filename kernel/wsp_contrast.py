"""WSP contrast driver: generate toolace calls, label right vs hallucinated,
capture per-layer gen-span mean activations, compute contrast directions.

    colab exec -s hrun4 -f kernel/wsp_contrast.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (default 30), GEN_LEN (512), SKIP (2500)

Mirrors stage-2 diagnosis of wsp-reproduction: d_hat = normalized
mean(act_hall) - mean(act_right) per layer; prints separation ranking so the
edit target can be chosen empirically. Saves results/wsp_contrast.npz + .json.
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
import re, json, hashlib, time                                   # noqa: E402
from datasets import load_dataset                                # noqa: E402

if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
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

N_ITEMS = int(os.environ.get("N_ITEMS", "30"))
GEN_LEN = int(os.environ.get("GEN_LEN", "512"))
SKIP = int(os.environ.get("SKIP", "2500"))
CAP = 4096
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(r'"name"\s*:\s*"([\w]+)"')

# activation capture: {layer_name: [gen-span mean vectors]}
LAYER_NAMES = ([f"enc_{i}" for i, _ in enumerate(hn.enc)]
               + [f"main_{i}" for i, _ in enumerate(hn.main)]
               + [f"dec_{i}" for i, _ in enumerate(hn.dec)])
ACTS = {n: [] for n in LAYER_NAMES}
GEN_START = [0]                       # mutable: positions >= this are "gen"


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
        x = torch.tensor(ids[-CAP:], device="cuda")
        bp, _, _ = hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
                      n_ctx=1, super_k=4, head_first=trunk["head"].first)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-6:] == list(b"\n\nUSER:") or ids[-2:] == list(b"\n\n"):
            break
    return ids


ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv if c.get("from") == "assistant"), None)
    gold = CALL_RE.findall(a) if a else []
    if u and gold:
        items.append({"sys": r["system"], "user": u, "gold": gold,
                      "tools": set(TOOL_RE.findall(r["system"]))})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items", flush=True)

# ---- pass 1: generate + label
pop = {"right": [], "hall": [], "none": []}
labels = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    ids = generate(pid, GEN_LEN)
    gen = bytes(ids[len(pid):]).decode("utf-8", "replace")
    calls = CALL_RE.findall(gen)
    if set(it["gold"]) & set(calls):
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
        x = torch.tensor(ids[-CAP:], device="cuda")
        hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
           n_ctx=1, super_k=4, head_first=trunk["head"].first)
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
