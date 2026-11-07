"""Branch graft: wrap trunk main layers as parallel base+instruct branches,
output-blended at ALPHA — append rather than overwrite. ALPHA=0 is exactly
the unmodified model.

    colab exec -s hrunX -f kernel/hnet_branch_graft.py
    env: MODEL, DONOR (default Qwen/Qwen3.5-2B), CKPT_PREFIX,
         ALPHA (0.5), N_ITEMS (15), GEN_LEN (512), SKIP (2500),
         BRANCH (main|all, default main)
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
from transformers import AutoModelForCausalLM                    # noqa: E402

if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.factory import (build_hnet, fit_adapters, load_trunk,   # noqa: E402
                         periodic)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base")
DONOR = os.environ.get("DONOR", "Qwen/Qwen3.5-2B")
ALPHA = float(os.environ.get("ALPHA", "0.5"))
BRANCH = os.environ.get("BRANCH", "main")

trunk = load_trunk(MODEL)
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


class Branch(torch.nn.Module):
    """out = (1-a)*A(x) + a*B(x) — A is the calibrated base path."""

    def __init__(self, a, b, alpha):
        super().__init__()
        self.a, self.b, self.alpha = a, b, alpha

    def forward(self, *args, **kw):
        oa = self.a(*args, **kw)
        ob = self.b(*args, **kw)
        ha = oa[0] if isinstance(oa, tuple) else oa
        hb = ob[0] if isinstance(ob, tuple) else ob
        h = (1 - self.alpha) * ha + self.alpha * hb
        return (h,) + oa[1:] if isinstance(oa, tuple) else h


print(f"[graft] loading donor {DONOR} ...", flush=True)
donor = AutoModelForCausalLM.from_pretrained(
    DONOR, dtype=torch.bfloat16, device_map="cuda")
dlayers = donor.model.layers
lin_idx, attn_idx = meta["lin_idx"], meta["attn_idx"]

wrapped = 0
if BRANCH in ("main", "all"):
    for i, l in enumerate(hn.main):
        hn.main[i] = Branch(l, dlayers[attn_idx[i]], ALPHA)
        wrapped += 1
if BRANCH == "all":
    for i, l in enumerate(hn.enc):
        hn.enc[i] = Branch(l, dlayers[lin_idx[i]], ALPHA)
        wrapped += 1
    for i, l in enumerate(hn.dec):
        hn.dec[i] = Branch(l, dlayers[lin_idx[-8:][i]], ALPHA)
        wrapped += 1
print(f"[graft] {wrapped} layers wrapped as base+instruct branches "
      f"(alpha={ALPHA}, branch={BRANCH})", flush=True)

N_ITEMS = int(os.environ.get("N_ITEMS", "15"))
GEN_LEN = int(os.environ.get("GEN_LEN", "512"))
SKIP = int(os.environ.get("SKIP", "2500"))
CAP = 4096
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(r'"name"\s*:\s*"([\w]+)"')


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
    return bytes(ids[len(prompt_ids):]).decode("utf-8", "replace")


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

right_n = call_n = hall_n = 0
rows = []
for k, it in enumerate(items):
    prompt = f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:"
    gen = generate(list(prompt.encode()), GEN_LEN)
    calls = CALL_RE.findall(gen)
    right = bool(set(it["gold"]) & set(calls))
    hall = [c for c in calls if c not in it["tools"]]
    right_n += right; call_n += bool(calls); hall_n += bool(hall)
    rows.append({"gold": it["gold"], "calls": calls, "right": right,
                 "hall": hall, "gen_head": gen[:200]})
    print(f"  item {k:2d} gold={it['gold'][:2]} calls={calls[:3]} "
          f"{'RIGHT' if right else 'miss'} hall={hall[:2]}", flush=True)

n = len(items)
res = {"donor": DONOR, "alpha": ALPHA, "branch": BRANCH,
       "right": right_n, "any": call_n, "hall": hall_n, "n": n,
       "wrapped": wrapped, "rows": rows}
os.makedirs(RESULTS, exist_ok=True)
rp = f"{RESULTS}/branch_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
json.dump(res, open(rp, "w"), indent=1)
print(f"[branch] right {right_n}/{n} any {call_n}/{n} hall {hall_n}/{n} "
      f"(base 4/12/8, full-delta 0/0/0)", flush=True)
print(f"[receipt] -> {rp}", flush=True)
print("[branch] complete", flush=True)
