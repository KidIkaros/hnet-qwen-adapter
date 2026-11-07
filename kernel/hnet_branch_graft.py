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
import re, json, time                                           # noqa: E402
from transformers import AutoModelForCausalLM                    # noqa: E402
from src.probe_common import (RESULTS, build_model, forward_bp,  # noqa: E402
                              load_ckpt, stream_toolace)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base")
DONOR = os.environ.get("DONOR", "Qwen/Qwen3.5-2B")
ALPHA = float(os.environ.get("ALPHA", "0.5"))
BRANCH = os.environ.get("BRANCH", "main")

trunk, hn, meta = build_model()
load_ckpt(hn)


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
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")


@torch.no_grad()
def generate(prompt_ids, max_new):
    ids = list(prompt_ids)
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-6:] == list(b"\n\nUSER:") or ids[-2:] == list(b"\n\n"):
            break
    return bytes(ids[len(prompt_ids):]).decode("utf-8", "replace")


items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

right_n = call_n = hall_n = 0
rows = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    gen = generate(pid, GEN_LEN)
    calls = CALL_RE.findall(gen)
    right = it["gold"] in calls
    hall = [c for c in calls if c not in it["tools"]]
    right_n += right; call_n += bool(calls); hall_n += bool(hall)
    rows.append({"gold": it["gold"], "calls": calls, "right": right,
                 "hall": hall, "gen_head": gen[:200]})
    print(f"  item {k:2d} gold={it['gold'][:20]} calls={calls[:3]} "
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
