"""Delta graft: apply (instruct - base) task vector to the trunk layers
inside the H-Net, then re-run the toolace generation probe.

    colab exec -s hrunX -f kernel/hnet_delta_graft.py
    env: MODEL, DONOR (default Qwen/Qwen3.5-2B), CKPT_PREFIX,
         ALPHA (1.0), N_ITEMS (15), GEN_LEN (512), SKIP (2500)

Rationale: W_instruct - W_base is the SFT-written task vector at exact
matching shapes; enc/dec/main are references into the trunk's own layer
modules so editing trunk["model"] layers in place re-wires the H-Net.
Embeddings/lm_head skipped (byte-level path doesn't use them).
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
ALPHA = float(os.environ.get("ALPHA", "1.0"))

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

# ---- delta graft: instruct - base on layer tensors
print(f"[graft] loading donor {DONOR} ...", flush=True)
donor = AutoModelForCausalLM.from_pretrained(
    DONOR, dtype=torch.bfloat16, device_map="cpu")
donor_sd = donor.state_dict()
base_sd = trunk["model"].state_dict()
applied, skipped = 0, 0
with torch.no_grad():
    for name, w in donor_sd.items():
        if not name.startswith("model.layers."):
            continue                       # skip embed/norm/lm_head
        b = base_sd.get(name)
        if b is None or b.shape != w.shape:
            skipped += 1
            continue
        # find the live tensor: trunk["model"] is the same object hn wraps
        parts = name.split(".")
        mod = trunk["model"]
        for p_ in parts[:-1]:
            mod = getattr(mod, p_) if not p_.isdigit() else mod[int(p_)]
        t = getattr(mod, parts[-1])
        if t.shape == w.shape:
            t.add_(ALPHA * (w.to(t.dtype).cuda() - t))
            applied += 1
print(f"[graft] applied delta to {applied} tensors, skipped {skipped} "
      f"(alpha={ALPHA})", flush=True)
del donor, donor_sd, base_sd
import gc; gc.collect(); torch.cuda.empty_cache()

# ---- same toolace gen exam as hnet_gen_eval
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
res = {"donor": DONOR, "alpha": ALPHA, "right": right_n, "any": call_n,
       "hall": hall_n, "n": n, "applied": applied, "rows": rows}
os.makedirs(RESULTS, exist_ok=True)
rp = f"{RESULTS}/graft_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
json.dump(res, open(rp, "w"), indent=1)
print(f"[graft] right {right_n}/{n} any {call_n}/{n} hall {hall_n}/{n} "
      f"(base probe was 4/12/8)", flush=True)
print(f"[receipt] -> {rp}", flush=True)
print("[graft] complete", flush=True)
