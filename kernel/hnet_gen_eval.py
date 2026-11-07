"""Generation probe: does the model EMIT valid tool calls, not just
compress them?  Locked-exam style: fixed unseen items, greedy decode,
mechanical scoring, receipt.

    colab exec -s hrun4 -f kernel/hnet_gen_eval.py
    env: MODEL, CKPT_PREFIX (default hnet_resume_best), N_ITEMS, GEN_LEN

Items: ToolACE rows past all trained/eval prefixes. Prompt = system +
"USER: <first user turn>\n\nASSISTANT:" — same serialization as training.
Scored: emitted `[name(...)]` calls vs gold first-assistant names +
system-declared tool names (hallucination check).
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
from src.factory import (build_hnet, enwik8, fit_adapters,       # noqa: E402
                         load_trunk, periodic)

trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_docs = [d[:4096] for d in all_docs[:4]]
g = fit_adapters(hn, trunk, meta, fit_docs, P=P)
print(f"[fit] adapters fit | {g} dechunk pairs", flush=True)

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
assert glue is not None, "checkpoint not found"
hn.w_sup.load_state_dict(glue["w_sup"])
for i, l in enumerate(hn.enc):
    l.load_state_dict(enc[f"enc_{i}"])
for i, l in enumerate(hn.dec):
    l.load_state_dict(dec[f"dec_{i}"])
print(f"[ckpt] loaded {PREFIX}_* from {d}", flush=True)
CKPT_SHA = hashlib.sha256(
    open(f"{d}/{PREFIX}_glue.pt", "rb").read()).hexdigest()[:16]
hn.train()                                   # last_p needs train mode

N_ITEMS = int(os.environ.get("N_ITEMS", "15"))
GEN_LEN = int(os.environ.get("GEN_LEN", "512"))
CAP = 4096
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(r'"name"\s*:\s*"([\w]+)"')
STOP = b"\n\nUSER:"


@torch.no_grad()
def generate(prompt_ids, max_new):
    ids = list(prompt_ids)
    for _ in range(max_new):
        x = torch.tensor(ids[-CAP:], device="cuda")
        bp, _, _ = hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
                      n_ctx=1, super_k=4, head_first=trunk["head"].first)
        nb = int(bp[0, -1].argmax())
        ids.append(nb)
        if ids[-len(STOP):] == list(STOP):
            break
    return bytes(ids[len(prompt_ids):]).decode("utf-8", "replace")


# ---- exam: toolace items past trained region (SEEN=1824 + margin)
ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < 2500:
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
print(f"[exam] {len(items)} toolace items (skipped past 2500)", flush=True)

rows, right_n, call_n, hall_n = [], 0, 0, 0
for k, it in enumerate(items):
    prompt = f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:"
    gen = generate(list(prompt.encode("utf-8")), GEN_LEN)
    calls = CALL_RE.findall(gen)
    right = bool(set(it["gold"]) & set(calls))
    hall = [c for c in calls if c not in it["tools"]]
    right_n += right; call_n += bool(calls); hall_n += bool(hall)
    rows.append({"item": k, "gold": it["gold"], "gen_calls": calls,
                 "right": right, "hallucinated": hall,
                 "gen_head": gen[:200]})
    print(f"  item {k:2d} gold={it['gold'][:2]} calls={calls[:3]} "
          f"{'RIGHT' if right else 'miss'} hall={hall[:2]}", flush=True)

n = len(items)
receipt = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ckpt_prefix": PREFIX, "ckpt_sha_glue": CKPT_SHA,
    "n_items": n, "gen_len": GEN_LEN,
    "right_tool_rate": right_n / n, "any_call_rate": call_n / n,
    "hallucination_rate": hall_n / n, "items": rows}
os.makedirs(RESULTS, exist_ok=True)
rp = f"{RESULTS}/gen_receipt_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
json.dump(receipt, open(rp, "w"), indent=1)
print(f"[gen] right-tool {right_n}/{n} | any-call {call_n}/{n} "
      f"| hallucinated {hall_n}/{n}", flush=True)
print(f"[receipt] -> {rp}", flush=True)
print("[gen] complete", flush=True)
