"""27B trunk compression probe — the real-host version of compress_check.

Loads the abliterated 27B (reuses kernel-global nf4 model if resident),
calibrates Hessians + GPTQ-quantizes + optionally Wanda-prunes trunk
linears IN BLOCKS (per-block H accumulation bounds CPU RAM — 27B
intermediate dims make full-model H ~60GB).

Verification = the pinned name-selection exam (teacher_score prefix
protocol, sum logprob) before/after on the same slice — baseline was
27/27 declared. This is the behavior gate for the compressed-host path.

CAVEAT (v1): loads bf16 with GPU/CPU offload (54GB > 40GB VRAM) —
slow forwards, and GPTQ on CPU-resident layers is the bottleneck.
quantize_linear needs REAL float weights: the resident nf4 model's
packed 4bit weights are NOT quantizable, so it is evicted first.

    colab exec -s hrun8 -f kernel/compress_trunk.py
    env: BITS (4), PRUNE (0), ROTATE (1), N_CALIB (48), BLOCK_LAYERS (4),
         N_ITEMS (40), VERIFY_SKIP (2500), SKIP_BASE (0 = always measure)
"""
import os, sys, subprocess, json, time, re

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"FAILED {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "fetch", "-q", "origin")
    up = set(sh("git", "-C", REPO, "ls-tree", "-r", "--name-only",
                "origin/main").split())
    for _r, _, _fs in os.walk(REPO):
        for _f in _fs:
            _p = os.path.join(_r, _f)
            _rel = os.path.relpath(_p, REPO)
            if _rel in up and "??" in sh(
                    "git", "-C", REPO, "status", "--porcelain", "--", _rel):
                os.remove(_p)
    sh("git", "-C", REPO, "merge", "-q", "origin/main")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)
for _m in [k for k in sys.modules if k.startswith("src.")]:
    del sys.modules[_m]

import torch                                                       # noqa: E402
from src.probe_common import RESULTS, stream_toolace                # noqa: E402
from src.compress import (calibrate_hessians, quantize_linear,     # noqa: E402
                          trunk_linears, wanda_prune)

BITS = int(os.environ.get("BITS", "4"))
ROTATE = os.environ.get("ROTATE", "1") == "1"
PRUNE = float(os.environ.get("PRUNE", "0"))
N_CALIB = int(os.environ.get("N_CALIB", "48"))
BLOCK = int(os.environ.get("BLOCK_LAYERS", "4"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
MODEL = os.environ.get(
    "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")

# ---- model: evict ALL resident GPU holders, load bf16 + offload ----
# nf4/8bit weights are packed — quantize_linear needs real float weights.
# The kernel accumulates globals across execs (2B trunk/hn, old teachers);
# evict every known big holder or the bf16 load OOMs on warmup.
_g = globals()
for _k in ("model", "tok", "trunk", "hn", "meta", "lux", "teacher_model",
           "teacher_tok", "proj", "opt", "sc", "rows", "items", "H"):
    _g.pop(_k, None)
import gc                                                       # noqa: E402
gc.collect()
torch.cuda.empty_cache()
from transformers import (AutoTokenizer, AutoModelForCausalLM)      # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "30GiB", "cpu": "70GiB"})
model.eval()
print(f"[trunk] loaded {MODEL} bf16 device_map=auto", flush=True)

items = stream_toolace(VSKIP, N_ITEMS)


@torch.no_grad()
def prefix_lp(ctx_ids, cand_ids):
    """Summed logprob of cand_ids after ctx_ids (teacher_score protocol)."""
    ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


@torch.no_grad()
def name_exam():
    """Pinned name-selection exam: argmax prefix score over tools+none."""
    right = none_n = 0
    decl = decl_right = 0
    for it in items:
        ctx = tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                  return_tensors="pt").input_ids[0].tolist()
        sc = {n: prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                           ["input_ids"])
              for n in it["tools"] + ["none"]}
        pred = max(sc, key=sc.get)
        declared = it["gold"] in it["tools"]
        decl += declared
        decl_right += declared and pred == it["gold"]
        right += pred == it["gold"]
        none_n += pred == "none"
    return {"right": right, "declared": decl, "decl_right": decl_right,
            "none": none_n, "n": len(items)}


print("[exam] measuring BASE name-selection...", flush=True)
base = name_exam()
print(f"[base] {base}", flush=True)

# ---- chunked Hessian calibration + quantize, per layer block ----
targets = trunk_linears(model)
print(f"[quant] {len(targets)} linear targets", flush=True)

# group targets by transformer layer index; sub-chunk oversized groups
import re as _re
blocks = {}
for name, lin in targets:
    m = _re.search(r"layers\.(\d+)\.", name)
    key = int(m.group(1)) // BLOCK if m else -1
    blocks.setdefault(key, []).append((name, lin))
flat = []
for bkey, blk in sorted(blocks.items()):
    for i in range(0, len(blk), BLOCK * 7):
        flat.append(blk[i:i + BLOCK * 7])
print(f"[quant] {len(flat)} calibration blocks", flush=True)

calib_items = stream_toolace(1224, N_CALIB)
calib_ids = [tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                 return_tensors="pt").input_ids[0].tolist()
             for it in calib_items]


def hf_forward(ids):
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device="cuda"))


meta_q = {}
for bi, blk in enumerate(flat):
    names = [n for n, _ in blk]
    H = calibrate_hessians(model, names, calib_ids, hf_forward)
    with torch.no_grad():
        for name, lin in blk:
            h = H.get(name)
            if h is None:
                continue
            if PRUNE > 0:
                wanda_prune(lin, h.to(lin.weight.device), PRUNE)
            meta_q[name] = quantize_linear(
                lin, h.to(lin.weight.device), bits=BITS, rotate=ROTATE)
    del H
    print(f"[quant] block {bi + 1}/{len(blocks)} done "
          f"({len(meta_q)} linears)", flush=True)

print(f"[quant] quantized {len(meta_q)} linears @ {BITS}bit", flush=True)

print("[exam] measuring QUANT name-selection...", flush=True)
quant = name_exam()
print(f"[quant] {quant}", flush=True)

res = {"model": MODEL, "bits": BITS, "prune": PRUNE, "rotate": ROTATE,
       "n_targets": len(meta_q), "n_calib": N_CALIB, "n": N_ITEMS,
       "skip": VSKIP, "base": base, "quant": quant}
json.dump(res, open(f"{RESULTS}/compress_trunk_{int(time.time())}.json",
                    "w"))
from src.edit_ledger import record                                # noqa: E402
record({"method": "gptq-trunk-27b",
        "target": f"{MODEL} linears x{len(meta_q)} @ {BITS}bit"
                  + (f" prune {PRUNE}" if PRUNE else ""),
        "teach_gain": f"declared {base['decl_right']}/{base['declared']}"
                      f" -> {quant['decl_right']}/{quant['declared']}",
        "neighbor_cost": f"none-picks {base['none']}->{quant['none']}",
        "n_eval": N_ITEMS,
        "receipt": f"compress_trunk_{int(time.time())}.json"})
print("[compress_trunk] complete", flush=True)
