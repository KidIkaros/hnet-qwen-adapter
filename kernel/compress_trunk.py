"""27B trunk compression probe — the real-host version of compress_check.

Loads the abliterated 27B (reuses kernel-global nf4 model if resident),
calibrates Hessians + GPTQ-quantizes + optionally Wanda-prunes trunk
linears IN BLOCKS (per-block H accumulation bounds CPU RAM — 27B
intermediate dims make full-model H ~60GB).

Verification = the pinned name-selection exam (teacher_score prefix
protocol, sum logprob) before/after on the same slice — baseline was
27/27 declared. This is the behavior gate for the compressed-host path.

CAVEAT (v2): loads bf16 with GPU/CPU offload (54GB > 40GB VRAM) —
slow forwards, and GPTQ on CPU-resident layers is the bottleneck.
CPU-offloaded Linear params stay on `meta`; real weights live in the
accelerate hook's weight maps. _RW resolves them up front (early abort
if unresolved) and quantize_real writes dequantized values back into
whichever tensor forward() actually uses. Learned the hard way: v1
crashed at block 9/16 on `Cannot copy out of meta tensor`, and hrun8
died before a receipt could be written — hence the early base dump and
progress log.

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
from src.compress import (calibrate_hessians, trunk_linears,        # noqa: E402
                          gptq_quantize, hadamard)

BITS = int(os.environ.get("BITS", "4"))
ROTATE = os.environ.get("ROTATE", "1") == "1"
PRUNE = float(os.environ.get("PRUNE", "0"))
N_CALIB = int(os.environ.get("N_CALIB", "48"))
BLOCK = int(os.environ.get("BLOCK_LAYERS", "4"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
SKIP_BASE = os.environ.get("SKIP_BASE", "0") == "1"
MODEL = os.environ.get(
    "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")
TS = int(time.time())

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


# ---- meta-weight resolution -------------------------------------------
# device_map="auto" offloads ~half the layers to CPU via accelerate hooks:
# those params stay on the `meta` device and the REAL bf16 tensors live in
# the hook's weight maps (swapped in during forward). quantize_linear's
# lin.weight.data access hits meta -> crash (seen at block 9/16). Resolve
# every target's effective tensor ONCE, up front, and abort early if any
# cannot be found — before burning exam+calibration time.
def _iter_hook_tensors(hook, depth=0):
    """Yield (key, real_tensor) from a module's offload hook.

    AlignDevicesHook.weights_map is a PrefixedDataset (NOT a dict):
    .keys() returns full param names, wm[short] fetches the tensor where
    short = key minus wm.prefix. Plain dicts (short keys) also handled.
    """
    if hook is None or depth > 5:
        return
    wm = getattr(hook, "weights_map", None)
    if wm is not None:
        pref = getattr(wm, "prefix", "") or ""
        try:
            ks = list(wm.keys())
        except Exception:
            ks = []
        for k in ks:
            short = k[len(pref):] if pref and k.startswith(pref) else k
            v = None
            for kk in (short, k):
                try:
                    v = wm[kk]
                    break
                except Exception:
                    pass
            if torch.is_tensor(v) and not v.is_meta:
                yield k, v
        if isinstance(wm, dict):
            for k, v in wm.items():
                if torch.is_tensor(v) and not v.is_meta:
                    yield k, v
    for sub in getattr(hook, "hooks", None) or []:
        yield from _iter_hook_tensors(sub, depth + 1)


_RW = {}
for _n, _m in model.named_modules():
    for _k, _v in _iter_hook_tensors(getattr(_m, "_hf_hook", None)):
        _full = _k if (_k == _n or _k.startswith(_n + ".")) else f"{_n}.{_k}"
        _RW[_full] = _v
print(f"[trunk] hook weight-map: {len(_RW)} real tensors", flush=True)


def _real_weight(name, lin):
    """Effective weight tensor for a Linear whose param may be meta."""
    w = lin.weight
    if not w.is_meta:
        return w
    t = _RW.get(name + ".weight")
    if t is None:
        raise RuntimeError(f"meta weight unresolved: {name}")
    return t


@torch.no_grad()
def quantize_real(name, lin, H, bits, rotate):
    """Rotate + GPTQ one Linear via its EFFECTIVE weight; compute on GPU.

    Writes dequantized values back into whichever tensor forward() uses
    (param itself, or the hook's offload map) so the exam sees the
    quantized model.
    """
    Wt = _real_weight(name, lin)
    Wf = Wt.detach().float().to("cuda")
    Hf = H.detach().float().to("cuda")
    if rotate and Wf.shape[1] & (Wf.shape[1] - 1) == 0:
        h = hadamard(Wf.shape[1]).to("cuda", torch.float32)
        Wf = Wf @ h
        Hf = h.T @ Hf @ h
    Wq, s, z = gptq_quantize(Wf, Hf, bits=bits)
    Wt.data.copy_(Wq.to(device=Wt.device, dtype=Wt.dtype))
    del Wf, Hf, Wq
    torch.cuda.empty_cache()
    return {"scale": s.cpu(), "zero": z.cpu(), "bits": bits}


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


# ---- targets + early meta-resolution check (before burning exam time) ----
targets = trunk_linears(model)
_meta_t = [(n, l) for n, l in targets if l.weight.is_meta]
_resolved = sum(1 for n, _ in _meta_t if _RW.get(n + ".weight") is not None)
print(f"[quant] {len(targets)} linear targets "
      f"({len(_meta_t)} meta-offloaded, {_resolved} resolved)", flush=True)
if _meta_t and _resolved < len(_meta_t):
    for n, l in _meta_t[:5]:
        h = getattr(l, "_hf_hook", None)
        print(f"  UNRESOLVED {n} hook={type(h).__name__ if h else None} "
              f"attrs={[a for a in dir(h)[:15]] if h else []}", flush=True)
    raise SystemExit("meta weights unresolved — aborting before exam")

prog = open(f"{RESULTS}/compress_trunk_{TS}.progress.log", "a")

print("[exam] measuring BASE name-selection...", flush=True)
base = {"skipped": True} if SKIP_BASE else name_exam()
print(f"[base] {base}", flush=True)
json.dump({"stage": "base", "base": base},
          open(f"{RESULTS}/compress_trunk_{TS}_base.json", "w"))
prog.write(f"base {base}\n"); prog.flush()

# ---- chunked Hessian calibration + quantize, per layer block ----
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
                Wt = _real_weight(name, lin)
                W = Wt.detach().float().to("cuda")
                imp = (W.abs() * torch.sqrt(
                    h.detach().float().diag().clamp(min=1e-8)).to("cuda"))
                keep = max(1, int(W.shape[1] * (1.0 - PRUNE)))
                idx = imp.argsort(1, descending=True)[:, :keep]
                mask = torch.zeros_like(W)
                mask.scatter_(1, idx, 1.0)
                Wt.data.copy_((W * mask).to(Wt.dtype).to(Wt.device))
                del W, imp, mask
                torch.cuda.empty_cache()
            meta_q[name] = quantize_real(name, lin, h, BITS, ROTATE)
    del H
    prog.write(f"block {bi + 1}/{len(flat)} done cum={len(meta_q)}\n")
    prog.flush()
    print(f"[quant] block {bi + 1}/{len(flat)} done "
          f"({len(meta_q)} linears)", flush=True)

print(f"[quant] quantized {len(meta_q)} linears @ {BITS}bit", flush=True)
prog.write(f"quantized {len(meta_q)}\n"); prog.flush()

print("[exam] measuring QUANT name-selection...", flush=True)
quant = name_exam()
print(f"[quant] {quant}", flush=True)
prog.write(f"quant_exam {quant}\n"); prog.flush()

res = {"model": MODEL, "bits": BITS, "prune": PRUNE, "rotate": ROTATE,
       "n_targets": len(meta_q), "n_calib": N_CALIB, "n": N_ITEMS,
       "skip": VSKIP, "base": base, "quant": quant}
_rcpt = f"{RESULTS}/compress_trunk_{TS}.json"
json.dump(res, open(_rcpt, "w"))
print(f"[compress_trunk] receipt {_rcpt}", flush=True)
from src.edit_ledger import record                                # noqa: E402
record({"method": "gptq-trunk-27b",
        "target": f"{MODEL} linears x{len(meta_q)} @ {BITS}bit"
                  + (f" prune {PRUNE}" if PRUNE else ""),
        "teach_gain": (f"declared {base.get('decl_right')}/"
                       f"{base.get('declared')} -> "
                       f"{quant['decl_right']}/{quant['declared']}"),
        "neighbor_cost": f"none-picks {base.get('none')}->{quant['none']}",
        "n_eval": N_ITEMS,
        "receipt": f"compress_trunk_{TS}.json"})
print("[compress_trunk] complete", flush=True)
