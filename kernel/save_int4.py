"""Pack the 27B trunk to TRUE int4 (nibbles + fp16 scales) — not dequant-bf16.

compress_trunk proved GPTQ-4bit is behaviorally lossless but save_pretrained
only stores dequantized bf16 (~54GB). This driver re-runs the same GPTQ
calibration while RETAINING integer codes, packs 2 codes/byte, and writes a
custom safetensors artifact (~20GB est).

Scope EXTENDED vs the bf16 artifact: original 7 proj suffixes PLUS the
hybrid linear-attention projections (in_proj_qkv / in_proj_z / out_proj on
48 layers) — 400 linears, ~25.3B params (~94%). lm_head + embeddings stay
bf16 (logit/embedding fidelity). in_proj_a/b are tiny — skipped.

NO Hadamard rotation: config dims (5120/6144/17408) are all non-pow2, so
compress_trunk's rotate=1 was INERT — manifest records rotate=0 honestly.

Verify in-run: packed tensors are dequantized back into the resident model
(fresh bf16 load) and the pinned 40-item name exam runs — same slice as
compress_trunk/base/quant (expect ~27/27 declared). After this script the
kernel's `model` IS the int4 model — big_exam MODE=resident can follow.

    colab exec -s hrun9 -f kernel/save_int4.py
    env: BITS (4), N_CALIB (48), BLOCK_LAYERS (4), OUT_DIR, N_ITEMS (40),
         VSKIP (2500), SHARD_GB (4), SKIP_QUANT (0 = reuse? no - always requant)
"""
import os, sys, subprocess, json, time

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
from src.compress import (calibrate_hessians, gptq_quantize_int,    # noqa: E402
                          pack_nibbles, dequant_int)

BITS = int(os.environ.get("BITS", "4"))
N_CALIB = int(os.environ.get("N_CALIB", "48"))
BLOCK = int(os.environ.get("BLOCK_LAYERS", "4"))
OUT_DIR = os.environ.get("OUT_DIR", "/content/qt/int4-27b-toolace")
SHARD_GB = float(os.environ.get("SHARD_GB", "4"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
VSKIP = int(os.environ.get("VSKIP", "2500"))
MODEL = os.environ.get(
    "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")
TS = int(time.time())

BASE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")
EXTRA_SUFFIXES = ("in_proj_qkv", "in_proj_z", "out_proj")

_g = globals()
_KEEP = {"sh", "stream_toolace", "RESULTS", "torch", "json", "os", "sys",
         "subprocess", "time", "gc", "BITS", "N_CALIB", "BLOCK", "OUT_DIR",
         "SHARD_GB", "N_ITEMS", "VSKIP", "MODEL", "TS", "BASE_SUFFIXES",
         "EXTRA_SUFFIXES", "_g", "_KEEP",
         "calibrate_hessians", "gptq_quantize_int", "pack_nibbles",
         "dequant_int"}


def _evict():
    for _k in list(_g.keys()):
        if _k not in _KEEP:
            _g.pop(_k, None)
    import gc
    gc.collect()
    torch.cuda.empty_cache()


_evict()

from transformers import (AutoTokenizer, AutoModelForCausalLM)      # noqa: E402
_g["tok"] = AutoTokenizer.from_pretrained(MODEL)
_g["model"] = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "72GiB", "cpu": "70GiB"})
_g["model"].eval()
print("[int4] model loaded", flush=True)

# ---- targets: standard 7 + linear_attn projections ----
targets = []
for n, m in _g["model"].named_modules():
    if not isinstance(m, torch.nn.Linear):
        continue
    s = n.split(".")[-1]
    if s in BASE_SUFFIXES or ("linear_attn" in n and s in EXTRA_SUFFIXES):
        targets.append((n, m))
print(f"[int4] {len(targets)} quantization targets", flush=True)
_meta = [n for n, l in targets if l.weight.is_meta]
if _meta:
    raise SystemExit(f"meta weights present ({len(_meta)}) — need all-GPU")

prog = open(f"{RESULTS}/save_int4_{TS}.progress.log", "a")

# ---- blockwise calibration + int-quantize (no rotation — was inert) ----
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
print(f"[int4] {len(flat)} calibration blocks", flush=True)

calib_items = stream_toolace(1224, N_CALIB)
calib_ids = [tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                 return_tensors="pt").input_ids[0].tolist()
             for it in calib_items]


def hf_forward(ids):
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device="cuda"))


packed = {}   # state_dict-key -> cpu tensor (uint8 packed or bf16)
t0 = time.time()
for bi, blk in enumerate(flat):
    names = [n for n, _ in blk]
    H = calibrate_hessians(model, names, calib_ids, hf_forward)
    with torch.no_grad():
        for name, lin in blk:
            h = H.get(name)
            if h is None:
                continue
            W = lin.weight.detach().float().to("cuda")
            Hf = h.detach().float().to("cuda")
            codes, scales, _z = gptq_quantize_int(W, Hf, bits=BITS)
            pk = pack_nibbles(codes).cpu()
            sc = scales.half().cpu()
            packed[f"{name}.qweight"] = pk
            packed[f"{name}.scales"] = sc
            del W, Hf, codes, scales, pk, sc
            torch.cuda.empty_cache()
    del H
    prog.write(f"block {bi + 1}/{len(flat)} done cum={len(packed) // 2}\n")
    prog.flush()
    print(f"[int4] block {bi + 1}/{len(flat)} done "
          f"({len(packed) // 2} linears, {time.time() - t0:.0f}s)",
          flush=True)
print(f"[int4] quantized {len(packed) // 2} linears @ {BITS}bit", flush=True)

# ---- non-target params stay bf16 ----
tset = {n for n, _ in targets}
nsd = 0
for k, v in model.state_dict().items():
    mod = k.rsplit(".", 1)[0]
    if mod in tset and k.endswith(".weight"):
        continue
    packed[k] = v.detach().to("cpu", torch.bfloat16).contiguous()
    nsd += 1
print(f"[int4] +{nsd} bf16 tensors (embed/lm_head/norms/conv/ssm)", flush=True)

# ---- shard + write safetensors ----
os.makedirs(OUT_DIR, exist_ok=True)
items = sorted(packed.items(), key=lambda kv: -kv[1].numel())
shards, cur, cur_sz = [], {}, 0
limit = int(SHARD_GB * 1e9)
for k, v in items:
    nb = v.numel() * v.element_size()
    if cur and cur_sz + nb > limit:
        shards.append(cur); cur, cur_sz = {}, 0
    cur[k] = v; cur_sz += nb
if cur:
    shards.append(cur)
from safetensors.torch import save_file                        # noqa: E402
for i, sd in enumerate(shards):
    fn = f"{OUT_DIR}/int4-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(sd, fn)
    print(f"[int4] wrote {fn} ({sum(t.numel() * t.element_size() for t in sd.values()) / 1e9:.2f}GB)", flush=True)
tot = sum(os.path.getsize(os.path.join(OUT_DIR, f))
          for f in os.listdir(OUT_DIR) if f.endswith(".safetensors"))
print(f"[int4] total packed size {tot / 1e9:.1f}GB", flush=True)

# ---- apply dequantized packed weights back into resident model, verify ----
print("[int4] dequantizing packed weights into model for exam...", flush=True)
with torch.no_grad():
    for name, lin in targets:
        pk = packed[f"{name}.qweight"].to("cuda")
        sc = packed[f"{name}.scales"].to("cuda").float()
        Wq = dequant_int(pk, sc, lin.weight.shape[1], bits=BITS)
        lin.weight.data.copy_(Wq.to(lin.weight.dtype))
        del pk, sc, Wq
        torch.cuda.empty_cache()
print("[int4] resident model now = packed-int4 weights", flush=True)


@torch.no_grad()
def _prefix_lp(ctx_ids, cand_ids):
    ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


@torch.no_grad()
def name_exam():
    right = none_n = decl = decl_right = 0
    for it in stream_toolace(VSKIP, N_ITEMS):
        ctx = tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                  return_tensors="pt").input_ids[0].tolist()
        sc_ = {n: _prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                             ["input_ids"])
               for n in it["tools"] + ["none"]}
        pred = max(sc_, key=sc_.get)
        declared = it["gold"] in it["tools"]
        decl += declared
        decl_right += declared and pred == it["gold"]
        right += pred == it["gold"]
        none_n += pred == "none"
    return {"right": right, "declared": decl, "decl_right": decl_right,
            "none": none_n, "n": N_ITEMS}


print("[exam] int4-packed name-selection...", flush=True)
exam = name_exam()
print(f"[exam] {exam}", flush=True)

manifest = {
    "artifact": "qwen27b-int4-toolace",
    "lineage": "PACKED int4 GPTQ trunk, EXTENDED scope (incl. linear_attn "
               "projections). Sibling of qwen27b-gptq4-toolace (bf16-dequant, "
               "256-target scope). Pre-H-Net base artifact.",
    "base_model": MODEL,
    "method": f"GPTQ {BITS}-bit sym blocksize=128, per-(row,block) scales, "
              f"nibble-packed codes, Hessian calib N={N_CALIB} "
              f"(stream_toolace@1224). rotate=0 (Hadamard inert: no pow2 "
              f"in-features in this hybrid).",
    "scope": {"n_linears": len(targets),
              "suffixes": list(BASE_SUFFIXES) +
              [f"linear_attn.{s}" for s in EXTRA_SUFFIXES],
              "excluded_bf16": ["embed_tokens", "lm_head", "norms",
                                "linear_attn.in_proj_a/b", "conv/ssm params"]},
    "packing": {"code_order": "even col=high nibble, odd col=low nibble",
                "scales_dtype": "float16", "zero": (2 ** BITS - 1) / 2,
                "blocksize": 128,
                "dequant": "W ~= (unpack(qweight) - zero) * "
                           "repeat(scales,128)[:, :in]"},
    "n_calib": N_CALIB, "bits": BITS,
    "exam": exam, "exam_slice": {"skip": VSKIP, "n": N_ITEMS},
    "ts": TS,
}
json.dump(manifest, open(f"{OUT_DIR}/INT4_MANIFEST.json", "w"), indent=1)
# reuse config/tokenizer/chat template from the bf16 artifact dir
import shutil
for f in ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "chat_template.jinja",
          "vocab.json", "merges.txt", "special_tokens_map.json"):
    src = f"/content/qt/quant27b-gptq4-toolace/{f}"
    if os.path.exists(src):
        shutil.copy(src, f"{OUT_DIR}/{f}")

from src.edit_ledger import record                                # noqa: E402
record({"method": "gptq-int4-packed-27b",
        "target": f"{MODEL} {len(targets)} linears @ {BITS}bit packed",
        "teach_gain": f"declared {exam['decl_right']}/{exam['declared']}",
        "neighbor_cost": f"none-picks {exam['none']}",
        "n_eval": N_ITEMS,
        "receipt": f"save_int4_{TS}.json"})
_rcpt = {"artifact": "qwen27b-int4-toolace", "exam": exam,
         "packed_gb": tot / 1e9, "n_targets": len(targets), "ts": TS}
json.dump(_rcpt, open(f"{RESULTS}/save_int4_{TS}.json", "w"))
json.dump(_rcpt, open(f"{OUT_DIR}/PACKED_RECEIPT.json", "w"), indent=1)
print(f"[int4] receipt {RESULTS}/save_int4_{TS}.json", flush=True)

# ---- optional immediate HF upload (session-death insurance) ----
# hrun9 died seconds after this script finished — artifact was lost.
# If HF_TOKEN is in env, push the packed dir NOW, before anything else.
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi                       # noqa: E402
    repo = os.environ.get("INT4_REPO", "KidIkaros/qwen27b-int4-toolace")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, private=True, exist_ok=True)
    card = f"""---
license: other
base_model: 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored
library_name: transformers
tags: [qwen, gptq, int4, packed, toolace, function-calling, compressed]
---

# qwen27b-int4-toolace (PACKED int4, extended scope)

TRUE 4-bit GPTQ artifact: nibble-packed codes + fp16 per-(row,128-block)
scales, ~{tot / 1e9:.1f}GB vs ~53.8GB bf16. Scope: {len(targets)} linears
(~94% of params) incl. linear_attn projections; embed/lm_head/norms bf16.

NOT loadable by vanilla transformers — use src/int4_load.py
(dequantize-on-load to bf16). See INT4_MANIFEST.json for the recipe.
Sibling of qwen27b-gptq4-toolace (bf16-dequant backup, 256-target scope).
Pre-H-Net base artifact.

Pinned exam: {exam['decl_right']}/{exam['declared']} declared
(skip={VSKIP}, n={N_ITEMS}).
"""
    open(f"{OUT_DIR}/README.md", "w").write(card)
    t0 = time.time()
    api.upload_folder(repo_id=repo, folder_path=OUT_DIR,
                      commit_message=f"qwen27b-int4-toolace packed artifact "
                                     f"(exam {exam['decl_right']}/"
                                     f"{exam['declared']}, ts={TS})")
    print(f"[int4] HF upload done {time.time() - t0:.0f}s -> {repo}",
          flush=True)
print("[int4] complete", flush=True)
