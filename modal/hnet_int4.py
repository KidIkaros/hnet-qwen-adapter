"""Modal app: packed-int4 GPTQ rerun of the 27B trunk (hrun9 successor).

Per AGENTS.md rules: deploy app + spawn via `modal run --detach` (never
plain `modal run`), results to a Modal Volume, everything committed.

    modal secret create hf-token HF_TOKEN=hf_...   # once
    modal run --detach modal/hnet_int4.py

Pipeline (port of kernel/save_int4.py, colab preamble removed):
  load bf16 27B -> GPTQ-4bit int-capture on 400 linears (incl linear_attn)
  -> nibble-pack + fp16 scales -> safetensors shards -> dequant back into
  resident model -> pinned 40-item exam -> manifest/receipt -> HF upload
  (KidIkaros/qwen27b-int4-toolace, private) -> volume backup.
"""
import modal

app = modal.App("hnet-int4")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy")
    .add_local_dir("src", "/root/repo/src")
)


@app.function(image=image, gpu="H200", timeout=4 * 3600, memory=131072,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def run_int4():
    import os, sys, json, time, glob, shutil
    os.environ["HF_HOME"] = "/hf"
    sys.path.insert(0, "/root/repo")
    import torch
    from src.probe_common import stream_toolace
    from src.compress import (calibrate_hessians, gptq_quantize_int,
                              pack_nibbles, dequant_int)

    BITS = int(os.environ.get("BITS", "4"))
    N_CALIB = int(os.environ.get("N_CALIB", "48"))
    BLOCK = int(os.environ.get("BLOCK_LAYERS", "4"))
    OUT_DIR = "/artifacts/int4-27b-toolace"
    SHARD_GB = float(os.environ.get("SHARD_GB", "4"))
    N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
    VSKIP = int(os.environ.get("VSKIP", "2500"))
    MODEL = os.environ.get(
        "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")
    TS = int(time.time())
    BASE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")
    EXTRA_SUFFIXES = ("in_proj_qkv", "in_proj_z", "out_proj")

    from transformers import (AutoTokenizer, AutoModelForCausalLM)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "130GiB", "cpu": "70GiB"})
    model.eval()
    print("[int4] model loaded", flush=True)

    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        s = n.split(".")[-1]
        if s in BASE_SUFFIXES or ("linear_attn" in n and s in EXTRA_SUFFIXES):
            targets.append((n, m))
    print(f"[int4] {len(targets)} quantization targets", flush=True)
    _meta = [n for n, l in targets if l.weight.is_meta]
    if _meta:
        raise RuntimeError(f"meta weights present ({len(_meta)})")

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

    packed = {}
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
                packed[f"{name}.qweight"] = pack_nibbles(codes).cpu()
                packed[f"{name}.scales"] = scales.half().cpu()
                del W, Hf, codes, scales
                torch.cuda.empty_cache()
        del H
        print(f"[int4] block {bi + 1}/{len(flat)} done "
              f"({len(packed) // 2} linears, {time.time() - t0:.0f}s)",
              flush=True)
    print(f"[int4] quantized {len(packed) // 2} linears @ {BITS}bit",
          flush=True)

    tset = {n for n, _ in targets}
    for k, v in model.state_dict().items():
        mod = k.rsplit(".", 1)[0]
        if mod in tset and k.endswith(".weight"):
            continue
        packed[k] = v.detach().to("cpu", torch.bfloat16).contiguous()
    print(f"[int4] packed dict complete ({len(packed)} tensors)", flush=True)

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
    from safetensors.torch import save_file
    for i, sd in enumerate(shards):
        fn = f"{OUT_DIR}/int4-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sd, fn)
        print(f"[int4] wrote {os.path.basename(fn)}", flush=True)
    tot = sum(os.path.getsize(os.path.join(OUT_DIR, f))
              for f in os.listdir(OUT_DIR) if f.endswith(".safetensors"))
    print(f"[int4] total packed size {tot / 1e9:.1f}GB", flush=True)
    art.commit()

    # dequantize packed weights back into resident model for exam
    print("[int4] dequantizing into model for exam...", flush=True)
    with torch.no_grad():
        for name, lin in targets:
            pk = packed[f"{name}.qweight"].to("cuda")
            sc = packed[f"{name}.scales"].to("cuda").float()
            Wq = dequant_int(pk, sc, lin.weight.shape[1], bits=BITS)
            lin.weight.data.copy_(Wq.to(lin.weight.dtype))
            del pk, sc, Wq
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _prefix_lp(ctx_ids, cand_ids):
        ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
        logits = model(ids).logits[0]
        s = len(ctx_ids) - 1
        lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
        return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))

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
    exam = {"right": right, "declared": decl, "decl_right": decl_right,
            "none": none_n, "n": N_ITEMS}
    print(f"[exam] {exam}", flush=True)

    manifest = {
        "artifact": "qwen27b-int4-toolace",
        "lineage": "PACKED int4 GPTQ trunk, EXTENDED scope (incl. "
                   "linear_attn projections). Sibling of "
                   "qwen27b-gptq4-toolace. Pre-H-Net base artifact.",
        "base_model": MODEL,
        "method": f"GPTQ {BITS}-bit sym blocksize=128, per-(row,block) "
                  f"scales, nibble-packed, Hessian calib N={N_CALIB} "
                  f"(stream_toolace@1224). rotate=0 (inert on this arch).",
        "scope": {"n_linears": len(targets),
                  "excluded_bf16": ["embed_tokens", "lm_head", "norms",
                                    "linear_attn.in_proj_a/b",
                                    "conv/ssm params"]},
        "packing": {"code_order": "even=high nibble, odd=low",
                    "scales_dtype": "float16", "zero": (2 ** BITS - 1) / 2,
                    "blocksize": 128},
        "n_calib": N_CALIB, "bits": BITS, "exam": exam,
        "exam_slice": {"skip": VSKIP, "n": N_ITEMS}, "ts": TS,
    }
    json.dump(manifest, open(f"{OUT_DIR}/INT4_MANIFEST.json", "w"),
              indent=1)
    rcpt = {"artifact": "qwen27b-int4-toolace", "exam": exam,
            "packed_gb": tot / 1e9, "n_targets": len(targets), "ts": TS}
    json.dump(rcpt, open(f"{OUT_DIR}/PACKED_RECEIPT.json", "w"), indent=1)
    json.dump(rcpt, open("/artifacts/save_int4_receipt.json", "w"),
              indent=1)

    # config/tokenizer from HF cache copy of the model dir
    src_dir = os.path.join(os.environ["HF_HOME"], "hub",
                           "models--" + MODEL.replace("/", "--"),
                           "snapshots")
    snap = sorted(glob.glob(f"{src_dir}/*"))[-1] if glob.glob(
        f"{src_dir}/*") else None
    if snap:
        for f in ("config.json", "generation_config.json",
                  "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "vocab.json", "merges.txt",
                  "special_tokens_map.json"):
            p = os.path.join(snap, f)
            if os.path.exists(p):
                shutil.copy(p, f"{OUT_DIR}/{f}")

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
Sibling of qwen27b-gptq4-toolace (bf16-dequant backup).
Pre-H-Net base artifact.

Pinned exam: {exam['decl_right']}/{exam['declared']} declared
(skip={VSKIP}, n={N_ITEMS}).
"""
    open(f"{OUT_DIR}/README.md", "w").write(card)
    art.commit()

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        repo = "KidIkaros/qwen27b-int4-toolace"
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, private=True, exist_ok=True)
        t0 = time.time()
        api.upload_folder(repo_id=repo, folder_path=OUT_DIR,
                          commit_message="qwen27b-int4-toolace packed "
                                         f"(exam {exam['decl_right']}/"
                                         f"{exam['declared']}, ts={TS})")
        print(f"[int4] HF upload done {time.time() - t0:.0f}s -> {repo}",
              flush=True)

    print("RESULT_JSON " + json.dumps(rcpt), flush=True)
    print("[int4] complete", flush=True)


@app.local_entrypoint()
def main():
    call = run_int4.spawn()
    print(f"spawned call_id={call.object_id}", flush=True)
