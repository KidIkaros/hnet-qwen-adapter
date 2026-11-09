"""Persist the resident GPTQ-quantized 27B trunk to disk for HF backup.

The compress_trunk run left the quantized model in kernel globals on
hrun9 (all weights real GPU tensors — 72GiB budget meant zero meta
offload). This driver saves it in standard transformers format so it
can be uploaded to HuggingFace and recalled anywhere.

IMPORTANT: weights are DEQUANTIZED bf16 — accuracy-equivalent to the
4-bit GPTQ result but NOT size-packed. int4 packing (codes+scales) is
separate work; this artifact prioritizes recall fidelity.

    colab exec -s hrun9 -f kernel/save_quant.py
"""
import os, json, time, subprocess

OUT = "/content/qt/quant27b-gptq4-toolace"
TS = int(time.time())

_g = globals()
model = _g.get("model")
tok = _g.get("tok")
assert model is not None and tok is not None, "no resident model/tok"

print(f"[save] saving quantized trunk -> {OUT}", flush=True)
t0 = time.time()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"[save] done in {time.time() - t0:.0f}s", flush=True)

du = subprocess.run(["du", "-sh", OUT], capture_output=True,
                    text=True).stdout.strip()
print(f"[save] {du}", flush=True)

manifest = {
    "artifact": "qwen27b-gptq4-toolace",
    "base_model": "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored",
    "method": "GPTQ 4-bit symmetric + Hadamard rotation, Hessian "
              "calibrated on 48 ToolACE prompt blocks, blockwise "
              "(16 blocks x 4 layers), 256 trunk linears",
    "dtype_stored": "bfloat16 (dequantized — accuracy-equivalent, "
                    "NOT int4-packed)",
    "verification": {
        "pinned_name_exam_40": "27/27 declared -> 27/27 declared, "
                               "none-picks 3->3 (identical)",
        "big_exam_500_skip0": "resident quant: 270/283 declared "
                              "(95.4%), 49 abstains",
        "receipts": ["results/compress_trunk_1794212023.json",
                     "results/big_exam_1794217042_resident.json"],
    },
    "ts": TS,
}
json.dump(manifest, open(f"{OUT}/QUANT_MANIFEST.json", "w"), indent=1)
print("[save] manifest written", flush=True)
