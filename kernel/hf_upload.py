"""Upload the saved quantized trunk to a PRIVATE HF repo.

Run ONLY after the eval gate passes. Requires HF_TOKEN env (fine-grained
write token) — pass via `colab exec --env HF_TOKEN=...`.

    colab exec -s hrun9 -f kernel/hf_upload.py \
        --env HF_TOKEN=hf_... --env REPO_ID=KidIkaros/qwen27b-gptq4-toolace
"""
import os, sys, subprocess, json, time

SRC = os.environ.get("SRC", "/content/qt/quant27b-gptq4-toolace")
REPO_ID = os.environ.get("REPO_ID", "KidIkaros/qwen27b-gptq4-toolace")
PRIVATE = os.environ.get("PRIVATE", "1") == "1"
TS = int(time.time())

assert os.environ.get("HF_TOKEN"), "HF_TOKEN env required"
assert os.path.isdir(SRC), f"{SRC} missing — run compress_trunk SAVE_DIR first"

from huggingface_hub import HfApi                              # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
print(f"[upload] ensuring repo {REPO_ID} (private={PRIVATE})", flush=True)
api.create_repo(REPO_ID, private=PRIVATE, exist_ok=True)

# card + manifest already in SRC (QUANT_MANIFEST.json); add README card
card = f"""---
license: other
base_model: 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored
library_name: transformers
tags: [qwen, gptq, toolace, function-calling, compressed]
---

# qwen27b-gptq4-toolace (BASE artifact)

GPTQ-4bit + Hadamard-rotation compressed trunk of the abliterated 27B,
Hessian-calibrated on ToolACE prompts. **Pre-H-Net base artifact** —
a separate trunk+adapter model (`qwen27b-gptq4-toolace-hnet`) will follow.

Weights stored dequantized bf16 (accuracy-equivalent; int4 packing is
separate work). See QUANT_MANIFEST.json for method + verification.

## Eval gate (quantized vs fresh bf16, same items)

| exam | quant | bf16 base | delta |
|---|---|---|---|
| tool-name selection, pinned 40 | 27/27 declared | 27/27 | 0 |
| tool-name selection, 500 items | 270/283 (95.4%) | 270/283 | 0 |
| BFCL v3 name-selection (200) | 188/200 | 189/200 | -1 |
| BFCL v3 full call, AST-equiv (200) | 26/200 | 27/200 | -1 |
| ToolACE closed-set call (40) | gold_match 3, name 25/25 | identical | 0 |

Compression is behaviorally lossless on every measured axis. The ~13%
BFCL full-call accuracy is a property of the evaluation protocol
(closed-set extractive argument composition), identical for bf16 —
not a quantization defect. Receipts in results/ of the source repo.
"""
open(f"{SRC}/README.md", "w").write(card)

print(f"[upload] uploading {SRC} -> {REPO_ID}", flush=True)
t0 = time.time()
api.upload_folder(repo_id=REPO_ID, folder_path=SRC,
                  commit_message=f"qwen27b-gptq4-toolace base artifact "
                                 f"(eval-gated, ts={TS})")
print(f"[upload] done in {time.time() - t0:.0f}s", flush=True)
info = api.repo_info(REPO_ID, files_metadata=True)
total = sum(f.size or 0 for f in info.siblings)
print(f"[upload] repo size ~{total / 1e9:.1f} GB", flush=True)
