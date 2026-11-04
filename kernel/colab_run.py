"""The one-shot Colab measurement: the adapter v0's first-byte bpb vs stock Qwen on the same text.

Fired from the repo root:   colab run --gpu A100 kernel/colab_run.py [--markdown]
The VM is ephemeral: clone this repo, download the model + tokenizer, run, print the table.

Corpus: the hnet-kaggle val split (the published 1.4177 protocol's data) when the three val files were
uploaded to the session (persistent-session workflow: colab upload), else the hnet-kaggle repo's own
markdown (the OOD mode the falsification test used).
"""
import os, subprocess, sys, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


print("[setup] cloning the repos", flush=True)
sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-kaggle.git", "/content/qt/hnet-kaggle")
sys.path.insert(0, REPO)
os.chdir(REPO)

print("[setup] downloading Qwen/Qwen3.5-0.8B (weights + tokenizer)", flush=True)
t0 = time.time()
import torch                                                     # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
src = snapshot_download("Qwen/Qwen3.5-0.8B")
print(f"[setup] done in {time.time() - t0:.0f}s", flush=True)

import numpy as np                                               # noqa: E402
from src.corpus import ValDocs, byte_token_map                   # noqa: E402
from src.adapter import ChunkLM, qwen_byte_bpb                   # noqa: E402

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda()
bmap = byte_token_map(tok)
print(f"[setup] byte-token map: {len(bmap)}/256 bytes mapped", flush=True)

# ---- the text
val_dir = "/content/val"
if all(os.path.exists(os.path.join(val_dir, f)) for f in ("val.bin", "val_docs.u32", "val_lens.u32")):
    docs = ValDocs(val_dir).docs(8, min_len=4096)
    corpus_name = "hnet val split (the published protocol's data)"
else:
    buf = []
    for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
        if name.endswith(".md"):
            buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
    joined = b"\n".join(buf)
    docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:8]
    corpus_name = "hnet-kaggle markdown (OOD)"
print(f"[corpus] {corpus_name}: {len(docs)} docs, {sum(len(d) for d in docs)} bytes", flush=True)

# ---- the measurement
PERIOD = 6
lm = ChunkLM(model, tok, period=PERIOD)
rows = []
tot_a = tot_q = 0.0; n_a = n_q = 0.0; covs = []
t0 = time.time()
for i, d in enumerate(docs):
    try:
        bpb, n, cov = lm.first_byte_bpb(d)
        tot_a += bpb * n; n_a += n; covs.append(cov)
    except ValueError as e:
        print(f"[doc {i}] skipped (adapter): {e}", flush=True); continue
    qb, _, _ = qwen_byte_bpb(model, tok, d)
    tot_q += qb * len(d); n_q += len(d)
    print(f"[doc {i}] {len(d)} bytes: adapter first-byte bpb {bpb:.4f} (n={n}, cov={cov:.3f}) | "
          f"qwen full bpb {qb:.4f}", flush=True)
print("", flush=True)
print(f"adapter v0 (period {PERIOD}): first-byte bpb {tot_a / max(n_a, 1):.4f} over {int(n_a)} predictions "
      f"(byte coverage {np.mean(covs):.4f})", flush=True)
print(f"stock Qwen3.5-0.8B (BPE):      full-byte bpb {tot_q / max(n_q, 1):.4f} over {int(n_q)} bytes", flush=True)
print(f"trained 18.6M H-Net (published, full-byte): 1.4177", flush=True)
print(f"[timing] {time.time() - t0:.0f}s", flush=True)
