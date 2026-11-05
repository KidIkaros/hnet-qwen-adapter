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

# ---- the measurement: the same-position control, then the period x aggregation sweep
# Period 1 is the control: each chunk IS one byte, so the adapter degenerates to "Qwen as a byte-level
# model through the byte-fallback rows" — decomposing the v0 loss into granularity cost (period 1 vs
# stock) + chunking cost (period >1 vs period 1).
NDOCS = int(os.environ.get("NDOCS", "8"))
docs = docs[:NDOCS]

print("[control] stock Qwen, same first-byte positions (byte-marginalized head at the byte-token that ENDS the chunk)", flush=True)


from src.adapter import ByteHead                                    # noqa: E402
_shared_head = ByteHead(tok)


def adapter_score(period, agg):
    lm = ChunkLM(model, tok, period=period, agg=agg, byte_head=_shared_head)
    tot = n = 0.0; covs = []
    for d in docs:
        try:
            bpb, k, c = lm.first_byte_bpb(d)
            tot += bpb * k; n += k; covs.append(c)
        except ValueError:
            continue
    return tot / max(n, 1), int(n), float(np.mean([c for c, _ in covs])), float(np.mean([t for _, t in covs]))


t0 = time.time()
qb_tot = qb_n = 0.0
for d in docs:
    qb, _, _ = qwen_byte_bpb(model, tok, d)
    qb_tot += qb * len(d); qb_n += len(d)
print(f"stock Qwen3.5-0.8B (native BPE): full-byte bpb {qb_tot / max(qb_n, 1):.4f} over {int(qb_n)} bytes", flush=True)

print(f"\n{'period':>6} {'agg':>5} {'first-byte bpb':>15} {'n_preds':>9} {'in_cov':>7} {'tg_cov':>7}", flush=True)
for period in (1, 2, 3, 6, 12):
    for agg in ("sum", "mean"):
        if period == 1 and agg == "mean":
            continue                                    # sum == mean for one-byte chunks
        bpb, n, ci, ct = adapter_score(period, agg)
        print(f"{period:>6} {agg:>5} {bpb:>15.4f} {n:>9} {ci:>7.4f} {ct:>7.4f}", flush=True)

print(f"\ntrained 18.6M H-Net (published, full-byte): 1.4177", flush=True)
print(f"[timing] {time.time() - t0:.0f}s", flush=True)
