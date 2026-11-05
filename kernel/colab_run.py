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

# ---- the measurement: stock control, then the construction sweep
# Constructions under test (all training-free):
#   chunkers:  periodic (baseline), EntropyChunker (BLT entropy patching — the period-1 byte-level
#              Qwen supplies its own boundary scores), SimChunker (consecutive-embedding cosine sim —
#              the streaming analog of LSA's Ward clustering).  Both adaptive chunkers calibrate their
#              threshold to mean length ~6 -> the same compression ratio as period 6.
#   patchers:  sum (LSA's centroid), mean, last (boundary-weighted exp-decay — ZHA's
#              query-answerability insight: preserve what the prediction needs).
NDOCS = int(os.environ.get("NDOCS", "8"))
docs = docs[:NDOCS]

from src.adapter import (ByteHead, EntropyChunker, PeriodicChunker, SimChunker,  # noqa: E402
                         effective_bpb)
_shared_head = ByteHead(tok)

lm1 = ChunkLM(model, tok, period=1, agg="sum", byte_head=_shared_head)
print("[entropy] byte-level scores + fallback NLLs from the period-1 model", flush=True)
entropy = [lm1.byte_entropy(d) for d in docs]
fb_nll = [lm1.byte_nll(d) for d in docs]                 # the byte-level fallback per position


def adapter_score(chunker, agg, need_aux):
    lm = ChunkLM(model, tok, agg=agg, byte_head=_shared_head, chunker=chunker)
    tot = n = 0.0; covs = []; tcovs = []; skipped = 0; mlen = []; eff_bits = eff_bytes = 0.0
    for i, d in enumerate(docs):
        sc = entropy[i] if need_aux else None
        try:
            pos, cnll, c, ns = lm.boundary_nll(d, aux=sc)   # one forward per config per doc
            tot += float(cnll.sum()); n += len(pos); covs.append(c)
            tcovs.append(len(pos) / max(ns - 1, 1))
            mlen.append(len(d) / ns)
            eff_bits += effective_bpb(d, pos, cnll, fb_nll[i]) * len(d); eff_bytes += len(d)
        except ValueError as e:
            skipped += 1
            print(f"        [skip] doc {i}: {e}", flush=True)
    ci = float(np.mean(covs)) if covs else float("nan")
    ct = float(np.mean(tcovs)) if tcovs else float("nan")
    return (tot / max(n, 1), int(n), ci, ct, skipped,
            float(np.mean(mlen)) if mlen else float("nan"), eff_bits / max(eff_bytes, 1))


t0 = time.time()
qb_tot = qb_n = 0.0
for d in docs:
    qb, _, _ = qwen_byte_bpb(model, tok, d)
    qb_tot += qb * len(d); qb_n += len(d)
print(f"stock Qwen3.5-0.8B (native BPE): full-byte bpb {qb_tot / max(qb_n, 1):.4f} over {int(qb_n)} bytes", flush=True)

fb_eff = sum(n.sum() for n in fb_nll) / sum(len(n) for n in fb_nll)
print(f"[control] byte-level adapter effective bpb (its own fallback): {fb_eff:.4f}", flush=True)
print(f"\n{'chunker':>10} {'agg':>5} {'mean_len':>8} {'first-b bpb':>12} {'EFF bpb':>8} {'n_preds':>9} {'in_cov':>7} {'tg_cov':>7} {'skip':>5}", flush=True)

CONFIGS = []
for period in (1, 2, 3, 6, 12):
    for agg in ("sum", "mean", "last"):
        if period == 1 and agg != "sum":
            continue                                    # one-byte chunks: mean == last == sum
        CONFIGS.append((f"period={period}", PeriodicChunker(period), agg, False))
for agg in ("sum", "mean", "last"):
    CONFIGS.append(("entropy~6", EntropyChunker(target_mean=6.0, max_len=16), agg, True))
    CONFIGS.append(("sim~6", SimChunker(lm1.embedder, target_mean=6.0, max_len=16), agg, False))
# the compression curve for the winning family (sum only): does more compression cost or help?
for tm in (3.0, 4.0, 8.0, 12.0):
    CONFIGS.append((f"entropy~{tm:g}", EntropyChunker(target_mean=tm, max_len=24), "sum", True))
    CONFIGS.append((f"sim~{tm:g}", SimChunker(lm1.embedder, target_mean=tm, max_len=24), "sum", False))

for name, chunker, agg, need_aux in CONFIGS:
    bpb, n, ci, ct, sk, ml, eb = adapter_score(chunker, agg, need_aux)
    print(f"{name:>10} {agg:>5} {ml:>8.2f} {bpb:>12.4f} {eb:>8.4f} {n:>9} {ci:>7.4f} {ct:>7.4f} {sk:>5}", flush=True)

print(f"\ntrained 18.6M H-Net (published, full-byte): 1.4177", flush=True)
print(f"[timing] {time.time() - t0:.0f}s", flush=True)
