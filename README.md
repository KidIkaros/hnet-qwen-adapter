# hnet-qwen-adapter

**Can a frozen Qwen3.5 be turned into an H-Net-style byte-level model with zero training — constructed
glue only?**

The follow-on program from the qwentensor fork's research gate (`docs/DECISION_HNET.md` there): the
companion report predicted the composed system runs *degraded, not competitive* (distribution shift
through every frozen layer; the WSP post's Gemma control: editing cannot add a capability), and nobody has
measured how degraded. This repo measures it.

## The v0 architecture (no trained weights anywhere)

```
bytes ──► periodic byte-phase chunker (every N bytes; the falsification test showed the trained
          18.6M router's decisions are indistinguishable from this: +0.0009/+0.0011 bpb)
     ──► patch embedder: each chunk = the SUM of its bytes' own token-embedding rows
         (221/256 bytes map to exactly one of Qwen's embedding rows; constructed, zero trained weights;
         MEGABYTE's patch embedder reduced to a sum)
     ──► Qwen3.5-0.8B frozen stack over chunk positions (inputs_embeds; the KV/state machinery unchanged)
     ──► LM head at each chunk position: the next chunk's FIRST byte-token
```

The distribution shift is minimized *by construction*: every patch vector is a linear combination of the
model's own token embeddings. Chunk sequences are ~1.3× shorter than BPE sequences for the same text
(6 bytes/chunk vs 4.57 bytes/token), so the sequence statistics are comparable.

## The measurement (the unmeasured question — now measured)

First-byte bpb of composed v0 on 8 hnet-kaggle markdown docs (65536 bytes, OOD for the H-Net but the
honest stress case), period-6 chunks, Qwen3.5-0.8B on Colab A100 (`results/a100_v0_period6.log`):

| system | what it predicts | bpb |
|---|---|---|
| trained 18.6M H-Net (the reference) | every byte | 1.4177 (published, its val split) |
| stock Qwen3.5-0.8B, native BPE | every byte (via tokens) | **4.2567** |

Construction sweep (`results/a100_construction_sweep.log`), byte-marginalized head throughout.
Adaptive chunkers calibrate to mean_len ≈ 6 — the same compression ratio as period 6, the fair
comparison; sim/entropy targets are *harder* bytes (boundaries land on transitions/surprises):

| chunker | sum | mean | last | n_preds (sum) | task |
|---|---|---|---|---|---|
| period=1 | **5.2250** | — | — | 65,132 | every byte — byte-level-Qwen control |
| period=3 | 6.4325 | 8.9436 | 9.4592 | 21,701 | every 3rd byte |
| period=6 | 9.1890 | 8.9554 | **8.7607** | 10,848 | every 6th byte |
| period=12 | 8.5120 | 8.9565 | 8.6820 | 5,421 | every 12th byte |
| **sim~6** | **6.3867** | 7.3099 | 7.1615 | 11,400 | boundary bytes, mean_len 5.55 |
| **entropy~6** | **7.3175** | 9.4454 | 10.1968 | 11,489 | boundary bytes, mean_len 5.58 |

Coverage: 221/256 bytes have a single-token row; ~0.6-3% of chunks contain an unmappable byte
(zero-vector holes, counted in input cov) and ~0.6-3% of targets are skipped (target cov).

> **Metric correction (2026-11-05):** all "stock BPE" numbers below were computed by a buggy
> reference (per-token surprisal, ~3× inflated). Corrected stock: **1.57 md / 0.82 val** —
> implausibly good, i.e. the eval data is very likely inside Qwen's pretraining corpus and
> stock-vs-adapter gaps are contamination-confounded. Adapter-family comparisons are unaffected.

**Cross-domain check — the H-Net val split** (85,796 bytes, `results/a100_val_eval.log`): the ordering
survives on real English. Stock BPE 3.7560, byte-level 4.3310, best constructed chunked EFF ~4.59
(entropy~12) — adaptive still beats periodic at matched compression, coverage rises to 0.998. The
compression optimum drifts toward ~8-12 on natural text vs ~6 on markdown.

**v1 — the learned glue (the ZHA recipe):**
- **Granularity cost is small — the surprise**: period 1 degenerates the adapter to "Qwen as a
  byte-level model" (byte-token rows in, marginalized head out) at **5.23 bits vs stock 4.26 — only
  +0.97**. The frozen stack handles byte-granular input far better than the distribution-shift
  prediction suggested.
- **Boundary placement is where the information lives**: at matched ~6× compression, content-aware
  boundaries beat periodic by 1.9-2.8 bits — and sim~6 (6.39 at mean 5.55) even beats period-3
  (6.43) on ~2× fewer positions. Chunk quality beats quantity. This rhymes with the frozen-router
  result from the trained H-Net: the router's learned decisions carry real information — but a
  training-free similarity rule captures most of it.
- **sum dominates mean everywhere** — consistent with LSA keeping unnormalized sums for centroids.
- **Boundary-weighting is conditional**: "last" helps under arbitrary periodic boundaries (the last
  byte carries position info) but hurts under semantic boundaries (the chunk is a coherent unit best
  served whole). ZHA's query-answerability insight confirmed — with a caveat.
- **Cross-config comparisons are task-confounded**: each chunker scores a different byte subset;
  adaptive boundaries predict *harder* bytes by construction, which makes their wins more
  meaningful, not less.
- The readout matters: scoring single-byte *tokens* directly reads ~3 bits worse than uniform — a
  metric artifact. `ByteHead` (P(byte) = mass of all tokens whose first byte is b) is the honest
  readout.

**v1 — the learned glue (the ZHA recipe):** a rank-32 residual projection on the patch sum (~160k
params, identity at init) trained 156 steps on the byte-marginalized task loss, trunk frozen:

| system (3 held-out eval docs) | bpb | effective |
|---|---|---|
| **v1 learned projection (3 seeds)** | **2.5585 ± 0.0515** | **5.2728 ± 0.0091** |
| constructed sum (same chunker/docs) | 4.1579 | 5.5558 |
| stock BPE on eval docs | 4.9226 | — |
| byte-level adapter on eval docs | 5.5730 | 5.5730 |

Training only the patch interface recovers **−1.66 bits** over the best construction — the ZHA
indexer pattern works at our scale (~2 min A100). Caveats: same-domain eval, boundary-byte task,
interior-byte decoding untouched. Full writeup: `docs/DECISION.md`.

Papers: MEGABYTE (2305.07185, the frozen-chunking shape), BLT (2412.09871, §2.3 entropy patching),
H-Net (2507.07955, the mechanism), ZHA (2610.42112, the train-only-the-indexer recipe + recursive
summaries), LSA (2610.42147, the centroid construction + training-free Ward clustering), the
companion report (`~/Documents/HNet_Qwen_Companion_Research_20261104/`), the weight-space-programming
post (rikiyat.github.io/blog/weight-space-programming/).

## Layout

```
src/       adapter.py (chunker, patch embedder, ChunkLM) · corpus.py (val-split loader, byte-token map)
kernel/    colab_run.py (the one-shot measurement; all compute runs on Colab)
tests/     fast CPU tests (logic only; the model runs live on Colab)
docs/      DESIGN.md · RESULTS.md
configs/   run configurations
```

## Compute policy

Everything runs on Colab (`colab run --gpu A100 kernel/colab_run.py`); the local machine edits code and
pushes. Check `colab usage` before each run.
