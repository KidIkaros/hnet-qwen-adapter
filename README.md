# hnet-qwen-adapter

**Can a frozen Qwen3.5 be turned into an H-Net-style byte-level model with zero training — constructed
glue only?**  Answer (final): yes, as a scorer — and with a small trained byte decoder it becomes a
real tokenizer-free LM below the frozen trunk's own byte-level floor. The decoder, not the trunk or
the glue, is the measured bottleneck. See **Final state** below.

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
| stock Qwen3.5-0.8B, native BPE | every byte (via tokens) | **1.57** (corrected — see note below) |

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
survives on real English. Stock BPE 0.82 (corrected; `a100_family_sweep.log`), byte-level 4.3310,
best constructed chunked EFF ~4.59 (entropy~12) — adaptive still beats periodic at matched
compression, coverage rises to 0.998. The compression optimum drifts toward ~8-12 on natural text
vs ~6 on markdown.

**v1 — the learned glue (the ZHA recipe):**
- **Frozen Qwen runs as a byte-level model — but "≈ stock" is retracted**: period 1 degenerates the
  adapter to "Qwen as a byte-level model" (byte-token rows in, marginalized head out) at **5.23 md /
  4.33 val vs corrected stock 1.57 / 0.82** — the real token-interface gap is ~3.5–4 bits (the
  "+0.97" figure came from the buggy stock baseline). The frozen stack still handles byte-granular
  input far better than the distribution-shift prediction suggested; it just isn't near-parity.
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
| ridge→BPE (closed-form, zero training) | 4.5760† | 5.4466† |
| stock BPE on eval docs (corrected) | 1.5692 | — |
| byte-level adapter on eval docs | 5.5730 | 5.5730 |

† periodic-6 chunker, `a100_construct_sweep.log` / `a100_construct_ledger.log`. On the val split
the closed-form ridge scores **4.32 effective** — already at the byte-level floor with no training
at all.

Training only the patch interface recovers **−1.66 bits** over the best construction — the ZHA
indexer pattern works at our scale (~2 min A100). Full writeup: `docs/DECISION.md`.

## Final state — where the program landed

The full pipeline ran to ground. Corrected headline (`docs/RESULTS.md` for the full tables and
log citations):

| component | verdict | evidence |
|---|---|---|
| trunk | **solved** — Qwen3.5-4B-Base is the best byte trunk (3.35 val / 4.53 decontam); or merge instruct→Base — the α-dial recovers byte competence linearly, no barrier | `a100_family_sweep.log`, `a100_merge_sweep.log`, `a100_decontam.log` |
| patch map | **solved** — closed-form ridge→BPE ≈ trained rank-32 (4.32 val effective, zero training) | `a100_construct_ledger.log` |
| byte decoder | **the bottleneck** — v2 (ridge patch map + 13.2M decoder) hits **4.99 md / 4.13 val** on 0.8B, below its byte-level floor; but the same recipe on the better 4B-Base trunk is *worse* (5.11–5.15 / 4.21–4.30) | `a100_v3_scaled_gen.log`, `a100_v2_4bbase.log` |
| cross-size glue transfer | **dead** — conjugating the 0.8B-learned ΔW to 4B-Base (R⁺ΔWᵀR) scores 5.95 vs identity 5.66; per-size refit (4.37) required | `a100_conjugate.log` |

Decontaminated protocol (`data/decontam/`, `kernel/decontam_eval.py`, 48 post-cutoff arxiv+GitHub
docs): stock stays ~1.0 bpb → mostly real modeling, not memorization; the byte interface costs
~4 bits through Qwen on clean data too. Next lever: **decoder data scaling** — the decoder is
data-bound, everything upstream of it is solved.

Papers: MEGABYTE (2305.07185, the frozen-chunking shape), BLT (2412.09871, §2.3 entropy patching),
H-Net (2507.07955, the mechanism), ZHA (2610.42112, the train-only-the-indexer recipe + recursive
summaries), LSA (2610.42147, the centroid construction + training-free Ward clustering), the
companion report (`~/Documents/HNet_Qwen_Companion_Research_20261104/`), the weight-space-programming
post (rikiyat.github.io/blog/weight-space-programming/).

## Layout

```
src/       adapter.py (chunkers, patch embedder, ChunkLM, ByteHead, corrected qwen_byte_bpb)
           construct.py (closed-form maps: norm_match, cov_recolor, ridge→BPE)
           learned.py (rank-32 residual patch map) · decoder.py (byte decoder) · generate.py
           corpus.py (val-split loader, byte-token map)
kernel/    colab_run.py · train_projection.py · train_decoder.py · construct_sweep.py
           family_sweep.py · merge_sweep.py · decontam_eval.py · conjugate_transfer.py
           (all compute runs on Colab)
tests/     fast CPU tests (logic only; the model runs live on Colab)
docs/      DECISION.md · RESULTS.md · RESEARCH_WEIGHT_TRANSFER.md
data/      decontam/ (post-cutoff arxiv+GitHub eval docs)
results/   a100_*.log (every number in the docs cites one) · byte_decoder.pt
```

## Compute policy

Everything runs on Colab (`colab run --gpu A100 kernel/colab_run.py`); the local machine edits code and
pushes. Check `colab usage` before each run.
