# Decision memo — adapter program after the construction sweep + v1 projection

Date: 2026-11-05. Context: the no-training adapter v0 measured (`results/a100_*.log`), the
construction sweep done, and the first trained-glue run (v1, low-rank patch projection) landed.

## Measured facts (Qwen3.5-0.8B, Colab A100, byte-marginalized head)

**Shared-8 markdown docs (65536 bytes):**

| system | task | bpb |
|---|---|---|
| stock BPE | every byte | 4.2560 |
| byte-level adapter (period 1) | every byte | 5.2250 |
| sim~6 + sum (constructed) | boundary bytes, mean 5.55 | 6.3867 |
| entropy~6 + sum (constructed) | boundary bytes, mean 5.58 | 7.3175 |
| period-6 + sum (constructed) | every 6th byte | 9.1890 |

**Compression curve (adaptive, sum):** U-shaped, optimum at mean_len ≈ 5.5–6 — matching H-Net's
target_ratio=6. sim: 7.27/7.07/6.39/7.66/9.74 at means 3/4/6/8/12; entropy: 8.42/7.91/7.32/7.64/8.24.

**v1 trained glue (3 held-out same-domain eval docs, 4349 targets):**

| system | bpb |
|---|---|
| learned rank-32 projection (156 steps, ~160k params) | **2.4940** |
| constructed sum baseline (same chunker, same docs) | 4.1579 |
| stock BPE on those docs | 4.9226 |
| byte-level adapter on those docs | 5.5730 |

## What the evidence says

1. **Frozen Qwen is byte-competent**: byte-level input costs only +0.97 bits vs native BPE. The
   granularity wall from the research gate did not exist at this scale.
2. **Boundary placement is learnable-by-construction**: training-free similarity boundaries capture
   most of what the trained router had (~2-3 bits over periodic at matched compression, on harder
   targets). Consistent with the frozen-router null result on the trained H-Net.
3. **The constructed patch vector is the bottleneck — and it is cheap to fix**: 160k learned
   parameters recover −1.66 bits on held-out docs. The ZHA pattern (train only the indexer, freeze
   the trunk) works at our scale with trivial compute (~2 min A100).
4. **Honest caveats**: eval docs are same-domain markdown (structured → easy boundaries); all
   multi-byte numbers are boundary-byte tasks, not full-byte bpb; cross-domain transfer unmeasured;
   the adapter still has no mechanism for interior-byte prediction — a real H-Net needs the decoder
   side (dechunking) which this program has not touched.

## Decisions

- **The patch interface should be learned, not constructed.** v0.x construction space (sum/mean/
  boundary-weighted × periodic/sim/entropy) is mapped; returns have diminished. The learned low-rank
  map is strictly better where measured.
- **Next: scale the v1 recipe, don't add machinery.** Options ranked by information-per-cost:
  (a) eval on genuinely different text (the H-Net val split via kaggle upload, or code/prose) —
      does the 2.49 transfer or is it markdown-specific?
  (b) rank/epochs sweep — where does the learned map saturate?
  (c) entropy boundaries + learned projection (boundary quality may still matter);
  (d) a learned byte-decoder for interior bytes — the missing half of a real byte LM, and the point
      where the "adapter" becomes a real model with a frozen core.
- **qwentensor stays the hosting instrument**: none of this changed the in-substrate story — the
  learned projection is host-side glue either way.

## Open questions parked

- Can a WSP-style construction produce what the rank-32 map learned? (The falsifiable WSP question —
  now with a concrete target: read the learned A B^T and see if it's interpretable/construcible.)
- Does the composed system transfer to the H-Net val split (needs the kaggle dataset on the VM)?
- Interior-byte decoding: is the gap closable with a learned dechunk head at all?
