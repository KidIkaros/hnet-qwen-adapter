# Results — a tokenizer-free adapter over frozen Qwen3.5-0.8B

**Question**: can a frozen token-level LLM be driven at byte/chunk level through constructed or
cheaply-learned glue — no trunk training?  Model: Qwen3.5-0.8B (frozen, bf16, Colab A100).  Head:
the LM head marginalized to bytes (P(b) = mass of all tokens whose first byte is b; 221/256 bytes
have a single-token embedding row).  Metric: **effective bpb** — total bits / total bytes, chunk
boundaries scored by the chunk model, all other positions by the byte-level fallback, position 0 at
the uniform 8 bits.  Every number below is on that one axis.

## Headline numbers

| system | interface cost | markdown (65,536 B) | H-Net val split (85,796 B) |
|---|---|---|---|
| stock BPE | native | 4.2560 | 3.7560 |
| byte-level (period-1) | none | 5.2940 | 4.3310 |
| best constructed chunker | zero training | 5.5614 (sim~6+sum) | 4.5931 (entropy~12+sum) |
| **v1 learned projection (rank-32)** | ~160k params, 156 steps | **5.2728 ± 0.0091** | — |
| trained 18.6M H-Net (reference) | fully trained | — | **1.4177** (published) |

## Findings

1. **Frozen Qwen is byte-competent.** Byte-token rows in, byte-marginalized head out: +0.97 bits on
   markdown, **+0.58 on natural English** vs native BPE. The granularity wall predicted by the
   research gate did not exist at this scale.

2. **Boundary placement carries the learnable signal — and training-free rules capture most of it.**
   At matched ~6x compression, content-aware boundaries beat periodic by 1.9-2.8 boundary-bpb
   (sim 6.39, entropy 7.32, periodic 9.19) on harder targets. Consistent with the frozen-router
   result on the trained H-Net (+0.001 bpb for frozen boundaries).

3. **Constructed patch vectors are the bottleneck.** Sum-of-byte-rows beats mean everywhere (the
   norm marks patch vectors as non-token — the LSA centroid choice). Boundary-weighting helps only
   under arbitrary boundaries.

4. **Training only the glue works (the ZHA recipe).** A rank-32 residual on the patch sum — 160k
   params, 156 steps, 2 min A100 — cut boundary bpb 4.16 → 2.56 on held-out docs (3 seeds,
   ±0.05) and flipped the effective axis net-positive (5.27 < 5.29 byte-level). The autopsy shows a
   structured ~0.25‖I‖ correction, not a perturbation.

5. **No constructed chunker beats plain byte-level on the full-byte axis.** Without a trained
   interface, chunking is a compute play, not a prediction play.

6. **Cross-domain holds.** The ordering, the adaptive>periodic margin, and the U-curve all survive
   on the H-Net val split; the compression optimum drifts ~6→8-12 on natural English.

## Caveats

- v1 trained/eval'd on same-domain markdown docs; its val-split transfer is unmeasured.
- The adapter predicts boundary bytes only — it is a scorer, not a complete LM. Interior-byte
  decoding (the BLT/MEGABYTE local-model half) is the missing piece for a real model.
- All runs: single A100, bf16, unauthenticated HF download of Qwen/Qwen3.5-0.8B.

## Logs

`results/a100_v0_period6.log` · `a100_sweep.log` · `a100_construction_sweep.log` ·
`a100_compression_curve.log` · `a100_effective.log` · `a100_v1_seeds.log` · `a100_val_eval.log`
