# Decision memo — adapter program: final record

Date: 2026-11-05 (final consolidation). Context: the no-training adapter v0 measured
(`results/a100_*.log`), the construction sweep done, the trained-glue runs (v1 low-rank patch
projection, v2 composed system + byte decoder) landed, the trunk question swept across the
Qwen3.5 family, the instruct↔Base merge measured, the decontaminated eval run, and the
cross-size conjugated-transfer test executed.

**Program status: complete.**  Verdict recorded in §"Final decision" below.  The early sections
are kept as the historical record; their stock-BPE numbers predate the metric correction (see
§"Metric correction + contamination flag").

## Measured facts (Qwen3.5-0.8B, Colab A100, byte-marginalized head)

**Shared-8 markdown docs (65536 bytes):** †

| system | task | bpb |
|---|---|---|
| stock BPE | every byte | 4.2560 † |
| byte-level adapter (period 1) | every byte | 5.2250 |
| sim~6 + sum (constructed) | boundary bytes, mean 5.55 | 6.3867 |
| entropy~6 + sum (constructed) | boundary bytes, mean 5.58 | 7.3175 |
| period-6 + sum (constructed) | every 6th byte | 9.1890 |

† stock-BPE values shown as logged pre-correction; corrected implementation gives
**1.57 md / 0.82 val** — see §"Metric correction".

**Compression curve (adaptive, sum):** U-shaped, optimum at mean_len ≈ 5.5–6 — matching H-Net's
target_ratio=6. sim: 7.27/7.07/6.39/7.66/9.74 at means 3/4/6/8/12; entropy: 8.42/7.91/7.32/7.64/8.24.

**v1 trained glue (3 held-out same-domain eval docs, 4349 targets):** †

| system | bpb |
|---|---|
| learned rank-32 projection (156 steps, ~160k params) | **2.4940** |
| constructed sum baseline (same chunker, same docs) | 4.1579 |
| stock BPE on those docs | 4.9226 † |
| byte-level adapter on those docs | 5.5730 |

## What the early evidence said

1. **Frozen Qwen is byte-competent** (amended): byte-level input costs ~4 bits vs the corrected
   stock baseline, not +0.97 — the granularity wall still did not exist in the sense that the
   frozen stack runs byte-granular input as a real LM (4.33 val), but the token-interface gap is
   ~3.5 bits, not a near-parity result.
2. **Boundary placement is learnable-by-construction**: training-free similarity boundaries capture
   most of what the trained router had (~2-3 bits over periodic at matched compression, on harder
   targets). Consistent with the frozen-router null result on the trained H-Net.
3. **The constructed patch vector was the bottleneck — and cheap to fix**: 160k learned parameters
   recover −1.66 bits on held-out docs; the closed-form ridge later matched this without any
   training (see construction staircase, RESULTS.md §6).
4. **Honest caveats** (status updated): eval docs were same-domain markdown — cross-domain
   transfer has since been measured (v2: 4.13 on the H-Net val split); interior-byte decoding has
   since been built (the v2 decoder — and it is the measured bottleneck).

## Round 1 results (2026-11-05, `results/a100_effective.log`, `a100_v1_seeds.log`, `a100_val_eval.log`)

**The effective metric changed the story.** With a byte-level fallback covering unpredicted
positions, every constructed chunker lands ~5.56-5.95 on markdown — and none beats byte-level alone
(5.29). Chunking never helps *prediction*; it can only help *compute*. The learned projection flips
the sign: eval 2.5585±0.0515 boundary bpb → effective 5.2728±0.0091, now below its own fallback —
chunking goes net-positive once the interface is trained.

**Cross-domain holds.** On the H-Net val split (85,796 bytes English): byte-level 4.3310, best
constructed chunked ~4.59 — same ordering, same adaptive>periodic margin, compression optimum
drifting toward 8-12 on natural text.  (Stock shown in that log, 3.7560, is the buggy metric —
corrected: 0.82.)

**W autopsy**: ||dW||/||I|| ≈ 0.23-0.29, top singular values 5-6 decaying — a real structured
correction, not a perturbation. Weights saved for the WSP question.

## The WSP answer (2026-11-05, `results/a100_construct_sweep.log`)

Partial yes — **~69% of the interface is closed-form constructible** (paired Procrustes-ridge
4.58 vs identity 6.55 vs SGD 3.68, periodic-6).  But `cos(dW, every construction) ≈ 0`: the
trained map occupies a different direction in weight space than any statistical reshaping.
The residual ~31% is task-conditional.  Also: `norm_match` helped despite RMSNorm scale-invariance
(an open anomaly); `cov_recolor` actively hurt.  WSP framing: *fitting ≠ training* — deterministic
one-shot solves count as construction, and they get you most of the way.

## Metric correction + contamination flag (2026-11-05)

`qwen_byte_bpb` computed mean per-token surprisal (~3× inflated) — all cited stock-bpb numbers were
wrong.  Corrected (`src/adapter.py`): 0.8B 1.57/0.82, 4B-Base 1.30/0.65 (md/val,
`a100_family_sweep.log`).  Consequences: (a) the "byte-level ≈ stock" claims are retracted — the
real interface gap is ~3.5 bits on val; (b) corrected stock is implausibly good → the eval corpora
are very likely inside Qwen3.5's pretraining data, so stock-relative gaps are
contamination-confounded.  Adapter-internal comparisons unaffected.
~~Next: a decontaminated eval set (post-cutoff or synthetic data) before any stock-relative
claim.~~ **Done** — see next section.

## Decontam + family + merge + v2 + conjugation (2026-11-05, endgame rounds)

**Decontaminated eval** (`a100_decontam.log`, `kernel/decontam_eval.py`, `data/decontam/` — 48
post-cutoff arxiv+GitHub docs, 393,216 B): stock 0.8B 1.0890 / 0.8B-Base 1.0227 / 4B-Base 0.8517 —
a ~0.2-bit delta vs the (contaminated) val numbers, so **the stock advantage is mostly real
modeling, not memorization**.  The byte interface costs ~4 bits through Qwen on clean data too;
that cost is intrinsic to the interface, not to contamination.

**Trunk sweep** (`a100_family_sweep.log`): byte-level floors — 0.8B 5.61 md/4.33 val,
0.8B-Base 4.92/3.80, 2B 4.63/3.43, 4B 4.61/3.59, 2B-Base 4.54/3.44, **4B-Base 4.40/3.35 — best
trunk**.  Decontam confirms the ordering (4B-Base 4.53 vs 0.8B-Base 5.03 vs 0.8B 5.57).
The instruct tax is ~0.5 bits at 0.8B on clean data.  9B-Base OOMs on A100-40GB.

**Merge sweep** (`a100_merge_sweep.log`, `kernel/merge_sweep.py`): W(α)=(1−α)·instruct+α·Base.
Byte-level recovers ~linearly (val 4.33→4.15→4.00→3.88→3.80; decontam 4.60→4.00) with **no
barrier**; stock stays ~flat.  The byte-capable directions persist inside the instruct weights.

**v2 composed system** (`a100_v2_decoder.log`, `a100_v2_ridge_val.log`, `a100_v3_scaled_gen.log`,
`a100_v2_4bbase.log`): ridge-init patch map + trained byte decoder = a real tokenizer-free LM.
On 0.8B: d384×6/13.2M → **4.99 md / 4.13 val**, below the byte-level floor.  On 4B-Base — the
better trunk — the same recipe is **worse** (5.11–5.15 / 4.21–4.30): a stronger trunk cannot
rescue a 13-doc decoder.  **The decoder is the data-bound bottleneck.**

**Conjugated transfer** (`a100_conjugate.log`, `kernel/conjugate_transfer.py`): the 0.8B-learned
ΔW transported to 4B-Base via a ridge bridge on 248,320 shared-vocab embedding pairs scores
**5.95 boundary bpb vs identity 5.66** — worse than doing nothing; the local closed-form refit
scores 4.37.  Learned glue is size-specific: activations bridge, task-functions don't.
Per-size refit required (cheap: minutes).

## Final decision

1. **Trunk: solved.**  Pick Qwen3.5-4B-Base (best byte floor measured, 3.35 val / 4.53 decontam).
   If an instruct model must be used, merge toward its Base — the α-dial recovers byte
   competence linearly with no barrier.
2. **Patch map: solved.**  Closed-form ridge→BPE ≈ trained rank-32 on the effective axis
   (4.32 val zero-training vs 5.27±0.01 md for SGD on its axis; same landing zone).
   Construction wins on cost; SGD wins marginally on boundary-task bpb.  Ship either — the
   ~160k-param problem is not where the remaining loss lives.
3. **Byte decoder: the bottleneck.**  The only component where added capacity still buys bits
   (4.87M→13.2M: −0.03/−0.04) and the only one that regressed on a better trunk.  **Next
   investment goes to decoder data scaling** (13 markdown docs is the current training set — the
   binding constraint), then capacity/context-window scaling.
4. **No cross-size glue transfer.**  The conjugation test is decided: re-fit per size.  The
   refit cost is trivial, so this is a policy, not a hardship.
5. **qwentensor stays the hosting instrument**: all glue is host-side; the in-substrate story is
   unchanged.

## Open questions — resolution ledger

- ~~Can a WSP-style construction produce what the rank-32 map learned?~~  **Answered**: ~69%
  closed-form under periodic-6, ~27% under sim~6; direction orthogonal (cos≈0) but endpoint
  equivalent on the effective axis.  Remaining autopsy: read ΔW's singular directions through
  the workspace lenses (`camilablank/workspace-lenses` — J+R pairs for Qwen3.5-4B/9B/27B) to see
  if the learned operator is an interpretable byte→subword affinity map.
- ~~Does the composed system transfer to the H-Net val split?~~  **Answered**: 4.13 val at
  d384×6 on 0.8B — yes, cross-domain holds.
- ~~Interior-byte decoding: is the gap closable with a learned dechunk head at all?~~
  **Answered**: yes — the learned decoder closes the composed system below the byte-level floor;
  it is now the bottleneck, not a missing piece.
- New: decoder data scaling curve; gauge-canonicalized (A,B) comparisons before any seed- or
  size-space interpolation (RESEARCH_WEIGHT_TRANSFER.md §3.1); reverse-direction conjugation
  (fit at 4B-Base, transport down) as a completeness check on the size-specificity result.
