# Spec — TwIL leg-3: scale-up + period comparison (GPU, 2×T4, ~6h)

Date: 2026-11-10. Predecessor: leg-2 v2 PASS — 800 steps moved full-eval
bpb 6.980 → 6.401 (delta +0.579), learning curve still descending
(~−0.0004 bpb/step at step 800). Route A confirmed; leg-3 tests how far
it goes and which compression point wins.

Compute: one Kaggle GPU session, ~6h (~20% of weekly quota).
**Owner approval required before push.**

## Questions

1. **How far does scaling go?** 3000 steps vs 800 — does the curve keep
   descending, and where does it flatten?
2. **Period-2 vs period-1** at matched training — which compression point
   gives the better bpb-per-trunk-position trade-off?

## Design — two arms, one session

Load trunk + corpus ONCE; train two adapters sequentially.

- **Arm A — period-2, 3000 steps** (isolates the step variable vs leg-2)
- **Arm B — period-1, 1500 steps** (every byte = one trunk position;
  probe showed period-1 effective bpb 2.41 — best observed. Half the
  steps because each step is ~2× the sequence length at same byte budget;
  matched trunk-token budget vs Arm A at 1500)
- Both arms: same corpus split, same eval docs → comparable numbers.

## Corpus — larger, fresh split

- N_TRAIN_TOTAL 400 → **1600 docs** per-source weight mix unchanged
  (agentic 0.40 / code 0.25 / reasoning 0.20 / prose 0.15).
- Fresh split: TRAIN_REGION = q*2048 per source (q ~4× leg-2's), eval
  region immediately after — disjoint by construction, but a NEW eval
  anchor (leg-2's eval docs are inside leg-3's train region now).
  **Therefore: re-measure the v0 baseline on the new eval docs in-kernel
  (Leg 1) — cross-leg comparison anchors on DELTA, not absolute bpb.**
- Per-source isolation + toolace chunking unchanged.

## Carried forward from leg-2 (proven)

- DOC_CAP_TRAIN 512 (zero OOM at T~256)
- bpb smoke gate at ok-step 50 — **revised after leg-3 v1**: abort only
  on NaN/non-finite E OR bpb worse than step-0 warm-start. (v1's ≥0.05
  bar false-failed Arm A's real +0.031 progress on the fresh eval set;
  gate should catch non-learning, not slow-start.)
- **v3 addition**: `clip_grad_norm_([E], 1.0)` + non-finite check after
  each opt.step — Arm B (period-1) diverged to NaN within 50 steps in v1;
  clip + immediate abort-on-nonfinite is the fix and the tripwire.
- 800-successful-step counting logic (leg-3: TARGET per arm)
- canonical first_byte_bpb everywhere; warm-start == v0 construction
- learning-curve evals on fixed 8-doc subset: {0, 250, 750, 1500,
  3000/final} per arm
- adapter checkpoints every 250 steps (both arms, distinct filenames)
- artifacts.zip + run_status.json; legs.train reflects reality
  (done/partial/aborted/skipped_empty)

## Restored gap

- **effective_bpb sanity check** (2 docs, both modes) — missing from
  leg2_eval.json; re-added per leg-1 spec.

## Verdict logic (pre-registered)

- **Arm A**: delta vs new baseline. Curve slope at 3000 tells the scale
  ceiling. Expect (from leg-2 slope) ~6.0–6.2 if linear; flattening
  earlier = saturation found.
- **Arm B vs A**: compare final full-eval bpb on the same eval docs.
  Period-1 wins → the H-Net substrate prefers 1:1 byte:position at this
  stage (matches probe's 2.41 effective bpb); period-2 wins → compression
  is worth its information loss.
- **Both fail to beat leg-2's +0.579 delta** → scaling saturates below
  leg-2 rates; investigate corpus/objective before more steps.

## Cost accounting

loads ~10min + corpus ~20min + baseline eval ~10min + Arm A (3000 steps
~3.3h + 5 curve evals ~20min) + Arm B (1500 steps ~1.7h + evals ~15min)
+ final evals ~20min ≈ **~6h**. One session. If session limit approaches,
Arm A artifacts are already checkpointed — a resume kernel can finish B.

## What this decides

The substrate's practical operating point: how many steps buy what bpb,
and period-1 vs period-2 — the two knobs that define the H-Net byte
substrate's configuration for everything downstream (learned boundaries,
S1/S2 operating points, deployment).
