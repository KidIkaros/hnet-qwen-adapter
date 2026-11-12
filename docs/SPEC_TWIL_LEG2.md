# Spec — TwIL leg-2: the FIRST real training run (GPU, 2×T4, ~2h)

Date: 2026-11-10. Supersedes the "scale leg-1" framing: leg-1's train
log (`reports/runs/twil-leg1-v4/leg1_train_log.json`) shows the run was
not undertrained — it was **mostly failed**:

- `steps_done: 16` of ~250 attempted; 24/40 logged entries `oom: true`
- loss trajectory 8.73 → 11.64 bits, noisy, no descent (16 steps can't
  teach a 2M-param table anyway)
- the +0.105 bpb "delta" is warm-start + noise, NOT a training result

**Therefore leg-1's flat result is void.** Leg-2 is the first legitimate
measurement of whether the frozen TwIL trunk can be taught byte-level
input at all.

Compute: Kaggle GPU `NvidiaTeslaT4` ×2, ~2h. **Owner approval required
before push.**

## Root cause of the OOM (from generator audit)

`train_step` runs **one full 2048-byte doc per step** → period-2 →
T≈1024 chunk positions → forward+backward through the entire 8B trunk
per step, even with `gradient_checkpointing_enable`. Activation memory
for T=1024 × ~40 layers does not fit 14.5GB/GPU when the trunk is
sharded across both. Fix is sequence length, not batch (already 1 doc).

## Design — same everything, three changes

1. **DOC_CAP_TRAIN 2048 → 512 bytes** (T≈256 chunks — ~4× less activation
   memory; byte-LM needs no long context for this measurement).
   DOC_CAP_EVAL stays 4096 (eval is inference-only, no backward).
2. **Smoke gate — revised after leg-2 v1 (evidence-based)**: v1's
   loss-descent gate (first3 vs last3 training loss) FALSE-FAILED:
   each step trains on a different document, and per-doc CE is
   dominated by document entropy (code vs prose vs tool-JSON), so the
   comparison measured doc difficulty, not learning. v1 also proved
   the OOM fix (10/10 attempts at T~256, zero OOM). New gate:
   **at ok-step 50, bpb on the fixed curve docs must have improved
   ≥0.05 vs step-0** — the actual metric, immune to per-doc noise.
   Fail → abort with diagnosis (objective/setup bug).
3. **Learning-curve eval**: first-byte bpb (canonical `_lm_rows` +
   `ByteHead.byte_probs` path) on a FIXED subset of 8 eval docs at
   steps {0 (=v0 warm-start), 50, 100, 300, 600, end} — the curve is
   the deliverable, not a single endpoint. Step 50 doubles as the
   smoke-gate point.

Everything else identical to leg-1 v4: same corpus mix + per-source
isolation, same LearnedPatchEmbedder warm-start, same byte-restricted
CE, same eval docs (deterministic seeds → same splits), same artifacts
contract + `adapter_leg2.pt`.

## Steps / cost

Target **800 successful steps** (attempts may exceed; OOM steps are
skipped and counted). At T≈256 a step is ~3-5s → ~60-80min training.
Total ~2h GPU.

## Verdict logic (pre-registered)

- **bpb@800 improves >0.5 vs the step-0 baseline on the same eval
  subset, curve still descending** → undertraining confirmed → leg-3
  scales data/steps; Route A alive.
- **curve flat (<0.2 total improvement) after 800 real steps** → strong
  evidence the trunk can't exploit byte-derived input → Route A pivot:
  byte-signal/trunk-layer audit (which layers carry byte info?), or
  Route B reconsideration.
- **loss still ~uniform/increasing after smoke+training** → objective
  or warm-start bug, not capacity — debug before any scale-up.

## What this decides

The single most important open measurement in the project: does the
H-Net substrate have a trainable byte pathway on TwIL at all. Leg-1
could not answer it; leg-2 can.
