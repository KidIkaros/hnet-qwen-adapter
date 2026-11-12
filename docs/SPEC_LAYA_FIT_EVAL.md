# SPEC — laya/vega `fit()` eval: the real S1 head-to-head (CPU lane)

Follow-up to `SPEC_LAYA_CPU_EVAL.md` (run `laya-cpu-eval` v1). v1 showed
zero-shot laya/vega ≈ BGE-level (66.7%/65.0% tool gold@1) with uninformative
confidence — expected, since the product's regime is *fitted* heads on your
labels. This spec tests that regime on our actual task.

Lane: Kaggle **CPU** kernel (zero GPU quota). Reuses the v1 kernel's ToolACE
parsing (`SKIP=2500` slice family) extended to more items for fit volume.

## Question

Does `VegaAgent.fit()` on ~100–300 ToolACE decision labels produce an S1 that
beats (a) zero-shot laya/vega, (b) BGE baseline 62.5%, and (c) approaches
StartLux-0.8B on the same items — with *calibrated* outputs good enough to
drive the glial gate?

## API (verified against laya docs, vegaml page)

```python
rows = [{"state": "...", "questions": questions, "gold": {"tool": "name"}}, ...]
report = agent.fit(rows)          # heads -> temperature -> conformal -> band, one call
report["serving"]                 # which questions' heads earned serving
agent.save_calibration("fit/")    # heads.npz + calibration.json -> artifact
agent.load_calibration("fit/")    # fresh agent, identical answers/routes
```

- `min_rows(questions)`: 5 per option of widest question + 25% held out inside
  `fit` for calibration, floor 10 → **35 rows for a 5-way choice, 20 for noul**.
- Question types: `choice` (criteria: ordered label→gloss map), `score`
  (ordered rubric), `noul` (returns P(true)).

## Legs

### L0 — data prep
- Pull ToolACE items (same HF dataset path as v1; extend SKIP window for
  volume — target ≥120 usable rows after filtering).
- **Open implementation question**: ToolACE items have *per-item* tool sets.
  `fit()` rows carry their own `questions`, but whether a fitted head tolerates
  varying `criteria` across rows is unverified. Two formulations, try in order:
  1. **Fixed candidate pool**: restrict to items sharing a common tool subset
     (or synthesize a fixed 8–12 tool pool per row with gold inside) —
     preferred if criteria must be static.
  2. **Per-row criteria**: pass each item's own tools as criteria; check
     `report["serving"]` and held-out accuracy — if heads ignore criteria
     variation, accuracy will be at floor.
- Splits: **fit / calibrate-holdout (internal to fit) / evaluate must be
  disjoint** — carve evaluate BEFORE calling fit; fit's internal 25% is its
  own calibration, ours is the honest estimate.

### L1 — fit + evaluate (choice question)
- `questions = {"tool": {"type": "choice", "instructions": "Which tool
  should handle this request?", "criteria": {name: desc, ...}}}`
- gold = gold tool name.
- Metrics on held-out evaluate rows:
  - tool gold@1 accuracy (head-served vs engine-fallback, reported separately
    via `report["serving"]`)
  - confidence/Brier/ECE on the fitted head
  - conformal `prediction_set` gold coverage vs nominal level
  - `route` distribution + route-gate simulation (auto vs review accuracy)
- Baselines on same rows: zero-shot vega engine (v1 numbers), BGE 62.5%.

### L2 — noul sufficiency head (glial-relevant)
- Second question type on same rows: `"ready": {"type": "noul",
  "instructions": "Can this be handled correctly by the chosen tool call
  alone?"}` — gold synthesized: 1.0 for items with unambiguous gold tool,
  0.0 for the v1-style corrupted/OOD variants (description-shuffled,
  missing-arg) — tests whether a fitted noul head produces a usable
  act/escalate signal.
- Metrics: Brier/ECE, risk-coverage curve (does fitted confidence finally
  rank correctness?), threshold sweep for escalate precision/recall.

### L3 — persistence check
- `save_calibration` → `load_calibration` on fresh agent → predict 10 rows →
  assert identical answers/routes (their claimed reproducibility; verify).

## Success criteria (pre-registered)

| metric | pass |
|---|---|
| fitted choice gold@1 | > 75% on held-out (beats BGE 62.5%, clears zero-shot 66.7% meaningfully) |
| conformal gold coverage | ≥ nominal − 5pp |
| noul risk-coverage | monotone-ish: acc@top-50%-conf > acc@bottom-50% by ≥10pp |
| route-gate sim | auto-path accuracy ≥ always-S1, review-path catches ≥50% of errors |

Any leg failing its criterion is a *result*, not a bug — record and stop;
do not tune on evaluate.

## Artifacts (single artifacts.zip + run_status.json)

- `fit_report.json` (report["serving"], per-question metrics)
- `laya_fit_L1_results.json`, `laya_fit_L2_results.json` (per-row preds)
- `calibration/` → heads.npz + calibration.json
- `run_status.json` with leg statuses

## Cost

CPU lane, ~30–60min: vega reader ~2GB HF download + fit on ≤300 rows (closed-
form-ish heads, minutes) + ~300 predicts. Zero GPU quota.

## What this decides

- **PASS** → laya/vega becomes a credible S1 *candidate* alongside StartLux;
  next step is StartLux-vs-fitted-laya on identical rows (still CPU).
- **FAIL on accuracy but calibrated outputs** → the head machinery works but
  needs more/better labels → informs whether to invest in label generation.
- **FAIL on calibration** → composite gate unusable; reinforces that OUR
  EnergyHead design (DESIGN_EBR_GLIAL.md) must be trained on-domain — which
  was the plan anyway.
