# SPEC — S1 head-to-head: StartLux-Decision vs fitted laya/vega (CPU lane)

The decisive S1 comparison, run once on identical rows. Zero-shot phase is
done (laya ≈ BGE 62.5%); this tests each engine in its *intended* regime —
StartLux native decision, laya `fit()` heads — plus the glial-relevant outputs
(conformal sets, route, calibrated confidence) on the same items.

Lane: Kaggle **CPU** kernel — zero GPU quota. Both engines are ≤0.8B
single-pass decision models; laya's v1 eval already proved the CPU envelope.

## Question

On ToolACE tool-choice + ready-to-act decisions, which S1 wins on
accuracy-per-cost — and does either produce glial-grade calibrated outputs
(gate on route/conformal, never raw confidence — the v1 lesson)?

## Shared substrate (identical for both engines)

- **Items**: ToolACE `train` split, `SKIP=2500`, extended window — target
  ≥120 scored rows for fit volume + ≥60 held-out evaluate rows. Same parser
  as `kernel/startlux_baseline.py` + `laya-cpu-eval` (user msg, system tools
  via `TOOL_RE`, gold = first called function name).
- **Splits**: `FIT` (laya fit rows only) / `EVAL` (held out, never fitted,
  never calibrated-on) — disjoint by construction; laya's internal 25%
  calibration holdout is inside FIT, ours is the honest estimate.
- **Questions** (same shape both engines where possible):
  - `tool` — choice over the item's declared tools (gold = gold fn name)
  - `ready` — noul/act-vs-escalate (gold synthesized: 1.0 clean items,
    0.0 corrupted/OOD variants — same recipe as laya eval v1)
- **Metrics** (computed identically per engine):
  gold@1 · Brier/ECE on top-prob · risk-coverage curve · conformal/prediction-
  set gold coverage vs nominal · route/auto-review split accuracy ·
  per-item paired results (enables McNemar-style comparison)

## Legs

### S0 — data prep (shared)
Parse + split + dump `items.json` (state text, tool criteria map, gold,
variant flag). Both engines consume this file — no parsing divergence.

### S1 — StartLux engine
- Runtime per `kernel/startlux_baseline.py`: `hf download
  startlux-models/StartLux-Decision-v1-0.8B` + `pip install -r
  requirements.txt` + `sys.path.insert` + `StartLuxDecision(MDIR)`.
- `m.decide(state, {"tool": {"type":"choice","instructions":...,"criteria":
  {name: desc}}})` → `answers.tool.probabilities` → argmax.
- **Risk flag**: runtime deps unknown on Kaggle base image; if import fails,
  record `startlux: unavailable` and continue — laya legs still produce value.
- Also run its `ready` equivalent if the API exposes noul/binary decisions;
  else map choice-confidence to a ready proxy and label it as such.

### S2 — laya/vega engine (fitted)
Per `SPEC_LAYA_FIT_EVAL.md`:
- `VegaAgent(vegaml=True)`; `agent.fit(fit_rows)` on FIT split only.
- `report["serving"]` recorded — heads that didn't earn serving fall back to
  engine (report both accuracies separately).
- `save_calibration`/`load_calibration` persistence check on 10 EVAL rows.

### S3 — head-to-head metrics
- Paired per-item table: engine × item → {pred, gold, conf, route, set}.
- Aggregate: gold@1 each, accuracy delta with item-level wins/losses,
  Brier/ECE each, risk-coverage each, conformal coverage (vega only —
  StartLux has no set output; record as N/A not zero).
- **Route-gate simulation**: for each engine with a route signal, compute
  acc(auto-only) vs acc(all) vs oracle-gate — does gating help?

## Success criteria (pre-registered)

| outcome | meaning |
|---|---|
| StartLux > fitted-laya by >5pp gold@1 | StartLux confirmed S1; laya demoted to baseline/conformal-reference |
| fitted-laya ≥ StartLux | laya earns S1 candidacy; AutoThink-vs-plain-decision becomes the real question |
| both <65% | task is hard for decision-heads; S1 needs the generative trunk's S1-mode (TwIL leg-C showed 12/12) — decision shifts to "trunk-as-S1" |
| either produces usable conformal/route | adopt that output contract for OUR glial layer regardless of winner |

## Artifacts (single artifacts.zip + run_status.json)

- `items.json` (shared substrate, for reproducibility)
- `startlux_results.json` (per-item preds/probs/latency)
- `laya_fit_report.json`, `laya_eval_results.json`, `calibration/`
- `headtohead_metrics.json` (paired comparison + gate sims)
- `run_status.json` (per-leg status; StartLux-unavailable is a recorded
  result, not a crash)

## Cost

CPU lane ~1–2h: two model downloads (~2–4GB total) + ~120 fit rows +
~180 predicts × 2 engines + fit solve (minutes). Zero GPU quota. TwIL v3
GPU run is unaffected (different lane).

## What this decides

The S1 slot in the one-mind. Winner becomes the fast path; loser's useful
parts (conformal contract, weight-edit technique, route semantics) get
absorbed into the glial layer design (`DESIGN_EBR_GLIAL.md`) regardless.

## OUTCOME (2026-11-10, kernel v2 COMPLETE)

RESOLVED — **StartLux wins decisively.** Clean-parse substrate, 60 EVAL
items (49 answerable), disjoint FIT for laya:

| engine | all-60 | answerable-49 |
|---|---|---|
| StartLux-0.8B | 41/60 = 68.3% | 41/49 = 83.7% |
| laya vega | 31/60 = 51.7% | 31/49 = 63.3% |
| laya encoder | 29/60 = 48.3% | 29/49 = 59.2% |

Paired sign tests: vs vega z=2.25 (p~0.02), vs encoder z=2.59 (p~0.01).

Key finding: `fit_report.serving=[]` — laya's own CV gate REFUSED to
serve fitted heads (they didn't beat zero-shot on held-out rows).
Per-item-varying tool pools break the fixed-option assumption the
closed-form heads need. vega's number is its zero-shot engine.

Glial contract validated where it matters: vega `route` split
auto 64.5% / review 37.9% acc — real gate separation. Conformal coverage
65% (under nominal). StartLux conf margin right=0.905/wrong=0.735.

DECISION: StartLux = S1 (confirmed empirically + structurally). Glial
layer adopts the route/conformal/abstain CONTRACT shape; gate signal
sourced from StartLux probability margins (or a fitted gate on them),
not from vega. Evidence: reports/runs/s1-headtohead-v2/.
