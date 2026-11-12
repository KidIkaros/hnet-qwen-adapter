# Spec — laya/vega CPU-lane eval + design inspirations for the one-mind

Date: 2026-11-10. Lane: **Kaggle CPU kernel** (no GPU quota). Purpose: measure whether
laya/vega's decision engine makes good S1/glial decisions on *our* tasks, and harvest
the design patterns worth adopting regardless of outcome.

Sources: `github.com/NandhaKishorM/laya` (Apache-2.0, `pip install laya`),
embedding.in blog "A decision skill written into the weights" (2026-11-10),
paper "From Laya to Vega" (Preprints 2026110781).

## Part 1 — what laya/vega actually is (verified from README)

- **Laya encoder**: non-autoregressive typed decisions (`choice`, `score`, `noul`)
  over text in ONE forward pass (~33 ms on T4; CPU-slow but fine for ~100 items).
  Trained with RLCD — RL against strictly proper scoring rules → calibrated probs.
- **Vega** (`laya.Agent(vegaml=True)`): frozen 0.8B reader (Qwen3.5-0.8B/4B, gemma-1b)
  + "physics engine" returning per-answer: `probabilities`, `prediction_set`
  (conformal, ~90% coverage), `abstain`, `unbound` (OOD flag), `route`
  (auto/review), `route_reason` (audit string).
- **`fit()`**: closed-form logistic head per question over *frozen* reader features —
  convex, no gradient through the model. Holds out ¼ rows for calibration; serves a
  head **only if it beats chance under CV AND beats the zero-shot engine on the same
  rows — a fit can lose and say so.**
- **Calibration = 3 numbers/question**: temperature + conformal q̂ threshold +
  review line (precision 0.95 @ 95% Wilson bound). Needs ~50+ calibration rows to
  certify a route; the output says so when it can't.
- **Router**: rule-based fast-encoder vs vega-reader selection with reasons.
- **Fast-edits** (blog): MEMIT-style closed-form solve writes a fitted decision
  policy into trunk weights — 3.1 ms, 262K weights, KL≤0.03 protection cap,
  dose-selected on held-out rows. 87 s end-to-end on 2×T4.

## Part 2 — CPU eval spec

### Items

Reuse the **same ToolACE slice** as `kernel/startlux_baseline.py` (SKIP=2500,
N_ITEMS=40+) for direct comparability with the StartLux numbers already collected,
plus ~40 BFCL single-turn items if cheap to embed. For each item construct laya
`questions`:

```python
questions = {
  "tool":    {"type": "choice", "instructions": "Which tool should handle this?",
              "criteria": {name: desc for each declared tool} | {"none": "no tool needed"}},
  "ready":   {"type": "noul",   "instructions": "Does the request contain enough
              information to call the tool correctly?"},
}
```

### Legs

- **L1 — encoder baseline**: `laya.Agent("convaiinnovations/laya")` (English
  checkpoint) on all items. Metrics: tool-choice accuracy vs GT, ready-probability
  Brier/ECE on items with known sufficiency labels (derive from ToolACE answer
  completeness or hand-label a subset).
- **L2 — vega engine** (if CPU time allows; 0.8B reader ~seconds/item): same items
  through `laya.Agent(vegaml=True)` — adds `prediction_set`, `abstain`, `unbound`,
  `route`. Measure **conformal coverage** (does the set hit nominal 90%?), abstain
  precision on insufficient-info items, OOD flag on deliberately corrupted inputs
  (shuffle tool descriptions).
- **L3 — route-gate simulation** (the glial question): treat `route` as the
  escalate/don't-escalate signal. Compare accuracy-per-cost of {always-S1,
  laya-gated, always-S2} where "S2" = a stronger scorer we already have
  (StartLux-AutoThink or the trunk itself on a subset). This is the number that
  matters: does the gate beat both extremes?

### Deliverables (`artifacts.zip`)

- `laya_results.jsonl` — per-item raw outputs (all fields)
- `laya_metrics.json` — accuracy, ECE, Brier, conformal coverage, abstain stats
- `route_gate.json` — the L3 comparison table
- `src/` bundle + `run_status.json`

### Non-goals

No GPU, no fine-tuning, no `fit()` on our labels yet (that's a follow-up once we
see zero-shot quality), no StartLux re-run (numbers exist).

## Part 3 — design inspirations worth adopting (independent of eval outcome)

These are stealable patterns for OUR glial/S1 layer — they cost nothing to adopt:

1. **The glial output contract.** Every regulatory decision should return
   `{decision, probabilities, prediction_set, abstain, ood_flag, route,
   route_reason}` — not a bare confidence float. `route_reason` (an audit string
   explaining *why*) is something we hadn't spec'd and is cheap to add: it makes
   escalation decisions debuggable. Adopt verbatim as our glial interface.

2. **Gate on `route`, never on raw confidence.** Their finding: top-probability
   confidence misleads; the conformal set + abstain + OOD composition is what
   stays calibrated. Our energy-head plan should emit the same composite signal —
   energy margin alone is the "confidence number" they warn against.

3. **Closed-form `fit()` on frozen features, CV-gated, losable.** Our planned
   energy/glial heads should fit logistic/energy heads on frozen trunk hidden
   states exactly this way: convex solve, held-out calibration split, serve only
   if it beats zero-shot on the same rows, and *report when it loses*. This is
   stricter honesty than our current eval conventions and matches AGENTS.md's
   disjoint fit/calibrate/evaluate rule — they implement it as code.

4. **Calibration = 3 numbers.** temperature + conformal threshold + review line,
   fitted on calibration rows disjoint from fit rows. We already have the
   temperature/conformal distinction in AGENTS.md ("temperature is two things") —
   their implementation is a reference for doing it correctly.

5. **RLCD — proper scoring rules as the training objective.** Any head we train
   for calibrated decisions should optimize a strictly proper scoring rule
   (log-loss/Brier), which makes calibration a *guarantee of the objective*, not
   a post-hoc patch. StartLux's RLCD² is the deliberative version of the same idea.

6. **Non-autoregressive typed decisions for S1.** For tool-routing/readiness/
   gating, one forward pass producing typed answers beats generation-then-parse.
   If TwIL hidden states support it, an S1 head in this style is cheaper than
   sampling tool calls.

7. **Fast-edits as the deployment path.** When a decision policy stabilizes,
   the closed-form weight edit bakes it into the trunk — zero runtime cost, no
   separate component to serve. Worth prototyping on StartLux decisions first
   (our own model, our own labels) before trusting it on TwIL.

8. **Honest-negative interface.** `fit()` returning "refused — didn't beat
   engine" is a feature. Our harnesses should report component failures as
   first-class results, not silent fallbacks.

## Results — run `laya-cpu-eval` v1 (2026-11-10, CPU, COMPLETE)

Artifacts: `reports/runs/laya-cpu-eval-v1/` (metrics, per-item rows, log).

| metric | L1 encoder | L2 vega |
|---|---|---|
| tool gold@1 (n=60, ToolACE SKIP=2500) | **66.7%** | **65.0%** |
| risk-coverage acc@50% cov | 63.3% | 63.3% |
| conformal gold coverage | — | **65%** (sets ~singleton, under-covered vs ~90% nominal) |
| route counts | — | 55 auto / 5 review (auto_acc 65.5%) |
| ready Brier / ECE (synthetic sufficiency) | 0.195 / 0.080 | — |
| unbound fired on OOD probes | 0/20 | — |

### Reading

- **Zero-shot laya/vega ≈ BGE-level on tool choice** (BGE was 25/40=62.5%,
  byte-scorer 21/40=52.5% on overlapping slice) — NOT a drop-in S1. The product's
  strength is `fit()` on *your* labels; zero-shot on a novel tool-calling domain
  is not its regime. Consistent with laya's own docs: fit a head on domain labels.
- **Confidence is uninformative here**: risk-coverage is flat-to-inverted
  (top-50% confidence → 63% ≈ overall 67%; several conf=1.0 predictions are wrong).
  Validates their own warning — gate on route/composite signals, never raw
  confidence. Direct evidence FOR adopting the composite contract.
- **Conformal sets under-cover zero-shot** (65% gold-in-set vs ~90% nominal,
  mean set size 1.08) — calibration does not transfer across domains without
  fit+calibrate on in-domain rows. Expected; the machinery exists for exactly
  this (`agent.fit` + calibration split).
- **route is too permissive zero-shot** (92% auto) — useless as escalation gate
  until fitted. The 5 review-routed items are the interesting subset to inspect.
- **unbound never fired** on description-shuffled probes — either encoder lacks
  the field or the corruption wasn't OOD enough; inconclusive, needs a stronger
  OOD probe (e.g., unrelated-domain requests).

### Decision impact

- **StartLux stays S1** — confirmed empirically: laya zero-shot doesn't beat the
  field on our task. The open question is now `fit()` on ToolACE/BFCL decision
  labels (closed-form heads, CPU-cheap) — the real head-to-head.
- **Adopt the output contract regardless** (§3 items 1–4 stand): route/abstain/
  conformal-set/route_reason is the glial interface spec.
- Follow-up kernel candidate (CPU, cheap): `agent.fit()` on ~100-300 ToolACE
  decision labels → re-measure the same metrics. That's the experiment that
  actually tests laya as S1.
