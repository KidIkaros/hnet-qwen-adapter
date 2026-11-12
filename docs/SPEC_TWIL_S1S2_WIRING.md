# Spec — one-mind wiring kernel: StartLux S1 → glial gate → TwIL S2 (GPU)

Date: 2026-11-10. Purpose: close the loop on the one-mind assembly for the
first time — measure whether escalating the calibrated gate's non-auto bands
to TwIL-S2 (`enable_thinking=True`) recovers the errors S1 makes, and at what
accuracy-per-cost. Every component is proven individually; this tests the
composition.

Prerequisite evidence (all complete):
- `reports/runs/s1-headtohead-v2/` — StartLux wins S1 decisively
- `reports/runs/startlux-gate-cal-v1/` — 160 rows, answerable acc 94.2%,
  margin gap at 0.5 (0/15 wrong above), calibrated config (0.50/0.30)
- `reports/runs/twil-probe-v3/` — TwIL-S2 real think traces, tool_call
  parse contract; Route A byte substrate (leg-1 training in flight)

Compute: Kaggle GPU, `NvidiaTeslaT4` ×2 (`machine_shape`), ~1.5 h budget.
**Owner approval required before push.**

## What it decides

The one-mind's core claim: fast S1 when confident, deliberate S2 when the
glial gate says so. Output: end-to-end accuracy under three policies +
recovery rate on the escalate band + accuracy-per-cost. Positive → the
wiring is real and leg-2/scale-up proceeds. Negative → the escalation
mechanism needs redesign before any scaling.

## Models on one box — the memory question

- **StartLux-Decision-v1-0.8B**: `StartLuxDecision(MDIR, device="cuda:0")`
  — the runtime accepts a `device` param and pins via `device_map={"":
  device}` (verified in `startlux_decision/model.py:277,291`). ~1.6GB bf16
  on cuda:0. requirements.txt filtered for CUDA-only deps exactly as
  s1-headtohead v2 (causal-conv1d/flash-linear-attention/torch/torchvision
  blocked; transformers>=5.8 etc kept; plain-torch path).
- **TwIL-8B**: `from_pretrained(REPO_ID, device_map='auto',
  max_memory={0:'12GiB',1:'14GiB'})` — cuda:0 gets StartLux first
  (~1.6GB), so TwIL's cuda:0 share shrinks from ~8.8GB to ~8.8GB
  (14.5 - 1.6 = 12.9GB headroom on cuda:0 — comfortable). GPU memory
  printed after both loads; assert cuda:0 free > 1GB before S2 leg.
- **Fallback**: if StartLux fails to co-locate → flag `SL_CPU=True`
  (device="cpu", ~47s/item) with items capped at 40 — recorded, not a crash.

## Items

ToolACE `train` streaming, same v2 parse (raw_decode tools, bracket gold,
normalized match, answerable flag, gold-guaranteed criteria). **Fresh slice:
stream positions 400–499** (100 items) — disjoint from v2 eval (140–199),
gate-cal (200–359), and leg-1 corpus (5000+). Items.json dumped.

## Legs

### Leg 0 — co-load (~10 min)

Both models + memory report + StartLux `decide` smoke test on 1 item.

### Leg 1 — S1 pass + gate (~25 min)

StartLux `decide` on all 100 items → probs. `glial_gate` (bundled
`src/glial.py`, calibrated defaults margin_auto=0.50,
margin_escalate=0.30) per item → route/abstain/forced/prediction_set.
Record: S1 acc all + answerable, route distribution, per-route acc,
forced count, gen_s per item.

### Leg 2 — S2 on non-auto bands (~40 min)

For every item routed `review` OR `escalate` (flag `S2_ON_REVIEW=True`
default — we want S2 data on both bands to choose the policy from
measurements, not assumptions): TwIL generate with
`enable_thinking=True`, `max_new_tokens=768`, greedy.

- Prompt: `tok.apply_chat_template(msgs, tools=twil_tools,
  tokenize=False, add_generation_prompt=True, enable_thinking=True)` —
  `twil_tools` converted from the same criteria (name/description/
  parameters shape the Granite template expects — verify format in-session
  against the v3 probe's toolcall cells).
- Parse: `parse_calls(body)` on the post-`</think>` region only (v3
  contract — reasoning text false-positives otherwise).
- S2 output → tool name → compare to gold + to S1 pred.
- Per-item: gen_s, has_think, format_valid, name_ok, calls[:3], full text
  (audit trail).

### Leg 3 — metrics + policies (~5 min)

- **S1-only**: StartLux acc (all + answerable).
- **Gate split**: route distribution; per-route acc (S1 answers).
- **S2 on non-auto**: acc on review-band and escalate-band separately;
  **recovery rate** = fraction of S1-wrong non-auto items S2 gets right.
- **End-to-end policies** (answer emitted = S1 for auto, S2 for
  review+escalate / escalate-only / all-S2): acc_all, acc_answerable,
  mean gen_s per policy → accuracy-per-cost at S2≈19s vs S1≈2s.
- **S2 think audit**: think_rate, think length distribution.
- Verdict block: `wiring_pass` = recovery_rate ≥ 0.5 AND
  e2e_answerable_acc ≥ S1_answerable_acc + 0.02 (material improvement).

### Leg 4 — artifacts (~1 min)

`artifacts.zip`: items.json, s1_rows.json, gate_verdicts.json,
s2_rows.json, wiring_metrics.json, run_status.json.

## Failure modes / guards

- StartLux cuda:0 OOM with TwIL resident → `SL_CPU` fallback (40 items).
- TwIL generate OOM at T≈1k tokens → reduce `max_new_tokens` to 512.
- S2 emits `<need-more-info>` (seen in v3 on underspecified items) →
  recorded as `clarify=True`, NOT counted wrong (correct behavior).
- Per-item try/except everywhere; error rows recorded, leg continues.
- Tool-schema format mismatch (Granite template vs our criteria shape) →
  Leg 0 smoke test catches it before the S2 leg burns time.

## Cost

~1.5 h GPU quota: loads ~6 min, S1 pass ~5 min (100 × ~2-3s GPU), S2 pass
~30-40 min (40-60 non-auto items × ~19-25s), metrics ~1 min, slack.

## What this decides

Whether the one-mind's escalation path is real. Positive → proceed to
leg-2 scale-up + eventually the wiring becomes the runtime loop. Negative
→ redesign the escalation signal (energy head? different S2 host?)
before scaling anything.
