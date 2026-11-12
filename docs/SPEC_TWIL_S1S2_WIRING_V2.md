# Spec — wiring v2: S2-as-verifier (GPU)

Date: 2026-11-10. Supersedes the escalate-to-S2 design in
`SPEC_TWIL_S1S2_WIRING.md` after v1 measured `wiring_pass=False`:
TwIL-S2 re-deciding tool choice recovered 0/12 S1 errors and cost
19pp end-to-end. Root cause: deliberation is good at *judgment over
options*, bad at this generation format — so S2 stops deciding and
starts *verifying*.

Compute: Kaggle GPU `NvidiaTeslaT4` ×2, ~1.5 h. **Owner approval
required before push.**

## The v2 loop

```
request → StartLux decide → glial_gate(margin, forced)
    ├─ auto (margin≥0.50)      → emit S1 answer
    ├─ review (0.30–0.50)      → S2 VERIFY → accept S1 / switch / abstain
    └─ escalate (<0.30/forced) → S2 VERIFY → accept S1 / switch / abstain
```

S2 never emits the tool call. It sees S1's decision + the candidate
set and returns a verdict. S1 remains the decider; S2 is the skeptic.

## S2 verifier contract

Prompt (chat template, `enable_thinking=True` — deliberation is the
point):

```
System: You are a verification layer for a tool-routing system.
A fast classifier proposed calling {s1_pred} for this request.
Candidate tools: {name: description, ...}
User request: {user}
S1 confidence: {conf}, margin: {margin}
Question: Is {s1_pred} the correct tool? Answer with exactly one of:
VERDICT: CONFIRM   — S1's pick is right
VERDICT: SWITCH <name> — a different candidate is correct (name it)
VERDICT: ABSTAIN   — none of the candidates fit / request underspecified
```

Parse: first `VERDICT:` line in post-`</think>` text →
`{verdict, switch_name}`. Robust to think-trace noise (search whole
body, not just first line).

Action mapping:
- CONFIRM → emit S1 answer
- SWITCH x → emit x (only if x ∈ candidate set, else treat as ABSTAIN)
- ABSTAIN → abstain (no emission — honest stop, feeds future metrics)

## Why this should work where v1 failed

- v1 asked S2 to *generate* a tool call in a format it only hit 55%
  valid on real schemas. v2 asks for a 3-way judgment — no format
  burden, no arg generation.
- SWITCH gives S2 a concrete alternative to point at rather than
  re-deriving the whole choice — the review band becomes "is the
  top-1 wrong and is the right one in the set?" which matches what
  the prediction_set machinery was built for.
- ABSTAIN is a first-class outcome: the gate's escalate band becomes
  "S1 unsure AND S2 can't rescue" → honest stop instead of a wrong
  auto-emit. That alone may beat v1's forced-wrong emissions.

## Kernel legs (reuse v1 skeleton)

- Leg 0: co-load StartLux cuda:0 + TwIL device_map auto (same as v1;
  SL_CPU fallback). NOTE: v1 fell back to CPU — investigate why
  cuda:0 co-load failed (likely max_memory too tight with StartLux
  resident; try loading StartLux AFTER TwIL, or max_memory 11GiB
  on cuda:0).
- Leg 1: identical — StartLux decide ×N fresh ToolACE items →
  glial_gate. **New slice: qualifying 460–559** (v1 used 360–459).
- Leg 2: S2-verify on review+escalate items (same bands, both — we
  need per-band verifier accuracy to set policy). max_new_tokens=512
  (verdict is short; think trace still allowed).
- Leg 3: metrics + artifacts (same zip contract).

## Metrics / verdict

- `s2_verify_acc`: fraction of non-auto items where verifier's final
  action (confirm→S1 pick, switch→switch name) matches gold.
- `switch_precision`: of SWITCH verdicts, fraction where the named
  tool == gold.
- `abstain_rate` + `abstain_correct`: abstentions on unanswerable
  items (the honest-stop payoff).
- e2e policies: P1 auto→S1 / non-auto→verdict-action; P2 only
  escalate→verdict; P3 S1-only baseline.
- `wiring_v2_pass` = e2e_P1_ans > s1_acc_ans + 0.02 AND
  switch_precision ≥ 0.5 (S2's overrides must be right at least
  half the time to be trusted).

## S2 host — open question (owner decision)

Default: TwIL-8B `enable_thinking=True` (already proven loadable,
real think traces). Alternatives if TwIL-as-verifier also fails:
function-calling fine-tunes (xLAM, watt-tool), Qwen3 dense/int4,
Mistral-Small int4. NOT gpt-oss (owner veto). The verifier prompt is
host-agnostic — swap is a one-line REPO_ID change.

## Known risks

- n will again be ~40 items if SL_CPU persists → widen cuda:0 fix
  first (load order / max_memory) since CPU StartLux capped v1 at 40.
- SWITCH name matching: S2 may emit spaced/normalized variants —
  compare via norm_name, reject out-of-set switches.
- Verifier sycophancy: S2 may CONFIRM everything (defer to S1) —
  confirm_rate is itself a metric; >90% confirm = verifier adds
  nothing, which is also a result.
