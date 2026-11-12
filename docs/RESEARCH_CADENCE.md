# Research note — Cadence (muellerberndt/cadence) vs the one-mind design

Date: 2026-11-10. Source: github.com/muellerberndt/cadence (Pragma Research,
GPL-3.0, alpha, NumPy-only, ~622 stars, v0.97.0 on PyPI as `cadence-net`).

## Why it is the closest existing expression of the one-mind vision

| one-mind concept | Cadence mechanism |
|---|---|
| S1 fast default | `live()` — greedy answer from one settle |
| S2 deliberate correction | observer regions feeding back into the SAME settlement (not a separate model stage) |
| glial/regulatory layer | `arousal` — learning gated by surprise/want; a neuromodulatory "when to adapt" signal |
| learned substrate | local plasticity — synapses learn from free-vs-nudged settle difference; NO backprop, no train/inference switch |
| one continuing mind | one continuing stream; working trace + associative/episodic memory carry state across moments |
| learns from live experience | `reward=` channel — the world pays for good acts and the brain updates continuously |

## The gap it exposes in our build

Our wiring experiments (twil-s1s2-wiring v1+v2) proved: a second
same-class model re-judging the SAME information adds nothing (0/12 and
0/N recovery). The deeper reason: **our system never sees an outcome.**
Every request is independent; gate thresholds are frozen constants fit
once. Cadence's core bet — outcome-driven online plasticity gated by
surprise — is exactly the missing channel.

## What to borrow (design, not code — GPL-3.0)

1. **Outcome channel**: tool-call success / user acceptance / parse
   validity must feed back and adapt routing. Without it, "escalation"
   can only ever be re-opinion, which we measured as worthless.
2. **Surprise/arousal as glial signal**: we gate on margin (uncertainty
   proxy); Cadence gates on prediction error |expected − observed|.
   Margin asks "was I unsure?"; surprise asks "was I WRONG?" — only the
   second one should trigger expensive correction.
3. **S2-as-observer**: S2 should constrain/influence the same decision
   (verification as propagation over S1's candidate set), not re-answer
   in a separate pipeline.

## What it cannot do

No language — NumPy rate-neuron equilibrium nets for control-scale
tasks. Cannot read ToolACE, cannot emit tool calls, cannot deliberate
in words. It is an architectural template and a PoC substrate, not a
component for the LLM pipeline.

## PoC (this repo, local, CPU)

`reports/runs/` archived StartLux rows (probs + gold + answerable,
~260 rows across head-to-head / gate-cal / wiring runs) feed a
`Brain.compose` router: obs = [conf, margin, log n_candidates, forced],
actions = {auto-emit, defer}. Reward: auto→+1/−1 by correctness,
defer→0. Question: does online-learned regulation beat the frozen
margin thresholds on held-out rows? See session notes / scratch script.
