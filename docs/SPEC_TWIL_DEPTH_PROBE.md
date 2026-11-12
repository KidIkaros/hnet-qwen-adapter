# Spec — depth-recovery probe: does the trunk's deliberate mode, seeded by its
# fast pass, recover what the fast pass missed?

Date: 2026-11-12. The one-mind's core untested claim, in the owner's words:
*"whether the trunk's deliberate mode, conditioned on the fast mode's
compressed state, recovers what the fast mode missed."*

Everything prior tested cross-model rescue (StartLux→TwIL escalate: 0/12
recovery; Spark verify: 0 recovery, 0 switches). Those measured committees.
This measures the mind: **same trunk, two depths, the deliberate pass seeded
by the fast pass's own compressed state.**

Compute: Kaggle GPU `NvidiaTeslaT4` ×2, ~2 h. **Owner approval required
before push.**

## Arms (same trunk, same items)

| arm | config | question |
|---|---|---|
| **A — S1 fast** | constrained prefix-scoring over the declared tools (`enable_thinking=False` template), argmax → pred; softmax over per-candidate scores → the gate's distribution | the fast baseline; also produces the **fast state** |
| **B — S2 unconditioned** | `enable_thinking=True`, generate from the prompt alone | is think-mode *depth* or just *length*? (assumption 1) |
| **C — S2 conditioned** | same as B, but the prompt's embeddings are **prefixed with 4 soft tokens = the fast pass's final hidden state** (identity projection, `inputs_embeds` — proven on TwIL in the v3 probe) | does seeding deliberation with the fast mind's own state recover what it missed? (the core claim) |

The fast state = `output_hidden_states=True`'s final-layer state at the last
position of the *predicted candidate's* scoring forward — the state the fast
mind held when it decided. Identity projection v0; the trained upgrade is
Coca-Cola's Action-Preserving Training (arXiv:2611.06500) if v0 shows signal.

## Items

ToolACE `train` streaming, qualifying slice **560–659** (100 items) —
disjoint from v2 eval (140–199), gate-cal (200–359), wiring v1 (360–459),
wiring v2 (460–559), and the leg corpus (5000+). Same parser as wiring v2
(raw_decode tools, bracket gold, normalized match, answerable flag).

## Metrics (paired, comparable to the wiring runs)

- **S1**: acc all + answerable; margin distribution; mean gen_s.
- **B and C**, each vs S1 on the same items: **recovery** (fraction of
  S1-wrong items the arm gets right), **breakage** (fraction of S1-right
  items it gets wrong), format_valid, think_rate, mean gen_s.
- **C vs B directly** — conditioning must beat unconditioned deliberation,
  not just S1.
- **E2E policies**: P1 = auto→S1, non-auto→B; P2 = auto→S1, non-auto→C;
  vs S1-only. Two denominators recorded: emitted-only (all items) and
  `e2e_v1_style` (answerable-only, non-emission counts wrong — the wiring
  runs' definition, so cross-session numbers are quotable). Gate bands
  provisional (0.50/0.30) — trunk-score margins are uncalibrated; the
  caveat is written into `depth_metrics.json` as `gate_caveat`.
- **C vs B directly**: paired outcome counts on S1-wrong items both arms
  completed (`c_vs_b_on_s1_wrong`: both_recover / c_only / b_only /
  neither), plus the marginal recoveries the verdict bands consume.
- **Margin distribution**: quartiles of S1's constructed-distribution
  margins in `depth_metrics.json`.

## Pre-registered verdict

Verdicts are evaluated on the **paired intersection** of items completed by
both B and C (`c_vs_b_on_s1_wrong` / `c_vs_b_on_s1_right`), so differing wall-
clock completion counts cannot bias `C > B`:

- `depth_pass` = recovery_C ≥ 0.5 AND breakage_C ≤ 0.10 AND recovery_C >
  recovery_B — the one-mind loop is real and the fast state is the ingredient.
- `think_is_depth` = recovery_B ≥ 0.3 — think-mode alone recovers; the gate
  needs only a confidence signal.
- **Both fail** → assumption 1 dies (think-mode is length, not depth); the
  required mechanism is the H-Net middle (full attention over compressed
  chunks), and the probe's result redirects the build there.
- `conditioning_neutral` = `|recovery_C − recovery_B| < 0.05` — the fast state
  adds nothing; the trained (action-preserving) version is the next question,
  not a bigger untrained injection.

## Guards & v2 execution protocol

- Per-item try/except everywhere; error rows recorded, legs continue.
- **Leg-0 smoke gate**: immediately after loading TwIL-8B, runs a 2-token
  generation through both `s2_generate_ids` (`input_ids=`) and
  `s2_generate_embeds` (`inputs_embeds=`, `attention_mask=`) on item 0.
- **Pinned v1 bundle**: v1 completed `items.json` (100 items, slice 560–659),
  `s1_rows.json` (100/100 rows, 0 errors), and `b_rows.json` (61/61 rows,
  `k=0..60`, 0 errors). v2 bundles these directly, verifies item-0 full-
  candidate scoring parity against `s1_rows.json[0]`, reconstructs all 100
  `FAST_STATE` vectors in 100 single forwards (~2.5 min), and saves
  `fast_states.pt`.
- **Execution order (C first)**: Arm C runs `k=0..99` first (65-min budget,
  `C_WALL_S=3900`) so `k=0..60` pair immediately with v1's 61 B rows; Arm B
  then resumes `k>=61` on C-completed items (30-min budget, `B_WALL_S=1800`).
- TwIL generate OOM → `max_new_tokens` 768→512 for the remainder.
- Skipped counts written into `depth_metrics.json` (`b_skipped`/`c_skipped`).
- S2 emits `<need-more-info>` → recorded as `clarify=True`, not wrong.
- Artifacts: single `artifacts.zip` (`items.json`, `s1_rows.json`,
  `b_rows.json`, `c_rows.json`, `fast_states.pt`, `depth_metrics.json`,
  `run_status.json`).

## Cost

~2 h GPU: loads ~10 min, S1 scoring ~10–15 min (100 items × ~8 candidates,
one forward each), B ~35–40 min, C ~35–40 min, metrics ~1 min.

## What this decides

Whether the one-mind's S1→S2 loop exists *inside one trunk*. Positive → the
glial layer's job is deciding when to pay for arm C, and leg-2/scale-up
proceeds on the wiring. Negative → the architecture needs the H-Net middle,
and the probe's measurements say so with evidence rather than assumption.
