# DESIGN — EBR-style energy head as the glial/regulatory layer

Trunk-agnostic design note. Maps Energy-Based Reasoners (arXiv:2611.08977,
`Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based
Model.pdf`) onto the one-mind stack: frozen trunk + S1 fast path + S2
deliberation + learned stop/escalate.

Status: design only — no code yet. Motivated by `laya-cpu-eval` v1 finding that
zero-shot confidence is uninformative on our tool-choice task (flat
risk-coverage) — a *trained* energy head is the fix, not a better zero-shot
model.

## 1. What EBR actually does (the transferable mechanics)

EBR replaces a learned recurrent transition `z_{t+1} = f_θ(z_t, x)` with
gradient descent on a learned scalar energy `E_θ(z; x)`:

```
z̃ ← RMS(z) + σξ                      # Langevin noise injection
z  ← Π_M(z̃ − α ∇_M E_θ(z̃; x))        # projected gradient step on sphere M
```

Three properties make it interesting for us — all from one object:

| property | mechanism | our use |
|---|---|---|
| **candidate scoring** | `E(z;x)` is a scalar quality measure | S1 fast candidate/tool ranking |
| **convergence signal** | stop when `‖∇_M E‖ < τ` and `‖z_n − z_{n−1}‖ < δ` | glial stop/continue/escalate |
| **iterative refinement** | the descent trajectory itself improves z | optional S2-in-latent-space |

EBR trains end-to-end with segment-wise truncated BPTT (last H steps),
second-order derivatives through the dynamics, stablemax CE + a convergence
loss, and SXM for trajectory diversity. Inference: sample z ~ N(0,I), descend
until the gradient-norm threshold, decode `ŷ_θ(z)`. Reported: 99.9%
Sudoku-Extreme at 345× less inference compute than the next-best looped model.

## 2. What transfers and what doesn't

**Doesn't transfer directly**: EBR trains the energy *and* the reasoner
end-to-end on algorithmic tasks with known solutions. Our trunk is frozen
(TwIL-8B bf16 on 2×T4; possibly the 27B int4 later) and our tasks are
tool-calling/agentic, not Sudoku. We cannot run EBR's full training loop.

**Transfers cleanly**: the *interface contract*. A discriminative head
`E(h, c)` over frozen trunk hidden states `h` and candidate `c` is already an
energy model — `p(c|h) ∝ exp(−E(h,c))` is the standard EBM↔softmax duality.
The glial layer needs exactly three reads: energy margin, energy calibration,
and a convergence/plateau signal. All obtainable from a small trained head on
frozen features — the same regime as laya's `fit()` (closed-form/logistic heads
on frozen embeddings) and our own `fit_adapters` ridge solves.

## 3. The concrete design: EnergyHead over frozen hidden states

```
prompt (+ tool schemas / candidate set)
   │
   ▼
frozen trunk forward (no generation)          ← one pass, ~30-80ms on T4
   │
   ▼
h = last-token hidden state (or pooled candidate-span states)
   │
   ├──► EnergyHead_φ(h, c_i) → E_i  for each candidate c_i
   │        (small MLP or 2-layer transformer over [h; c_i embedding])
   │
   ├──► S1 decision: argmin E_i  + margin ΔE = E_(2) − E_(1)
   │
   └──► glial signals: {ΔE, entropy(p), E_top1 calibrated, OOD score}
            │
            ├── ΔE large + in-distribution → AUTO: emit c_1 (S1 act)
            ├── ΔE small / OOD / low margin → ESCALATE: S2 (<think> on)
            └── verifier fails → retry/escalate further
```

The generative trunk still emits the actual tool call / answer text — the
energy head only *routes and gates*. This keeps the head tiny (≪1% of trunk
params), trainable on Kaggle-cheap frozen features, and swappable per task
family without touching the trunk.

### Candidate encodings c_i (task-dependent)

- **Tool routing**: c_i = tool name + description embedding (from trunk's own
  embed_tokens, mean-pooled) — score each offered tool, pick argmin E.
- **Ready-to-act / sufficiency**: c ∈ {act, clarify, escalate} — three-way
  energy → the glial route decision directly.
- **Answer verification**: c = candidate answer span hidden vs. prompt hidden —
  scores a *generated* candidate post-hoc (the verifier role).

### Glial signal spec (what the regulator reads)

| signal | definition | action |
|---|---|---|
| `margin` | E_(2) − E_(1) in nats | < τ_margin → escalate |
| `p_top1` | softmax(−E)_1, temperature-calibrated | < τ_p → escalate |
| `energy_abs` | E_(1) vs. in-distribution energy quantiles | OOD → escalate/abstain |
| `set_size` | conformal-style: smallest set covering τ mass | > k_max → escalate |
| `route_reason` | audit string (which signal fired) | logging/debug — stolen from vega contract |

Gate on the **composition** of these, never raw confidence — the laya eval
showed top-probability alone is uninformative zero-shot; a *trained* head plus
the composite gate is the bet.

## 4. Training recipe (Kaggle-feasible)

1. **Features**: run frozen trunk once per training prompt, cache last-token
   (or candidate-span) hidden states → `.npz` on the volume. One pass per
   example; ~100–2k examples is enough for a small head.
2. **Labels**: gold tool/route from ToolACE/BFCL slices we already parse
   (`kernel/bfcl_*.py`, `startlux_baseline.py` item format).
3. **Head**: 2-layer MLP (D→1024→n_candidates) or bilinear `hᵀWc_i`;
   ~1–4M params.
4. **Loss**: proper scoring rule — cross-entropy on gold candidate is the
   baseline; optionally add laya-style RLCD calibration term or conformal
   calibration split (fit / calibrate / evaluate disjoint per AGENTS.md).
5. **Solve**: plain Adam few hundred steps on cached features — CPU-trainable;
   GPU only needed if we want fresh features per epoch (not needed v1).
6. **Honest-negative gate**: serve the head only if it beats the zero-shot
   baseline (laya encoder 66.7%, BGE 62.5%) on held-out evaluate split —
   same discipline as laya's `fit()`.

## 5. The EBR-faithful extension (parked, research-risk)

The full EBR move — energy *descent on latent states* as S2 — would treat the
trunk's last hidden state as z and take gradient steps on E w.r.t. a learned
low-dim projection, then re-decode. That is second-order training through a
frozen 8B trunk: technically possible (hidden states are differentiable w.r.t.
input embeddings) but expensive and fragile. Park until the scorer head proves
the energy signal carries real information. If it does, the descent loop is a
strict upgrade path, not a rewrite.

## 6. Experiment ladder

| step | question | cost | success = |
|---|---|---|---|
| E0 | do trunk hidden states linearly separate gold tool/route? | linear probe on cached h | accuracy > zero-shot floor |
| E1 | EnergyHead trained on ToolACE slice beats laya/BGE on held-out | cached features + small head | >70% gold@1, calibrated |
| E2 | composite gate beats always-S1 and always-S2 on acc/cost | simulation on E1 outputs | Pareto improvement |
| E3 | energy margin predicts S2-escalation need | correlate margin vs. S2 gain | rank correlation > 0 |
| E4 | (parked) latent energy descent | research | beats S2-trunk on hard items |

E0 is nearly free (probe on features we can cache in the next GPU session's
idle time or a small dedicated run). E1–E3 are CPU-trainable once features
exist.

## 7. Open questions

- **Which hidden state?** last-token vs. pooled tool-span vs. per-candidate
  span states — E0 should compare; per-candidate states likely win for
  verification, last-token for routing.
- **Does TwIL's `<think>` already implement the energy role?** Its RLCD²-style
  length control may make an external head redundant for *escalation* — the
  head then only needs the *scoring* role. The TwIL probe's S1/S2 leg
  (`toolcall_results.json`) is the first evidence.
- **Calibration under shift**: tool schemas at inference differ from training
  — energy_abs/OOD signal must be validated on held-out schema families, not
  just held-out items.
- **Relation to fast-edits**: once an EnergyHead policy stabilizes, the
  closed-form weight-edit technique could bake the routing decision into an
  MLP layer — zero runtime head. Prototype on StartLux decisions first.
