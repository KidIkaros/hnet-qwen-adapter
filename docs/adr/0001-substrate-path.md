---
status: proposed
---

# TwIL-8B is the trunk; the byte H-Net line is the endgame substrate

The project had been conflating two roles that must stay separate: the **trunk** (the model
that *is* the mind, today) and the **substrate** (the model that will host the Companion,
endgame).

## The two paths

- **Path A — frozen TwIL-8B + learned byte adapter** (this repo): a learned patch embedder
  maps raw bytes into the frozen trunk's `inputs_embeds`; the trunk is never trained
  end-to-end.
- **Path B — from-scratch byte H-Net** (`hnet-kaggle`): `nn.Embedding(256, D)` over every
  byte, SSM encoder → attention main → SSM decoder, learned boundary routing.

## Evidence

**Path A is the trunk path, and it works.** leg-2: +0.58 bpb at 800 steps. leg-3 Arm A
(period-2, 3000 steps): **3.618 vs 6.964 bpb, delta +3.35**, curve still falling at the
budget's end. TwIL is HF-transformers native (`inputs_embeds` proven in the v3 probe),
hybrid with a native operating-point toggle (`enable_thinking`), and byte-compatible
(207/256). Spark-X2.6-4B is GGUF-only — no `inputs_embeds` path — and is disqualified as
substrate.

**Path B is the endgame substrate, and it is owned.** The 18.6M-param byte H-Net reached
1.42–1.49 bpb on FineWeb-Edu; all 256 byte rows are learned parameters, so Path A's
unmapped-byte impedance cannot exist there; its anatomy is the one-mind design (fast SSM =
S1, attention over compressed chunks = S2, learned boundaries as the control surface); it
carries the gate/injection-test culture. Its scaling measurement (b = 0.111) says compute,
not architecture, is the lever.

## Decision

1. **Trunk = TwIL-8B.** The one-mind system is built on TwIL now. No from-scratch training
   runs on the one-mind critical path.
2. **Substrate = the byte H-Net line, endgame.** It is the Companion's intended owned home;
   work on it is endgame work, not the one-mind critical path.
3. **Path A's adapter result stands** as the proven byte-input mechanism (+3.35 bpb,
   archived in `reports/runs/twil-leg3-v3/`).

## Consequences

- Development continues on TwIL: the byte adapter (period-1 diagnostic, leg-4), S1/S2 depth
  work, and the glial regulator.
- `hnet-kaggle` remains the endgame substrate reference; its results are not re-run as
  substitutes for the trunk.
- The outcome channel is the deepest open problem: no run — adapter or from-scratch — has
  seen a tool-execution or user-correction signal.
