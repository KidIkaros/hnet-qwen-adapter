# Adapter vocabulary

**Chunk / patch**:
A contiguous span of `period` bytes the chunker emits as one unit for the frozen stack. Chunks are not
tokens; the model has no vocabulary of them.
_Avoid_: token, wordpiece, subword

**Byte-phase chunking**:
Boundaries at a fixed offset on the byte grid (every `period` bytes at phase 0). The falsification test
on the trained 18.6M H-Net showed its learned router's decisions are statistically indistinguishable
from this (+0.0009/+0.0011 bpb on the published val split) — the router is a byte-phase segmenter.
_Avoid_: fixed tokenization, static chunking (static is right but vague)

**Patch embedder**:
The construction that turns a chunk's bytes into one D-dim vector: the sum of the bytes' own
token-embedding rows. Zero trained weights; every patch vector lives in the model's own embedding space,
which is what minimizes the distribution shift by construction.
_Avoid_: patch encoder (it has no parameters to train), byte pooling

**Byte-token map**:
byte value → the id of Qwen's single-byte token for it. 221/256 bytes have one (all printable text
bytes); the missing ones are unprintable controls. This map is the bridge that lets a byte-level
pipeline reuse the model's own embedding table and LM head.
_Avoid_: byte fallback (a tokenizer-internal term), byte vocab

**First-byte bpb**:
Bits per byte of the FIRST-byte predictions only: at chunk position m the model predicts byte (m+1)·P
(the next chunk's first byte). The v0 comparable metric — the full-byte bpb needs a local module (v1).
Stock Qwen is scored on the same byte positions for the control.
_Avoid_: bpb unqualified (that means every byte, which v0 does not predict)

**Chunk resolution**:
The model runs at one position per chunk (MEGABYTE's global module). The sequence is ~1.3× shorter than
the BPE sequence for the same text (6 bytes/chunk vs 4.57 bytes/token).
_Avoid_: patch space, coarse resolution

**Distribution shift**:
The gap between what the frozen stack was trained on (BPE token-embedding sequences) and what the
adapter feeds it (constructed patch vectors). The report's Finding 4 predicts this degrades capability
until a training pass adapts the network; the patch embedder's construction minimizes the shift's size.
The measurement's purpose is to quantify the degradation, not to argue it away.
_Avoid_: domain gap, OOD (overloaded)

**Constructed weights**:
Weights written by reasoning/construction, never by gradient descent (the weight-space-programming
post's sense; the qwentensor lineage's sense). The patch embedder's sum is the trivial case: weights
all 1.
_Avoid_: hand-crafted (vague), zero-training (the property, not the artifact)

## One-mind vocabulary

**Trunk**:
The frozen generative model that *is* the mind. Today: TwIL-8B. Not a component, and not the
endgame substrate.
_Avoid_: substrate, backbone, base model

**Substrate**:
The model that hosts the Companion (endgame: the byte H-Net line). A different thing from the
trunk.
_Avoid_: trunk

**S1**:
The trunk's fast depth — the fast local layers in the H-Net anatomy (SSM/DeltaNet), or one
pass without deliberation on a dense trunk (`enable_thinking=False`).
_Avoid_: fast model, decision model, StartLux

**S2**:
The trunk's deliberate depth — the full-attention middle over compressed chunks in the H-Net
anatomy, or `enable_thinking=True` on a dense trunk. Never a second model.
_Avoid_: verifier, second opinion, escalation model

**Glial**:
The regulator inside the forward pass: a slow-state reader that recruits depth. Two functions:
brake = whether an answer leaves (the calibrated gate); steering = what engages (the learned
regulator).
_Avoid_: gate (alone), router between models

**StartLux**:
A peripheral closed-set decision head — a reflex for one task class, like a spinal circuit. Not
the mind and not the fast path of the mind.
_Avoid_: S1, the fast path

**One mind**:
The trunk and its internal depth regulation. Not a committee of models voting, not a pipeline.
_Avoid_: ensemble, router, pipeline
