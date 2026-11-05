# Research — vetting the H-Net-of-Qwen re-slice against the literature

Date: 2026-11-05. Design under test (`src/hnet.py`, `kernel/hnet_adapt.py`,
`kernel/hnet_probe.py`): a hierarchical byte model re-sliced from a frozen Qwen3.5
checkpoint (3:1 Gated DeltaNet : Gated Attention hybrid stack, `layer_types` /
`block_type`).  First `enc_layers=4` DeltaNet layers = byte encoder; all
`full_attention` layers = main module over pooled chunk states; last
`dec_layers=14` DeltaNet layers = byte decoder; learned q/k boundary heads
(`BoundaryHeads`, H-Net cosine routing); every cut surface fitted with a
closed-form ridge on activations harvested from the intact trunk at byte level.

Measured state (`results/a100_hnet_probe.log`, `results/a100_hnet_adapt.log`):
unadapted re-slice (4B-Base) = **13.69 byte bpb — worse than the 8-bit uniform
floor, i.e. anti-predictive**; adapted re-slice (0.8B) = **7.70–7.92 bpb** on
8,192 B of a decontam GitHub doc; w_main contributes +0.004 bits; retrieval ctx
beats prev-chunk broadcast by ~0.15.  For scale, the same trunk's byte-level
floor is 5.57 (0.8B) / 4.53 (4B-Base) decontam (`a100_decontam.log`).

Qwen3.5 architecture confirmed against HF source/docs: 3 DeltaNet : 1 attention
blocks, per-layer `block_type`, `Qwen3_5GatedDeltaNet` (conv1d + chunked gated
delta rule + gated RMSNorm), `Qwen3_5DecoderLayer` with per-layer
`input_layernorm`.
(https://huggingface.co/docs/transformers/model_doc/qwen3_5,
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## 1. Layer reuse / subnetwork surgery — SUPPORTED, with a "needs healing" caveat

Extracting contiguous layer blocks from a pretrained model is an established
technique — the entire depth-pruning literature does exactly this:

- **Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers"
  (ICLR 2025)** — up to ~half of LLaMA's layers can be deleted with minimal QA
  degradation **after** a small healing finetune; deep layers are the most
  removable.  https://arxiv.org/abs/2403.17887
- **ShortGPT (2403.03853)**, **Shortened LLaMA (2402.02834)**,
  **LLM-Streamline (2403.19135)**, **A Deeper Look at Depth Pruning
  (2407.16286)** — same picture: contiguous-block excision works, but every
  paper heals with retraining or a trained replacement layer; none claim
  zero-shot extraction is free.  https://arxiv.org/abs/2403.03853,
  https://arxiv.org/abs/2402.02834, https://arxiv.org/abs/2403.19135,
  https://arxiv.org/abs/2407.16286
- Recency: **"No Free Swap" (2605.16234)** and **"Rethinking Layer Redundancy"
  (2604.24938)** — 2026 papers finding layer redundancy is
  protocol/objective-dependent, not an intrinsic property; a cut that looks
  free under one eval is not free under another.
  https://arxiv.org/abs/2605.16234, https://arxiv.org/abs/2604.24938
- **SOLAR DUS (2312.15166)** — concatenating duplicated mid-layer blocks works
  because "layer distance at the seam" is minimized; performance drops then
  rapidly recovers with continued pretraining.  Same lesson: the seam is the
  cost.  https://arxiv.org/abs/2312.15166
- **MOHAWK (2408.10189)** and **The Mamba in the Llama (2408.15237)** — the
  hybrid/SSM-side prior art: attention layers *can* be replaced/reused as
  Mamba-style layers, but only via staged distillation (matrix→hidden-state→
  end-to-end alignment, 3–5B tokens).  **LoLCATs (2410.10254)** — even swapping
  softmax→linear attention requires an output-matching fit ("attention
  transfer") + LoRA.  https://arxiv.org/abs/2408.10189,
  https://arxiv.org/abs/2408.15237, https://arxiv.org/abs/2410.10254
- **Tuned lens (Belrose et al., 2303.08112)** — the direct prior-art analogue
  of `w_out`: one affine probe per layer translating mid-stack states into the
  basis the unembedding expects; exists *because* the raw logit lens (identity
  map) is brittle at mid layers.  https://arxiv.org/abs/2303.08112
- ELMo (1802.05365) — task-side repurposing of frozen layer stacks via learned
  combinations; historical precedent only.
- Architecture precedent for the E–M–D re-slice itself: the community
  parameter-golf reproduction trains a 1-stage H-Net as an encoder/main/decoder
  split of a *single* transformer backbone — but from scratch, not re-sliced.
  (https://github.com/openai/parameter-golf/pull/1104 — community artifact.)

**Does the literature support "mid-stack layers are off-manifold when fed raw
embeddings"?** Yes — convergent evidence:

- Token embeddings are degenerate into a narrow cone under tied-weight LM
  training (Gao et al., Representation Degeneration,
  https://arxiv.org/abs/1907.12009); mid-layer residual states do not live in
  that cone — the two are distributionally different manifolds by construction.
- Mid-stack residual states carry input-agnostic massive-activation bias dims
  created by the early layers (Sun et al., COLM 2024,
  https://arxiv.org/abs/2402.17762) — a pattern raw embeddings lack; attention
  layers consume it.
- Vanilla LLMs cannot early-exit or skip layers without being *trained* to
  (LayerSkip, 2404.16710) — layer inputs are conditioned on their predecessors.
- Linear transport between residual-stream sites decays with leap distance:
  per-feature R² ≈ 0.95 for small leaps, degrading for large ones, and later
  layers prioritize freshly synthesized (non-linearly written) features over
  transported ones (Activation Transport Operators,
  https://arxiv.org/abs/2508.17540).  Feeding layer k a vector the network
  never would have produced is the large-leap case.
- Weight-level function similarity decays with layer distance (DOCS,
  https://arxiv.org/abs/2501.16650); the residual stream's effective depth is a
  learned, correlated-update regime — measured **on Qwen3.5 specifically**
  (D_eff 32–44%, https://arxiv.org/abs/2609.31098).

Our 13.69-bpb unadapted probe is exactly the predicted signature.

**Caveat that cuts both ways**: the pruning literature says the *deepest*
layers are the most redundant — our decoder slice is the last 14 DeltaNet
layers.  Two readings: (a) safe — they do the least damage reused; (b) weak —
they are the least-load-bearing slice, capping decoder capacity.  Worth an
enc/dec fraction sweep.

## 2. Closed-form interface adapters — SUPPORTED

Ridge/least-squares linear maps between activation spaces are canonical:

- **Cross-lingual embedding alignment** — the original closed-form setting:
  Mikolov et al. 2013 (least-squares W, https://arxiv.org/abs/1309.4168),
  Xing et al. 2015 (orthogonal constraint → Procrustes, closed form via SVD,
  https://aclanthology.org/N15-1104/), Conneau et al. MUSE
  (https://arxiv.org/abs/1710.04087).  Our "paired-Procrustes ridge→BPE" is
  literally this recipe on the shared-vocab pairing.
- **Model stitching** — Lenc & Vedaldi 2015 (origin,
  https://arxiv.org/abs/1411.5908); Bansal, Nakkiran & Barak, NeurIPS 2021
  (https://arxiv.org/abs/2106.07682): bottom of A + linear stitch + top of B,
  networks frozen, stitch trained briefly — works without performance drop even
  across seeds/datasets/objectives.  Csiszárik et al., NeurIPS 2021 "Dr.
  Frankenstein" (https://arxiv.org/abs/2110.14633): a *single affine* stitch
  matches representations across initializations, and they compare stitch
  classes — closest to "which closed form".  Hernandez et al. 2023
  (https://arxiv.org/abs/2303.11277): stitching across *different*
  architectures/shapes works but warns stitches can find surrogate
  representations that are not true matches.
- **Direct matching vs task-loss matching** — arXiv:2412.11299 shows
  task-loss-fitted stitches create *out-of-distribution* representations to win
  the task metric, while direct matching (minimize mapped-representation
  distance — i.e., our ridge recipe) is the more faithful interface.
  https://arxiv.org/abs/2412.11299 — supports closed-form over end-task-fitted.
- **Closed-form specifically**: the cross-model KV-cache ridge
  (arXiv:2608.03893 — per-head ridge on 500 sequences, Qwen family, see
  RESEARCH_WEIGHT_TRANSFER.md §1.3) and Wiring's closed-form compensation
  formula (arXiv:2608.02829, §7, in-doc §1.1) are the strongest positive
  existence proofs; ATO (2508.17540) fits linear inter-layer maps
  gradient-free.
- Caveats imported from the stitching-critique literature: functional alignment
  can mislead — even random-noise senders can be stitched into a trained
  receiver (Smith et al., PMLR v267,
  https://proceedings.mlr.press/v267/smith25a.html); a good ridge R² does not
  certify the receiver is using the *right* information.  And ridge is linear:
  the KV paper found failures where a nonlinear map adds +37pp by relocating
  residual error out of sensitive directions — measure *where* the ridge
  residual lands, not just its size.

## 3. Decoder conditioning — HAS-BETTER-ALTERNATIVE (known mechanisms we are missing)

The attributed bottleneck ("decoder lacks byte-level context") has documented
solutions we have not yet implemented:

- **H-Net (2507.07955)** — decoder input is
  `z = Dechunk(ẑ_main, p) + Linear(x̂_encoder)`:
  (a) a **learned, zero-init linear projection of the per-byte ENCODER states**
  added into the decoder input (`residual_proj` in goombalab/hnet
  `hnet/models/hnet.py`) — the U-Net skip that carries byte-local detail across
  the chunk bottleneck.  **Our design feeds raw byte embeddings into the
  decoder, never the encoder states** (`src/hnet.py`: `x = hs.clone()` where
  `hs = emb(...)`; `enc` is used only for pooling/routing).  This is the single
  most likely fix for the "byte-level context" gap — and it is ridge-fittable
  closed-form on the same harvest.  (b) The dechunk is an **EMA over chunk
  outputs weighted by boundary probability** (Eq. 5, implemented via the Mamba2
  scan), then zero-order-hold upsample; (c) upsample semantics: each byte
  position receives **its own chunk's** main-network output — our `ctx="prev"`
  broadcasts the *previous* chunk's state, i.e. context from before the chunk
  began.  H-Net's choice is causal (chunk output depends only on the
  chunk-start encoder state) and strictly more informative.
  https://arxiv.org/abs/2507.07955 · https://github.com/goombalab/hnet ·
  annotated: https://vpj.github.io/paper_implementations/hnet/hnet.html
- **BLT (2412.09871, ACL 2025)** — the local decoder alternates byte-level
  transformer layers with **cross-attention layers where byte queries attend
  to the global patch representations** (K/V from patch latents incl. the
  current patch), repeated after each transformer layer — not a one-shot input
  injection.  Our `ctx="retrieve"` (`decoder_attn_pass`) is a single
  input-side approximation of this.  BLT also keeps byte-local information via
  n-gram hash embeddings — another "don't lose the bytes" mechanism.
  https://arxiv.org/abs/2412.09871 · https://github.com/facebookresearch/blt
- **MEGABYTE (2305.07185)** — patch representation is broadcast/added into the
  per-byte local-model inputs (same family as our broadcast dechunk).
  https://arxiv.org/abs/2305.07185
- **Hourglass (Nawrot et al., 2110.13711)** — hierarchical transformer whose
  upsampling is combined with **skip connections from pre-pooled full-resolution
  states**; ablations pick the skip-connected variant.  Same mechanism as
  H-Net's residual_proj.  https://arxiv.org/abs/2110.13711
- **SpaceByte (2404.14408)** — the no-dechunk alternative: global blocks are
  interleaved *inside* the byte stack and run only at boundary positions, so no
  separate decoder interface exists.  If the dechunk interface stays lossy,
  this is the structurally different plan B.  https://arxiv.org/abs/2404.14408

Verdict: the gap is real but the mechanism to close it is known —
**add an encoder-state residual into the decoder input** (`x = w_enc(enc) +
dechunk_ctx` or fold into the w_in harvest), and consider current-chunk (not
prev-chunk) dechunk semantics + interleaved cross-attention if retrieval stays
bottlenecked.

## 4. Chunking eval — NEEDS-FIX: the sweep was vacuous by construction

The boundary-rule sweep showing "no difference" is not merely "on an unadapted
model" — it is **structurally guaranteed**: `kernel/hnet_probe.py:45` runs
`hn.dechunk.weight.data.zero_()`, so `ctx_v = dechunk(glob) ≡ 0`; the main
module's output never reaches the decoder in any mode (prev broadcast adds
zeros; retrieve attends over zero ctx).  All 9 cells in
`a100_hnet_probe.log` printing 13.6892 is a tautology about a disconnected
path, not evidence about boundary rules.  The sweep must be re-run **after**
the dechunk refit (which `hnet_adapt.py` does perform — but the adapted eval
only ever used periodic-6 boundaries).

- Internal precedent for the documented failure mode: RESULTS.md §8/DECISION —
  "no constructed chunker beats byte-level on the full-byte axis *without a
  trained decoder*" and "chunking goes net-positive once the interface is
  trained" (the v1 learned projection flips the sign).  This repo has already
  measured that boundary quality only expresses through a working interface.
- External corroboration is indirect but consistent: stitching work shows a
  composite with an unfit/random interface is chance-level regardless of the
  sender's content (2106.07682); H-Net's boundary quality is measured only
  under joint training, and its ablations show the boundary *mechanism* matters
  through trained dynamics (smoothing removal → unstable compression ratios,
  https://arxiv.org/abs/2507.07955); BLT's entropy-vs-space-vs-fixed patching
  differences are all measured on trained models
  (https://arxiv.org/abs/2412.09871).
- Flag: "boundary choice matters only after the interface is fit" is
  single-source — our own logs are the strongest evidence; the literature
  supports the mechanism but nobody states this exact claim.

## 5. Red flags — SSM/linear-attention layers on OOD inputs — MIXED

- **State blowup: unlikely.**  The gated delta update
  `S_t = S_{t−1}·(α_t(I−β_t k_tk_tᵀ)) + β_t v_t k_tᵀ` is contractive by
  construction (α∈(0,1) gate, sigmoid β) — the state cannot grow without bound
  regardless of input (Yang et al., https://arxiv.org/abs/2412.06464).  Mamba
  SSMs are Lyapunov-stable learners — bounded input deviations do not produce
  exponentially diverging outputs (https://arxiv.org/abs/2406.00209).
- **But every input writes to the state.**  Unlike softmax attention (which can
  attend ~zero to junk), a DeltaNet layer *must* fold each input into S_t; a
  garbage dechunk ctx at position t corrupts all subsequent reads of that
  state.  The decoder slice is the worst place in the stack to inject an
  unfitted residual — the error is recurrent, not per-position.
- **Gate calibration is input-dependent and OOD-sensitive.**  α/β/dt come from
  projections of the (normed) layer input; inputs outside the trained
  distribution shift decay/write rates off their operating point — fast
  forgetting or state saturation, no explosion.  Documented analogue:
  length-extrapolation failure in Mamba is traced to OOD state-transition
  dynamics and the A-matrix spectrum ("Mamba Modulation",
  https://arxiv.org/abs/2509.19633); the generalization theory is in
  https://arxiv.org/abs/2502.01473.
- **Norms partially mitigate.**  The re-slice calls `layer()` wholesale, so the
  trained per-layer `input_layernorm` still rescales every cut-surface input —
  Jamba needed exactly these in-Mamba norms for stability at scale
  (https://arxiv.org/abs/2403.19887).  Norm fixes magnitude, not direction;
  the residual risk is directional (which subspaces are active), not scalar.
- **The attention main module has its own OOD risk:** pooled DeltaNet
  chunk-end states lack the massive-activation bias dims and attention-sink
  pattern attention layers were trained against (2402.17762; StreamingLLM
  2309.17453) — garbage keys/values, not blowup.  w_main's ~null gain
  (7.7082→7.7047) is consistent with a too-big leap for one linear map
  (ATO: transport efficiency plateaus with leap size; consider per-attention-
  layer w_main, fit at each attention layer's input, not only `main_start`).
- **Perturbation-sensitivity is position-dependent:** the last block and the
  output projection are the most perturbation-sensitive components of a Mamba
  stack (HPD, https://arxiv.org/abs/2508.11935) — consistent with w_out being
  the adapter that matters.
- **Instrument, don't assume:** log per-decoder-layer state norms and α/β gate
  histograms under adapted vs harvested inputs — the divergence signature will
  be visible long before the bpb gap closes.

## 6. Honesty ledger / single-source flags

- **Zero-training hybrid re-slice is novel ground.**  All hybrid/subnetwork
  prior art (MOHAWK, Mamba-in-Llama, LoLCATs, pruning-healing, DUS) requires
  *some* training/distillation.  Nobody has published re-slicing a pretrained
  hybrid checkpoint into an H-Net with only closed-form adapters; 7.7 bpb
  functional-but-degraded is consistent with, not contradicted by, the
  literature's "surgery needs an interface" consensus.
- "Boundaries matter only post-interface": strongest evidence is our own logs;
  external support is mechanism-level, not a stated result.
- The parameter-golf E–M–D H-Net split is a community reproduction artifact,
  trained from scratch — architecture precedent only.
- Csiszárik/Frankenstein and ATO measure *cross-seed/within-model* stitching;
  same-family-layer-slice is our extrapolation.
- Depth-pruning evidence is pure-transformer; no published depth-pruning
  numbers on Gated-DeltaNet hybrids found.

## What to try next (ranked by expected bits per hour)

1. **Encoder-skip residual** — feed `enc` (not `emb`) into the decoder path via
   a ridge-fitted `w_enc: enc → dec-input residual`, or re-harvest w_in with
   enc as source.  H-Net's residual_proj, closed-form, directly targets the
   stated bottleneck.  *Trivial — one harvest + one ridge.*
2. **Re-run the boundary sweep post-adaptation** — dechunk refit makes the main
   path live; periodic/spacelike/sim should now separate.  *Trivial.*
3. **Current-chunk dechunk semantics** — upsample ẑ_j to chunk j's bytes
   (H-Net-faithful) vs the current prev-chunk broadcast.  *One-line change,
   one eval.*
4. **Per-layer w_main** — ridge at every attention layer's input (ATO leap
   finding says one map can't span the pooled→first-attn gap).  *Same recipe,
   more harvests.*
5. **Interleaved decoder cross-attention** (BLT-faithful): reuse 1–2 checkpoint
   attention layers between decoder DeltaNet layers over the chunk-state K/V.
   *Moderate — the existing decoder_attn_pass is the prototype.*
6. **Gate/state instrumentation** — state-norm and α/β histograms adapted-vs-
   harvested before/after each decoder layer.  *Instrumentation only.*
7. **enc/dec fraction sweep** — 4/14 was chosen by H-Net analogy, not measured;
   deep layers are the most redundant (2403.17887), so the decoder slice may be
   over-provisioned relative to the encoder.

## Verdict summary

1. **Q1 layer reuse — SUPPORTED.** Depth-pruning (Gromov ICLR'25, ShortGPT, Shortened-LLaMA), SOLAR-DUS, tuned-lens, and stitching lit establish subset-of-layers reuse; "off-manifold mid-layers" is corroborated by representation-degeneration, massive-activations, LayerSkip, ATO, DOCS. Caveat: every prior case heals/fits *something* — nobody gets it free; our 13.69→7.7 arc matches.
2. **Q2 closed-form adapters — SUPPORTED.** Procrustes/ridge on paired activations is canonical since Mikolov'13/Xing'15/MUSE; stitching (Lenc-Vedaldi, Bansal-Nakkiran-Barak, Csiszárik's Dr. Frankenstein) is the same method; 2412.11299 explicitly prefers direct-matching (our ridge) over task-loss stitching.
3. **Q3 decoder conditioning — HAS-BETTER-ALTERNATIVE.** Missing H-Net's documented mechanism: `dechunk + Linear(encoder_states)` skip; current-chunk (not prev-chunk) upsampling; BLT's interleaved byte→patch cross-attention.
4. **Q4 chunking eval — NEEDS-FIX.** The null sweep is vacuous: `dechunk.zero_()` disconnects the main path; all rules *must* print identical bpb. Re-run post-adaptation.
5. **Q5 red flags — MIXED.** State blowup unlikely (contractive gated delta rule), but every byte writes to SSM state → bad ctx corrupts recurrently; attention main gets no massive-activation/sink structure; the output end is perturbation-sensitive — w_out is where risk lives.
6. **Bottom line: proceed, but implement the encoder-skip + redo the boundary sweep before drawing conclusions about chunking or the main module.**
