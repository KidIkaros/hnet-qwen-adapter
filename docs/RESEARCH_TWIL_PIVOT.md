# Research — TwIL-8B pivot assessment + LSA selection-fidelity probe spec

Date: 2026-11-10. Context: the 27B trunk (Qwen3.5-27B int4) is compute-blocked — Modal
frozen for billing, Kaggle 2×T4 is the largest free GPU (32 GB aggregate), Colab units
unverified. Candidate pivot trunk: `webAI-Official/TwIL-8B` (Granite-4.2-8B base,
8.79B dense, bf16 16.38 GiB, Q4_K_M GGUF 4.98 GiB published, 131k context,
SFT→GRPO/DAPO→length-aware RL post-training, `<think>` reasoning format, balanced +
verbose variants). License: `webai-non-commercial-license-ver.-1.0` — research/private
use only until cleared.

## 1. H-Net port analysis — what actually transfers

Two H-Net implementations exist; they have different trunk requirements:

- `src/adapter.py` (v0, "no-training"): bytes → `PeriodicChunker` → patch vector =
  **sum of the bytes' own token-embedding rows** → frozen trunk over chunk positions →
  `ByteHead` (LM head marginalized to bytes). **Trunk-agnostic** — works on any causal
  LM with an embedding table and byte-covering tokenizer.
- `src/hnet.py` (full): **re-slices the checkpoint's own layers by `block_type`** —
  first `enc_layers` `linear_attention` (DeltaNet) layers = byte encoder, all
  `full_attention` layers = main module over pooled chunks, last `dec_layers`
  DeltaNet layers = byte decoder, `BoundaryHeads` q/k routing, ridge-fitted `w_*`
  glue. The efficiency thesis requires the 3:1 linear:full hybrid: byte positions get
  O(T) SSM mixing, chunk positions get O(C²) attention (C ≈ bytes/period).

TwIL-8B is a **dense** transformer — no `linear_attention` layers exist to re-slice.
The re-slicing shortcut does not transfer; running full-attention "encoder" layers
over raw byte positions costs O(bytes²) and destroys the hierarchical-compute thesis.

Byte-interface compatibility verified: Granite tokenizer is ByteLevel BPE (GPT-2
byte-unicode scheme, same as Qwen) — `byte_token_map` in `src/corpus.py` ports
directly (single-byte token ids exist for the printable byte range; unprintable
controls unmapped, same 221/256 coverage class as Qwen).

### Routes for byte-level modeling on a dense trunk

| route | trunk | mechanism | status |
|---|---|---|---|
| A. H-Net-as-modules | TwIL-8B | v0 adapter now → trained small enc/dec modules later (BLT/H-Net formulation: local encoder/decoder need not be trunk layers) | feasible on Kaggle |
| B. Re-sliced H-Net | Nemotron-H-8B-Reasoning-128K | Mamba-2 layers ≈ DeltaNet role; `layers_block_type` vocabulary matches `"linear_attention"`/`"full_attention"` | needs layer-call interface port (Mamba fwd signature ≠ DeltaNet); NVIDIA research-only license |
| C. Re-sliced H-Net | Kimi-Linear-48B-A3B | KDA = literal Gated DeltaNet, 3:1 hybrid identical to Qwen3.5 | 48B total — too big for free compute; revisit if small hybrid appears |

Route A recommended: TwIL's GRPO/DAPO reasoning post-training is the S1/S2 substrate
(IFBench 74.4); dense arch = every quant/eval tool works; bf16 fits Kaggle 2×T4.
Honest cost: v0 (periodic chunker, no enc/dec) will likely land worse bpb than the
trained 27B version — but it runs today and the trained-enc/dec upgrade is a
Kaggle-feasible leg at 8B scale.

### One-mind layering on TwIL-8B (target architecture)

```
bytes → chunker (periodic → learned BoundaryHeads later)
      → patch embedder (embedding-row sums — exists)
      → [optional small trainable local enc/dec — Route A upgrade]
      → TwIL-8B frozen over chunk positions
      → byte head (ByteHead marginalization — exists)

S1: TwIL with enable_thinking=False (see §1a — native fast mode)
S2: TwIL with enable_thinking=True (<think> deliberation) or latent descent (EBR)
Glial: energy/gradient-norm/convergence signals → stop/escalate/route;
       escalation = choosing which generation prompt to emit (§1a)
```

### 1a. Chat-template findings (verified 2026-11-10, `chat_template.jinja`)

The template **natively encodes the S1/S2 split** — no wrapper machinery needed:

- `enable_thinking=False` → generation prompt emits `<|im_start|>assistant\n<think></think>`
  (pre-closed empty think) → model answers/calls tools directly. **This is S1-fast,
  built into the template.**
- `enable_thinking=True` (default) → open `<think>` block → deliberate reasoning = S2.
- `reasoning_effort="low"` → appends `{reasoning effort: low}` to the last user turn —
  soft length control; the model was RL-trained with length-aware rewards, so it has a
  *trained* signal for modulating reasoning cost, not a heuristic we invent.
- `truncate_history_thinking=True` → strips think blocks from history turns — free
  context management for multi-turn agent loops.
- Tool calls are native: `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n
  </parameter>\n</function>\n</tool_call>` — multi-call capable; tool results return as
  `<tool_response>` blocks. **Integration task**: our BFCL harness expects its own call
  format — needs a parser/adapter for this XML-ish scheme (straightforward regex/parse,
  values may span multiple lines).
- Assistant turns interleave think → tool_call correctly; reasoning-before-call is the
  trained order.

**Design consequence**: the balanced/verbose variants are two pre-trained operating
points on the same reasoning-length spectrum; `enable_thinking` + `reasoning_effort`
give a continuous dial. Glial escalation = prompt selection, and EBR-style energy can
later decide *when* to flip the dial — a cleaner interface than anything we'd build on
the 27B (whose template has no such controls).

## 2. Sparse attention — pattern ≠ kernel ≠ weight sparsity

Three distinct axes, often conflated:

- **Attention pattern** (algorithm): which keys each query sees — window, strided,
  sink+window, semantic clusters.
- **Attention kernel** (systems): how the pattern executes — FA3 page-table, FlexAttention,
  block-sparse. A sparse pattern through dense SDPA still burns O(T²) FLOPs.
- **Weight sparsity** (LTH/pruning): zero weights — needs sparse GEMM kernels; unrelated
  to attention patterns. The LTH paper (Frankle & Carbin ICLR'19) is MNIST/CIFAR-scale;
  its 90% claim does not transfer to LLMs (capability-selective degradation — same wall
  our int4 hit: scoring preserved, emission broken).

`src/hnet.py::causal_mask(sparse=(w, stride, sinks))` already implements the LSA-style
pattern — but through dense attention it is a mask, not a speedup.

### Sparse-attention papers in the library

- **Latent Sparse Attention** (arXiv 2610.42147, Prefrontal-AI) — keys clustered in a
  learned latent space; queries score cluster centroids (flat or tree beam-search →
  O(log T)/query). **Demonstrated on GLM-5.3-Flash = Qwen3.8-27B** — our trunk family.
  D-LSA distills onto frozen pretrained models; ships Triton indexer kernels.
  Repo: `github.com/Prefrontal-AI/latent-sparse-attention`.
- **Zyphra Hierarchical Attention** (arXiv 2610.42112) — tree-structured KV indexer,
  subquadratic; fixes quadratic indexer cost of prior sparse attention.
- **Dynamic Token Pooling** (arXiv 2211.09761) — learned segment pooling, ancestor of
  the chunking lineage.
- **You don't need long context** — compaction/SSSM argument; context management, not
  attention sparsity.

## 3. D-LSA on TwIL-8B — feasibility verdict

**Mechanically attachable.** Per attention layer: two small index projections
`k^(idx) = k^(attn) W_K^(idx)`, `q^(idx) = q^(attn) W_Q^(idx)` to dim `d^(idx)`,
L2-normalized; keys pooled (n_pool) → Ward-clustered into buckets → centroid cache;
query selects top-p buckets → expands to k long-range keys + sinks + window →
attention over J_i only. Handles RoPE by projecting only RoPE-free dims (Granite uses
RoPE — covered). **Distillation trains only W + τ per layer, base frozen** — forward-KL
between frozen attention distribution and indexer softmax. Training-Free variant exists
(PCA projections, zero training).

**Three catches:**

1. **Kernel reality on sm_75**: indexer has Triton kernels ("runs on any GPU"), but the
   attention side reads J_i as a FlashAttention-3 page table — FA3 is Hopper-only; T4
   needs FA2 (sm_80+) → gather-attention needs FlexAttention or custom Triton. Real
   kernel work, the riskiest part.
2. **Solves a problem we don't have yet**: LSA wins at 128k–10M tokens (k=4096 tuned
   >1M). TwIL context is 131k — heavy but not prohibitive. Our actual sparse need is
   the H-Net byte encoder, where *local* structure dominates → sliding-window+sink
   (already implemented as a pattern) likely suffices; semantic clustering earns its
   complexity only at ultra-long range.
3. **Serving-stack method**: built for SGLang inference; using it in our training loop
   means reimplementing selection in PyTorch regardless.

**Decision: defer D-LSA; run the selection-fidelity probe first** (§4). LSA's existence
on Qwen3.8-27B means the 27B trunk already has a sparse-attention retrofit path — a
reason to keep that work warm.

## 4. Probe spec — "LSA-Selection Fidelity on TwIL-8B"

Answers one question cheaply: **does semantic key selection preserve what TwIL's
attention actually does, on our workloads?** Necessary-but-not-sufficient filter before
any distillation/kernel work.

### Setup

- TwIL-8B bf16 (~16.4 GB) on Kaggle 2×T4. `output_attentions` off — extract K/Q via
  forward hooks on `self_attn.q_proj`/`k_proj` per layer; compute metrics inside the
  hook, discard tensors (memory-safe at 131k).
- **Data = our workloads, not random text**: BFCL-style prompts with tool schemas;
  code+docs at 4k / 16k / 64k / 131k token sweep; ~50 sequences per bucket; 256 sampled
  query positions per sequence (stratified early/mid/late).
- **Selection simulation** (mirrors LSA Training-Free variant): PCA-project K,Q to
  `d_idx` dims per layer → L2-normalize → pool `n_pool` consecutive keys → k-means
  cluster into buckets (Ward's is the paper's choice; k-means is an acceptable probe
  approximation — record as limitation) → query scores centroids → top-p buckets →
  expand to k keys + sink/window baseline. Sweep k ∈ {512, 2048, 4096, 8192}.

### Metrics — pass thresholds (pre-registered)

| metric | definition | pass |
|---|---|---|
| Mass recall | fraction of full-attention prob mass (excl. sink+window) captured by J_i | ≥90% at k=4096, ≥64k ctx |
| KL divergence | KL(full_attn ‖ renormalized sparse) — same loss D-LSA distills | ≤0.1 nats median |
| Top-k overlap | fraction of attention's true top-k keys inside J_i | ≥80% |
| Baselines | sliding-window-only and recency-only at same budgets | semantic must beat both, else clustering adds nothing |

Per-layer breakdown essential — expected heterogeneous outcome (retrieval/induction
heads sparse-friendly, diffuse heads not). "Partial yes" (some layers pass) is still
actionable: hybrid sparse/dense per layer.

### Result interpretation

- **YES** → D-LSA distillation justified; spec Kaggle leg (frozen 8B + tiny projections).
- **PARTIAL** → sparse only on compatible layers; narrower but real speedup.
- **NO** → LSA doesn't fit our scale/workloads; `{sink+window+strided}` mask is the
  ceiling; redirect effort.

### What it does NOT test

End-to-end generation quality, kernel speed, actual distillation. Passing means "worth
building," not "works."

### Cost

~1–2 h Kaggle session (dominated by 16 GB weight download + bf16 load). Harness is
trunk-agnostic — reusable verbatim on Nemotron-H or the 27B later.

## 5. Verification log — resolved 2026-11-10 (no weights needed)

- **Granite attention layout — VERIFIED** against `transformers/models/granite/
  modeling_granite.py` (transformers 5.22.1): `model.layers[i].self_attn.{q,k,v,o}_proj`
  standard names; RoPE applied *after* projection via `apply_rotary_pos_emb` with
  `(cos, sin)` position_embeddings — probe must capture post-RoPE Q/K (hook rotary
  output or re-apply RoPE to hooked projections). Config: 40 layers, 32 Q heads,
  8 KV heads, head_dim 128, hidden 4096, vocab 100352.
- **Byte coverage — VERIFIED locally** (tokenizer files only, synthetic single bytes):
  `byte_token_map` gives **207/256** on TwIL-8B. All printable ASCII (32–126) covered;
  missing 49 are high/control bytes (0x80–0x9F class). Slightly below Qwen's 221/256 —
  fine for text corpora; binary-heavy byte modeling needs a fallback mapping for the
  uncovered bytes (flag for NEXT_MIX-type corpora audit).
- **Chat template / S1-S2 controls — VERIFIED**: see §1a. `enable_thinking`,
  `reasoning_effort`, `truncate_history_thinking`, native `<tool_call>` format.
- **Remaining for Kaggle**: v0 adapter forward + byte-bpb on TwIL bf16; BFCL-format
  tool-call emission test (needs parser for the XML-ish `<tool_call>` scheme); LSA
  selection-fidelity probe (§4) if session budget allows.
- **License**: webAI Non-Commercial 1.0 — fine for research; flag before any
  redistribution or commercial deployment.
