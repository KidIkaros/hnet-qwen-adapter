# Spec — TwIL-8B leg-1: learned byte adapter on frozen trunk (GPU kernel)

Date: 2026-11-10. Purpose: train a **learned local patch embedder** (replacing
`PatchEmbedder`'s zero-param embedding-row sum) on frozen TwIL-8B hidden
states, and measure whether it closes the v0 byte-bpb gap on **real,
non-repeated, deployment-matched text**.

Prerequisite evidence: `reports/runs/twil-probe-v3/` — Route A viable
end-to-end; period-6 first-byte bpb 6.74/6.56 (untrained embedder ≈ uniform),
effective_bpb 2.41 on the period-1 path **but on ×30-repeated docs (flattered)**;
native S1/S2 tool-call works. Reference points: 27B v0 ≈ 4.26 bpb,
27B trained leg-1 ≈ 2.79 held.

Compute: Kaggle GPU, `NvidiaTeslaT4` ×2 (`machine_shape` mandatory),
~1.5–2 h wall-time budget. **Owner approval required before push.**

## What leg-1 must answer

Does a *learned* byte→trunk-embedding map transfer LM ability to the byte
pathway? v3 proved mechanics (inputs_embeds, logits_to_keep, coverage);
it did not prove semantics — the trunk's DeltaNet layers were trained on
*token* inputs, and whether byte-derived vectors carry usable signal is
the open question. This is the existential measurement for Route A.

## Corpus spec — train on what it's used for

Deployment distribution = agentic text: tool schemas, call syntax
(`<tool_call>`, `[func(args)]`), instructions, retrieved context, and the
model's own `<think>` reasoning traces. Generic web prose is the wrong
prior. All sources resolve through `DOLMA3_SOURCES` / `corpus_stream_next`
in `src/factory.py` (weighted multi-source streaming + `_text_of` joins
conversation lists — agentic rows serialize correctly).

| bucket | weight | sources (DOLMA3_SOURCES keys) | rationale |
|---|---|---|---|
| agentic / tool-use | 0.40 | `nemotron_terminal` 0.15, `open_swe_traces` 0.15, **ToolACE convs** 0.10 | trajectories + actual call syntax — the deployment format |
| code | 0.25 | `stack_edu` | byte-dense structure; agents read/write code |
| reasoning / instruction | 0.20 | `finemath_3plus` 0.12, `proofpile_arxiv` 0.08 | matches TwIL's GRPO/DAPO post-training domain |
| general prose | 0.15 | `common_crawl` 0.10, `dolma_wiki` 0.05 | coverage baseline — keeps adapter from going narrow |

- **ToolACE source**: custom loader in-kernel (~30 lines, parser proven in
  s1-headtohead v2) — `load_dataset("Team-ACE/ToolACE", split="train",
  streaming=True)`, serialize `system+user+assistant` turns to text.
  Take a contiguous ~2k-row slice disjoint from the SKIP=2500 eval windows
  used elsewhere (use rows 5000+ to stay clear).
- **Splits (three-way, disjoint)**: per source, stream position A = train
  docs, position B (later, non-overlapping) = eval docs. `SEEN`-dedup
  pattern from `factory.py`. ~300–500 train docs (~1–2 MB text),
  ~4 eval docs per source per domain (~28 docs total).
- **Eval docs**: real held-out text, **non-repeated**, per-domain bpb
  reported separately — the v3 ×30-repeat caveat is explicitly retired.
- **Tracked gap (not a v0 blocker)**: byte coverage 207/256 — the 49
  unmapped bytes are unprintable/high-byte values; deployment text is
  mostly ASCII. Multilingual/UTF-8 stress is a leg-2 item.

## Architecture — what trains, what freezes

- **Frozen**: entire TwIL-8B trunk (`requires_grad_(False)`, `eval()`),
  `ByteHead` readout (logits→byte-prob marginalization is a fixed map),
  `PeriodicChunker`.
- **Trained**: `LearnedPatchEmbedder` — replaces `PatchEmbedder.embed`.
  v0 form: per-byte-value × per-position-in-chunk learned table
  `E[256, period, D]` summed over the chunk (generalizes the current
  sum-of-rows construction; ~2.1M params at period=2, D=4096).
  Init: E[b, p, :] = trunk embedding row of byte b's token (warm start
  from the v0 construction — the sum-embedder is the identity init).
- **Period**: **2** for the smoke (T = bytes/2; period-1 at T≈1024–2048
  is the fallback if period-2 stalls — v3 showed period-1 dominates but
  costs 2× positions; compression is the upgrade, not the start).
- **Doc cap**: 2048 bytes/doc for training (T≈1024 positions — bounded
  activation memory on T4); eval docs up to 4096 bytes.
- **Objective**: next-chunk-first-byte CE via `ByteHead.byte_probs`
  marginalization (same readout as v0 — the head IS a byte head for
  mapped bytes). Teacher-forced, one doc per step.
- **Backward path**: gradients flow embedder→trunk→logits. Trunk params
  frozen but forward must NOT be `no_grad` — enable
  `model.gradient_checkpointing_enable()` to bound activation memory
  (recompute-on-backward, ~2× forward cost, standard for frozen-trunk
  adapter training).
- **Optimizer**: AdamW, lr 3e-4–1e-3 (embedding-table scale), ~200–300
  steps, batch=1 doc. bf16 trunk, fp32 adapter params + master weights
  optional (table is small — plain fp32 fine).

## Kernel legs

### Leg 0 — load + baseline (~10 min)

Same device-map-safe shim as twil-probe-v3 (never `.to()` the split
model; forward via `model.model(inputs_embeds=...)`, cast hidden rows to
model dtype before `lm_head`, `attention_mask=None` for sdpa is_causal).
Load TwIL-8B bf16 2×T4. Verify `inputs_embeds`/`logits_to_keep` in 30 s.

### Leg 1 — v0 baseline on real eval docs (~10 min)

`first_byte_bpb` + `effective_bpb` per eval doc with the **untrained**
sum-embedder at period=2 — the honest pre-training baseline on real text
(v3 numbers were repeated-doc; these are not). Per-domain breakdown.

### Leg 2 — training (~40–60 min)

`LearnedPatchEmbedder` warm-started from v0 construction; train stream
docs teacher-forced; log train CE every 20 steps; checkpoint adapter
state_dict (`adapter_leg1.pt`) to `/kaggle/working/` periodically
(crash insurance — save_int4 lesson).

### Leg 3 — eval (~10 min)

Same eval docs through the learned embedder: `first_byte_bpb`,
`effective_bpb`, per-domain. Delta vs Leg-1 baseline. Save
`adapter_leg1.pt` + `leg1_train_log.json` + `leg1_eval.json`.

### Leg 4 — artifacts (~1 min)

`artifacts.zip`: adapter checkpoint, eval JSONs, train log, corpus
manifest (sources/weights/positions), `run_status.json`.

## Metrics & pass signal

- **Primary**: effective_bpb delta (learned − v0) on held-out real docs,
  per-domain. v0 baseline expected ~4–6 bpb on real text (v3's 2.41 was
  flattered); trained leg-1 target ≤ ~3.5 would be encouraging,
  ≤ ~2.8 matches the 27B trained reference.
- **Secondary**: first_byte_bpb per domain; train-CE curve (did it
  descend?); coverage stats.
- **Go/no-go**: learned embedder improves effective_bpb by >0.5 on
  held-out real docs → Route A viable, scale to leg-2 (more data,
  period-1, learned boundaries). Flat/worse → pivot assessment: bigger
  adapter, trunk byte-signal audit (which layers carry it), or Route B
  (H-Net re-slice).

## Failure modes to watch

- Activation OOM on backward → already mitigated by gradient checkpointing
  + 2048-byte cap; if still OOM, drop to 1024-byte docs.
- sdpa math-backend attention blowup at T≈1024 — unlikely (v3 survived
  T=4096 period-6 ≈ T=683 positions; period-2 T=1024 is similar scale).
- Adapter learns dataset quirks → mitigated by multi-source mix +
  disjoint eval; per-domain eval exposes narrowness.
- `decide()`-style API mismatch — N/A here (pure LM forward/backward).

## Cost

~1.5–2 h of ~30 h/week GPU quota: model load ~3 min, baseline ~10 min,
training ~40–60 min, eval ~10 min, slack. CPU lane cannot do this
(frozen-trunk backward needs the GPU).

## What this decides

Whether the byte substrate is worth scaling. Positive → leg-2 (bigger
corpus, period-1, learned/semantic boundary module — the H-Net++ reading
item). Negative → the trunk lacks exploitable byte-level signal and the
one-mind re-plans around token-native trunk + glial/StartLux S1 (which
is already proven).
