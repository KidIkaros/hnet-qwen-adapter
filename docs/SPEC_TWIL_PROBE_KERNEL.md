# Spec — bundled Kaggle TwIL-8B probe kernel

Date: 2026-11-10. Purpose: answer the four open TwIL-pivot questions in **one** Kaggle
GPU session (quota hygiene: one session, maximum information). Build target:
`NvidiaTeslaT4` ×2 via `machine_shape`, ~1 h wall-time budget. Owner approval required
before push (AGENTS.md).

Prerequisite research: `docs/RESEARCH_TWIL_PIVOT.md` (all local verification done —
attention layout, byte coverage 207/256, chat-template S1/S2 controls).

## Model loading

- `webAI-Official/TwIL-8B` root = **balanced** variant (bf16, ~16.4 GB).
- `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16,
  device_map="auto", max_memory={0:"14GiB",1:"14GiB"})` — 2×T4 split. Granite
  `GraniteForCausalLM` supports `inputs_embeds` (needed by `ChunkLM._logits`) and
  `logits_to_keep` — verify both in-session before legs B/C.
- Tokenizer: `AutoTokenizer` — byte coverage already verified locally (207/256,
  all printable ASCII); leg A re-verifies in 2 s for the record.

## Legs

### Leg A — byte map + interface sanity (~1 min)

`byte_token_map(tok)` coverage count; `PatchEmbedder.embed` on 3 synthetic chunks;
assert `model(inputs_embeds=...)` forward returns logits of shape (1, M, V).
**Gate**: if `inputs_embeds` or `logits_to_keep` unsupported → log and skip B/C.

### Leg B — v0 byte-adapter bpb (~15 min)

`ChunkLM(model, tok, period=6)` from `src/adapter.py` — zero new params, frozen trunk.

- Corpus: ~32–64 KB of held-out text. Source: small HF dataset sample (e.g.
  `wikitext-103-raw-v1` test split slice) or embedded markdown — **no training-data
  overlap concerns for a smoke metric; record the source string in artifacts**.
- Metric: byte-level bpb via `ByteHead.byte_probs` marginalization over chunk
  positions (same protocol as the 27B v0 runs for comparability).
- Also record: mean chunk coverage, per-position byte-entropy profile (feeds the
  EntropyChunker design question later).
- **Reference points** (context, not pass/fail): 27B v0 ≈ 4.26 bpb (0.8B stock);
  trained leg-1 ≈ 2.79 held. TwIL-8B is a stronger trunk than 0.8B — v0 landing
  ≤ ~3.5 bpb would be encouraging; the number matters less than "coherent byte
  predictions at all."

### Leg C — tool-call emission, S1 vs S2 (~15 min)

- Prompt set: 10–20 BFCL-style tool-use examples (single-turn, 1–3 tools).
  Source: embed a small JSONL in the kernel (curated, not the full BFCL download —
  keeps the session lean) OR pull `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
  test split slice if network allows.
- For each example, run generation twice via `tokenizer.apply_chat_template`:
  - **S1**: `enable_thinking=False` → expect direct `<tool_call>` emission
  - **S2**: `enable_thinking=True` → expect `<think>...</think>` then `<tool_call>`
- Parse emitted text with a small regex/parser for
  `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>...` (values may
  span lines; multi-call possible).
- Metrics: format-valid rate, function-name accuracy vs GT, parameter accuracy,
  think-block presence/length (S2), tokens-generated (cost signal).
- **This doubles as the BFCL-format adapter test** — the parser written here becomes
  the harness's TwIL call-parser.

### Leg D — LSA selection-fidelity probe (~30 min, FLAG-GATED)

Run only if legs A–C pass and session time remains (`RUN_LSA_PROBE` cell flag).
Spec: `docs/RESEARCH_TWIL_PIVOT.md` §4 — PCA index projections per layer, k-means
buckets, mass-recall/KL/top-k-overlap vs baselines on our workload texts at
4k/16k/64k contexts (131k only if memory allows). Output: per-layer score matrix
`.npz` + verdict table. If it doesn't fit, defer to a second session — do NOT let it
compromise legs A–C artifacts.

## Artifacts (single `artifacts.zip` per AGENTS.md)

- `run_status.json` — leg verdicts, timings, GPU metadata (`torch.cuda.get_device_name`)
- `twil_bpb.json` — leg B metrics + corpus source + coverage stats
- `toolcall_results.jsonl` — per-example emissions (S1+S2), parsed calls, scores
- `toolcall_parser.py` — the reusable TwIL `<tool_call>` parser (source, for reuse)
- `lsa_probe.npz` + `lsa_verdict.json` — only if leg D ran
- `src/` bundle used (adapter.py, corpus.py, factory stubs as needed)

## Session hygiene

- `machine_shape: "NvidiaTeslaT4"`, assert 2×T4 in cell 1.
- HF token via Kaggle secret `HF_TOKEN` if repo gated (TwIL is public — likely
  unneeded; keep the secret path anyway).
- Delete kernel after artifact pull unless it becomes the canonical TwIL probe
  notebook (it probably should — name slug accordingly, e.g. `twil-probe-v1`).
- Pre-push checklist per AGENTS.md: rebuild → cell-by-cell `ast.parse` → dep-order
  audit → never-run legs traced → `tests/` torch-free subset locally.

## Explicit non-goals

- No training (leg-1 H-Net adapter, energy head, D-LSA distillation) — probe only.
- No int4/native-loader path — bf16 only this session.
- No 27B anything — separate staged kernel remains frozen pending approval.
