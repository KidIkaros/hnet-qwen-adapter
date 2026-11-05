# Model Card — qwen3.5-byte-adapter (v2)

**A tokenizer-free byte LM built on a frozen Qwen3.5-0.8B trunk + ~13.4M learned glue.**

| | |
|---|---|
| Base model | Qwen/Qwen3.5-0.8B (frozen, bf16) — used through byte-token rows + byte-marginalized head |
| Glue | ridge→BPE-initialized rank-32 patch map (~160k) + pair-window ByteDecoder (13.2M) |
| Interface | periodic-6 chunks; boundary bytes from the trunk head, interior bytes from the decoder |
| Training | 13 markdown docs (~105kB), ~2 min A100 total; trunk untouched |
| Metric | full-byte bpb = total bits / total bytes |

## Evaluated results

| corpus | stock BPE | byte-level | **v2 composed** |
|---|---|---|---|
| markdown eval docs (24,576 B) | 4.9226 | 5.5730 | **4.9400** |
| hnet val split, cross-domain (85,796 B) | 3.7560 | 4.3310 | **4.1309** |

## Intended use

Research artifact: demonstrates that a frozen token-level LLM can be driven at byte level with
~0.4 bits/byte overhead through ~13M parameters of learned glue — no tokenizer, no trunk training.

## Limitations

- Trained on 13 markdown documents (~105kB) — narrow data; generation is weak/fragmentary.
  The model is a strong *scorer*; coherent *generation* needs a longer-context decoder.
- Boundary chunking is periodic (period-6); adaptive chunkers exist in the repo.
- No instruction tuning, no safety measures; byte-level only.
- Evaluated on one domain pair (markdown→English); other domains unmeasured.

## Reproduce

`kernel/train_decoder.py` on a Colab A100; results logged in `results/a100_v3_scaled_gen.log`.
Weights: `results/byte_decoder.pt`.
