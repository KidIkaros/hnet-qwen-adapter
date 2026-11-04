# hnet-qwen-adapter

**Can a frozen Qwen3.5 be turned into an H-Net-style byte-level model with zero training — constructed
glue only?**

The follow-on program from the qwentensor fork's research gate (`docs/DECISION_HNET.md` there): the
companion report predicted the composed system runs *degraded, not competitive* (distribution shift
through every frozen layer; the WSP post's Gemma control: editing cannot add a capability), and nobody has
measured how degraded. This repo measures it.

## The v0 architecture (no trained weights anywhere)

```
bytes ──► periodic byte-phase chunker (every N bytes; the falsification test showed the trained
          18.6M router's decisions are indistinguishable from this: +0.0009/+0.0011 bpb)
     ──► patch embedder: each chunk = the SUM of its bytes' own token-embedding rows
         (221/256 bytes map to exactly one of Qwen's embedding rows; constructed, zero trained weights;
         MEGABYTE's patch embedder reduced to a sum)
     ──► Qwen3.5-0.8B frozen stack over chunk positions (inputs_embeds; the KV/state machinery unchanged)
     ──► LM head at each chunk position: the next chunk's FIRST byte-token
```

The distribution shift is minimized *by construction*: every patch vector is a linear combination of the
model's own token embeddings. Chunk sequences are ~1.3× shorter than BPE sequences for the same text
(6 bytes/chunk vs 4.57 bytes/token), so the sequence statistics are comparable.

## The measurement (the unmeasured question)

First-byte bpb on the hnet-kaggle val split (the published 1.4177 protocol's data):

| system | what it predicts | bpb |
|---|---|---|
| trained 18.6M H-Net (the reference) | every byte | 1.4177 (published) |
| stock Qwen3.5-0.8B, BPE | every byte (via tokens) | measured in-repo |
| **adapter v0** | each chunk's first byte, at chunk resolution | **the number this repo exists for** |
| stock Qwen3.5-0.8B, same positions | the same first bytes, token resolution | the control |

Papers: MEGABYTE (2305.07185, the frozen-chunking shape), BLT (2412.09871, §2.3 entropy patching — the
v1 chunker), H-Net (2507.07955, the mechanism), the companion report
(`~/Documents/HNet_Qwen_Companion_Research_20261104/`), the weight-space-programming post
(rikiyat.github.io/blog/weight-space-programming/).

## Layout

```
src/       adapter.py (chunker, patch embedder, ChunkLM) · corpus.py (val-split loader, byte-token map)
kernel/    colab_run.py (the one-shot measurement; all compute runs on Colab)
tests/     fast CPU tests (logic only; the model runs live on Colab)
docs/      DESIGN.md · RESULTS.md
configs/   run configurations
```

## Compute policy

Everything runs on Colab (`colab run --gpu A100 kernel/colab_run.py`); the local machine edits code and
pushes. Check `colab usage` before each run.
