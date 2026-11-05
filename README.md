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

## The measurement (the unmeasured question — now measured)

First-byte bpb of composed v0 on 8 hnet-kaggle markdown docs (65536 bytes, OOD for the H-Net but the
honest stress case), period-6 chunks, Qwen3.5-0.8B on Colab A100 (`results/a100_v0_period6.log`):

| system | what it predicts | bpb |
|---|---|---|
| trained 18.6M H-Net (the reference) | every byte | 1.4177 (published, its val split) |
| stock Qwen3.5-0.8B, native BPE | every byte (via tokens) | **4.2560** |
| **adapter v0, period 6** | each chunk's first byte, byte-marginalized head | **9.1890** (10,848 preds) |

Coverage: 221/256 bytes have a single-token row (input cov 0.994); 0.6% of targets are unmappable
bytes and are skipped (target cov 0.994).

**Reading the number honestly:**
- 9.19 bits is ~1.2 bits *better than uniform* (8 bits): the frozen stack extracts weak-but-real
  signal from summed patch embeddings.
- The tasks differ by construction — the adapter predicts one byte per six (chunk onsets; it has no
  mechanism for interior bytes); Qwen's 4.26 covers every byte. The fair summary: the zero-training
  patch interface is **dramatically lossy** — the distribution-shift wall (report Finding 4)
  measured.
- The readout matters: scoring the head on single-byte *tokens* directly reads 12.58 (worse than
  uniform — a metric artifact); the byte-marginalized head (`ByteHead`, P(byte) = mass of all tokens
  whose first byte is b) is the honest readout.

**Untested cheap knobs (v0.x):** mean vs sum patch aggregation (the patch norm is ~√6× a token row —
pre-norm helps but the direction is off-manifold), patch-vector length correction, period sweep.

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
