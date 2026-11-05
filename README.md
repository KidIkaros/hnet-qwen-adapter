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
| stock Qwen3.5-0.8B, native BPE | every byte (via tokens) | **4.2567** |

Period × aggregation sweep (`results/a100_sweep.log`), byte-marginalized head throughout:

| period | sum | mean | n_preds (sum) | the task |
|---|---|---|---|---|
| **1** | **5.2250** | — | 65,132 | every next byte — the byte-level-Qwen control |
| 2 | 7.7184 | 9.2854 | 32,558 | every 2nd byte |
| 3 | **6.4325** | 8.9436 | 21,701 | every 3rd byte |
| 6 | 9.1890 | 8.9554 | 10,848 | every 6th byte |
| 12 | 8.5120 | 8.9565 | 5,421 | every 12th byte |

Coverage: 221/256 bytes have a single-token row; ~0.6% of chunks contain an unmappable byte (the hole
positions embed as zero vectors, counted in input cov 0.994) and ~0.6% of targets are skipped
(target cov 0.994).

**Reading the numbers honestly:**
- **Granularity cost is small — the surprise**: period 1 degenerates the adapter to "Qwen as a
  byte-level model" (byte-token rows in, marginalized head out) at **5.23 bits vs stock 4.26 — only
  +0.97**. The frozen stack handles byte-granular input far better than the distribution-shift
  prediction suggested.
- **The patch construction is the lossy part**: summing multiple byte rows pushes boundary-byte
  prediction to 6.4–9.2. The shift lives in the patch interface, not the byte granularity.
- **sum > mean at small periods** (up to +2.5 bits @3): the larger norm seems to mark patch vectors
  as non-token — a useful signal the mean destroys.
- **Cross-period comparisons are task-confounded**: each period scores a different byte subset
  (boundary alignment vs UTF-8/word structure differs), so the non-monotonicity (3 < 2, 12 < 6) is
  not pure chunking cost. A same-target-set comparison is future work.
- The readout matters: scoring single-byte *tokens* directly reads ~3 bits worse than uniform — a
  metric artifact. `ByteHead` (P(byte) = mass of all tokens whose first byte is b) is the honest
  readout.

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
