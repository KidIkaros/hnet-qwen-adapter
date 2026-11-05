"""The no-training H-Net Qwen adapter v0.

Architecture (README.md; papers: MEGABYTE 2305.07185 for the shape, BLT 2412.09871 for the v1 chunker,
H-Net 2507.07955 for the mechanism this freezes):

    bytes -> PeriodicChunker (every `period` bytes at phase 0)
          -> PatchEmbedder: each chunk = the SUM of its bytes' own token-embedding rows
             (constructed, zero trained weights; every patch vector lives in the model's own
             embedding space, which minimizes the distribution shift by construction)
          -> the model's frozen stack over chunk positions (inputs_embeds)
          -> the model's own LM head: position m's distribution IS the next chunk's first byte

The falsification test on the trained 18.6M H-Net (qwentensor fork, scripts/probe_frozen_chunker.py)
showed the learned router's boundary decisions are statistically indistinguishable from a periodic byte
phase (+0.0009/+0.0011 bpb on the published val split) — so v0's chunker loses nothing the router had.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .corpus import byte_token_map


def _bytes_to_unicode():
    """GPT-2's byte-unicode table (byte value -> the printable char the tokenizer's vocab strings use)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ByteHead:
    """The LM head marginalized to BYTES: P(byte b | prefix) = the summed mass of every token whose
    FIRST byte is b.  A BPE head's mass sits on multi-byte tokens (its native task), so the honest
    byte-level readout of the adapter is this marginalization — scoring single-byte tokens directly
    reads ~3 bits worse than uniform, a metric artifact, not a system property."""

    def __init__(self, tok):
        strings = tok.convert_ids_to_tokens(range(len(tok)))
        u2b = {c: b for b, c in _bytes_to_unicode().items()}
        first = np.full(len(strings), -1, dtype=np.int64)
        for i, s in enumerate(strings):
            if s and s[0] in u2b:
                first[i] = u2b[s[0]]
        self.first = torch.tensor(first)
        self._src = self._dst = None                       # lazily sized to the logits' width
        self._width = None

    def byte_probs(self, logits):
        """(M, V_logits) logits -> (M, 256) byte probabilities.  V_logits can exceed len(tok)
        (transformers pads the vocab to a multiple of 256): the pad columns have no token and are
        excluded from every byte's mass."""
        p = logits.softmax(-1)
        out = torch.zeros(logits.shape[0], 256, device=logits.device, dtype=logits.dtype)
        if self._width != logits.shape[1]:
            first = self.first
            if first.numel() < logits.shape[1]:
                first = torch.cat([first, torch.full((logits.shape[1] - first.numel(),), -1)])
            tok_ids = (first >= 0).nonzero().squeeze(1)      # the columns carrying byte mass
            self._dst = first[tok_ids]                       # each column's byte bucket
            self._src = tok_ids
            self._width = logits.shape[1]
        src = self._src.to(logits.device)
        out.index_add_(1, self._dst.to(logits.device), p[:, src])
        return out


class PeriodicChunker:
    """Boundaries every `period` bytes at phase 0: chunks are [0:P), [P:2P), ...  The last partial chunk
    is dropped (the model predicts the next chunk's first byte; a partial chunk has none)."""

    def __init__(self, period):
        assert period >= 1
        self.period = period

    def chunks(self, data):
        P = self.period
        n = (len(data) // P) * P
        return [data[i:i + P] for i in range(0, n, P)]


class PatchEmbedder:
    """chunk bytes -> one D-dim vector: the sum of the bytes' own token-embedding rows.

    Constructed, zero trained weights.  Bytes without a single-byte token are SKIPPED (and the
    coverage recorded) — for text bytes the coverage is 1.0 (221/256 bytes are mapped; the missing
    ones are unprintable controls).
    """

    def __init__(self, model, tok, agg="sum"):
        assert agg in ("sum", "mean")
        self.bmap = byte_token_map(tok)
        self.emb = model.get_input_embeddings().weight          # (V_vocab, D), frozen
        self.D = self.emb.shape[1]
        self.agg = agg

    def embed(self, chunk):
        ids = [self.bmap[b] for b in chunk if b in self.bmap]
        cov = len(ids) / max(len(chunk), 1)
        if not ids:
            # A chunk where EVERY byte is unmappable (a single UTF-8 continuation byte at period 1,
            # or a 2-byte UTF-8 char aligned to a period-2 chunk) becomes a zero vector — a "hole"
            # the model sees at that position.  The coverage reports it; the doc is not dropped.
            return torch.zeros(self.D), 0.0
        with torch.no_grad():
            v = self.emb[ids].float().sum(0)                    # (D,) fp32 sum: ONE rounding, not five
            if self.agg == "mean":
                v = v / len(ids)
        return v, cov

    def embed_all(self, chunks):
        vs, covs = zip(*(self.embed(c) for c in chunks))
        return torch.stack(vs), float(np.mean(covs))


class ChunkLM:
    """The composed system: bytes -> chunks -> patch vectors -> the frozen stack -> the LM head.

    first_byte_logits(data): (M, V_vocab) — position m's distribution over the next chunk's FIRST byte
    (the head's next-token distribution restricted to nothing: the byte-token map makes every mapped
    byte a token, so the head IS a byte head for mapped bytes).
    """

    def __init__(self, model, tok, period=6, agg="sum", device="cuda", dtype=torch.bfloat16,
                 byte_head=None):
        self.model = model.eval().to(device, dtype)
        self.tok = tok
        self.chunker = PeriodicChunker(period)
        self.embedder = PatchEmbedder(self.model, tok, agg=agg)
        self.byte_head = byte_head if byte_head is not None else ByteHead(tok)
        self.period, self.device = period, device
        self.dtype = next(model.parameters()).dtype              # the model's native dtype (bf16)

    def first_byte_logits(self, data):
        chunks = self.chunker.chunks(data)
        if len(chunks) < 2:
            raise ValueError(f"a sequence of {len(data)} bytes gives {len(chunks)} chunks at period {self.period}")
        pv, cov = self.embedder.embed_all(chunks)               # (M, D) fp32
        with torch.no_grad():
            out = self.model(inputs_embeds=pv.unsqueeze(0).to(self.device, self.dtype))
        return out.logits[0].float(), cov                       # (M, V_vocab)

    def first_byte_bpb(self, data):
        """Bits per byte of the FIRST-byte predictions: position m predicts byte (m+1)*P, scored on
        the BYTE-MARGINALIZED head (ByteHead).  Targets without a single-byte token (unprintables,
        UTF-8 continuation bytes at a chunk boundary — the byte-phase finding in action) are SKIPPED
        and counted in the target coverage.  Returns (bpb, n_predictions, (input_coverage,
        target_coverage))."""
        logits, cov = self.first_byte_logits(data)
        chunks = self.chunker.chunks(data)
        ms, bs = [], []
        for m in range(len(chunks) - 1):
            b = int(data[(m + 1) * self.period])
            if b in self.embedder.bmap:
                ms.append(m); bs.append(b)
        if not ms:
            raise ValueError("no mapped target bytes in this sequence")
        pb = self.byte_head.byte_probs(logits[ms])               # (M', 256)
        tb = torch.tensor(bs, device=pb.device)
        nll = -torch.log2(pb.gather(1, tb[:, None]).clamp_min(1e-12))
        tcov = len(ms) / max(len(chunks) - 1, 1)
        return float(nll.sum() / len(ms)), len(ms), (cov, tcov)


def qwen_byte_bpb(model, tok, data, device="cuda"):
    """Stock Qwen's next-byte bpb at its native BPE granularity on the same bytes (the same-weights
    reference): token CE summed x token byte-lengths / total bytes / ln 2."""
    ids = tok.encode(data.decode("latin-1"), add_special_tokens=False)
    x = torch.tensor(ids, device=device)[None]
    with torch.no_grad():
        logits = model(input_ids=x).logits[0].float()       # (L, V)
    ce = F.cross_entropy(logits[:-1], x[0, 1:], reduction="none")
    lens = np.array([len(tok.encode(tok.decode([i]), add_special_tokens=False)) for i in ids[1:]])
    total_bytes = float(lens.sum())
    return float((ce.cpu().numpy() * lens).sum() / math.log(2) / total_bytes), len(ids), total_bytes
