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

    def byte_probs(self, logits, block=2048):
        """(M, V_logits) logits -> (M, 256) byte probabilities, computed in row blocks so the
        (M, V) softmax temporaries stay bounded on long docs.  V_logits can exceed len(tok)
        (transformers pads the vocab to a multiple of 256): the pad columns have no token and are
        excluded from every byte's mass."""
        if self._width != logits.shape[1]:
            first = self.first
            if first.numel() < logits.shape[1]:
                first = torch.cat([first, torch.full((logits.shape[1] - first.numel(),), -1)])
            tok_ids = (first >= 0).nonzero().squeeze(1)      # the columns carrying byte mass
            self._dst = first[tok_ids]                       # each column's byte bucket
            self._src = tok_ids
            self._width = logits.shape[1]
        src = self._src.to(logits.device)
        dst = self._dst.to(logits.device)
        out = torch.zeros(logits.shape[0], 256, device=logits.device, dtype=torch.float32)
        for i in range(0, logits.shape[0], block):
            p = logits[i:i + block].float().softmax(-1)
            out[i:i + block].index_add_(1, dst, p[:, src])
        return out.float()


class PeriodicChunker:
    """Boundaries every `period` bytes at phase 0: chunks are [0:P), [P:2P), ...  The last partial chunk
    is dropped (the model predicts the next chunk's first byte; a partial chunk has none)."""

    def __init__(self, period):
        assert period >= 1
        self.period = period

    def spans(self, data, aux=None):
        P = self.period
        n = (len(data) // P) * P
        return [(i, data[i:i + P]) for i in range(0, n, P)]

    def chunks(self, data):
        return [b for _, b in self.spans(data)]


def _calibrated_spans(data, scores, target_mean, min_len, max_len, high_scores_cut):
    """Shared boundary search for the adaptive chunkers: rank-select the ~n/target_mean most
    boundary-worthy positions (deterministic, tie-robust), then a greedy walk honoring min/max
    length.  `scores[t]` is the boundary-worthiness of cutting BEFORE byte t (-inf = never)."""
    n = len(data)
    n_cuts = max(int(round(n / target_mean)) - 1, 0)
    cand = [t for t in range(max(min_len, 1), n) if np.isfinite(scores[t])]
    key = (lambda t: -scores[t]) if high_scores_cut else (lambda t: scores[t])
    cut_at = set(sorted(cand, key=key)[:n_cuts])
    out, start = [], 0
    for t in range(1, n):
        L = t - start
        if L >= max_len or (L >= min_len and t in cut_at):
            out.append((start, data[start:t])); start = t
    if start < n and n - start >= min_len:
        out.append((start, data[start:n]))
    return out


class EntropyChunker:
    """BLT's entropy patching (2412.09871 §2.3), training-free: cut before byte t when the next-byte
    prediction entropy is high — unpredictable bytes open new chunks.  `aux` to spans() is the
    per-position boundary score (scores[t] = entropy of predicting byte t); the number of cuts is
    calibrated per document to hit ~`target_mean` — the same compression ratio as the periodic
    baseline, the fair comparison.  The entropy source can be anything; the driver uses the period-1
    ChunkLM itself (byte-level Qwen as its own boundary estimator)."""

    def __init__(self, target_mean=6.0, min_len=1, max_len=16):
        self.target_mean, self.min_len, self.max_len = target_mean, min_len, max_len

    def spans(self, data, aux=None):
        scores = aux
        if scores is None or len(scores) != len(data):
            raise ValueError("EntropyChunker needs a per-byte boundary-score array via aux")
        return _calibrated_spans(data, np.asarray(scores, dtype=np.float64),
                                 self.target_mean, self.min_len, self.max_len, True)


class SimChunker:
    """Streaming content-similarity chunker — the training-free analog of LSA's Ward clustering:
    cut before byte t when consecutive byte-token embeddings are dissimilar (a semantic transition),
    the number of cuts calibrated to `target_mean` for the same-compression comparison.  Positions
    touching an unmappable byte are treated as unknown content -> eligible cuts."""

    def __init__(self, embedder, target_mean=6.0, min_len=1, max_len=16):
        self.embedder = embedder
        self.target_mean, self.min_len, self.max_len = target_mean, min_len, max_len

    def spans(self, data, aux=None):
        emb = self.embedder.emb.detach().float()
        n = len(data)
        scores = np.full(n, 0.0, dtype=np.float64)           # cos(e_{t-1}, e_t); low -> cut
        prev = None
        for t in range(n):
            cur = self.embedder.bmap.get(data[t])
            if prev is not None and cur is not None and t > 0:
                a, b = emb[prev], emb[cur]
                scores[t] = float((a @ b) / (a.norm() * b.norm() + 1e-12))
            prev = cur
        scores[0] = np.inf
        return _calibrated_spans(data, scores, self.target_mean, self.min_len,
                                 self.max_len, False)


class PatchEmbedder:
    """chunk bytes -> one D-dim vector: the sum of the bytes' own token-embedding rows.

    Constructed, zero trained weights.  Bytes without a single-byte token are SKIPPED (and the
    coverage recorded) — for text bytes the coverage is 1.0 (221/256 bytes are mapped; the missing
    ones are unprintable controls).
    """

    def __init__(self, model, tok, agg="sum", decay=0.5):
        assert agg in ("sum", "mean", "last")
        self.bmap = byte_token_map(tok)
        self.emb = model.get_input_embeddings().weight          # (V_vocab, D), frozen
        self.D = self.emb.shape[1]
        self.agg = agg
        self.decay = decay

    def embed(self, chunk):
        ids = [self.bmap[b] for b in chunk if b in self.bmap]
        cov = len(ids) / max(len(chunk), 1)
        if not ids:
            # A chunk where EVERY byte is unmappable (a single UTF-8 continuation byte at period 1,
            # or a 2-byte UTF-8 char aligned to a period-2 chunk) becomes a zero vector — a "hole"
            # the model sees at that position.  The coverage reports it; the doc is not dropped.
            return torch.zeros(self.D, dtype=torch.float32, device=self.emb.device), 0.0
        with torch.no_grad():
            rows = self.emb[ids].float()
            if self.agg == "last":
                # Boundary-weighted (ZHA's query-answerability insight: preserve what the prediction
                # needs — the bytes nearest the boundary): position i gets decay^(L-1-i).
                w = torch.tensor([self.decay ** (len(chunk) - 1 - i)
                                  for i, b in enumerate(chunk) if b in self.bmap],
                                 dtype=rows.dtype, device=rows.device)
                v = (rows * w[:, None]).sum(0)
            else:
                v = rows.sum(0)                               # (D,) fp32 sum: ONE rounding, not five
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
                 byte_head=None, chunker=None):
        self.model = model.eval().to(device, dtype)
        self.tok = tok
        self.chunker = chunker if chunker is not None else PeriodicChunker(period)
        self.embedder = PatchEmbedder(self.model, tok, agg=agg)
        self.byte_head = byte_head if byte_head is not None else ByteHead(tok)
        self.period, self.device = period, device
        self.dtype = next(model.parameters()).dtype              # the model's native dtype (bf16)

    def _logits(self, spans, keep=None):
        pv, cov = self.embedder.embed_all([b for _, b in spans])  # (M, D) fp32
        kw = {}
        if keep is not None:                                    # only the scored positions' logits —
            kw["logits_to_keep"] = torch.as_tensor(keep, device=self.device)   # bounds the (M,V) peak
        with torch.no_grad():
            out = self.model(inputs_embeds=pv.unsqueeze(0).to(self.device, self.dtype), **kw)
        return out.logits[0], cov                               # (K, V) model dtype — ByteHead blocks cast

    def first_byte_logits(self, data, aux=None):
        spans = self.chunker.spans(data, aux=aux)
        if len(spans) < 2:
            raise ValueError(f"a sequence of {len(data)} bytes gives {len(spans)} chunks")
        logits, cov = self._logits(spans)
        return logits, cov

    def byte_entropy(self, data):
        """(T,) next-byte entropies from a period-1 run — the boundary scores an EntropyChunker
        consumes.  scores[t] = entropy of predicting byte t."""
        logits, _ = self.first_byte_logits(data)
        pb = self.byte_head.byte_probs(logits)                   # (T, 256): pos m predicts byte m+1
        ent = -(pb.clamp_min(1e-12).log2() * pb).sum(1).cpu().numpy()
        scores = np.full(len(data), -np.inf)
        scores[1:] = ent[:len(data) - 1]
        return scores

    def byte_nll(self, data):
        """(T,) per-position byte NLL in bits from a period-1 run: position m's output predicts byte
        m+1.  Position 0 gets the uniform 8 bits — no model conditions on an empty prefix.  This is
        the fallback scorer for every position a chunked system does not predict."""
        logits, _ = self.first_byte_logits(data)
        pb = self.byte_head.byte_probs(logits)                   # (T, 256)
        nll = np.full(len(data), 8.0)
        tgt = torch.tensor(list(data[1:]), device=pb.device)
        nll[1:] = -torch.log2(pb[:len(data) - 1].gather(1, tgt[:, None])
                              .clamp_min(1e-12)).double().cpu().numpy().ravel()
        return nll

    def boundary_nll(self, data, aux=None):
        """(positions, nll_bits, input_cov, n_spans): the chunk-start positions this chunked system
        predicts, the byte-marginalized NLL of each, the patch-input coverage, and the chunk count.
        Positions whose byte is unmappable are absent (the fallback scorer covers them)."""
        spans = self.chunker.spans(data, aux=aux)
        if len(spans) < 2:
            raise ValueError(f"a sequence of {len(data)} bytes gives {len(spans)} chunks")
        ms, bs, pos = [], [], []                                # scored positions are known pre-forward
        for m in range(len(spans) - 1):
            t = spans[m + 1][0]; b = int(data[t])
            if b in self.embedder.bmap:
                ms.append(m); bs.append(b); pos.append(t)
        if not ms:
            raise ValueError("no mapped target bytes in this sequence")
        logits, cov = self._logits(spans, keep=ms)              # (len(ms), V) — bounded
        pb = self.byte_head.byte_probs(logits)
        tb = torch.tensor(bs, device=pb.device)
        nll = -torch.log2(pb.gather(1, tb[:, None]).clamp_min(1e-12)).double().cpu().numpy().ravel()
        return np.array(pos), nll, cov, len(spans)

    def first_byte_bpb(self, data, aux=None):
        """Bits per byte of the FIRST-byte predictions: position m predicts the next chunk's first
        byte, scored on the BYTE-MARGINALIZED head (ByteHead).  Targets without a single-byte token
        (unprintables, UTF-8 continuation bytes at a chunk boundary — the byte-phase finding in
        action) are SKIPPED and counted in the target coverage.  `aux` is passed to the chunker
        (EntropyChunker's boundary scores).  Returns (bpb, n_predictions, (input_coverage,
        target_coverage))."""
        spans = self.chunker.spans(data, aux=aux)
        if len(spans) < 2:
            raise ValueError(f"a sequence of {len(data)} bytes gives {len(spans)} chunks")
        ms, bs = [], []
        for m in range(len(spans) - 1):
            b = int(data[spans[m + 1][0]])
            if b in self.embedder.bmap:
                ms.append(m); bs.append(b)
        if not ms:
            raise ValueError("no mapped target bytes in this sequence")
        logits, cov = self._logits(spans, keep=ms)               # (len(ms), V) — bounded
        pb = self.byte_head.byte_probs(logits)                   # (M', 256)
        tb = torch.tensor(bs, device=pb.device)
        nll = -torch.log2(pb.gather(1, tb[:, None]).clamp_min(1e-12))
        tcov = len(ms) / max(len(spans) - 1, 1)
        return float(nll.sum() / len(ms)), len(ms), (cov, tcov)


def effective_bpb(data, pos, chunk_nll, byte_nll_arr):
    """THE comparable metric: total bits / total bytes for a chunked system.  Chunk-start positions
    are scored by the chunk model; every other position (including unmapped-byte targets and the
    periodic tail the chunker drops) falls back to the byte-level NLL.  Position 0 pays the uniform
    8 bits for every system — so period-1, periodic-6, sim~6 and BPE all sit on the same axis."""
    total = float(byte_nll_arr.sum())
    total -= float(byte_nll_arr[pos].sum())
    total += float(chunk_nll.sum())
    return total / len(data)


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
