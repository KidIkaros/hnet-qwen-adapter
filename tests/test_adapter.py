"""Logic tests for the adapter v0: chunker, byte-token map, patch embedder, bpb accounting — all with
fake tokenizer/model stubs, no downloads (the model runs live on Colab)."""
import math
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adapter import ChunkLM, PatchEmbedder, PeriodicChunker, qwen_byte_bpb   # noqa: E402
from src.corpus import byte_token_map, bytes_to_token_ids                       # noqa: E402


class FakeTok:
    """A tokenizer stub: bytes 65..90 ('A'..'Z') are single-byte tokens with id = b - 64; everything
    else splits into TWO tokens (so it is not a single-byte token -> unmapped)."""

    def encode(self, s, add_special_tokens=False):
        out = []
        for ch in s:
            b = ord(ch) & 0xFF
            if 65 <= b <= 90:
                out.append(b - 64)
            else:
                out.extend([256 + b, 512 + b])
        return out

    def decode(self, ids, **kw):
        inv = {b - 64: chr(b) for b in range(65, 91)}
        return "".join(inv.get(i, "?") for i in ids)


class FakeModel(torch.nn.Module):
    """get_input_embeddings -> a (V, D) embedding; forward(inputs_embeds=...) -> logits = pv @ W."""

    def __init__(self, V=300, D=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(V, D)
        self.W = torch.randn(D, V)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, input_ids=None, **kw):
        pv = inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        return types.SimpleNamespace(logits=pv @ self.W)


def test_periodic_chunker():
    c = PeriodicChunker(6)
    data = bytes(range(20))
    chunks = c.chunks(data)
    assert len(chunks) == 3 and chunks[0] == data[:6] and chunks[2] == data[12:18]   # the partial [18:20] dropped


def test_byte_token_map_and_coverage():
    m = byte_token_map(FakeTok())
    assert m[65] == 1 and m[90] == 26                    # 'A'..'Z' -> 1..26
    assert all(b not in m for b in range(65 - 1))        # non-letters unmapped in the stub
    ids, cov = bytes_to_token_ids(b"ABC", m)
    assert ids == [1, 2, 3] and cov == 1.0
    ids, cov = bytes_to_token_ids(b"AB\x01", m)
    assert ids == [1, 2] and abs(cov - 2 / 3) < 1e-9     # the unmapped control byte skipped, coverage recorded


def test_patch_embedder_is_the_sum_of_embedding_rows():
    model = FakeModel()
    tok = FakeTok()
    pe = PatchEmbedder(model, tok)
    v, cov = pe.embed(b"ABC")
    want = model.emb.weight[torch.tensor([1, 2, 3])].sum(0)
    assert torch.allclose(v, want) and cov == 1.0        # the constructed patch vector: zero trained weights


def test_first_byte_bpb_accounting():
    model = FakeModel()
    tok = FakeTok()
    lm = ChunkLM(model, tok, period=4, device="cpu", dtype=torch.float32)
    data = b"ABCDEFGH"                                   # chunks ABCD, EFGH -> M=2, one prediction
    bpb, n, cov = lm.first_byte_bpb(data)
    assert n == 1 and cov == 1.0
    # manual: chunk vectors -> logits -> CE against byte 'E' (id 5)
    pv = (model.emb.weight[torch.tensor([1, 2, 3, 4])].sum(0)).unsqueeze(0)
    logits = (pv @ model.W)[0]
    want = float(torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([model.emb.weight.shape[0] and 5]), reduction="sum") / math.log(2))
    assert abs(bpb - want) < 1e-6


def test_qwen_byte_bpb_accounting():
    """The same-weights reference: token CE weighted by each token's byte length, over total bytes."""
    model = FakeModel()
    tok = FakeTok()
    data = b"ABCD"                                       # 4 single-byte tokens, each 1 byte
    bpb, ntok, nbytes = qwen_byte_bpb(model, tok, data, device="cpu")
    ids = torch.tensor([tok.encode(c)[0] for c in "ABCD"])
    pv = model.emb.weight[ids].unsqueeze(0)
    logits = (pv @ model.W)[0]
    ce = torch.nn.functional.cross_entropy(logits[:-1], ids[1:], reduction="none")
    want = float(ce.sum() / math.log(2) / 3.0)           # 3 predictions x 1 byte each
    assert abs(bpb - want) < 1e-6 and ntok == 4 and nbytes == 3.0
