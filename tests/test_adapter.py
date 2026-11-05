"""Logic tests for the adapter v0: chunker, byte-token map, patch embedder, bpb accounting — all with
fake tokenizer/model stubs, no downloads (the model runs live on Colab)."""
import math
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adapter import (ChunkLM, EntropyChunker, PatchEmbedder, PeriodicChunker,  # noqa: E402
                         SimChunker, qwen_byte_bpb)
from src.corpus import byte_token_map, bytes_to_token_ids                       # noqa: E402


class FakeTok:
    """A tokenizer stub: bytes 65..90 ('A'..'Z') are single-byte tokens with id = b - 64; everything
    else splits into TWO tokens (so it is not a single-byte token -> unmapped).  convert_ids_to_tokens
    speaks the GPT-2 byte-unicode dialect: printable ASCII chars map to themselves."""

    VOCAB = 768

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

    def convert_ids_to_tokens(self, ids):
        out = []
        for i in ids:
            if 1 <= i <= 26:
                out.append(chr(64 + i))          # 'A'..'Z': first byte 65..90
            elif 256 <= i < 512:
                out.append(chr((i - 256) if 32 <= i - 256 < 127 else 0x100 + (i - 256)))
            else:
                out.append("")                   # no first byte -> excluded from the byte head
        return out

    def __len__(self):
        return self.VOCAB


class FakeModel(torch.nn.Module):
    """get_input_embeddings -> a (V, D) embedding; forward(inputs_embeds=...) -> logits = pv @ W."""

    def __init__(self, V=768, D=8, seed=0):
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
    bpb, n, (cov, tcov) = lm.first_byte_bpb(data)
    assert n == 1 and cov == 1.0 and tcov == 1.0
    # manual: chunk vector -> logits -> softmax -> the BYTE-MARGINALIZED prob of 'E' (byte 69):
    # tokens with first byte 69 are id 5 ('E') and id 256+69=325 (the stub's junk token for 'E')
    pv = (model.emb.weight[torch.tensor([1, 2, 3, 4])].sum(0)).unsqueeze(0)
    logits = (pv @ model.W)[0]
    p = logits.softmax(-1)
    want = float(-torch.log2(p[5] + p[325]))
    assert abs(bpb - want) < 1e-4          # fp32 summation-order noise; the accounting is what's tested


def test_byte_head_padded_vocab():
    """transformers pads the model vocab to a multiple of 256 (248320 for Qwen's 248077 tokenizer):
    logits wider than len(tok) must exclude the pad columns rather than crash."""
    model = FakeModel(V=800)                              # 800 > FakeTok.VOCAB=768
    lm = ChunkLM(model, FakeTok(), period=4, device="cpu", dtype=torch.float32)
    bpb, n, (cov, tcov) = lm.first_byte_bpb(b"ABCDEFGH")
    assert n == 1 and cov == 1.0 and tcov == 1.0
    logits = (model.emb.weight[torch.tensor([1, 2, 3, 4])].sum(0).unsqueeze(0) @ model.W)[0]
    p = logits.softmax(-1)
    assert abs(bpb - float(-torch.log2(p[5] + p[325]))) < 1e-4


def test_unmapped_chunk_becomes_zero_vector():
    """A chunk whose every byte is unmappable (a lone UTF-8 continuation byte at period 1) must not
    kill the doc: it embeds as a zero 'hole' and shows up in the input coverage."""
    lm = ChunkLM(FakeModel(), FakeTok(), period=1, device="cpu", dtype=torch.float32)
    bpb, n, (cov, tcov) = lm.first_byte_bpb(b"AB\x80CD")   # 0x80 is unmapped: 5 chunks, one hole
    assert cov == 0.8                                     # one of five chunks empty
    assert n == 3                                         # 4 prediction slots; the 0x80 target skipped


def test_last_weighted_aggregation():
    emb = PatchEmbedder(FakeModel(), FakeTok(), agg="last", decay=0.5)
    v, cov = emb.embed(b"ABC")                            # ids 1,2,3 with weights .25/.5/1
    want = 0.25 * emb.emb[1].float() + 0.5 * emb.emb[2].float() + 1.0 * emb.emb[3].float()
    assert cov == 1.0 and torch.allclose(v, want)


def test_entropy_chunker_boundaries():
    ch = EntropyChunker(target_mean=3.0, min_len=1, max_len=16)
    data = b"ABCDEFGH"
    scores = np.array([-np.inf, 1, 9, 1, 1, 9, 1, 1])     # high entropy before bytes 2 and 5
    spans = ch.spans(data, aux=scores)
    assert [s for s, _ in spans] == [0, 2, 5]
    assert bytes(spans[1][1]) == b"CDE"


def test_sim_chunker_boundaries():
    class StubEmbedder:
        bmap = {ord(c): i for i, c in enumerate("AAaaBBbb")}   # A=0,1 a=2,3 B=4,5 b=6,7
        emb = torch.tensor([[1., 0.]] * 4 + [[0., 1.]] * 4)    # A/a cluster on x, B/b on y
    ch = SimChunker(StubEmbedder(), target_mean=4.0, min_len=1, max_len=16)
    spans = ch.spans(b"AAaaBBbb")                          # sim 1 within groups, 0 across
    starts = [s for s, _ in spans]
    assert 4 in starts and len(spans) <= 4                  # cut at the A->B transition


def test_variable_length_targets():
    """With a non-periodic chunker the target is the next chunk's first byte wherever it starts."""
    ch = EntropyChunker(target_mean=3.0, min_len=1, max_len=16)
    lm = ChunkLM(FakeModel(), FakeTok(), agg="sum", device="cpu", dtype=torch.float32, chunker=ch)
    scores = np.array([-np.inf, 1, 9, 1, 1, 9, 1, 1])
    bpb, n, (cov, tcov) = lm.first_byte_bpb(b"ABCDEFGH", aux=scores)
    assert n == 2 and tcov == 1.0                           # targets: 'C' (start 2) and 'F' (start 5)


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
