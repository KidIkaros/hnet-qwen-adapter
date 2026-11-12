"""Interface tests for HNetQwen — the seam every caller crosses.

A synthetic trunk (no checkpoint, no weights) exercises the contract:
one forward returns HNetOutput, HNetConfig bundles the processing modes,
head_first returns byte log-probs instead of logits, boundary_p is
returned rather than stashed on the module, and the stack stays causal.

Run: python -m pytest tests/test_hnet.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.hnet import HNetConfig, HNetOutput, HNetQwen

D, V, T = 32, 300, 48


class _CausalLayer(nn.Module):
    """Minimal stand-in for a decoder layer: causal self-attention that
    honours the additive mask the stack passes, plus a pointwise map."""

    def __init__(self, d, block_type):
        super().__init__()
        self.block_type = block_type
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, hs, position_embeddings=None, attention_mask=None,
                position_ids=None):
        t = hs.shape[1]
        scores = (hs @ hs.transpose(1, 2))[:, None]        # (B, 1, T, T)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool,
                                       device=hs.device), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask
        att = scores.softmax(-1)                           # (B, 1, T, T)
        return hs + (att @ hs[:, None]).squeeze(1) + 0.1 * self.proj(hs)


class _Text(nn.Module):
    def __init__(self, d, vocab, n_lin, n_full):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": d})()
        self.layers = nn.ModuleList(
            [_CausalLayer(d, "linear_attention") for _ in range(n_lin)]
            + [_CausalLayer(d, "full_attention") for _ in range(n_full)])
        self.embed_tokens = nn.Embedding(vocab, d)
        self.norm = nn.LayerNorm(d)
        self.rotary_emb = None


class _Model(nn.Module):
    def __init__(self, d, vocab, n_lin=4, n_full=2):
        super().__init__()
        self.model = _Text(d, vocab, n_lin, n_full)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def _net():
    torch.manual_seed(0)
    return HNetQwen(_Model(D, V), tok=None,
                    byte_map={b: b for b in range(256)},
                    enc_layers=2, dec_layers=2)


def _ids(n=T):
    return torch.randint(0, 256, (n,))


def _fixed_boundaries(every=4):
    b = torch.zeros(T, dtype=torch.float32)
    b[::every] = 1.0
    return b


def _first():
    first = torch.full((V,), -1, dtype=torch.long)
    first[:256] = torch.arange(256)
    return first


def test_output_contract():
    hn = _net()
    out = hn(_ids())
    assert isinstance(out, HNetOutput)
    assert out.logits.shape == (1, T, V) and out.byte_logprobs is None
    assert out.boundaries.shape == (T,)
    assert out.boundary_p.shape == (T,)
    assert out.spans[0][0] == 0 and out.spans[-1][1] == T
    assert len(out.spans) == int(out.boundaries.sum())
    # the old side effect is gone — the probabilities travel with the output
    assert not hasattr(hn, "last_p")


def test_head_first_returns_byte_logprobs():
    hn = _net()
    out = hn(_ids(), head_first=_first())
    assert out.logits is None
    assert out.byte_logprobs.shape == (1, T, 256)
    assert torch.isfinite(out.byte_logprobs).all()
    # marginalization excludes unmapped vocab columns: mass never exceeds 1
    mass = out.byte_logprobs.exp().sum(-1)
    assert (mass <= 1.0 + 1e-4).all() and (mass > 0.5).all()


def test_cfg_modes_are_finite():
    hn = _net()
    ids, b = _ids(), _fixed_boundaries()
    for cfg in (HNetConfig(), HNetConfig(ctx="own"),
                HNetConfig(ctx="retrieve"),
                HNetConfig(ctx="retrieve", super_k=4)):
        out = hn(ids, boundaries=b, cfg=cfg)
        assert torch.isfinite(out.logits).all(), cfg


def test_super_k_survives_few_chunks():
    hn = _net()
    b = _fixed_boundaries(every=32)          # 2 chunks < super_k=4
    out = hn(_ids(), boundaries=b, cfg=HNetConfig(ctx="retrieve", super_k=4))
    assert torch.isfinite(out.logits).all()


def test_unknown_ctx_raises():
    hn = _net()
    try:
        hn(_ids(), cfg=HNetConfig(ctx="bogus"))
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fixed_boundaries_are_respected():
    hn = _net()
    b = _fixed_boundaries()
    out = hn(_ids(), boundaries=b)
    assert torch.equal(out.boundaries, b)
    assert [s for s, _ in out.spans] == list(range(0, T, 4))


def test_causality_with_fixed_boundaries():
    hn = _net()
    ids, b = _ids(), _fixed_boundaries()
    a = hn(ids, boundaries=b).logits
    ids2 = ids.clone()
    ids2[-1] = (ids[-1] + 1) % 256
    c = hn(ids2, boundaries=b).logits
    # the last chunk is [44, 48); everything before it must be untouched
    assert torch.allclose(a[:, :44], c[:, :44], atol=1e-5)


def test_deterministic():
    hn = _net()
    ids = _ids()
    assert torch.allclose(hn(ids).logits, hn(ids).logits)
