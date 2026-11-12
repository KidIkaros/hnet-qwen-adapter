"""Round-trip tests for pack.py — the rANS seam.

The encoder and decoder are two functions sharing one table contract;
the only honest test is encode -> decode -> identical codes, plus the
compression claim on a skewed distribution.

Run: python -m pytest tests/test_pack.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.pack import bits_per_weight, rans_decode, rans_encode, vocab_trim


def test_rans_roundtrip_uniform():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=5000)
    data, table = rans_encode(codes, n_sym=16)
    assert np.array_equal(rans_decode(data, table), codes)


def test_rans_roundtrip_skewed():
    codes = np.array([0] * 900 + [1] * 50 + [2] * 50, dtype=np.int64)
    data, table = rans_encode(codes, n_sym=3)
    assert np.array_equal(rans_decode(data, table), codes)


def test_rans_compresses_below_fixed_width():
    codes = np.array([0] * 1000 + [1] * 24, dtype=np.int64)   # ~0.2 bits/sym
    data, _ = rans_encode(codes, n_sym=2)
    assert len(data) * 8 < codes.size * 4                     # beats 4-bit fixed


def test_vocab_trim_roundtrip():
    word = np.arange(30).reshape(10, 3)
    trimmed, id_map, ids = vocab_trim(word, {1, 4, 7})
    assert np.array_equal(trimmed, word[[1, 4, 7]])
    assert np.array_equal(ids, np.array([1, 4, 7]))
    assert id_map[1] == 0 and id_map[4] == 1 and id_map[7] == 2
    assert id_map[0] == -1


def test_bits_per_weight_matches_entropy():
    codes = np.array([0] * 8 + [1] * 8)
    assert abs(bits_per_weight(codes, n_sym=2) - 1.0) < 1e-9
