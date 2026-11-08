"""Packing stage of the 10% pipeline: entropy-coded indices + vocab trim.

GPTQ gives quantized weights (integer codes + per-row scale/zero). The
packing stage makes the artifact small:
  1. vocab trim   — drop token rows the calibration set never uses
  2. rANS         — entropy-code the integer codes near their empirical
                    entropy instead of fixed-width bits

Size accounting, not model quality: unpacking restores exactly the codes
GPTQ produced.
"""
import numpy as np

RANS_L = 1 << 23
SCALE_BITS = 8
SCALE = 1 << SCALE_BITS


def vocab_trim(word, used_ids):
    """Keep only used token rows; return (trimmed, id_map, inverse)."""
    ids = np.asarray(sorted(used_ids), dtype=np.int64)
    id_map = -np.ones(word.shape[0], dtype=np.int64)
    id_map[ids] = np.arange(len(ids))
    return word[ids], id_map, ids


def _tables(codes, n_sym):
    counts = np.bincount(codes.astype(np.int64).ravel(), minlength=n_sym)
    counts = np.maximum(counts, 1).astype(np.float64)
    freq = np.floor(counts / counts.sum() * SCALE).astype(np.int64)
    freq = np.maximum(freq, 1)
    freq[np.argmax(freq)] += SCALE - int(freq.sum())   # exact table sum
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]])
    return freq, cum


def rans_encode(codes, n_sym):
    """Static-model byte rANS. Returns (bytes, table dict)."""
    codes = codes.astype(np.int64).ravel()
    freq, cum = _tables(codes, n_sym)
    x, out = RANS_L, bytearray()
    for c in codes[::-1]:
        f, s = int(freq[c]), int(cum[c])
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << 8) + (x % f) + s
    return (x.to_bytes(4, "big") + bytes(out[::-1]),
            {"freq": freq.tolist(), "cum": cum.tolist(),
             "n": int(codes.size), "n_sym": n_sym})


def rans_decode(data, table):
    """Inverse of rans_encode; returns the original code array."""
    freq = np.asarray(table["freq"], dtype=np.int64)
    cum = np.asarray(table["cum"], dtype=np.int64)
    slot = np.repeat(np.arange(len(freq)), freq)
    n, x, pos = int(table["n"]), 0, 0
    for _ in range(4):
        x = (x << 8) | data[pos]
        pos += 1
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = x & 0xFF
        sym = int(slot[s])
        out[i] = sym
        x = int(freq[sym]) * (x >> 8) + s - int(cum[sym])
        while x < RANS_L:
            x = (x << 8) | data[pos]
            pos += 1
    return out


def bits_per_weight(codes, n_sym):
    """Empirical entropy of the codes in bits/weight."""
    counts = np.bincount(codes.astype(np.int64).ravel(), minlength=n_sym)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())
