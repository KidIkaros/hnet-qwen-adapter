"""host_profile — single source of per-host geometry constants.

Protocols transfer across hosts; geometry does not. Every layer tap,
hidden size, probe threshold, and scale that was derived on one trunk
must be re-derived (or explicitly marked host-specific) before use on
another. This module is the registry: code that needs host geometry
asks here instead of hardcoding literals.

Hosts:
  "qwen27b-int4"  — packed GPTQ 4-bit artifact (current trunk)
  "qwen27b-bf16"  — bf16 reference (same geometry, different numerics)
  "qwen2b"        — retired 2B trunk (kept for historical receipts only)

Add a host by copying an existing profile and re-deriving constants
on that host — never copy constants across hosts.
"""

# ---- geometry (verified against each host's config) -------------------
GEOMETRY = {
    "qwen27b-int4": {
        "hidden_size": 5120,        # TODO verify vs INT4_MANIFEST/config
        "n_layers": 64,             # TODO verify
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "notes": "Qwen3.8-27B hybrid linear-attention; verify n_layers "
                 "and which layers are full-attention vs linear",
    },
    "qwen27b-bf16": {
        "hidden_size": 5120,
        "n_layers": 64,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "notes": "same geometry as int4; bf16 numerics",
    },
    "qwen2b": {
        "hidden_size": None,        # fill from retired host config
        "n_layers": None,
        "vocab_size": None,
        "tie_word_embeddings": None,
        "notes": "RETIRED — historical receipts only; do not use for "
                 "new derivations",
    },
}

# ---- derived constants (re-derive per host; None = not yet derived) ---
# Layer taps for state probing / glial field. The 2B taps (L14/L21/L22)
# DO NOT transfer — pick taps by the same *protocol* (e.g. quartiles of
# depth, or attention/linear boundary layers) on the new host.
TAPS = {
    "qwen27b-int4": {"early": None, "mid": None, "late": None},
    "qwen27b-bf16": {"early": None, "mid": None, "late": None},
    "qwen2b": {"early": 14, "mid": 21, "late": 22},   # historical only
}

# Probe/calibration thresholds — per host, fit on that host's corpus.
THRESHOLDS = {
    "qwen27b-int4": {},
    "qwen27b-bf16": {},
    "qwen2b": {},
}


def profile(host="qwen27b-int4"):
    """Return the geometry+derived bundle for a host.

    Raises KeyError on unknown host — silent fallback to another host's
    constants is exactly the bug this module exists to prevent.
    """
    if host not in GEOMETRY:
        raise KeyError(f"unknown host {host!r}; known: {list(GEOMETRY)}")
    return {
        "geometry": GEOMETRY[host],
        "taps": TAPS[host],
        "thresholds": THRESHOLDS[host],
    }
