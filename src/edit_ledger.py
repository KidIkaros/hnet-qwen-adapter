"""Append-only ledger of verified behavior changes — the runbook for the
autowsp loop.

An entry is a plain dict. Required keys are validated; anything else is
carried through verbatim (forward-compatible). The ledger is the source
of truth for lineage: what changed, by which mechanism, with what
evidence, and what it cost on neighbors.

    from src.edit_ledger import record, load, REQUIRED
    record(entry)                    # validates + appends JSONL
    entries = load()                 # all entries, oldest first
"""
import json
import os
import time

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
LEDGER = os.path.join(RESULTS, "edits.jsonl")

REQUIRED = (
    "method",        # direction-subtraction | decode-constraint | row-write
    "target",        # e.g. "L21.down_proj[64 cols]" / "greedy_emit"
    "teach_gain",    # measured improvement on the targeted behavior
    "neighbor_cost", # measured regression (loops/closed/etc on holdout)
    "n_eval",        # items in the verification slice
    "receipt",       # path/name of the raw result artifact
)


def record(entry, path=LEDGER):
    """Validate required keys, stamp time, append one JSONL line."""
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"ledger entry missing {missing}")
    stamped = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **entry}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(stamped, sort_keys=True) + "\n")
    return stamped


def load(path=LEDGER):
    """All entries, oldest first; empty list if the ledger doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
