"""Emit-level domain rules: what counts as a loop, a closed call,
a malformed emit. Pure functions over emitted bytes — the task
semantics component (grows with each new failure phenotype)."""
import re


RUNAWAY_ONSET = 240
_ARG = re.compile(rb'(\w+)="')


def dup_class(text: bytes):
    """(onset, pheno): 'argdup' | 'runaway' | (None, None) if clean."""
    seen, pos = set(), None
    for m in _ARG.finditer(text):
        if m.group(1) in seen:
            pos = m.start()
            break
        seen.add(m.group(1))
    m2 = re.search(rb"\]\s*\[", text)
    if m2 and (pos is None or m2.start() < pos):
        pos = m2.start()
    if pos is not None:
        return pos, "argdup"
    if len(text) > RUNAWAY_ONSET and b"]" not in text:
        return RUNAWAY_ONSET, "runaway"
    return None, None


def greedy_emit(hn, trunk, ids, max_new):
    """Greedy byte decode from prompt ids; returns emitted bytes."""
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return bytes(ids)
