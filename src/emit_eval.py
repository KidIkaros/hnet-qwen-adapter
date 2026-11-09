"""Emit-level domain rules: what counts as a loop, a closed call,
malformed emit, and — new — semantic correctness of the emitted call.
Pure functions over emitted bytes + prompt context; the task-semantics
component (grows with each new failure phenotype).

score_emission is the locked-exam layer: symbolic by design (it is the
examiner, not the mind). Its labels are what a learned regulator
(glial gate) trains against — richer labels, better regulator.

All scanners work on str via latin-1 decode (byte↔char 1:1), so emitted
byte streams with non-UTF8 tails still parse positionally.
"""
import re


RUNAWAY_ONSET = 240
_ARG = re.compile(rb'(\w+)="')
_NAME_CALL = re.compile(r'([A-Za-z_][\w]*)\s*\(')
_TOOL_NAME = re.compile(r'"name"\s*:\s*"([\w]+)"')
_PROPS = re.compile(r'"properties"\s*:\s*\{')
_PARAMS = re.compile(r'"parameters"\s*:\s*\{')
_SCHEMA_WORDS = frozenset({
    "type", "required", "properties", "items", "description", "enum",
    "default", "additionalProperties", "title", "$schema"})
_KV = re.compile(r'(\w+)\s*=\s*')
_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _s(text) -> str:
    return text if isinstance(text, str) else text.decode("latin-1")


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


def _scan_quoted(s, i):
    """Index just past the closing quote of s[i]=='"' (handles \\ escapes)."""
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == '"':
            return j + 1
        j += 1
    return len(s)


def _match_close(s, i):
    """s[i] is an opener → (index past match, matched?). Quote-aware."""
    want = _PAIRS[s[i]]
    depth, j = 1, i + 1
    while j < len(s) and depth:
        c = s[j]
        if c == '"':
            j = _scan_quoted(s, j)
            continue
        if c == s[i]:
            depth += 1
        elif c == want:
            depth -= 1
        j += 1
    return j, depth == 0


def _split_top(body: str):
    """Split on commas at bracket depth 0, quote-aware."""
    parts, depth, start, j = [], 0, 0, 0
    while j < len(body):
        c = body[j]
        if c == '"':
            j = _scan_quoted(body, j)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(body[start:j])
            start = j + 1
        j += 1
    parts.append(body[start:])
    return [p.strip() for p in parts if p.strip()]


def parse_call(text) -> dict | None:
    """First `name(k=v, ...)` call in bytes/str →
    {name, args: [(k, v_str)], closed}. None if no call found.
    Tolerates a truncated tail: closed=False when ')' never arrives."""
    s = _s(text)
    m = _NAME_CALL.search(s)
    if not m:
        return None
    end, matched = _match_close(s, m.end() - 1)
    body = s[m.end():end - 1] if matched else s[m.end():]
    args = []
    for part in _split_top(body):
        kv = _KV.match(part)
        if kv:
            args.append((kv.group(1), part[kv.end():].strip()))
    return {"name": m.group(1), "args": args, "closed": matched}


def tool_schemas(sys_text: str) -> dict:
    """{tool_name: tuple(arg_names_in_schema_order) | None} parsed from
    the prompt's tool JSON. None = tool seen but no verifiable schema.
    Order preserved (schema order) for deterministic arg synthesis."""
    marks = [(m.start(), m.group(1)) for m in _TOOL_NAME.finditer(sys_text)]
    out = {}
    for idx, (pos, name) in enumerate(marks):
        stop = marks[idx + 1][0] if idx + 1 < len(marks) else len(sys_text)
        block = sys_text[pos:stop]
        pm = _PROPS.search(block) or _PARAMS.search(block)
        if not pm:
            out[name] = None
            continue
        obj_end, _ = _match_close(block, pm.end() - 1)
        keys = _top_keys(block[pm.end():obj_end - 1])
        if pm.re is _PARAMS:
            keys = [k for k in keys if k not in _SCHEMA_WORDS]
        out[name] = tuple(keys) if keys else None
    return out


def _top_keys(props: str):
    """Depth-0 `"key":` names inside a JSON object body (props excludes
    the outer braces, so arg names sit at depth 0)."""
    keys, depth, j = [], 0, 0
    while j < len(props):
        c = props[j]
        if c == '"':
            k = _scan_quoted(props, j)
            if depth == 0 and re.match(r'\s*:', props[k:k + 8]):
                keys.append(props[j + 1:k - 1])
            j = k
            continue
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        j += 1
    return keys


def _canon(call: dict):
    """Canonical (name, {k: v}) for equality; v whitespace/quote-normalized."""
    if call is None:
        return None
    return (call["name"],
            {k: re.sub(r"\s+", "", v).strip('"') for k, v in call["args"]})


def score_emission(emitted: bytes, *, tools, sys_text="", gold_name=None,
                   gold_call=None) -> dict:
    """Locked-exam verdict for one emission. Four independent labels:

      parse_ok   — a `name(...)` call was found
      tool_valid — name is declared in this prompt's tool list
      args_valid — arg names ⊆ schema (None when schema unverifiable)
      gold_match — canonical equality with the gold call
                   (falls back to name match when gold unparseable)

    Plus legacy phenotype/closed fields for continuity with loop metrics.
    """
    onset, pheno = dup_class(emitted)
    call = parse_call(emitted)
    v = {"phenotype": pheno, "onset": onset,
         "closed": b"]" in emitted,
         "parse_ok": call is not None,
         "tool": call["name"] if call else None,
         "tool_valid": None, "args_valid": None,
         "gold_match": False}                     # no call ⇒ wrong answer
    if call is None:
        return v
    v["tool_valid"] = call["name"] in set(tools)
    schemas = tool_schemas(sys_text) if sys_text else {}
    if call["name"] in schemas:
        spec = schemas[call["name"]]
        v["args_valid"] = (set(k for k, _ in call["args"]) <= set(spec)
                           if spec else None)
    gold = parse_call(gold_call) if gold_call else None
    if gold is not None:
        v["gold_match"] = _canon(call) == _canon(gold)
    elif gold_name:
        v["gold_match"] = call["name"] == gold_name
    return v
