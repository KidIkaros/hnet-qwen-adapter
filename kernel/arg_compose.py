"""Schema-aware candidate composition for constrained call synthesis.

The teacher ceiling run (results/call_synth_teacher_1794193104.json) fixed
the failure taxonomy: with name selection solved (27/27), gold_match stalls
at 6/27 because the *value* closed set is too narrow. Three classes:

  structured-JSON args  ~7   extraction cannot produce arrays/objects
  arg-name subset       ~11  model picks plausible but wrong schema args
  value span/format     ~12  right arg, wrong or unnormalized literal

This module widens the candidate set without widening the search: the
model still picks by score, but the set it picks from now contains what
the gold actually is. All functions are deterministic and model-free, so
recall is measurable on CPU before any GPU run.

  arg_specs(sys_text)          {tool: {arg: spec}} — type/enum/desc/
                               required/items/properties from tool JSON
  arg_prior(spec, user)        deterministic arg-name bonuses (required,
                               lexically grounded, type-matched)
  value_candidates(...)        literal candidates for one arg: quoted
                               literals, enum members, spans, number/date
                               forms, entities, composed JSON
  candidate_recall(...)        the metric: fraction of gold literals that
                               appear in the candidate set
"""
import json
import re

from src.emit_eval import _match_close, _s

_WS = re.compile(r"\s+")
_EDGE = " \t\n\"'`,.;:!?()[]{}"
_WORD = re.compile(r"[A-Za-z0-9_]+")
_QUOTED = re.compile(r'"([^"\n]{1,160})"|\'([^\'\n]{1,160})\'')
_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_BARE_OK = re.compile(r"^[\w./:-]+$")


def _tokens(text):
    return _WORD.findall((text or "").lower())


# ---------------------------------------------------------------- schemas
_TYPES = {"int": "integer", "integer": "integer", "float": "number",
          "number": "number", "str": "string", "string": "string",
          "bool": "boolean", "boolean": "boolean", "dict": "object",
          "object": "object", "list": "array", "array": "array"}


def _norm_spec(p):
    """One property schema, with ToolACE's Python-style types normalized
    ("int"→integer, "dict"→object, "list"→array) and nested props kept."""
    p = p or {}
    out = {"type": _TYPES.get(str(p.get("type") or "").lower(),
                              p.get("type")),
           "enum": p.get("enum"),
           "desc": p.get("description") or "",
           "items": _norm_spec(p["items"]) if p.get("items") else None,
           "props": None,
           "required": False}
    props = p.get("properties")
    if isinstance(props, dict):
        out["props"] = {k: _norm_spec(v) for k, v in props.items()}
    return out


def arg_specs(sys_text):
    """{tool: {arg: spec}} — parsed by top-level brace matching, not by
    `"name"` marks: a tool whose properties include a field called "name"
    otherwise truncates its own block and the whole schema fails to parse
    (measured: 19/65 gold args lost that way).
    """
    s = _s(sys_text)
    out = {}
    i = 0
    while i < len(s):
        if s[i] == "{":
            end, ok = _match_close(s, i)
            if ok:
                frag = s[i:end]
                try:
                    obj = json.loads(frag)
                except Exception:
                    obj = None
                if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                    params = obj.get("parameters") or obj.get("properties")
                    if isinstance(params, dict) and \
                            isinstance(params.get("properties"), dict):
                        req = set(params.get("required") or [])
                        out[obj["name"]] = {
                            k: dict(_norm_spec(v), required=(k in req))
                            for k, v in params["properties"].items()}
                i = end
                continue
        i += 1
    return out


def arg_prior(spec, user):
    """Deterministic arg-name bonuses added to model scores.

    Prior, not gate: every schema arg stays a candidate, but required and
    query-grounded args start ahead — the measured failure mode is the
    model skipping a grounded arg (topic) and emitting an ungrounded one
    (page).
    """
    ut = set(_tokens(user))
    out = {}
    for a, s in (spec or {}).items():
        p = 0.0
        if s.get("required"):
            p += 0.6
        if set(_tokens(a)) & ut:
            p += 0.4
        elif set(_tokens(s.get("desc"))) & ut:
            p += 0.15
        if s.get("enum") and any(str(e).lower() in ut
                                 for e in s["enum"]):
            p += 0.2
        out[a] = round(p, 3)
    return out


# --------------------------------------------------------------- values
def _spans(user, top=144):
    """Extractive word n-grams (n<=4), edge punctuation stripped.

    Quota per n (top//4 each) instead of filling the cap with unigrams:
    with a flat cap the first 96 single tokens exhausted the set and
    multi-word gold values ("senior software engineer") were never
    offered — the measured verbatim miss.
    """
    toks = []
    for t in _WS.split(user or ""):
        s = t.strip(_EDGE)
        # keep a trailing period on abbreviations (D.C., U.S.A.) — gold
        # values carry it and stripping orphans the span
        if t.endswith(".") and s and "." in s:
            s += "."
        if s:
            toks.append(s)
    out, seen = [], set()
    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            sp = " ".join(toks[i:i + n])
            if '"' in sp or "\n" in sp or sp in seen:
                continue
            seen.add(sp)
            out.append(sp)
            if len(out) >= top:
                return out
    return out


def _spans_raw(user, top=144):
    """Punctuation-preserving n-grams — gold values carry interior commas
    and periods ("New York, NY", "Washington D.C.") that edge-stripping
    destroys. Added as extra candidates, not a replacement."""
    toks = [t for t in _WS.split(user or "")
            if t and '"' not in t and "\n" not in t]
    out, seen = [], set()
    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            sp = " ".join(toks[i:i + n]).strip()
            if not sp or sp in seen:
                continue
            seen.add(sp)
            out.append(sp)
            if len(out) >= top:
                return out
    return out


_COORD = re.compile(
    r"[A-Za-z]?\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_RANGE2 = re.compile(
    r"\w+\s*=\s*(-?\d+(?:\.\d+)?)\s+to\s+\w+\s*=\s*(-?\d+(?:\.\d+)?)")


def _pair(a, b):
    """[a, b] plus a float-normalized variant."""
    return [f"[{a}, {b}]", f"[{float(a)}, {float(b)}]"]


def _entity_spans(user):
    """Capitalized runs — the shape of names, places, titles."""
    out, cur = [], []
    for tok in _WS.split(user or ""):
        t = tok.strip(_EDGE)
        if t and t[0].isupper() and not t.isdigit():
            cur.append(t)
        else:
            if len(cur) >= 1:
                out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return [e for e in out if len(e) > 1][:24]


def _json_literals(user):
    """Brace/bracket-balanced JSON fragments already present in the query."""
    out = []
    for i, c in enumerate(user or ""):
        if c in "{[":
            try:
                end, _ = _match_close(user, i)
            except Exception:
                continue
            frag = user[i:end]
            if len(frag) > 3:
                try:
                    out.append(json.dumps(json.loads(frag)))
                except Exception:
                    out.append(frag)
    return out[:4]


def _sub_values(prop, user):
    """Candidate literal forms for one scalar property, from the query."""
    typ = (prop or {}).get("type")
    out = []
    for m in _NUM.finditer(user or ""):
        n = m.group(0)
        if typ in ("number", "integer"):
            out.append(n if "." in n or typ == "integer" else n)
            if "." not in n:
                out.append(n + ".0")
        else:
            out.append(f'"{n}"')
    for m in _DATE.finditer(user or ""):
        out.append(f'"{m.group(0)}"')
    for e in (prop or {}).get("enum") or []:
        out.append(json.dumps(e) if not isinstance(e, str) else f'"{e}"')
    if typ == "boolean":
        out += ["true", "false"]
    for e in _entity_spans(user):
        out.append(f'"{e}"')
    return out[:24]


def _compose_object(props, user):
    """Compose {k: v} from per-property sub-values.

    Numeric siblings consume query numbers in order (min_price=500,
    max_price=1500), not the same first number twice — the measured
    price_range failure.
    """
    if not props:
        return []
    nums = _NUM.findall(user or "")
    ni = 0
    pairs = []
    for k, p in props.items():
        typ = (p or {}).get("type")
        if typ in ("number", "integer") and ni < len(nums):
            n = nums[ni]
            ni += 1
            if typ == "integer" and "." in n:
                n = n.split(".")[0]
            pairs.append((k, n))
            continue
        vals = _sub_values(p, user)
        if vals:
            pairs.append((k, vals[0]))
    if not pairs:
        return []
    body = ", ".join(f'"{k}": {v}' for k, v in pairs)
    return ["{" + body + "}"]


def _compose_array_of_objects(props, user):
    """Compose repeated objects by aligning property occurrences in order.

    Heuristic: each property contributes its extracted values in query
    order; the i-th object takes the i-th value of each property. Covers
    the ToolACE shape where a list of records is described in prose.
    """
    if not props:
        return []
    cols = {}
    n = 0
    for k, p in props.items():
        vals = _sub_values(p, user)
        cols[k] = vals
        n = max(n, len(vals))
    if n == 0:
        return []
    objs = []
    for i in range(min(n, 8)):
        pairs = [(k, v[i]) for k, v in cols.items() if i < len(v)]
        if pairs:
            objs.append("{" + ", ".join(f'"{k}": {v}'
                                        for k, v in pairs) + "}")
    return ["[" + ", ".join(objs) + "]"]


def _compose_array_scalars(items, user):
    """Compose [a, b, ...] from an enumeration in the query."""
    out = []
    typ = (items or {}).get("type")
    # numeric range: "between 1 and 3" / "1-3" → [1, 3] and [1.0, 3.0]
    for m in re.finditer(r"\b(\d+)\s*(?:-|–|to|and)\s*(\d+)\b", user or ""):
        a, b = m.group(1), m.group(2)
        out.append(f"[{a}, {b}]")
        out.append(f"[{a}.0, {b}.0]")
    # quoted items: "Confidentiality, Termination"
    for m in re.finditer(r'"([^"\n]{2,40})"\s*(?:,|and)\s*"([^"\n]{2,40})"',
                         user or ""):
        vals = [m.group(1), m.group(2)]
        out.append("[" + ", ".join(f'"{v}"' for v in vals) + "]")
    # capitalized enumeration: A, B and C
    caps = re.findall(r"\b([A-Z][\w-]+(?:\s[A-Z][\w-]+)?)\b", user or "")
    if len(caps) >= 2:
        out.append("[" + ", ".join(f'"{c}"' for c in caps[:6]) + "]")
    # lowercase "x and y" / "x, y" enumeration (2–3 short noun phrases)
    for m in re.finditer(
            r"\b([a-z][\w-]{2,20})\s*(?:,|and)\s*([a-z][\w-]{2,20})"
            r"(?:\s*(?:,|and)\s*([a-z][\w-]{2,20}))?\b", user or ""):
        vals = [g for g in m.groups() if g]
        out.append("[" + ", ".join(f'"{v}"' for v in vals) + "]")
    if typ in ("number", "integer"):
        nums = _NUM.findall(user or "")
        if len(nums) >= 2:
            out.append("[" + ", ".join(nums[:6]) + "]")
    return out


def value_candidates(spec, aname, user, top_spans=96):
    """Ordered literal candidates for arg `aname` (deduped, capped).

    96 spans, not 48: the measured misses include gold values that ARE
    verbatim in the query ("senior software engineer") but fell past the
    old cap, so the set never offered them.
    """
    s = (spec or {}).get(aname) or {}
    typ = s.get("type") or "string"
    out = []

    def add(c):
        if c and c not in out:
            out.append(c)

    for m in _QUOTED.finditer(user or ""):
        add(f'"{m.group(1) or m.group(2)}"')
    for e in s.get("enum") or []:
        add(json.dumps(e) if not isinstance(e, str) else f'"{e}"')
    if typ == "boolean":
        add("true")
        add("false")
    if typ == "array":
        items = s.get("items") or {}
        if items.get("type") == "object" or items.get("properties"):
            for c in _compose_array_of_objects(items.get("properties") or {},
                                               user):
                add(c)
        for c in _compose_array_scalars(items, user):
            add(c)
    elif typ == "object":
        for c in _compose_object(s.get("props") or {}, user):
            add(c)
    for frag in _json_literals(user):
        add(frag)
    for sp in _entity_spans(user):
        add(f'"{sp}"')
    for m in _DATE.finditer(user or ""):
        add(f'"{m.group(0)}"')
    for m in _NUM.finditer(user or ""):
        n = m.group(0)
        if typ in ("number", "integer"):
            add(n)
            if "." not in n:
                add(n + ".0")
        else:
            add(f'"{n}"')
    # parenthesized coordinate pairs: A(3,4) and B(1,2) → [3, 4] / [1, 2]
    for m in _COORD.finditer(user or ""):
        for p in _pair(m.group(1), m.group(2)):
            add(p)
    # ranges with variable binding: "from x=1 to x=3" → [1, 3]
    for m in _RANGE2.finditer(user or ""):
        for p in _pair(m.group(1), m.group(2)):
            add(p)
    for sp in _spans(user, top_spans):
        add(f'"{sp}"')
        if "^" in sp:
            add(f'"{sp.replace(chr(94), "**")}"')
    for sp in _spans_raw(user, top_spans):
        add(f'"{sp}"')
    return out


# ---------------------------------------------------------------- metric
def _norm(lit):
    """Canonical comparable value: parsed JSON when possible, else text."""
    if isinstance(lit, (int, float, bool)):
        return ("val", lit)
    s = str(lit).strip()
    try:
        return ("val", json.loads(s))
    except Exception:
        return ("val", s.strip('"\''))


def _eq(a, b):
    """Value equality across the two representations in play: emitted-form
    strings (ToolACE golds) and native Python values (BFCL golds)."""
    va = _norm(a)[1]
    vb = _norm(b)[1]
    if isinstance(va, str) and isinstance(vb, str):
        return va == vb or va.lower() == vb.lower()
    if isinstance(va, bool) or isinstance(vb, bool):
        return va == vb
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return abs(float(va) - float(vb)) < 1e-9
    if isinstance(va, str) != isinstance(vb, str):
        sa, sb = str(va), str(vb)
        if sa == sb or sa.lower() == sb.lower():
            return True
        try:
            return abs(float(sa) - float(sb)) < 1e-9
        except Exception:
            return False
    return va == vb


def candidate_recall(rows, items):
    """Per-gold-arg membership in the candidate set (CPU-only metric).

    rows: call_synth rows (k, gold_args); items: same-slice ToolACE items
    with sys/user. Returns per-item detail + aggregate recall.
    """
    by_k = {r["k"]: r for r in rows}
    detail, hit, total = [], 0, 0
    for k, it in enumerate(items):
        r = by_k.get(k)
        if not r or not r.get("gold_args"):
            continue
        specs = arg_specs(it["sys"])
        spec = specs.get(r["gold"]) or {}
        names = set(spec)
        per = []
        for aname, gold in r["gold_args"]:
            total += 1
            ok_name = aname in names
            cands = value_candidates(spec, aname, it["user"]) if ok_name else []
            ok_val = any(_eq(c, gold) for c in cands)
            hit += bool(ok_val)
            per.append({"arg": aname, "gold": gold[:80],
                        "name_in_schema": ok_name,
                        "value_recovered": ok_val,
                        "n_cands": len(cands)})
        detail.append({"k": k, "tool": r["gold"], "args": per})
    return {"n_gold_args": total, "recovered": hit,
            "recall": round(hit / total, 4) if total else 0.0,
            "detail": detail}
