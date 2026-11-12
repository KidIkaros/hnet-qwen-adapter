"""Candidate-recall harness — the model-free gate for arg_compose.

Streams the same pinned ToolACE slice as call_synth (VERIFY_SKIP=2500,
N=40) and answers, per gold arg, three separable questions:

  declared?   is the gold tool among the prompt's declared tools at all?
              (an undeclared gold has no schema — not a composer miss)
  parsed?     did arg_specs recover the arg's schema?
  recovered?  is the gold literal in the candidate set offered to the
              scorer?

The comparison number is the extractive baseline (43/65 = 66%, gold value
verbatim in the query) recomputed on the SAME subset, so the delta means
something.

    colab exec -s hpar -f kernel/arg_recall.py
    env: VERIFY_SKIP (2500), N_ITEMS (40)
"""
import json
import os
import sys
import time

_ROOT = os.environ.get("REPO", "/content/qt/hnet-qwen-adapter")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernel"))
# the colab kernel persists across execs — drop a stale cached module
sys.modules.pop("arg_compose", None)

from src.probe_common import RESULTS, stream_toolace          # noqa: E402
from src.emit_eval import parse_call                          # noqa: E402
from arg_compose import arg_specs, value_candidates, _eq       # noqa: E402

VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))


def _unquote(lit):
    s = str(lit).strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s


def _verbatim_in_query(gold, user):
    """Extractive baseline: gold value text appears in the query."""
    g = _unquote(gold)
    if not g:
        return False
    if g in (user or ""):
        return True
    # json form (e.g. [1, 3]) — compare against a whitespace-normalized copy
    norm = " ".join((user or "").split())
    return g.replace(" ", "") in norm.replace(" ", "")


items = stream_toolace(VSKIP, N_ITEMS)
print(f"[recall] {len(items)} items from ToolACE skip={VSKIP}", flush=True)

detail = []
n_all = n_declared = n_parsed = n_rec = 0
b_declared = b_rec = 0
undeclared_tools, parse_miss_tools, value_miss = set(), set(), []
for k, it in enumerate(items):
    g = parse_call(it.get("gold_full") or "")
    if not g or not g.get("args"):
        continue
    declared = it.get("tools") or []
    specs = arg_specs(it["sys"])
    per = []
    for aname, gold in g["args"]:
        n_all += 1
        is_decl = it.get("gold") in declared
        spec = specs.get(it.get("gold")) or {}
        is_parsed = aname in spec
        cands = value_candidates(spec, aname, it["user"]) if is_parsed else []
        ok = any(_eq(c, gold) for c in cands)
        verb = _verbatim_in_query(gold, it["user"])
        if is_decl:
            n_declared += 1
            b_declared += bool(verb)
            if is_parsed:
                n_parsed += 1
                n_rec += bool(ok)
                b_rec += bool(verb)
        else:
            undeclared_tools.add(it.get("gold"))
        if is_decl and not is_parsed:
            parse_miss_tools.add(it.get("gold"))
        if is_decl and is_parsed and not ok:
            value_miss.append({"k": k, "tool": it.get("gold"), "arg": aname,
                               "gold": str(gold)[:80],
                               "verbatim": verb,
                               "type": (spec.get(aname) or {}).get("type")})
        per.append({"arg": aname, "gold": str(gold)[:80],
                    "declared": is_decl, "parsed": is_parsed,
                    "recovered": ok, "verbatim": verb})
    detail.append({"k": k, "tool": it.get("gold"), "args": per})

res = {
    "n_gold_args": n_all,
    "declared_args": n_declared,
    "undeclared_args": n_all - n_declared,
    "parsed_args": n_parsed,
    "recovered_declared_parsed": n_rec,
    "recall_parsed": round(n_rec / n_parsed, 4) if n_parsed else 0.0,
    "baseline_verbatim_parsed": round(b_rec / n_parsed, 4) if n_parsed else 0.0,
    "delta_vs_extractive": (round((n_rec - b_rec) / n_parsed, 4)
                            if n_parsed else 0.0),
    "undeclared_tools": sorted(undeclared_tools),
    "parse_miss_tools": sorted(parse_miss_tools),
    "value_misses": value_miss,
    "detail": detail,
}

path = f"{RESULTS}/arg_recall_{int(time.time())}.json"
json.dump(res, open(path, "w"), indent=1)
print(f"[recall] gold args {n_all} | declared {n_declared} "
      f"(undeclared {res['undeclared_args']}) | parsed {n_parsed}", flush=True)
print(f"[recall] recovered {n_rec}/{n_parsed} = {res['recall_parsed']:.3f} "
      f"vs extractive {b_rec}/{n_parsed} = {res['baseline_verbatim_parsed']:.3f} "
      f"→ delta {res['delta_vs_extractive']:+.3f}", flush=True)
print(f"[recall] parse misses: {', '.join(res['parse_miss_tools'][:10])}",
      flush=True)
print(f"[recall] receipt {path}", flush=True)
