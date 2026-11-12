"""BFCL breadth — categories the pinned N=40 slice didn't cover.

Extends bfcl_recall.py's model-free questions to:

  live_simple / live_multiple / live_parallel / live_parallel_multiple
      real-world API scenarios — same GT shape, drop-in recall.
  irrelevance
      no possible_answer file; GT is "emit no call". Metric here is the
      false-positive surface: schema parse rate + mean candidate density
      per declared arg. Dense candidates on irrelevant queries are what
      tempt the pipeline into hallucinating a call.
  multi_turn_base
      GT is per-turn lists of Python call strings ("cd(folder='x')").
      Parsed best-effort; user context = all user turns joined (binding
      across turns is the scorer's job — this measures candidate recall).

    env: N_PER_CAT (default 100), CATS (comma list)
"""
import json
import os
import re
import sys
import time

_ROOT = os.environ.get("REPO", "/kaggle/working/repo")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernel"))
sys.modules.pop("arg_compose", None)

from huggingface_hub import hf_hub_download                  # noqa: E402
from arg_compose import arg_specs, value_candidates, _eq     # noqa: E402

DS = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
N = int(os.environ.get("N_PER_CAT", "100"))
CATS = os.environ.get(
    "CATS",
    "live_simple,live_multiple,live_parallel,live_parallel_multiple,"
    "irrelevance,multi_turn_base").split(",")


def _jsonl(name):
    p = hf_hub_download(DS, name, repo_type="dataset")
    return [json.loads(l) for l in open(p)]


def _answers(cat):
    try:
        return {a["id"]: a["ground_truth"]
                for a in _jsonl(f"possible_answer/BFCL_v3_{cat}.json")}
    except Exception:
        return {}


def _user(it):
    msgs = []
    for m in it.get("question") or []:
        if isinstance(m, list):
            msgs.extend(x for x in m if isinstance(x, dict))
        elif isinstance(m, dict):
            msgs.append(m)
    return "\n".join(m.get("content", "") for m in msgs
                     if m.get("role") == "user")


def _schema(it):
    funcs = it.get("function") or []
    return json.dumps(
        [{"name": f["name"], "description": f.get("description", ""),
          "parameters": f.get("parameters")} for f in funcs])


def _check_call(specs, user, fname, args, cat, tot, rows):
    """args: {arg_name: [acceptable,...]} — shared recall accounting."""
    spec = specs.get(fname) or {}
    for aname, golds in (args or {}).items():
        acc = golds if isinstance(golds, list) else [golds]
        optional = ("" in acc) or not (
            (spec.get(aname) or {}).get("required"))
        tot["args"] += 1
        okp = aname in spec
        tot["parsed"] += okp
        cands = value_candidates(spec, aname, user) if okp else []
        ok = any(_eq(c, g) for c in cands for g in acc)
        tot["recovered"] += ok
        tot["verbatim"] += any(
            str(g) in user for g in acc if isinstance(g, str) and g)
        # explicit buckets — the old abstain_ok-subtraction formula was
        # not a true required-subset recall
        if okp:
            if optional:
                tot["opt_total"] = tot.get("opt_total", 0) + 1
                tot["opt_hit"] = tot.get("opt_hit", 0) + ok
            else:
                tot["req_total"] = tot.get("req_total", 0) + 1
                tot["req_hit"] = tot.get("req_hit", 0) + ok
        if not ok and optional:
            tot["abstain_ok"] = tot.get("abstain_ok", 0) + 1
        elif okp and not ok:
            rows.append({"cat": cat, "tool": fname, "arg": aname,
                         "gold": str(acc)[:60],
                         "type": (spec.get(aname) or {}).get("type")})


_CALL = re.compile(r"(\w+)\((.*)\)")
_KV = re.compile(r"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|"
                 r"[^,]+)")


def _parse_call_string(s):
    """'cd(folder='document')' -> {'cd': {'folder': ['document']}}"""
    out = {}
    for m in _CALL.finditer(s):
        fname, body = m.group(1), m.group(2)
        args = {}
        for kv in _KV.finditer(body):
            v = kv.group(2).strip()
            if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
                v = v[1:-1]
            try:
                v = json.loads(v)
            except Exception:
                pass
            args[kv.group(1)] = [v]
        out[fname] = args
    return out


def run_cat(cat):
    tot = {"args": 0, "parsed": 0, "recovered": 0, "verbatim": 0,
           "items": 0, "schema_ok": 0, "abstain_ok": 0}
    rows = []
    extra = {}
    req = _jsonl(f"BFCL_v3_{cat}.json")[:N]
    ans = _answers(cat)

    if cat == "irrelevance":
        # No GT calls exist. Measure the hallucination surface instead:
        # how many schemas parse, and how dense are the candidate sets
        # the scorer would have to reject.
        dens = []
        for it in req:
            specs = arg_specs(_schema(it))
            tot["items"] += 1
            tot["schema_ok"] += bool(specs)
            user = _user(it)
            for fname, spec in specs.items():
                for aname in spec:
                    dens.append(len(value_candidates(spec, aname, user)))
        extra["mean_cands_per_arg"] = round(
            sum(dens) / max(1, len(dens)), 1)
        extra["n_arg_slots"] = len(dens)
        return tot, rows, extra

    for it in req:
        specs = arg_specs(_schema(it))
        tot["items"] += 1
        tot["schema_ok"] += bool(specs)
        user = _user(it)
        gt = ans.get(it["id"], [])
        if cat.startswith("multi_turn"):
            # GT: list of turns, each a list of call strings. Join all
            # user turns as context (cross-turn binding is scorer-side).
            for turn in gt:
                for callstr in (turn if isinstance(turn, list) else [turn]):
                    for fname, args in _parse_call_string(callstr).items():
                        _check_call(specs, user, fname, args, cat, tot, rows)
        else:
            for call in gt:
                for fname, args in call.items():
                    _check_call(specs, user, fname, args, cat, tot, rows)
    return tot, rows, extra


allrows, per_cat = [], {}
for cat in CATS:
    try:
        tot, rows, extra = run_cat(cat)
    except Exception as e:
        per_cat[cat] = {"error": str(e)[:200]}
        print(f"[breadth] {cat}: ERROR {e}", flush=True)
        continue
    per_cat[cat] = dict(tot, **extra)
    allrows += [{**r, "cat": cat} for r in rows]
    req_recall = round(tot.get("req_hit", 0) / max(1, tot.get("req_total", 0)), 3)
    print(f"[breadth] {cat}: args {tot['args']} parsed {tot['parsed']} "
          f"req_recall {req_recall} items {tot['items']}", flush=True)

# pooled across recall-capable cats — explicit req/opt buckets
pool = {"args": 0, "parsed": 0, "recovered": 0, "verbatim": 0,
        "abstain_ok": 0, "req_total": 0, "req_hit": 0,
        "opt_total": 0, "opt_hit": 0}
for cat, t in per_cat.items():
    if "error" in t or cat == "irrelevance":
        continue
    for k in pool:
        pool[k] += t.get(k, 0)
res = {
    "n_per_cat": N, "cats": CATS, "per_cat": per_cat,
    "pooled": dict(
        pool,
        recall_required=round(pool["req_hit"] / max(1, pool["req_total"]), 4),
        recall_optional=round(pool["opt_hit"] / max(1, pool["opt_total"]), 4),
        recall_parsed=round(pool["recovered"] / max(1, pool["parsed"]), 4)),
    "misses": allrows[:400],
}
path = f"/kaggle/working/bfcl_breadth_{int(time.time())}.json"
json.dump(res, open(path, "w"), indent=1)
print(f"[breadth] receipt {path}", flush=True)
print(json.dumps(res["pooled"], indent=1), flush=True)
