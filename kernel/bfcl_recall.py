"""BFCL candidate-recall — does the composer generalize off ToolACE?

ToolACE is synthetic and its golds are measurably noisy: on the pinned
slice 19 of 65 gold args name a tool that is not in the prompt at all.
BFCL is real API documentation with verified golds, so the same
model-free questions can be asked on a cleaner substrate:

  parse?      does arg_specs recover the schema (BFCL uses JSON Schema)
  recovered?  is the gold literal in the candidate set
  baseline    is the gold literal verbatim in the user turn

If recall holds or improves here, the composer is not overfit to
ToolACE's synthetic phrasing and the eval substrate can move to BFCL.

    colab exec -s hpar -f kernel/bfcl_recall.py
    env: N_PER_CAT (40), CATS (simple,multiple,parallel,parallel_multiple)
"""
import json
import os
import sys
import time

_ROOT = os.environ.get("REPO", "/content/qt/hnet-qwen-adapter")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernel"))
sys.modules.pop("arg_compose", None)

from huggingface_hub import hf_hub_download                  # noqa: E402
from arg_compose import arg_specs, value_candidates, _eq     # noqa: E402

DS = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
N = int(os.environ.get("N_PER_CAT", "40"))
CATS = os.environ.get(
    "CATS", "simple,multiple,parallel,parallel_multiple").split(",")


def _jsonl(name):
    p = hf_hub_download(DS, name, repo_type="dataset")
    return [json.loads(l) for l in open(p)]


def _user(it):
    """User turns — BFCL v3 question is a list of dicts for single-turn
    categories and a list of *lists* for multi-turn ones."""
    msgs = []
    for m in it.get("question") or []:
        if isinstance(m, list):
            msgs.extend(x for x in m if isinstance(x, dict))
        elif isinstance(m, dict):
            msgs.append(m)
    return "\n".join(m.get("content", "") for m in msgs
                     if m.get("role") == "user")


rows, tot = [], {"args": 0, "parsed": 0, "recovered": 0,
                 "verbatim": 0, "items": 0, "schema_ok": 0}
for cat in CATS:
    req = _jsonl(f"BFCL_v3_{cat}.json")[:N]
    ans = {a["id"]: a["ground_truth"]
           for a in _jsonl(f"possible_answer/BFCL_v3_{cat}.json")}
    for it in req:
        funcs = it.get("function") or []
        schema = json.dumps(
            [{"name": f["name"], "description": f.get("description", ""),
              "parameters": f.get("parameters")} for f in funcs])
        specs = arg_specs(schema)
        tot["items"] += 1
        tot["schema_ok"] += bool(specs)
        user = _user(it)
        for call in ans.get(it["id"], []):
            for fname, args in call.items():
                spec = specs.get(fname) or {}
                for aname, golds in (args or {}).items():
                    # BFCL gives a list of *acceptable* values per arg, and
                    # includes ''/defaults when the argument may legitimately
                    # be omitted. Those are abstain cases, not composer
                    # misses — counted separately so recall means something.
                    acc = golds if isinstance(golds, list) else [golds]
                    optional = ("" in acc) or not (
                        (spec.get(aname) or {}).get("required"))
                    tot["args"] += 1
                    okp = aname in spec
                    tot["parsed"] += okp
                    cands = (value_candidates(spec, aname, user)
                             if okp else [])
                    ok = any(_eq(c, g) for c in cands for g in acc)
                    tot["recovered"] += ok
                    tot["verbatim"] += any(
                        str(g) in user for g in acc
                        if isinstance(g, str) and g)
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
                        rows.append({"cat": cat, "tool": fname,
                                     "arg": aname,
                                     "gold": str(acc)[:60],
                                     "type": (spec.get(aname) or {}).get("type")})
    print(f"[bfcl] {cat}: cumulative {tot}", flush=True)

res = dict(tot)
res["recall_parsed"] = round(tot["recovered"] / max(1, tot["parsed"]), 4)
res["baseline_verbatim"] = round(tot["verbatim"] / max(1, tot["parsed"]), 4)
res["parse_rate"] = round(tot["parsed"] / max(1, tot["args"]), 4)
# TRUE required-subset recall — explicit buckets, not the old
# abstain_ok-subtraction approximation
res["req_total"] = tot.get("req_total", 0)
res["req_hit"] = tot.get("req_hit", 0)
res["opt_total"] = tot.get("opt_total", 0)
res["opt_hit"] = tot.get("opt_hit", 0)
res["recall_required"] = round(
    tot.get("req_hit", 0) / max(1, tot.get("req_total", 0)), 4)
res["recall_optional"] = round(
    tot.get("opt_hit", 0) / max(1, tot.get("opt_total", 0)), 4)
res["misses"] = rows
path = f"{_ROOT}/results/bfcl_recall_{int(time.time())}.json"
json.dump(res, open(path, "w"), indent=1)
print(f"[bfcl] args {tot['args']} parsed {tot['parsed']} "
      f"({res['parse_rate']:.3f})", flush=True)
print(f"[bfcl] recall {tot['recovered']}/{tot['parsed']} = "
      f"{res['recall_parsed']:.3f} vs verbatim {res['baseline_verbatim']:.3f} "
      f"→ {res['recall_parsed'] - res['baseline_verbatim']:+.3f}", flush=True)
print(f"[bfcl] receipt {path}", flush=True)
