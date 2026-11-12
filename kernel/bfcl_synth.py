"""BFCL end-to-end constrained call synthesis on the 27B scorer.

Adapted from call_synth.py (ToolACE) to the BFCL substrate:

  phase A  tool selection — closed-set argmax over declared names +
           "DONE" (multi-call cats loop until DONE or MAX_CALLS)
  phase B  arg names   — sequential argmax over remaining schema args
           + ")"
  phase C  arg values  — argmax over arg_compose.value_candidates
           (the schema-aware composer, not raw spans)

Every selection is scored by the host LM (mean token logprob after the
chat-templated context) — never free-generated. Verification compares
emitted calls against BFCL acceptable-value lists via _eq.

Model-agnostic: the Modal driver injects `score_fn(ctx_text, cands)`.

    env: CATS (simple,multiple,parallel,parallel_multiple),
         N_PER_CAT (40), MAX_CALLS (4), MAX_ARGS (6)
"""
import json
import os
import sys
import time

_REPO = os.environ.get("REPO", "/root/repo")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "kernel"))
sys.modules.pop("arg_compose", None)

from huggingface_hub import hf_hub_download                  # noqa: E402
from arg_compose import arg_specs, value_candidates, _eq     # noqa: E402
from src.emit_eval import parse_call                         # noqa: E402

DS = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
CATS = os.environ.get(
    "CATS", "simple,multiple,parallel,parallel_multiple").split(",")
N_PER_CAT = int(os.environ.get("N_PER_CAT", "40"))
OFFSET = int(os.environ.get("OFFSET", "0"))
MAX_CALLS = int(os.environ.get("MAX_CALLS", "4"))
MAX_ARGS = int(os.environ.get("MAX_ARGS", "6"))
VALUE_TOPK = int(os.environ.get("VALUE_TOPK", "0"))  # 0 = no cap


def _jsonl(name):
    p = hf_hub_download(DS, name, repo_type="dataset")
    return [json.loads(l) for l in open(p)]


def _user(it):
    msgs = []
    for m in it.get("question") or []:
        if isinstance(m, list):
            msgs.extend(x for x in m if isinstance(x, dict))
        elif isinstance(m, dict):
            msgs.append(m)
    return "\n".join(m.get("content", "") for m in msgs
                     if m.get("role") == "user")


def _schema_json(it):
    return json.dumps(
        [{"name": f["name"], "description": f.get("description", ""),
          "parameters": f.get("parameters")}
         for f in (it.get("function") or [])])


def _ctx_text(user, schema_json, emitted):
    return (f"{user}\n\nAvailable functions:\n{schema_json}\n"
            f"Assistant: [{emitted}")


def verify_calls(emitted_calls, gt_calls, names, specs):
    """Pure verification stage: emitted call strings -> metrics.

    parse_call is quote/bracket-aware (emit_eval._split_top) — naive
    comma-splitting corrupts nested JSON/array arg values.
    Returns (pred_calls, tool_valid, args_valid, gt_matched).
    """
    pred_calls = []
    for cs in emitted_calls:
        pc = parse_call(cs)
        if pc and pc.get("name"):
            pred_calls.append(
                (pc["name"], {k: v for k, v in pc.get("args", [])}))

    # per-call arg match: for each GT call, exists a pred call with same
    # fname where every GT arg is satisfied — a missing arg counts iff
    # its acceptable list contains '' (BFCL's omission convention,
    # same rule as the recall harness's abstain_ok)
    gt_matched = 0
    used = set()
    for gcall in gt_calls:
        for fname, gargs in gcall.items():
            for pi, (pf, par) in enumerate(pred_calls):
                if pi in used or pf != fname:
                    continue
                if all(
                        (par.get(an) is not None
                         and any(_eq(par[an], g) for g in accs))
                        or (par.get(an) is None and "" in accs)
                        for an, accs in gargs.items()):
                    used.add(pi)
                    gt_matched += 1
                    break

    tool_valid = all(pf in names for pf, _ in pred_calls)
    args_valid = all(
        an in (specs.get(pf) or {}) for pf, par in pred_calls
        for an in par)
    return pred_calls, tool_valid, args_valid, gt_matched


def synth_item(it, ans_entry, score_fn, cat="?"):
    """One BFCL item -> row dict. score_fn(ctx, cands) -> {cand: score}."""
    user = _user(it)
    schema_json = _schema_json(it)
    funcs = it.get("function") or []
    names = [f["name"] for f in funcs]
    specs = arg_specs(schema_json)
    # GT: list of calls, each {fname: {aname: [acceptable,...]}}
    gt_calls = []
    for call in (ans_entry or []):
        gt_calls.append({f: {a: (v if isinstance(v, list) else [v])
                             for a, v in args.items()}
                         for f, args in call.items()})

    emitted_calls = []
    emitted = ""
    name_picks = []
    for _call_i in range(MAX_CALLS):
        # ---- tool selection: names + DONE ----
        prefix = ", " if emitted else ""
        cands = [f"{prefix}{n}(" for n in names] + ["DONE"]
        sc = score_fn(_ctx_text(user, schema_json, emitted), cands)
        pick = max(sc, key=sc.get)
        if pick == "DONE" or pick not in [f"{prefix}{n}(" for n in names]:
            break
        fname = pick[len(prefix):-1]
        name_picks.append(fname)
        spec = specs.get(fname) or {}
        remaining = [a for a in spec]
        call = f"{fname}("
        # ---- arg names ----
        for _ in range(MAX_ARGS):
            comma = "" if call.endswith("(") else ", "
            ac = [f"{comma}{a}=" for a in remaining] + [")"]
            cur = emitted + call
            s2 = score_fn(_ctx_text(user, schema_json, cur), ac)
            ap = max(s2, key=s2.get)
            if ap == ")":
                call += ")"
                break
            an = ap[len(comma):-1]
            if an not in remaining:
                call += ")"
                break
            remaining.remove(an)
            call += f"{comma}{an}="
            # ---- value: composer candidates ----
            vc = value_candidates(spec, an, user)
            if VALUE_TOPK:
                vc = vc[:VALUE_TOPK]
            if not vc:
                vc = ['""']
            cur = emitted + call
            vs = score_fn(_ctx_text(user, schema_json, cur), vc)
            vp = max(vs, key=vs.get)
            call += vp
        else:
            call += ")"
        if not call.endswith(")"):
            call += ")"
        emitted_calls.append(call)
        emitted += (", " if emitted_calls[:-1] else "") + call

    # ---- verify: pure stage, separately testable ----
    pred_calls, tool_valid, args_valid, gt_matched = verify_calls(
        emitted_calls, gt_calls, names, specs)
    exact_calls = (len(pred_calls) == len(gt_calls)
                   and gt_matched == len(gt_calls) and len(gt_calls) > 0)

    return {
        "id": it["id"], "cat": cat,
        "n_gt_calls": len(gt_calls), "n_pred_calls": len(pred_calls),
        "name_picks": name_picks,
        "gt": str(gt_calls)[:300], "emit": emitted[:300],
        "tool_valid": tool_valid, "args_valid": args_valid,
        "call_match": gt_matched, "gold_match": exact_calls,
        "pred_calls": pred_calls, "gt_calls": gt_calls,
    }


def run_all(score_fn, on_ckpt=None):
    """score_fn(ctx_text, cands) -> {cand: float}. Returns rows.
    on_ckpt() is called after each incremental receipt write (e.g.
    volume commit) so partials survive a kill."""
    rows = []
    t0 = time.time()
    for cat in CATS:
        req = _jsonl(f"BFCL_v3_{cat}.json")[OFFSET:OFFSET + N_PER_CAT]
        ans = {a["id"]: a["ground_truth"] for a in
               _jsonl(f"possible_answer/BFCL_v3_{cat}.json")}
        ckpt_every = int(os.environ.get("CKPT_EVERY", "10"))
        ckpt_path = os.environ.get(
            "CKPT_PATH", "/artifacts/bfcl_synth_partial.json")
        for i, it in enumerate(req):
            ti = time.time()
            try:
                r = synth_item(it, ans.get(it["id"]), score_fn, cat=cat)
            except Exception as e:
                r = {"id": it.get("id"), "cat": cat, "error": str(e)[:200]}
            r["item_s"] = round(time.time() - ti, 1)
            if r["item_s"] > 120:
                print(f"[synth] SLOW {cat} item {i+1} "
                      f"{r['item_s']}s id={it.get('id')}", flush=True)
            rows.append(r)
            if (i + 1) % ckpt_every == 0:
                # incremental receipt — a kill loses <= ckpt_every items,
                # not the whole run (learned from the disconnect kill)
                try:
                    json.dump({"partial": True, "rows": rows,
                               "agg": aggregate(rows)},
                              open(ckpt_path, "w"))
                    if on_ckpt:
                        on_ckpt()
                except Exception:
                    pass
            if (i + 1) % 10 == 0:
                print(f"[synth] {cat} {i+1}/{len(req)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def aggregate(rows):
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    agg = {
        "n": n, "errors": len(rows) - n,
        "tool_valid": sum(r["tool_valid"] for r in ok),
        "args_valid": sum(r["args_valid"] for r in ok),
        "gold_match": sum(r["gold_match"] for r in ok),
        "call_match_total": sum(r["call_match"] for r in ok),
        "gt_calls_total": sum(r["n_gt_calls"] for r in ok),
        "pred_calls_total": sum(r["n_pred_calls"] for r in ok),
    }
    agg["call_recall"] = round(
        agg["call_match_total"] / max(1, agg["gt_calls_total"]), 4)
    for k in ("tool_valid", "args_valid", "gold_match"):
        agg[k] = {"n": agg[k], "rate": round(agg[k] / max(1, n), 4)}
    return agg


if __name__ == "__main__":
    # standalone smoke: uniform scorer (no model) — verifies plumbing
    def flat(ctx, cands):
        return {c: 0.0 for c in cands}
    rows = run_all(flat)
    print(json.dumps(aggregate(rows), indent=1))
