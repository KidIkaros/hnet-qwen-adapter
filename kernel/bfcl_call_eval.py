"""BFCL v3 full-call eval on the quantized 27B trunk — Phase B.

Extends bfcl_name_eval: after name selection, synthesize arguments
(schema arg names in order, extractive value candidates scored by
prefix logprob conditioned on the arg name) and score the emitted call
against BFCL ground_truth with official-equivalent AST semantics:

  valid = name in gold AND every emitted arg value matches an acceptable
          value in ground_truth[arg] (type-normalized), AND all required
          args present.

Categories: simple + multiple (single-call). Value candidates are
extractive spans of the user query — computed/inferred values are out
of scope for v1 and count as failures, which is the honest measure of
the extractive pipeline on a recognized exam.

    colab exec -s hrun9 -f kernel/bfcl_call_eval.py
    env: N_PER_CAT (100), CATS (simple,multiple), MODEL_PATH ("" = resident)
"""
import os, sys, subprocess, json, re, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"FAILED {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "fetch", "-q", "origin")
    up = set(sh("git", "-C", REPO, "ls-tree", "-r", "--name-only",
                "origin/main").split())
    for _r, _, _fs in os.walk(REPO):
        for _f in _fs:
            _p = os.path.join(_r, _f)
            _rel = os.path.relpath(_p, REPO)
            if _rel in up and "??" in sh(
                    "git", "-C", REPO, "status", "--porcelain", "--", _rel):
                os.remove(_p)
    sh("git", "-C", REPO, "merge", "-q", "origin/main")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)
for _m in [k for k in sys.modules if k.startswith("src.")]:
    del sys.modules[_m]

import torch                                                       # noqa: E402
from src.probe_common import RESULTS                                # noqa: E402

N_PER_CAT = int(os.environ.get("N_PER_CAT", "100"))
CATS = os.environ.get("CATS", "simple,multiple").split(",")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
TS = int(time.time())
_g = globals()

from huggingface_hub import hf_hub_download                    # noqa: E402
DS = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def _load_jsonl(repo_file):
    p = hf_hub_download(DS, repo_file, repo_type="dataset")
    return [json.loads(l) for l in open(p)]


if _g.get("model") is None or _g.get("tok") is None:
    from transformers import (AutoTokenizer, AutoModelForCausalLM)  # noqa: E402
    mp = MODEL_PATH or os.environ.get(
        "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")
    _g["tok"] = AutoTokenizer.from_pretrained(mp)
    _g["model"] = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "72GiB", "cpu": "70GiB"})
    _g["model"].eval()
tok, model = _g["tok"], _g["model"]
print(f"[bfcl] model ready ({MODEL_PATH or 'resident'})", flush=True)


@torch.no_grad()
def prefix_lp(ctx_ids, cand_ids):
    """Summed logprob of cand_ids after ctx_ids."""
    ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


def score_text(ctx_ids, text):
    """Mean logprob of text tokens after ctx (for value candidates)."""
    cand = tok(text, add_special_tokens=False)["input_ids"]
    if not cand:
        return -1e9
    return prefix_lp(ctx_ids, cand) / len(cand)


def value_candidates(user):
    """Extractive value spans: numbers (+.0 variants), quoted strings,
    capitalized/noun chunks, bools, common enum words."""
    cands = set()
    for m in re.finditer(r'"([^"]{1,60})"|\'([^\']{1,60})\'', user):
        cands.add(m.group(1) or m.group(2))
    for m in re.finditer(r"-?\d+\.?\d*", user):
        v = m.group(0)
        cands.add(v)
        if "." not in v:
            cands.add(v + ".0")
            cands.add(v + ".")
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", user):
        cands.add(m.group(0))
    for w in re.findall(r"\b[a-zA-Z][a-zA-Z_]{2,30}\b", user):
        if w.lower() not in {"the", "and", "for", "with", "that", "this",
                             "find", "calculate", "compute", "what", "how"}:
            cands.add(w)
    for w in ("true", "false", "all", "none", "yes", "no"):
        if w in user.lower():
            cands.add(w)
    return sorted(cands, key=len, reverse=True)[:40]


def norm_val(v):
    """Type-normalize for gold comparison."""
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        try:
            return int(float(s)) if float(s).is_integer() else float(s)
        except ValueError:
            return s
    return v


def ast_ok(pred_name, pred_args, gold_entries):
    """Official-equivalent check: name in gold, required args covered,
    every emitted arg value acceptable."""
    if pred_name not in [list(g.keys())[0] for g in gold_entries]:
        return False, "wrong_name"
    gold = next(iter(gold_entries[0].values())) if gold_entries else {}
    # emitted args must be a subset of gold's arg set (official checker
    # rejects unexpected args)
    for arg in pred_args:
        if arg not in gold:
            return False, f"extra_arg:{arg}"
    # required args must all be present and correct
    for arg, acc in gold.items():
        if arg not in pred_args:
            return False, f"missing_arg:{arg}"
        pv = norm_val(pred_args[arg])
        ok = any(norm_val(a) == pv or str(a) == str(pred_args[arg])
                 for a in acc)
        if not ok:
            return False, f"bad_value:{arg}"
    return True, "ok"


def synth_call(item):
    """Closed-set pipeline on the trunk: name -> arg names -> values."""
    funcs = item["function"]
    user = item["question"][0][-1]["content"]
    schema = json.dumps(
        [{"name": f["name"], "description": f.get("description", ""),
          "parameters": f["parameters"]} for f in funcs])
    ctx_s = (f"You are a function-calling assistant. Available functions:\n"
             f"{schema}\n\nUser: {user}\nAssistant: [")
    ctx = tok(ctx_s, return_tensors="pt").input_ids[0].tolist()

    # 1. name selection
    cands = [f["name"] for f in funcs] + ["none"]
    sc = {n: prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                       ["input_ids"]) for n in cands}
    pred = max(sc, key=sc.get)
    if pred == "none":
        return None, sc
    fdesc = next(f for f in funcs if f["name"] == pred)
    props = fdesc["parameters"].get("properties", {})
    required = fdesc["parameters"].get("required", [])
    arg_order = [a for a in required] + [a for a in props if a not in required]
    emit_args = arg_order[:len(required)]  # required only (v1)

    # 2./3. per-arg value selection, conditioned on arg name
    vals = value_candidates(user)
    arguments = {}
    prefix_args = ""
    for ai, arg in enumerate(emit_args):
        atype = props.get(arg, {}).get("type", "string")
        # score: name(arg="<cand>") — quote style by type
        sub_ctx_ids = tok(
            ctx_s + f"{pred}({prefix_args}{arg}=\"",
            return_tensors="pt").input_ids[0].tolist()
        best, bs = None, -1e18
        for v in vals:
            s = score_text(sub_ctx_ids, v + '"')
            if s > bs:
                best, bs = v, s
        if best is None:
            continue
        # coerce by schema type
        if atype in ("integer", "number"):
            try:
                fv = float(best.rstrip("."))
                arguments[arg] = int(fv) if fv.is_integer() else fv
            except ValueError:
                try:
                    arguments[arg] = int(float(best))
                except ValueError:
                    arguments[arg] = best
        elif atype == "boolean":
            arguments[arg] = best.lower() in ("true", "yes")
        else:
            arguments[arg] = best
        prefix_args += f'{arg}="{arguments[arg]}", '
    return {"name": pred, "arguments": arguments}, sc


all_rows = []
for cat in CATS:
    req = _load_jsonl(f"BFCL_v3_{cat}.json")[:N_PER_CAT]
    ans = {a["id"]: a["ground_truth"]
           for a in _load_jsonl(f"possible_answer/BFCL_v3_{cat}.json")}
    right = 0
    for i, it in enumerate(req):
        pred, sc = synth_call(it)
        gold = ans[it["id"]]
        if pred is None:
            ok, why = False, "abstain"
        else:
            ok, why = ast_ok(pred["name"], pred["arguments"], gold)
        right += ok
        all_rows.append({"cat": cat, "id": it["id"], "ok": ok, "why": why,
                         "pred": pred, "gold": gold})
        if (i + 1) % 20 == 0:
            print(f"  [{cat}] {i + 1}/{len(req)} acc={right / (i + 1):.3f}",
                  flush=True)
    acc = right / len(req)
    print(f"[{cat}] call-acc {right}/{len(req)} = {acc:.3f}", flush=True)

res = {"cats": CATS, "n_per_cat": N_PER_CAT, "ts": TS,
       "total_right": sum(r["ok"] for r in all_rows),
       "total": len(all_rows), "rows": all_rows}
json.dump(res, open(f"{RESULTS}/bfcl_call_{TS}.json", "w"))
print(f"[bfcl] TOTAL {res['total_right']}/{res['total']} "
      f"receipt bfcl_call_{TS}.json", flush=True)
