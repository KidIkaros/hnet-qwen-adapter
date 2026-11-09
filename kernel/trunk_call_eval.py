"""Trunk call-synthesis eval on ToolACE — the H-Net replacement comparison.

Runs the call_synth closed-set pipeline (name -> schema arg names ->
per-arg-conditioned extractive values) but scored by the TRUNK's own
prefix logprobs instead of the 2B byte-scorer. Same pinned slice and
same score_emission labels => direct comparability with call_synth
receipts (gold_match 3/40, name_acc ~49%, args_valid 36/39).

    colab exec -s hrun9 -f kernel/trunk_call_eval.py
    env: N_ITEMS (40), VERIFY_SKIP (2500 = pinned slice), SKIP (alt slice),
         MODEL_PATH ("" = resident model globals)
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
from src.probe_common import RESULTS, stream_toolace                # noqa: E402
from src.emit_eval import score_emission, tool_schemas              # noqa: E402

N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
SKIP = int(os.environ.get("SKIP", str(VSKIP)))
MODEL_PATH = os.environ.get("MODEL_PATH", "")
TS = int(time.time())
_g = globals()

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
print(f"[trunkcall] model ready skip={SKIP} n={N_ITEMS}", flush=True)


@torch.no_grad()
def prefix_lp(ctx_ids, cand_ids):
    ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


def score_text(ctx_ids, text):
    cand = tok(text, add_special_tokens=False)["input_ids"]
    if not cand:
        return -1e9
    return prefix_lp(ctx_ids, cand) / len(cand)


def value_candidates(user):
    cands = set()
    for m in re.finditer(r'"([^"]{1,60})"|\'([^\']{1,60})\'', user):
        cands.add(m.group(1) or m.group(2))
    for m in re.finditer(r"-?\d+\.?\d*", user):
        v = m.group(0)
        cands.add(v)
        if "." not in v:
            cands.add(v + ".0")
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


def synth(item):
    """Closed-set call on the trunk. Returns emitted call text or None."""
    ctx_s = f"{item['sys']}\nUSER: {item['user']}\nASSISTANT: ["
    ctx = tok(ctx_s, return_tensors="pt").input_ids[0].tolist()
    tools = item["tools"]
    sc = {n: prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                       ["input_ids"]) for n in tools + ["none"]}
    pred = max(sc, key=sc.get)
    name_scores = sc
    if pred == "none" or pred not in tools:
        return None, name_scores, pred
    schemas = tool_schemas(item["sys"])
    spec = schemas.get(pred)
    if not spec:
        return None, name_scores, pred
    vals = value_candidates(item["user"])
    parts, prefix_args = [], ""
    for arg in spec:
        sub_ctx = tok(ctx_s + f"{pred}({prefix_args}{arg}=\"",
                      return_tensors="pt").input_ids[0].tolist()
        best, bs = None, -1e18
        for v in vals:
            s = score_text(sub_ctx, v + '"')
            if s > bs:
                best, bs = v, s
        if best is None:
            continue
        parts.append(f'{arg}="{best}"')
        prefix_args += f'{arg}="{best}", '
    emit = f"[{pred}({', '.join(parts)})]"
    return emit, name_scores, pred


items = stream_toolace(SKIP, N_ITEMS)
rows, agg = [], {"parse_ok": 0, "tool_valid": 0, "args_valid": 0,
                 "gold_match": 0, "n": 0, "abstained": 0, "loops": 0}
decl = {"tool_valid": 0, "args_valid": 0, "gold_match": 0, "n": 0}
name_right = name_ans = 0
for i, it in enumerate(items):
    emit, nsc, pred = synth(it)
    if emit is None:
        agg["abstained"] += 1
        v = score_emission(b"", tools=it["tools"], sys_text=it["sys"],
                           gold_name=it.get("gold"))
        v["emit"] = ""
    else:
        v = score_emission(emit.encode("latin-1"), tools=it["tools"],
                           sys_text=it["sys"], gold_name=it.get("gold"),
                           gold_call=it.get("gold_full"))
        v["emit"] = emit
    for k in ("parse_ok", "tool_valid", "args_valid", "gold_match"):
        if v.get(k):
            agg[k] += 1
    if v.get("loops") or v.get("phenotype"):
        agg["loops"] += 1
    declared = it["gold"] in it["tools"]
    if declared:
        decl["n"] += 1
        for k in ("tool_valid", "args_valid", "gold_match"):
            if v.get(k):
                decl[k] += 1
        if emit is not None:
            name_ans += 1
            name_right += pred == it["gold"]
    agg["n"] += 1
    rows.append({"i": i, "gold": it.get("gold"), "pred": pred,
                 "declared": declared, **{k: v.get(k) for k in
                 ("parse_ok", "tool_valid", "args_valid", "gold_match",
                  "emit", "phenotype")}})
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(items)} gm={agg['gold_match']} "
              f"tv={agg['tool_valid']}", flush=True)

res = {"skip": SKIP, "n": N_ITEMS, "ts": TS, "agg": agg,
       "declared": decl, "name_acc_answered": name_right,
       "name_answered": name_ans, "rows": rows}
json.dump(res, open(f"{RESULTS}/trunk_call_{TS}.json", "w"))
print(f"[trunkcall] {json.dumps(agg)} decl={decl} "
      f"name_acc={name_right}/{name_ans}", flush=True)
print(f"[trunkcall] receipt trunk_call_{TS}.json", flush=True)
