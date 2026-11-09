"""BFCL v3 name-selection eval on the quantized 27B trunk — Phase A.

Recognized-benchmark arm: BFCL_v3_{simple,multiple} from
gorilla-llm/Berkeley-Function-Calling-Leaderboard, scored with the same
prefix-logprob protocol as the pinned exams — but the ITEMS are the
recognized benchmark's, not our slice. Answers: does the compressed
trunk select the gold function on BFCL-distribution prompts?

Phase B (args+values -> official ast_checker) is a follow-up; this
isolates selection, the component compression could plausibly harm.

    colab exec -s hrun9 -f kernel/bfcl_name_eval.py
    env: N_PER_CAT (100), CATS (simple,multiple), MODEL_PATH ("" = resident
         model globals; else load path e.g. saved quant dir)
"""
import os, sys, subprocess, json, time

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

# ---- data: BFCL v3 JSONL from the official HF dataset ----
from huggingface_hub import hf_hub_download                    # noqa: E402
DS = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def _load_jsonl(repo_file):
    p = hf_hub_download(DS, repo_file, repo_type="dataset")
    return [json.loads(l) for l in open(p)]


# ---- model: resident globals, or load MODEL_PATH/base ----
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
    ids = torch.tensor([ctx_ids + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    lp = torch.log_softmax(logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


def ctx_for(item):
    """Serialize BFCL functions + user turn into a scoring context."""
    funcs = item["function"]
    user = item["question"][0][-1]["content"]  # last msg of first turn
    schema = json.dumps(
        [{"name": f["name"], "description": f.get("description", ""),
          "parameters": f["parameters"]} for f in funcs])
    return (f"You are a function-calling assistant. Available functions:\n"
            f"{schema}\n\nUser: {user}\nAssistant: ["), user


all_rows = []
for cat in CATS:
    req = _load_jsonl(f"BFCL_v3_{cat}.json")
    ans = {a["id"]: a["ground_truth"]
           for a in _load_jsonl(f"possible_answer/BFCL_v3_{cat}.json")}
    req = req[:N_PER_CAT]
    right = decl = 0
    for i, it in enumerate(req):
        gold_names = list(ans[it["id"]][0].keys())
        ctx_s, _ = ctx_for(it)
        ctx = tok(ctx_s, return_tensors="pt").input_ids[0].tolist()
        cands = [f["name"] for f in it["function"]] + ["none"]
        sc = {n: prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                           ["input_ids"]) for n in cands}
        pred = max(sc, key=sc.get)
        ok = pred in gold_names
        right += ok
        decl += 1
        all_rows.append({"cat": cat, "id": it["id"], "pred": pred,
                         "gold": gold_names, "ok": ok,
                         "n_cands": len(cands)})
        if (i + 1) % 25 == 0:
            print(f"  [{cat}] {i + 1}/{len(req)} acc={right / decl:.3f}",
                  flush=True)
    print(f"[{cat}] name-acc {right}/{decl} = {right / decl:.3f}",
          flush=True)

res = {"cats": CATS, "n_per_cat": N_PER_CAT, "ts": TS,
       "by_cat": {c: {"right": sum(r["ok"] for r in all_rows
                                   if r["cat"] == c),
                      "n": sum(1 for r in all_rows if r["cat"] == c)}
                  for c in CATS},
       "total_right": sum(r["ok"] for r in all_rows),
       "total": len(all_rows), "rows": all_rows}
json.dump(res, open(f"{RESULTS}/bfcl_name_{TS}.json", "w"))
print(f"[bfcl] TOTAL {res['total_right']}/{res['total']} "
      f"receipt bfcl_name_{TS}.json", flush=True)
