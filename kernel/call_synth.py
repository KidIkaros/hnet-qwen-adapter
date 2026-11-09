"""Constrained call synthesis — the schema-structured alternative to
free-generation, motivated by emit_score decomposition:

  free-gen:      tool_valid 6/40   args_valid 4/40   gold_match 1/40
  name-forced:  tool_valid 27/40  args_valid 6/40   gold_match 1/40

Every level is a closed-set selection scored by the byte-LM
(cand_score), never free-generated:

  1. tool name  — argmax cand_score over declared names + "none("
  2. arg names  — sequential argmax over remaining schema args + ")"
  3. arg values — argmax cand_score over user-query spans (extractive)

Assembled call is verified by score_emission (the locked exam).
Success metric: args_valid / gold_match on the pinned slice vs the
free-gen baselines above.

    colab exec -s hrun8 -f kernel/call_synth.py
    env: VERIFY_SKIP (2500), N_ITEMS (40), MAX_ARGS (4), TOP_SPANS (48)
"""
import os, sys, subprocess, json, time, re

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"FAILED {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "fetch", "-q", "origin")
    # drop untracked files that upstream now tracks (committed-back results)
    up = set(sh("git", "-C", REPO, "ls-tree", "-r", "--name-only",
                "origin/main").split())
    for root, _, fs in os.walk(REPO):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, REPO)
            if rel in up and "??" in sh(
                    "git", "-C", REPO, "status", "--porcelain", "--", rel):
                os.remove(p)
    sh("git", "-C", REPO, "merge", "-q", "origin/main")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)
for _m in [k for k in sys.modules if k.startswith("src.")]:
    del sys.modules[_m]                     # stale-module guard (colab kernel)

import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, cand_score,    # noqa: E402
                              load_ckpt, stream_toolace)
from src.emit_eval import score_emission, tool_schemas             # noqa: E402

VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
MAX_ARGS = int(os.environ.get("MAX_ARGS", "4"))
TOP_SPANS = int(os.environ.get("TOP_SPANS", "48"))

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

items = stream_toolace(VSKIP, N_ITEMS)
_WS = re.compile(r"\s+")
_EDGE = " \t\n\"'`,.;:!?()[]{}"


def value_spans(user: str):
    """Extractive candidates: word n-grams (n<=4) from the user query,
    edge-punctuation stripped, quotes/newlines excluded."""
    toks = [t.strip(_EDGE) for t in _WS.split(user)]
    toks = [t for t in toks if t]
    spans = []
    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            s = " ".join(toks[i:i + n])
            if '"' not in s and "\n" not in s:
                spans.append(s)
    # dedupe preserving order, cap
    seen, out = set(), []
    for s in spans:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:TOP_SPANS]


_BARE_OK = re.compile(r"^[\w./:-]+$")


def argmax_score(pid, cands):
    """{cand_str: mean_logprob}; cand bytes appended after prompt ids."""
    return {c: cand_score(hn, trunk, pid, list(c.encode()))
            for c in cands}


rows = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())

    # ---- phase A: tool name selection (closed set + abstain) ----
    names = list(it["tools"]) + ["none"]
    nsc = argmax_score(pid, [f"{n}(" for n in names])
    pred = max(nsc, key=nsc.get)
    pred = pred[:-1] if pred.endswith("(") else pred
    if pred == "none":
        v = score_emission(b"", tools=it["tools"], sys_text=it["sys"],
                           gold_name=it["gold"],
                           gold_call=it.get("gold_full"))
        v.update({"k": k, "gold": it["gold"], "pred": "none",
                  "name_scores": {n: round(s, 3)
                                  for n, s in nsc.items()}})
        rows.append(v)
        print(f"  item {k:2d} [ABSTAIN] gold={str(it['gold'])[:16]}",
              flush=True)
        continue

    # ---- phase B/C: constrained arg synthesis ----
    schemas = tool_schemas(it["sys"])
    spec = schemas.get(pred)
    remaining = list(spec) if spec else []
    spans = value_spans(it["user"])
    emitted = f"{pred}("
    call_ids = pid + list(emitted.encode())

    for _ in range(MAX_ARGS):
        comma = "" if emitted.endswith("(") else ", "
        arg_cands = [f"{comma}{a}=" for a in remaining] + [")"]
        sc = argmax_score(call_ids, arg_cands)
        pick = max(sc, key=sc.get)
        if pick == ")":
            emitted += ")"
            break
        aname = pick[len(comma):-1]
        remaining.remove(aname)
        emitted += f"{comma}{aname}="
        call_ids = pid + list(emitted.encode())

        # value: extractive span selection — quoted form for all spans,
        # plus bare form for digit-bearing spans (gold emits id=123 unquoted)
        vcands = [f'"{s}"' for s in spans]
        vcands += [s for s in spans
                   if _BARE_OK.match(s) and any(c.isdigit() for c in s)]
        vsc = argmax_score(call_ids, vcands) if vcands else {}
        vpick = max(vsc, key=vsc.get) if vsc else '""'
        emitted += vpick
        call_ids = pid + list(emitted.encode())
    else:
        emitted += ")"                          # MAX_ARGS hit, force close
    emitted += "]"

    v = score_emission(emitted.encode(), tools=it["tools"],
                       sys_text=it["sys"], gold_name=it["gold"],
                       gold_call=it.get("gold_full"))
    v.update({"k": k, "gold": it["gold"], "pred": pred,
              "emit": emitted[:200],
              "name_scores": {n: round(s, 3)
                              for n, s in nsc.items()}})
    rows.append(v)
    print(f"  item {k:2d} pred={pred[:20]:20s} gold={str(it['gold'])[:14]:14s} "
          f"parse={v['parse_ok']} tv={v['tool_valid']} av={v['args_valid']} "
          f"gm={v['gold_match']} | {emitted[:70]}", flush=True)

n = len(rows)
answered = [r for r in rows if r.get("pred") != "none"]
agg = {key: sum(1 for r in answered if r[key] is True)
       for key in ("parse_ok", "tool_valid", "args_valid", "gold_match",
                   "closed")}
agg.update({"n": n, "abstained": n - len(answered),
            "name_acc_answered": sum(
                1 for r in answered if r.get("pred") == r["gold"]),
            "loops": sum(1 for r in rows if r["phenotype"])})
decl = [r for i, r in enumerate(rows)
        if i in {0,1,2,5,6,7,9,10,11,12,13,15,16,17,19,21,23,26,28,29,30,31,
                 34,35,37,38,39}]
agg["declared"] = {key: sum(1 for r in decl if r[key] is True)
                   for key in ("tool_valid", "args_valid", "gold_match")}
json.dump({"skip": VSKIP, "n": n, "agg": agg, "rows": rows},
          open(f"{RESULTS}/call_synth_{int(time.time())}.json", "w"))
print(f"[synth] {agg}", flush=True)
print("[synth] complete", flush=True)
