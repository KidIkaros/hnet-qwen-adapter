"""Big-slice name-selection exam — saturation check for compress_trunk.

The compress_trunk verdict (27/27 -> 27/27 @ 4bit) was on the pinned
40-item slice, which the teacher ALSO maxed — possible ceiling effect.
This driver measures the same prefix-scoring protocol on a fresh
N-item slice, twice:

  phase 1 "resident": whatever model is already in kernel globals
                       (on hrun9 that is the GPTQ-4bit quantized 27B)
  phase 2 "fresh":    evict + reload bf16 -> base reference, SAME items

Receipts are written after EACH phase so a session death cannot lose
the first result. Items come from stream_toolace(SKIP, N) — deterministic,
so both phases see identical prompts/candidates.

    colab exec -s hrun9 -f kernel/big_exam.py
    env: SKIP (0), N (500), MODEL (same 27B), RELOAD_BASE (1)
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
from src.probe_common import RESULTS, stream_toolace                # noqa: E402

SKIP = int(os.environ.get("SKIP", "0"))
N = int(os.environ.get("N", "500"))
RELOAD_BASE = os.environ.get("RELOAD_BASE", "1") == "1"
MODE = os.environ.get("MODE", "both")  # resident | base | both
MODEL = os.environ.get(
    "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")
TS = int(time.time())

_g = globals()

# Whitelist eviction: anything not needed for phase-2 dies. compress_trunk
# leaves `flat`/`blocks`/`targets` holding all 256 Linear modules — an
# allowlist-pop missed them once and pinned ~35GB, OOMing the reload.
_KEEP = {"items", "_exam", "_evict", "_load", "sh", "stream_toolace",
         "RESULTS", "torch", "json", "os", "sys", "subprocess", "time",
         "gc", "SKIP", "N", "RELOAD_BASE", "MODE", "MODEL", "TS", "_g"}


def _evict():
    for _k in list(_g.keys()):
        if _k not in _KEEP:
            _g.pop(_k, None)
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _load():
    from transformers import (AutoTokenizer,          # noqa: E402
                              AutoModelForCausalLM)
    _g["tok"] = AutoTokenizer.from_pretrained(MODEL)
    _g["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "72GiB", "cpu": "70GiB"})
    _g["model"].eval()


@torch.no_grad()
def _exam(items):
    """Pinned name-selection protocol (sum logprob, tools+none)."""
    tok, model = _g["tok"], _g["model"]
    right = none_n = decl = decl_right = 0
    rows = []
    for i, it in enumerate(items):
        ctx = tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                  return_tensors="pt").input_ids[0].tolist()
        sc = {}
        for n in it["tools"] + ["none"]:
            cand = tok(f"{n}(", add_special_tokens=False)["input_ids"]
            ids = torch.tensor([ctx + cand], device="cuda")
            logits = model(ids).logits[0]
            s = len(ctx) - 1
            lp = torch.log_softmax(logits[s:s + len(cand)].float(), -1)
            sc[n] = float(sum(lp[j, t] for j, t in enumerate(cand)))
        pred = max(sc, key=sc.get)
        declared = it["gold"] in it["tools"]
        decl += declared
        decl_right += declared and pred == it["gold"]
        right += pred == it["gold"]
        none_n += pred == "none"
        rows.append({"i": i, "pred": pred, "gold": it["gold"],
                     "declared": declared, "abstain": pred == "none"})
        if (i + 1) % 50 == 0:
            print(f"  exam {i + 1}/{len(items)}", flush=True)
    return {"right": right, "declared": decl, "decl_right": decl_right,
            "none": none_n, "n": len(items), "rows": rows}


items = stream_toolace(SKIP, N)
print(f"[exam] {len(items)} items skip={SKIP} mode={MODE}", flush=True)

res_q = res_b = None

# ---- phase 1: resident model (quantized on hrun9) ----
if MODE in ("resident", "both"):
    if _g.get("model") is None or _g.get("tok") is None:
        print("[exam] no resident model — loading bf16", flush=True)
        _load()
    res_q = _exam(items)
    print(f"[resident] { {k: v for k, v in res_q.items() if k != 'rows'} }",
          flush=True)
    json.dump({"phase": "resident", "skip": SKIP, "n": N,
               **{k: v for k, v in res_q.items() if k != "rows"},
               "rows": res_q["rows"]},
              open(f"{RESULTS}/big_exam_{TS}_resident.json", "w"))
    print(f"[resident] receipt big_exam_{TS}_resident.json", flush=True)

# ---- phase 2: fresh bf16 base on identical items ----
if MODE in ("base", "both") and (RELOAD_BASE or MODE == "base"):
    print("[exam] evicting + reloading fresh bf16 for base", flush=True)
    _evict()
    _load()
    res_b = _exam(items)
    print(f"[fresh] { {k: v for k, v in res_b.items() if k != 'rows'} }",
          flush=True)
    json.dump({"phase": "fresh-bf16", "skip": SKIP, "n": N,
               **{k: v for k, v in res_b.items() if k != "rows"},
               "rows": res_b["rows"]},
              open(f"{RESULTS}/big_exam_{TS}_fresh.json", "w"))
    print(f"[fresh] receipt big_exam_{TS}_fresh.json", flush=True)

if res_q is not None and res_b is not None:
    json.dump({"resident": {k: v for k, v in res_q.items() if k != "rows"},
               "fresh": {k: v for k, v in res_b.items() if k != "rows"},
               "skip": SKIP, "n": N},
              open(f"{RESULTS}/big_exam_{TS}.json", "w"))
print("[big_exam] complete", flush=True)
