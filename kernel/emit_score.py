"""Correctness re-scoring of emitted calls on the pinned slice.

For each item: greedy emit under MODE, then score_emission verdict
(parse_ok / tool_valid / args_valid / gold_match + legacy loop fields).
Answers the question loop-count couldn't: are the "closed" calls
actually correct?

    colab exec -s hrun8 -f kernel/emit_score.py
    env: MODE (base|constrain), VERIFY_SKIP (2500), N_ITEMS (40),
         GEN_LEN (300)
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
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)
for _m in [k for k in sys.modules if k.startswith("src.")]:
    del sys.modules[_m]                     # stale-module guard (colab kernel)

import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, dup_class,     # noqa: E402
                              forward_bp, load_ckpt, stream_toolace)
from src.emit_eval import score_emission                           # noqa: E402

MODE = os.environ.get("MODE", "base")
VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

items = stream_toolace(VSKIP, N_ITEMS)
prompts = [f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode()
           for it in items]


@torch.no_grad()
def emit(pid, constrain, force=b""):
    ids = list(pid) + list(force)
    n0 = len(pid)                     # tail includes forced prefix
    for _ in range(GEN_LEN):
        bp = forward_bp(hn, trunk, ids)
        logits = bp[0, -1].clone()
        if constrain and len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
            logits[ids[-1]] = -1e9                      # run-mask
        ids.append(int(logits.argmax()))
        tail = bytes(ids[n0:])
        onset, _ = dup_class(tail)
        if constrain and onset is not None and onset < len(tail):
            ids = ids[:n0 + onset]
            if b"(" in tail[:onset] and b"]" not in tail[:onset]:
                ids.extend(b"]")
            break
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return bytes(ids[n0:])


rows = []
for k, (it, pid) in enumerate(zip(items, prompts)):
    force = (f"{it['gold']}(".encode()
             if MODE == "goldname" and it["gold"] else b"")
    t = emit(pid, constrain=(MODE == "constrain"), force=force)
    v = score_emission(t, tools=it["tools"], sys_text=it["sys"],
                       gold_name=it["gold"],
                       gold_call=it.get("gold_full") or it["gold_call"])
    v.update({"k": k, "gold": it["gold"], "emit": t[:200].decode(
        "latin-1", "replace")})
    rows.append(v)
    print(f"  item {k:2d} gold={str(it['gold'])[:16]:16s} "
          f"tool={str(v['tool'])[:16]:16s} parse={v['parse_ok']} "
          f"tv={v['tool_valid']} av={v['args_valid']} "
          f"gm={v['gold_match']} ph={v['phenotype']}", flush=True)

n = len(rows)
agg = {key: sum(1 for r in rows if r[key] is True)
       for key in ("parse_ok", "tool_valid", "args_valid", "gold_match",
                   "closed")}
agg["loops"] = sum(1 for r in rows if r["phenotype"])
json.dump({"mode": MODE, "skip": VSKIP, "n": n, "agg": agg, "rows": rows},
          open(f"{RESULTS}/emit_score_{MODE}_{int(time.time())}.json", "w"))
print(f"[score:{MODE}] n={n} {agg}", flush=True)
print("[score] complete", flush=True)
