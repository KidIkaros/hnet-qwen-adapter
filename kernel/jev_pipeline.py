"""Jev pipeline — end-to-end three-tier tool call:
  S1 encoder (cosine user vs tool name+desc) -> margin gate -> escalate to
  S2 byte-scorer when margin < TAU -> pick -> emit [name(args)] via H-Net.
Also scores undeclared-gold items to measure Noul behavior.

    colab exec -s hrun6 -f kernel/jev_pipeline.py
    env: ENCODER, TAU (0.01), N_ITEMS (60), SKIP (1224), GEN_LEN (300)
"""
import os, sys, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import re, json, time                                           # noqa: E402
from datasets import load_dataset                                # noqa: E402
from transformers import AutoModel, AutoTokenizer                # noqa: E402
from src.probe_common import (RESULTS, build_model, cand_score,  # noqa: E402
                              forward_bp, load_ckpt)

ENC = os.environ.get("ENCODER", "BAAI/bge-small-en-v1.5")
TAU = float(os.environ.get("TAU", "0.01"))
NOUL_TAU = float(os.environ.get("NOUL_TAU", "0.55"))
N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
GEN_LEN = int(os.environ.get("GEN_LEN", "300"))
SKIP = int(os.environ.get("SKIP", "1224"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(
    r'"name"\s*:\s*"([\w]+)"\s*,\s*"description"\s*:\s*"([^"]{0,300})')
ARG_RE = re.compile(r'"(properties|parameters)"\s*:\s*\{\s*"([\w]+)"')

trunk, hn, meta = build_model()
load_ckpt(hn)
tok = AutoTokenizer.from_pretrained(ENC)
enc = AutoModel.from_pretrained(ENC, dtype=torch.float32).cuda()


@torch.no_grad()
def embed(texts):
    t = tok(texts, padding=True, truncation=True, max_length=256,
            return_tensors="pt").to("cuda")
    h = enc(**t).last_hidden_state
    m = t["attention_mask"].unsqueeze(-1)
    v = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(v, dim=-1)


@torch.no_grad()
def greedy_from(ids, max_new):
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-2:] == list(b"]\n") or ids[-6:] == list(b"\n\nUSER"):
            break
    return ids


ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv if c.get("from") == "assistant"), None)
    gold = CALL_RE.findall(a) if a else []
    tools = TOOL_RE.findall(r["system"])
    if u and gold and tools:
        items.append({"sys": r["system"], "user": u,
                      "gold": gold[0], "tools": tools[:24]})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items | S1={ENC} tau={TAU}", flush=True)

pick1 = esc_n = closed_n = noul_hit = declared = noul_n = 0
for k, it in enumerate(items):
    names = [t[0] for t in it["tools"]]
    ue = embed([it["user"]])
    te = embed([f"{n}: {d}" for n, d in it["tools"]])
    sims = (ue * te).sum(1).cpu().numpy()
    order = np.argsort(-sims)
    margin = (sims[order[0]] - sims[order[1]]) if len(sims) > 1 else 9.9
    gold_in = it["gold"] in names
    declared += gold_in

    if sims[order[0]] < NOUL_TAU:        # abstention floor: nothing fits
        noul_n += 1
        print(f"  item {k:2d} [NOUL] top-sim {sims[order[0]]:.3f} "
              f"gold={it['gold'][:20]} declared={gold_in}", flush=True)
        continue
    if margin >= TAU:                    # S1 answers
        pred, tier = names[order[0]], "S1"
    else:                                # S2 byte-scorer (schema-arg probe)
        esc_n += 1
        pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
        arg_names = dict((m.group(1), m.group(2))
                         for m in map(ARG_RE.search,
                                      it["sys"].split('"name"')) if m)
        def s2cand(t):
            a = arg_names.get(t)
            return f"[{t}({a}" if a else f"[{t}("
        sc = sorted(((t, cand_score(hn, trunk, pid,
                     list(s2cand(t).encode()))) for t in names + ["none"]),
                    key=lambda s: -s[1])
        pred, tier = sc[0][0], "S2"
        if pred == "none":
            print(f"  item {k:2d} [NOUL] abstained (gold={it['gold'][:20]})",
                  flush=True)
            pick1 += gold_in and False
            continue
    pick1 += gold_in and pred == it["gold"]

    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    ids = greedy_from(pid + list(f"[{pred}(".encode()), GEN_LEN)
    gen = bytes(ids[len(pid):]).decode("utf-8", "replace")
    closed = "]" in gen
    closed_n += closed
    if not gold_in and closed:
        noul_hit += 1                     # emitted call though gold undeclared
    print(f"  item {k:2d} [{tier}] gold={it['gold'][:20]:20s} "
          f"pred={pred[:20]:20s} margin={margin:.3f} "
          f"call={gen[:55]!r}", flush=True)

n = len(items)
res = {"tau": TAU, "noul_tau": NOUL_TAU, "n": n, "declared": declared,
       "pick1": pick1, "escalated": esc_n, "closed": closed_n,
       "noul": noul_n, "noul_emit_on_undeclared": noul_hit}
json.dump(res, open(f"{RESULTS}/pipeline_{int(time.time())}.json", "w"))
print(f"[pipeline] pick@1 {pick1}/{declared} declared ({n} total) | "
      f"S2 esc {esc_n} | NOUL {noul_n} | closed {closed_n}/{n} | "
      f"undeclared-emitted {noul_hit}", flush=True)
print("[pipeline] complete", flush=True)
