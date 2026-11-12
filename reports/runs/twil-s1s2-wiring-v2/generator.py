#!/usr/bin/env python3
"""Build the Kaggle wiring-v2 kernel: S2-as-verifier with Spark-X2.6-4B.

Per docs/SPEC_TWIL_S1S2_WIRING_V2.md. S2 never generates a tool call —
it returns VERDICT: CONFIRM | SWITCH <name> | ABSTAIN over S1's pick
and the candidate set. Host: SC117/Spark-X2.6-4B-abliterated-FIT-GGUF
(BALANCED Q4_K_M) via llama-cpp-python CUDA. Small footprint ->
StartLux stays on cuda:0, no CPU fallback expected.
"""
import json

OUT = "/tmp/twil_s1s2_v2.ipynb"
REPO = "/home/ikaaros/Coding/Gold/hnet-qwen-adapter"
SRC_FILES = ["src/__init__.py", "src/glial.py"]
GGUF_REPO = "SC117/Spark-X2.6-4B-abliterated-FIT-GGUF"
GGUF_FILE = "Spark-X2.6-4B-abliterated-FIT-BALANCED-3G-Q4_K_M.gguf"


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.split("\n")]}


def code_cell(text):
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in text.split("\n")]}


def src_writer_cell():
    import base64, os
    parts = ["import os, base64\n",
             "os.makedirs('/kaggle/working/src', exist_ok=True)\n"]
    for rel in SRC_FILES:
        content = open(os.path.join(REPO, rel), "rb").read()
        b64 = base64.b64encode(content).decode()
        parts.append(
            f"open('/kaggle/working/src/{os.path.basename(rel)}','wb')"
            f".write(base64.b64decode('{b64}'))\n")
    parts.append("import sys; sys.path.insert(0, '/kaggle/working')\n")
    parts.append("from src.glial import glial_gate\n")
    parts.append("print('src bundle OK')\n")
    return code_cell("".join(parts))


cells = [
    md_cell("""# Wiring v2 — S1 → glial gate → Spark-4B VERIFIER (2xT4)

Per `docs/SPEC_TWIL_S1S2_WIRING_V2.md`. v1 showed escalate-to-S2
re-decide destroys accuracy (0/12 recovery). v2 flips the role: the
S2 host judges S1's pick — CONFIRM / SWITCH / ABSTAIN — never
generates a call. Host: Spark-X2.6-4B GGUF Q4_K_M via llama.cpp."""),

    code_cell("""# Cell 1: deps — torch check (StartLux), llama-cpp-python CUDA
# build (~15min compile, do it first), transformers/datasets.
import subprocess, sys, os, json, re, time

v = subprocess.run([sys.executable, '-c',
    'import torch; print(torch.__version__)'],
    capture_output=True, text=True).stdout.strip()
print('torch', v)
if not v or tuple(int(x) for x in v.split('.')[:2]) < (2, 8):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'torch==2.8.0', '--index-url',
        'https://download.pytorch.org/whl/cu121'], check=True)
# llama-cpp-python — CPU path only. v2's CUDA build failed (nvcc/toolkit
# mismatch on the Kaggle image); verifier is a light task so CPU Q4 is
# fine (~10-20 tok/s -> ~15s/verdict). Try prebuilt wheel first.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'cmake', 'ninja'], check=False)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'llama-cpp-python', '--only-binary', ':all:'],
    capture_output=True)
if r.returncode != 0:
    print('no prebuilt wheel — source build (CPU)', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'llama-cpp-python', '--no-cache-dir'], check=False)
else:
    print('llama-cpp-python wheel OK', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=5.8', 'accelerate', 'datasets', 'huggingface_hub'],
    check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
for i in range(2):
    print(f'cuda:{i}', torch.cuda.get_device_name(i))
try:
    import llama_cpp
    print('llama_cpp', llama_cpp.__version__, flush=True)
except Exception as e:
    print('llama_cpp IMPORT FAIL:', e, flush=True)
"""),

    src_writer_cell(),

    code_cell("""# Cell 2: data — ToolACE qualifying items 460-559 (fresh slice;
# v1 wiring used 360-459). Keeps full tool JSON for verifier prompt.
from datasets import load_dataset

SKIP = 2500
NEED = 560
N_TOOLS_MAX = 24

def norm_name(s):
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

def parse_tools(sys_text):
    i = sys_text.find('[{')
    if i < 0:
        return []
    try:
        arr, _ = json.JSONDecoder().raw_decode(sys_text[i:])
    except Exception:
        return []
    out = []
    for t in arr:
        n = t.get('name')
        if n:
            out.append({'name': n,
                        'description': (t.get('description') or '')[:400],
                        'parameters': t.get('parameters') or
                        {'type': 'object', 'properties': {}}})
    return out

def parse_gold(a):
    m = re.search(r'\\[([^\\]]*)\\]', a or '')
    if not m:
        return None
    calls = re.findall(r'([A-Za-z][A-Za-z0-9_ ]*?)\\s*\\(', m.group(1))
    return calls[0].strip() if calls else None

ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
all_items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv
              if c.get("from") == "assistant"), None)
    gold = parse_gold(a) if a else None
    tools = parse_tools(r["system"])
    if u and gold and tools:
        gnorm = norm_name(gold)
        gold_tool, answerable = None, False
        for t in tools:
            if norm_name(t['name']) == gnorm:
                gold_tool, answerable = t['name'], True
                break
        sel = tools[:N_TOOLS_MAX]
        if answerable and gold_tool not in {t['name'] for t in sel}:
            gt = next(t for t in tools if t['name'] == gold_tool)
            sel = tools[:N_TOOLS_MAX - 1] + [gt]
        all_items.append({"sys": r["system"], "user": u,
                          "gold": gold, "gold_tool": gold_tool or gold,
                          "answerable": answerable, "tools": sel})
    if len(all_items) >= NEED:
        break

items = all_items[460:560]
assert len(items) >= 60, f"only {len(items)} items in slice"
n_ans = sum(it["answerable"] for it in items)
json.dump(items, open("/kaggle/working/items.json", "w"))
print(f"[data] {len(items)} items (qualifying 460-559); "
      f"answerable {n_ans}/{len(items)}", flush=True)
"""),

    code_cell("""# Cell 3: Leg 0 — co-load StartLux (cuda:0) + Spark GGUF.
# Spark ~3GB -> StartLux cuda:0 coexists; llama.cpp grabs a GPU itself.
from huggingface_hub import snapshot_download, hf_hub_download

SL_CPU = False
sl = None
try:
    MDIR = "/kaggle/working/StartLux-0.8B"
    if not os.path.exists(os.path.join(MDIR, "model.safetensors.index.json")):
        print("downloading StartLux...", flush=True)
        snapshot_download("startlux-models/StartLux-Decision-v1-0.8B",
                          local_dir=MDIR)
    req = os.path.join(MDIR, "requirements.txt")
    if os.path.exists(req):
        BLOCK = ("causal-conv1d", "flash-linear-attention", "mlx-lm",
                 "torch", "torchvision")
        kept = [l for l in open(req) if l.strip()
                and not l.strip().startswith("#")
                and not any(b in l for b in BLOCK)]
        open("/kaggle/working/req_filtered.txt", "w").write("\\n".join(kept))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                        '/kaggle/working/req_filtered.txt'], check=True)
    sys.path.insert(0, MDIR)
    from startlux_decision import StartLuxDecision
    sl = StartLuxDecision(MDIR, device="cuda:0")
    print(f"[startlux] cuda:0 fast_kernels="
          f"{getattr(sl, 'fast_kernels_active', '?')}", flush=True)
except Exception as e:
    print(f"[startlux] cuda:0 failed ({type(e).__name__}: "
          f"{str(e)[:120]}) -> CPU", flush=True)
    try:
        sl = StartLuxDecision(MDIR, device="cpu")
        SL_CPU = True
    except Exception as e2:
        print(f"[startlux] UNAVAILABLE: {type(e2).__name__}", flush=True)
        sl = None

spark = None
try:
    from llama_cpp import Llama
    gpath = hf_hub_download(
        "SC117/Spark-X2.6-4B-abliterated-FIT-GGUF",
        "Spark-X2.6-4B-abliterated-FIT-BALANCED-3G-Q4_K_M.gguf")
    print(f"[spark] gguf {gpath}", flush=True)
    spark = Llama(model_path=gpath, n_ctx=8192, n_gpu_layers=0,
                  n_threads=8, verbose=False)
    print("[spark] loaded", flush=True)
except Exception as e:
    print(f"[spark] UNAVAILABLE: {type(e).__name__}: {str(e)[:150]}",
          flush=True)
    spark = None

for i in range(2):
    print(f"cuda:{i} reserved="
          f"{torch.cuda.memory_reserved(i)/1e9:.1f}GB", flush=True)
"""),

    code_cell("""# Cell 4: Leg 1 — S1 pass + glial gate on all items.
from src.glial import glial_gate

SL_ROWS, GATE = [], []
run_items = items[:40] if SL_CPU else items
for k, it in enumerate(run_items):
    crit = {t['name']: (t['description'] or t['name']) for t in it["tools"]}
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"]}
    try:
        ti = time.time()
        ans, _ = sl.decide(
            {"request": it["user"][:1500], "system": it["sys"][:2000]},
            {"tool": {"type": "choice",
                      "instructions": "Which function should be called?",
                      "criteria": crit}})
        probs = ans["tool"]["probabilities"]
        pred = max(probs, key=probs.get)
        row.update({"pred": pred, "conf": probs[pred], "probs": probs,
                    "gen_s": round(time.time() - ti, 2)})
        v = glial_gate(probs)
        GATE.append({"i": k, "route": v.route, "abstain": v.abstain,
                     "forced": v.forced, "top": v.top,
                     "n_set": len(v.prediction_set),
                     "route_reason": v.route_reason})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        GATE.append({"i": k, "route": "error", "abstain": True,
                     "forced": False, "top": None, "n_set": 0,
                     "route_reason": row["error"]})
    SL_ROWS.append(row)
    if (k + 1) % 25 == 0:
        print(f"[s1] {k+1}/{len(run_items)}", flush=True)

json.dump({"s1": SL_ROWS, "gate": GATE},
          open("/kaggle/working/s1_gate.json", "w"))
from collections import Counter
print("[gate] routes:", Counter(g["route"] for g in GATE), flush=True)
print(f"[s1] mean gen_s="
      f"{sum(r.get('gen_s',0) for r in SL_ROWS)/max(len(SL_ROWS),1):.2f}",
      flush=True)
"""),

    code_cell("""# Cell 5: Leg 2 — Spark VERIFIER on review+escalate bands.
VERDICT_RE = re.compile(
    r"VERDICT:\\s*(CONFIRM|SWITCH\\s+(\\S+)|ABSTAIN)", re.I)

SYS_V = ("You are a verification layer for a tool-routing system. "
         "A fast classifier proposed one tool for a user request. "
         "Decide if the proposal is right. Reply with EXACTLY ONE line: "
         "VERDICT: CONFIRM   (proposal is correct) | "
         "VERDICT: SWITCH <tool-name>   (a different candidate is "
         "correct — name it exactly) | "
         "VERDICT: ABSTAIN   (no candidate fits / request unclear)")

def verifier_prompt(it, s1_pred, conf):
    cand = {t["name"]: (t["description"] or t["name"])
            for t in it["tools"]}
    user = (f"User request:\\n{it['user'][:2000]}\\n\\n"
            f"Candidate tools:\\n{json.dumps(cand, ensure_ascii=False)[:4500]}\\n\\n"
            f"Proposed tool: {s1_pred} (confidence {conf:.2f})\\n\\n"
            f"Is the proposed tool correct? One VERDICT line only.")
    return [{"role": "system", "content": SYS_V},
            {"role": "user", "content": user}]

def parse_verdict(text):
    m = VERDICT_RE.search(text or "")
    if not m:
        return {"verdict": "NONE", "switch": None}
    kind = m.group(1).upper()
    if kind.startswith("SWITCH"):
        return {"verdict": "SWITCH", "switch": m.group(2)}
    return {"verdict": kind, "switch": None}

V_ROWS = []
nonauto = [g["i"] for g in GATE
           if g["route"] in ("review", "escalate") and g.get("top")]
print(f"[verify] {len(nonauto)} non-auto items", flush=True)
if spark is None:
    print("[verify] spark unavailable — skipping", flush=True)
    nonauto = []
for j, k in enumerate(nonauto):
    it = run_items[k]
    s1p = SL_ROWS[k].get("pred")
    conf = SL_ROWS[k].get("conf", 0.0)
    row = {"i": k, "gold": it["gold_tool"],
           "answerable": it["answerable"], "s1_pred": s1p,
           "s1_route": GATE[k]["route"], "s1_conf": conf}
    try:
        msgs = verifier_prompt(it, s1p, conf)
        t0 = time.time()
        resp = spark.create_chat_completion(
            messages=msgs, max_tokens=256, temperature=0.0)
        dt = time.time() - t0
        text = resp["choices"][0]["message"].get("content") or ""
        vd = parse_verdict(text)
        # switch name must norm-match a real candidate, else ABSTAIN
        names = {norm_name(t["name"]): t["name"] for t in it["tools"]}
        sw = vd["switch"]
        sw_ok = sw and norm_name(sw) in names
        final = (s1p if vd["verdict"] == "CONFIRM"
                 else (names[norm_name(sw)] if sw_ok else None))
        row.update({"gen_s": round(dt, 1), "raw": text[-600:],
                    "verdict": vd["verdict"],
                    "switch_raw": sw, "switch_valid": bool(sw_ok),
                    "final_pred": final,
                    "final_ok": bool(final) and
                    norm_name(final) == norm_name(it["gold_tool"]),
                    "s1_ok": bool(s1p) and
                    norm_name(s1p) == norm_name(it["gold_tool"])})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    V_ROWS.append(row)
    if (j + 1) % 10 == 0:
        print(f"[verify] {j+1}/{len(nonauto)}", flush=True)

json.dump(V_ROWS, open("/kaggle/working/verify_rows.json", "w"))
print(f"[verify] done {len(V_ROWS)} rows", flush=True)
"""),

    code_cell("""# Cell 6: Leg 3 — metrics, policies, verdict, artifacts.
M = {}

def nmatch(a, b):
    return bool(a) and bool(b) and norm_name(a) == norm_name(b)

s1_ok = [r for r in SL_ROWS if "error" not in r]
M["n_items"] = len(run_items)
M["SL_CPU"] = SL_CPU
M["s1_acc_all"] = (sum(nmatch(r.get("pred"), r["gold"]) for r in s1_ok)
                   / max(len(s1_ok), 1))
ans_rows = [r for r in s1_ok if run_items[r["i"]]["answerable"]]
M["s1_acc_ans"] = (sum(nmatch(r.get("pred"), r["gold"]) for r in ans_rows)
                   / max(len(ans_rows), 1))
M["s1_mean_gen_s"] = sum(r.get("gen_s", 0) for r in s1_ok) / max(len(s1_ok), 1)

gate_by_i = {g["i"]: g for g in GATE}
for route in ("auto", "review", "escalate"):
    rows = [r for r in s1_ok if gate_by_i[r["i"]]["route"] == route]
    if rows:
        M[f"route_{route}_n"] = len(rows)
        M[f"route_{route}_s1_acc"] = (
            sum(nmatch(r.get("pred"), r["gold"]) for r in rows) / len(rows))

v_ok = [r for r in V_ROWS if "error" not in r]
M["verify_n"] = len(v_ok)
if v_ok:
    M["verify_acc_final"] = (sum(r["final_ok"] for r in v_ok)
                             / len(v_ok))
    M["verify_acc_ans"] = (sum(r["final_ok"] for r in v_ok
                               if r["answerable"]) /
                           max(sum(1 for r in v_ok if r["answerable"]), 1))
    M["s1_acc_on_verified"] = (sum(r["s1_ok"] for r in v_ok) / len(v_ok))
    sw = [r for r in v_ok if r["verdict"] == "SWITCH"]
    M["switch_n"] = len(sw)
    M["switch_precision"] = (sum(r["final_ok"] for r in sw) / len(sw)
                             if sw else None)
    M["confirm_rate"] = (sum(r["verdict"] == "CONFIRM" for r in v_ok)
                         / len(v_ok))
    M["abstain_rate"] = (sum(r["verdict"] == "ABSTAIN" for r in v_ok)
                         / len(v_ok))
    M["verdict_none_rate"] = (sum(r["verdict"] == "NONE" for r in v_ok)
                              / len(v_ok))
    M["verify_mean_gen_s"] = sum(r["gen_s"] for r in v_ok) / len(v_ok)
    # recovery: S1-wrong verified items the verifier fixed
    fixed = [r for r in v_ok if not r["s1_ok"] and r["final_ok"]]
    brok = [r for r in v_ok if r["s1_ok"] and not r["final_ok"]]
    M["recovery_on_s1_wrong"] = (len(fixed) /
        max(sum(1 for r in v_ok if not r["s1_ok"]), 1))
    M["breakage_on_s1_right"] = (len(brok) /
        max(sum(1 for r in v_ok if r["s1_ok"]), 1))
    # abstain quality on unanswerable verified items
    unans = [r for r in v_ok if not r["answerable"]]
    M["abstain_on_unanswerable"] = (sum(r["verdict"] == "ABSTAIN"
        for r in unans) / len(unans)) if unans else None

# e2e policies on answerable items
def emit(k):
    # P1: auto->S1; non-auto->verifier final (None=abstain/wrong).
    g = gate_by_i[k]
    if g["route"] == "auto":
        return SL_ROWS[k].get("pred")
    r = next((x for x in v_ok if x["i"] == k), None)
    return r.get("final_pred") if r else None

def emit_p2(k):
    # P2: auto+review->S1; escalate->verifier.
    g = gate_by_i[k]
    if g["route"] != "escalate":
        return SL_ROWS[k].get("pred")
    r = next((x for x in v_ok if x["i"] == k), None)
    return r.get("final_pred") if r else None

def e2e(fn):
    tot = cor = 0
    for k, it in enumerate(run_items):
        if not it["answerable"]:
            continue
        tot += 1
        if nmatch(fn(k), it["gold_tool"]):
            cor += 1
    return cor / max(tot, 1)

M["e2e_P1_ans"] = e2e(emit)
M["e2e_P2_ans"] = e2e(emit_p2)
M["route_counts"] = {r: sum(1 for g in GATE if g["route"] == r)
                     for r in ("auto", "review", "escalate", "error")}

sp = M.get("switch_precision")
M["wiring_v2_pass"] = bool(
    M["e2e_P1_ans"] > M["s1_acc_ans"] + 0.02 and
    sp is not None and sp >= 0.5 and
    M.get("confirm_rate", 1.0) <= 0.90)

print(json.dumps(M, indent=1), flush=True)
json.dump({"metrics": M},
          open("/kaggle/working/wiring_v2_metrics.json", "w"), indent=1)

import zipfile
with zipfile.ZipFile("/kaggle/working/artifacts.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    for fn in ["items.json", "s1_gate.json", "verify_rows.json",
               "wiring_v2_metrics.json"]:
        z.write(f"/kaggle/working/{fn}", fn)
    z.writestr("run_status.json", json.dumps(
        {"status": "COMPLETE", "kernel": "twil-s1s2-wiring-v2",
         "SL_CPU": SL_CPU, "spark": spark is not None,
         "n_items": len(run_items), "metrics": M}, indent=1))
print("artifacts.zip written", flush=True)
"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(OUT, "w"), indent=1)

import ast
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
        print(f"cell {i}: syntax OK ({len(''.join(c['source']))} chars)")
print(f"wrote {OUT}: {len(cells)} cells")
