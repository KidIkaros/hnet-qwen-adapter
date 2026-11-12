#!/usr/bin/env python3
"""Build the Kaggle one-mind wiring kernel (2xT4, ~1.5h).

Per docs/SPEC_TWIL_S1S2_WIRING.md:
  Leg 0  co-load StartLux (cuda:0) + TwIL-8B (device_map auto)
  Leg 1  StartLux decide x100 fresh ToolACE items -> probs -> glial_gate
  Leg 2  TwIL-S2 (enable_thinking=True) on review+escalate bands
  Leg 3  metrics: S1-only / gate split / S2 recovery / e2e policies / cost
  Leg 4  artifacts.zip
"""
import json

OUT = "/tmp/twil_s1s2.ipynb"
REPO = "/home/ikaaros/Coding/Gold/hnet-qwen-adapter"
SRC_FILES = ["src/__init__.py", "src/adapter.py", "src/corpus.py",
             "src/glial.py"]


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
    md_cell("""# One-mind wiring — StartLux S1 → glial gate → TwIL S2 (2xT4)

Per `docs/SPEC_TWIL_S1S2_WIRING.md`. Closes the loop for the first time:
fast S1 decisions, calibrated glial routing, deliberate S2 escalation.
Measures whether S2 recovers S1's gate-flagged errors — the one-mind's
core claim."""),

    code_cell("""# Cell 1: deps — torch>=2.8 check in fresh subprocess (in-process
# upgrade can't swap an imported compiled module); CUDA-only reqs filtered.
import subprocess, sys, os, json, re, time

v = subprocess.run([sys.executable, '-c',
    'import torch; print(torch.__version__)'],
    capture_output=True, text=True).stdout.strip()
print('torch', v)
if not v or tuple(int(x) for x in v.split('.')[:2]) < (2, 8):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'torch==2.8.0', '--index-url',
        'https://download.pytorch.org/whl/cu121'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=5.8', 'accelerate', 'datasets', 'huggingface_hub'],
    check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
for i in range(2):
    print(f'cuda:{i}', torch.cuda.get_device_name(i))
"""),

    src_writer_cell(),

    code_cell("""# Cell 2: data — ToolACE qualifying items 360-459 (fresh slice,
# disjoint from v2 eval 140-199, gate-cal 200-359, leg1 corpus 5000+).
# Keeps full tool JSON (parameters) for the TwIL chat-template tools arg.
from datasets import load_dataset

SKIP = 2500
NEED = 460                      # collect through qualifying index 459
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

items = all_items[360:460]
assert len(items) >= 60, f"only {len(items)} items in slice"
n_ans = sum(it["answerable"] for it in items)
json.dump(items, open("/kaggle/working/items.json", "w"))
print(f"[data] {len(items)} items (qualifying 360-459); "
      f"answerable {n_ans}/{len(items)}", flush=True)
"""),

    code_cell("""# Cell 3: Leg 0 — co-load StartLux (cuda:0) + TwIL-8B (auto).
# StartLux ~1.6GB bf16 pinned to cuda:0; TwIL max_memory leaves it room.
# Fallback: SL_CPU=True -> StartLux on CPU (slow, capped items).
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    print(f"[startlux] loaded on cuda:0 "
          f"fast_kernels={getattr(sl, 'fast_kernels_active', '?')}",
          flush=True)
except Exception as e:
    print(f"[startlux] cuda:0 load failed ({type(e).__name__}: "
          f"{str(e)[:150]}) — CPU fallback", flush=True)
    try:
        sl = StartLuxDecision(MDIR, device="cpu")
        SL_CPU = True
        print("[startlux] loaded on CPU (slow path)", flush=True)
    except Exception as e2:
        print(f"[startlux] UNAVAILABLE: {type(e2).__name__}", flush=True)
        sl = None

# TwIL — device_map auto; cuda:0 keeps ~2.5GB headroom for StartLux+activations
REPO_ID = "webAI-Official/TwIL-8B"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(REPO_ID)
twil = AutoModelForCausalLM.from_pretrained(
    REPO_ID, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0: '12GiB', 1: '14GiB'}, attn_implementation='sdpa')
twil.eval()
print(f"[twil] load {time.time()-t0:.0f}s", flush=True)
for i in range(2):
    print(f"cuda:{i} reserved="
          f"{torch.cuda.memory_reserved(i)/1e9:.1f}GB free="
          f"{torch.cuda.mem_get_info(i)[0]/1e9:.1f}GB", flush=True)
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
    if (k + 1) % 20 == 0:
        print(f"[s1] {k+1}/{len(run_items)}", flush=True)

json.dump({"s1": SL_ROWS, "gate": GATE},
          open("/kaggle/working/s1_gate.json", "w"))
from collections import Counter
print("[gate] routes:", Counter(g["route"] for g in GATE), flush=True)
print(f"[s1] mean gen_s="
      f"{sum(r.get('gen_s',0) for r in SL_ROWS)/max(len(SL_ROWS),1):.2f}",
      flush=True)
"""),

    code_cell("""# Cell 5: Leg 2 — TwIL-S2 (enable_thinking=True) on review+escalate.
CALL_RE = re.compile(r"<tool_call>\\s*<function=([\\w.-]+)>(.*?)</function>"
                     r"\\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\\w.-]+)>\\s*(.*?)\\s*</parameter>",
                      re.S)

def parse_calls(text):
    out = []
    for m in CALL_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        args = {k: v.strip() for k, v in PARAM_RE.findall(body)}
        out.append({"name": name, "args": args})
    return out

def twil_tools(it):
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in it["tools"]]

S2_ROWS = []
nonauto = [g["i"] for g in GATE
           if g["route"] in ("review", "escalate") and g.get("top")]
print(f"[s2] {len(nonauto)} non-auto items to deliberate", flush=True)
for j, k in enumerate(nonauto):
    it = run_items[k]
    row = {"i": k, "gold": it["gold_tool"],
           "answerable": it["answerable"],
           "s1_pred": SL_ROWS[k].get("pred"),
           "s1_route": GATE[k]["route"]}
    try:
        msgs = [{"role": "user", "content": it["user"][:3000]}]
        prompt = tok.apply_chat_template(
            msgs, tools=twil_tools(it), tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        ids = tok(prompt, return_tensors='pt').input_ids.to('cuda:0')
        t0 = time.time()
        eos = tok.eos_token_id
        pad_id = (tok.pad_token_id if tok.pad_token_id is not None
                  else (eos if isinstance(eos, int) else 0))
        with torch.no_grad():
            out = twil.generate(ids, max_new_tokens=768,
                                do_sample=False, pad_token_id=pad_id)
        dt = time.time() - t0
        new = out[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=False)
        body = (text.rsplit("</think>", 1)[-1]
                if "</think>" in text else text)
        calls = parse_calls(body)
        # name match: TwIL emits the function name; normalize for spaced names
        s2_name = calls[0]["name"] if calls else None
        row.update({
            "gen_s": round(dt, 1), "has_think": "</think>" in text,
            "format_valid": bool(calls), "clarify": "need-more-info" in text
            or "need_more_info" in text,
            "n_calls": len(calls), "s2_pred": s2_name,
            "s2_name_ok": bool(s2_name) and
            norm_name(s2_name) == norm_name(it["gold_tool"]),
            "calls": calls[:3], "text": text[-2000:]})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    S2_ROWS.append(row)
    if (j + 1) % 10 == 0:
        print(f"[s2] {j+1}/{len(nonauto)}", flush=True)

json.dump(S2_ROWS, open("/kaggle/working/s2_rows.json", "w"))
print(f"[s2] done {len(S2_ROWS)} rows", flush=True)
"""),

    code_cell("""# Cell 6: Leg 3 — metrics, policies, verdict, artifacts.
M = {}

def acc(rows, key_pred="pred", key_gold="gold"):
    ok = [r for r in rows if "error" not in r and r.get(key_pred)]
    return (sum(norm_name(r[key_pred]) == norm_name(r[key_gold])
                for r in ok) / max(len(ok), 1), len(ok))

ans_idx = [k for k, it in enumerate(run_items) if it["answerable"]]
s1_ok = [r for r in SL_ROWS if "error" not in r]
M["s1_acc_all"] = acc(s1_ok)[0]
M["s1_acc_ans"] = acc([r for r in s1_ok
                       if run_items[r["i"]]["answerable"]])[0]
M["s1_mean_gen_s"] = sum(r.get("gen_s", 0) for r in s1_ok) / max(len(s1_ok), 1)

# per-route S1 acc
gate_by_i = {g["i"]: g for g in GATE}
for route in ("auto", "review", "escalate"):
    rows = [r for r in s1_ok if gate_by_i[r["i"]]["route"] == route]
    if rows:
        a, n = acc(rows)
        M[f"route_{route}_n"] = n
        M[f"route_{route}_s1_acc"] = a

# S2 on non-auto
s2_ok = [r for r in S2_ROWS if "error" not in r]
for route in ("review", "escalate"):
    rows = [r for r in s2_ok if r["s1_route"] == route]
    if rows:
        a, n = acc(rows, "s2_pred")
        M[f"s2_{route}_n"] = n
        M[f"s2_{route}_acc"] = a
        M[f"s2_{route}_mean_s"] = sum(r["gen_s"] for r in rows) / n

# recovery: S1-wrong non-auto items -> S2 right?
recov = []
for r in s2_ok:
    s1_right = norm_name(r.get("s1_pred") or "") == norm_name(r["gold"])
    s2_right = r.get("s2_name_ok")
    if not s1_right:
        recov.append(bool(s2_right))
M["s2_recovery_on_s1_wrong"] = (sum(recov) / len(recov)) if recov else None
M["s2_recovery_n"] = len(recov)

# end-to-end policies: emit S1 for auto; S2 answer for routed items
# P1: auto->S1, review+escalate->S2 ; P2: auto+review->S1, escalate->S2
# P3: all->S2 (only measured on non-auto; auto uses S1 as proxy=upper bound)
def e2e(pred_for):
    # pred_for(i) -> ('s1'|'s2', name) ; None -> abstain (count wrong)
    tot = cor = 0
    for k, it in enumerate(run_items):
        if not it["answerable"]:
            continue
        src, name = pred_for(k)
        tot += 1
        if name and norm_name(name) == norm_name(it["gold_tool"]):
            cor += 1
    return cor / max(tot, 1)

s2_by_i = {r["i"]: r for r in s2_ok}
def p1(k):
    g = gate_by_i[k]
    if g["route"] == "auto":
        return "s1", SL_ROWS[k].get("pred")
    r = s2_by_i.get(k)
    return ("s2", r.get("s2_pred")) if r else ("none", None)

def p2(k):
    g = gate_by_i[k]
    if g["route"] == "escalate":
        r = s2_by_i.get(k)
        return ("s2", r.get("s2_pred")) if r else ("none", None)
    return "s1", SL_ROWS[k].get("pred")

M["e2e_P1_autoS1_restS2_ans"] = e2e(p1)
M["e2e_P2_autoReviewS1_escS2_ans"] = e2e(p2)
M["e2e_S1only_ans"] = M["s1_acc_ans"]
M["route_counts"] = {r: sum(1 for g in GATE if g["route"] == r)
                     for r in ("auto", "review", "escalate", "error")}
M["s2_think_rate"] = (sum(r.get("has_think", False) for r in s2_ok)
                      / max(len(s2_ok), 1))
M["s2_format_valid"] = (sum(r.get("format_valid", False) for r in s2_ok)
                        / max(len(s2_ok), 1))
M["SL_CPU"] = SL_CPU

# verdict: recovery >=0.5 AND P1 beats S1-only on answerable
rec = M["s2_recovery_on_s1_wrong"]
M["wiring_pass"] = bool(
    rec is not None and rec >= 0.5 and
    M["e2e_P1_autoS1_restS2_ans"] > M["s1_acc_ans"] + 0.02)

print(json.dumps(M, indent=1), flush=True)
json.dump({"metrics": M},
          open("/kaggle/working/wiring_metrics.json", "w"), indent=1)

import zipfile
zpath = "/kaggle/working/artifacts.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ["items.json", "s1_gate.json", "s2_rows.json",
               "wiring_metrics.json"]:
        z.write(f"/kaggle/working/{fn}", fn)
    z.writestr("run_status.json", json.dumps(
        {"status": "COMPLETE", "kernel": "twil-s1s2-wiring",
         "SL_CPU": SL_CPU, "n_items": len(run_items),
         "metrics": M}, indent=1))
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
