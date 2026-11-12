"""Depth-recovery probe generator — does the trunk's deliberate mode, seeded
by its fast pass, recover what the fast pass missed?

Per docs/SPEC_TWIL_DEPTH_PROBE.md. Three arms on one trunk (TwIL-8B):
A = fast constrained scoring (also captures the fast state),
B = think unconditioned, C = think conditioned on the fast state.
Owner approval required before push.
"""
import json

OUT = "/tmp/twil_depth_probe.ipynb"
REPO = "/home/ikaaros/Coding/Gold/hnet-qwen-adapter"
SRC_FILES = ["src/__init__.py", "src/glial.py",
             "src/toolcall.py"]


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
    md_cell("""# Depth-recovery probe — one trunk, two depths (2xT4)

Per `docs/SPEC_TWIL_DEPTH_PROBE.md`. The one-mind's core claim: the same
trunk, running deeper **conditioned on its fast pass's compressed state**,
recovers what the fast pass missed. Arms: A = fast scoring (+ fast state),
B = think unconditioned, C = think conditioned."""),

    code_cell("""# Cell 1: deps — torch>=2.8 check in fresh subprocess.
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

    code_cell("""# Cell 2: data — ToolACE qualifying items 560-659 (fresh slice;
# disjoint from v2 eval 140-199, gate-cal 200-359, wiring v1 360-459,
# wiring v2 460-559, leg corpus 5000+).
from datasets import load_dataset

SKIP = 2500
NEED = 660
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

items = all_items[560:660]
assert len(items) >= 60, f"only {len(items)} items in slice"
n_ans = sum(it["answerable"] for it in items)
json.dump(items, open("/kaggle/working/items.json", "w"))
print(f"[data] {len(items)} items (qualifying 560-659); "
      f"answerable {n_ans}/{len(items)}", flush=True)
"""),

    code_cell("""# Cell 3: Leg 0 — load TwIL-8B (2xT4, device_map auto).
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "webAI-Official/TwIL-8B"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(REPO_ID)
twil = AutoModelForCausalLM.from_pretrained(
    REPO_ID, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0: '13GiB', 1: '14GiB'}, attn_implementation='sdpa')
twil.eval()
EMB = twil.get_input_embeddings()
assert str(EMB.weight.device) == "cuda:0", f"embeddings on {EMB.weight.device}"
eos = tok.eos_token_id
PAD_ID = (tok.pad_token_id if tok.pad_token_id is not None
          else (eos if isinstance(eos, int) else 0))
print(f"[twil] load {time.time()-t0:.0f}s", flush=True)
for i in range(2):
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{i} free {free/2**30:.1f}GiB / {total/2**30:.1f}GiB",
          flush=True)
print(f"[leg0] {len(items)} items ready", flush=True)
"""),

    code_cell("""# Cell 4: Arm A — S1 fast pass: constrained prefix-scoring over the
# declared tools. One forward per candidate; softmax over per-candidate
# mean-token logprobs = the gate's distribution (constructed, recorded as
# such). The predicted candidate's forward also yields the FAST STATE:
# final hidden state at its last position — what the fast mind held.
def twil_tools(it):
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in it["tools"]]

def fast_prompt(it):
    msgs = [{"role": "user", "content": it["user"][:3000]}]
    return tok.apply_chat_template(msgs, tools=twil_tools(it),
                                   tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)

S1_ROWS, FAST_STATE = [], {}
for k, it in enumerate(items):
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"]}
    try:
        t0 = time.time()
        ctx = tok(fast_prompt(it), return_tensors="pt").input_ids[0].tolist()
        scores, states = {}, {}
        for t in it["tools"]:
            cd = tok(f"{t['name']}(", add_special_tokens=False).input_ids
            ids = torch.tensor([ctx + cd], device="cuda:0")
            with torch.no_grad():
                out = twil(ids, output_hidden_states=True)
            logits = out.logits[0].float()
            s = len(ctx) - 1
            lp = torch.log_softmax(logits[s:s + len(cd)], -1)
            sc = float(sum(lp[i, cd[i]].item()
                           for i in range(len(cd)))) / len(cd)
            scores[t["name"]] = sc
            states[t["name"]] = out.hidden_states[-1][0, -1].float().cpu()
        names = sorted(scores, key=lambda n: -scores[n])
        vals = torch.tensor([scores[n] for n in names])
        probs = torch.softmax(vals, 0)
        pred = names[0]
        row.update({"pred": pred, "conf": float(probs[0]),
                    "margin": (float(probs[0] - probs[1])
                               if len(probs) > 1 else 1.0),
                    "probs": {n: float(p) for n, p in zip(names, probs)},
                    "gen_s": round(time.time() - t0, 2),
                    "ok": norm_name(pred) == norm_name(it["gold_tool"])})
        FAST_STATE[k] = states[pred]
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    S1_ROWS.append(row)
    if (k + 1) % 10 == 0:
        print(f"[s1] {k+1}/{len(items)}", flush=True)

json.dump(S1_ROWS, open("/kaggle/working/s1_rows.json", "w"))
ok = sum(r.get("ok", False) for r in S1_ROWS if "error" not in r)
print(f"[s1] done: acc {ok}/{len(S1_ROWS)}; fast states {len(FAST_STATE)}",
      flush=True)
"""),

    code_cell("""# Cell 4b: S2 helpers — call parsing, think prompt, generation.
from src.toolcall import parse_calls, post_think

def think_prompt(it):
    msgs = [{"role": "user", "content": it["user"][:3000]}]
    return tok.apply_chat_template(msgs, tools=twil_tools(it),
                                   tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=True)

MAX_NEW = [768]

def s2_generate(model_input, attn=None):
    t0 = time.time()
    kw = {"max_new_tokens": MAX_NEW[0], "do_sample": False,
          "pad_token_id": PAD_ID}
    if attn is not None:
        kw["attention_mask"] = attn
    try:
        out = twil.generate(model_input, **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        MAX_NEW[0] = 512
        kw["max_new_tokens"] = MAX_NEW[0]
        out = twil.generate(model_input, **kw)
    return out, time.time() - t0

def judge(text, it):
    calls = parse_calls(post_think(text))
    name = calls[0]["name"] if calls else None
    return {"has_think": (chr(60) + "/think>") in text,
            "format_valid": bool(calls),
            "clarify": "need-more-info" in text or "need_more_info" in text,
            "pred": name,
            "ok": bool(name) and
            norm_name(name) == norm_name(it["gold_tool"]),
            "text": text[-1500:]}

print("[leg0] helpers ready", flush=True)
"""),

    code_cell("""# Cell 5: Arm B — S2 unconditioned (think from the prompt alone).
B_ROWS = []
B_WALL_S = 2700
tb, b_skipped = time.time(), 0
for k, it in enumerate(items):
    if time.time() - tb > B_WALL_S:
        b_skipped += 1
        continue
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"],
           "s1_pred": S1_ROWS[k].get("pred"), "s1_ok": S1_ROWS[k].get("ok")}
    try:
        ids = tok(think_prompt(it), return_tensors="pt").input_ids.to("cuda:0")
        out, dt = s2_generate(ids)
        gen = tok.decode(out[0, ids.shape[1]:],
                         skip_special_tokens=False)
        row.update({"gen_s": round(dt, 1), **judge(gen, it)})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    B_ROWS.append(row)
    if len(B_ROWS) % 10 == 0:
        print(f"[B] {len(B_ROWS)} done", flush=True)

json.dump(B_ROWS, open("/kaggle/working/b_rows.json", "w"))
print(f"[B] done {len(B_ROWS)}, skipped {b_skipped}", flush=True)
"""),

    code_cell("""# Cell 6: Arm C — S2 conditioned on the fast state.
# 4 soft prefix tokens = the fast pass's final hidden state (identity
# projection, v0). The deliberate pass starts from what the fast mind held.
N_SOFT = 4
C_WALL_S = 3300

C_ROWS = []
tc, c_skipped = time.time(), 0
for k, it in enumerate(items):
    if time.time() - tc > C_WALL_S:
        c_skipped += 1
        continue
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"],
           "s1_pred": S1_ROWS[k].get("pred"), "s1_ok": S1_ROWS[k].get("ok")}
    if k not in FAST_STATE:
        row["error"] = "no fast state (s1 leg failed)"
        C_ROWS.append(row)
        continue
    try:
        pids = tok(think_prompt(it), return_tensors="pt").input_ids.to("cuda:0")
        soft = FAST_STATE[k].to("cuda", dtype=EMB.weight.dtype)
        soft = soft[None, None].expand(1, N_SOFT, -1)
        inputs_embeds = torch.cat([soft, EMB(pids)], dim=1)
        attn = torch.ones(1, N_SOFT + pids.shape[1], device="cuda:0",
                          dtype=torch.long)
        out, dt = s2_generate(inputs_embeds, attn=attn)
        row.update({"gen_s": round(dt, 1), **judge(tok.decode(
            out[0], skip_special_tokens=False), it)})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    C_ROWS.append(row)
    if len(C_ROWS) % 10 == 0:
        print(f"[C] {len(C_ROWS)} done", flush=True)

json.dump(C_ROWS, open("/kaggle/working/c_rows.json", "w"))
print(f"[C] done {len(C_ROWS)}, skipped {c_skipped}", flush=True)
"""),

    code_cell("""# Cell 7: metrics, verdict, artifacts.
import zipfile

M = {"n_items": len(items)}

def acc(rows):
    ok = [r for r in rows if "error" not in r and r.get("pred")]
    return (sum(r["ok"] for r in ok) / max(len(ok), 1), len(ok))

s1_done = [r for r in S1_ROWS if "error" not in r]
M["s1_acc_all"], M["s1_n"] = acc(s1_done)
M["s1_acc_ans"] = acc([r for r in s1_done
                       if items[r["i"]]["answerable"]])[0]
M["s1_mean_gen_s"] = round(sum(r.get("gen_s", 0) for r in s1_done)
                           / max(len(s1_done), 1), 2)

from src.glial import glial_gate
GATE = {}
for r in s1_done:
    if r.get("probs"):
        GATE[r["i"]] = glial_gate(r["probs"]).to_dict()
routes = {}
for g in GATE.values():
    routes[g["route"]] = routes.get(g["route"], 0) + 1
M["routes"] = routes

def arm_stats(rows, tag):
    done = [r for r in rows if "error" not in r]
    a, n = acc(done)
    s1_wrong = [r for r in done if r.get("s1_ok") is False]
    s1_right = [r for r in done if r.get("s1_ok") is True]
    M[tag + "_acc"], M[tag + "_n"] = a, n
    M[tag + "_recovery"] = ((sum(r["ok"] for r in s1_wrong)
                             / len(s1_wrong)) if s1_wrong else None)
    M[tag + "_recovery_n"] = len(s1_wrong)
    M[tag + "_breakage"] = ((sum(not r["ok"] for r in s1_right)
                             / len(s1_right)) if s1_right else None)
    M[tag + "_breakage_n"] = len(s1_right)
    M[tag + "_format_valid"] = (sum(r.get("format_valid", False)
                                    for r in done) / max(len(done), 1))
    M[tag + "_think_rate"] = (sum(r.get("has_think", False)
                                  for r in done) / max(len(done), 1))
    M[tag + "_clarify"] = sum(r.get("clarify", False) for r in done)
    M[tag + "_mean_gen_s"] = round(sum(r.get("gen_s", 0)
                                       for r in done) / max(len(done), 1), 1)

arm_stats(B_ROWS, "b")
arm_stats(C_ROWS, "c")

def e2e(rows):
    by_i = {r["i"]: r for r in rows if "error" not in r}
    emit, correct = 0, 0
    for r in s1_done:
        k = r["i"]
        route = GATE.get(k, {}).get("route", "escalate")
        if route == "auto":
            emit += 1
            correct += bool(r.get("ok"))
        elif k in by_i and by_i[k].get("pred"):
            emit += 1
            correct += bool(by_i[k].get("ok"))
    return round(correct / max(emit, 1), 4), emit

M["e2e_b"], M["e2e_b_n"] = e2e(B_ROWS)
M["e2e_c"], M["e2e_c_n"] = e2e(C_ROWS)

# v1-comparable e2e: answerable-only, non-emission counts wrong —
# the wiring runs' definition, so cross-session numbers are quotable.
def e2e_v1(rows):
    by_i = {r["i"]: r for r in rows if "error" not in r}
    correct = 0
    for r in s1_done:
        k = r["i"]
        if not items[k]["answerable"]:
            continue
        route = GATE.get(k, {}).get("route", "escalate")
        if route == "auto":
            correct += bool(r.get("ok"))
        elif k in by_i and by_i[k].get("pred"):
            correct += bool(by_i[k].get("ok"))
    return round(correct / max(len(s1_done), 1), 4)

M["e2e_v1_style_b"] = e2e_v1(B_ROWS)
M["e2e_v1_style_c"] = e2e_v1(C_ROWS)
M["e2e_semantics"] = ("e2e_b/e2e_c: emitted-only denominator, all "
                      "items; e2e_v1_style_*: answerable-only, "
                      "non-emission counts wrong (wiring-run "
                      "comparable)")

# C vs B directly: paired outcome counts on S1-wrong items both arms did.
b_by_i = {r["i"]: r for r in B_ROWS if "error" not in r}
c_by_i = {r["i"]: r for r in C_ROWS if "error" not in r}
both = [k for k in b_by_i if k in c_by_i
        and b_by_i[k].get("s1_ok") is False]
M["c_vs_b_on_s1_wrong"] = {
    "n": len(both),
    "both_recover": sum(b_by_i[k]["ok"] and c_by_i[k]["ok"]
                        for k in both),
    "c_only": sum(not b_by_i[k]["ok"] and c_by_i[k]["ok"]
                  for k in both),
    "b_only": sum(b_by_i[k]["ok"] and not c_by_i[k]["ok"]
                  for k in both),
    "neither": sum(not b_by_i[k]["ok"] and not c_by_i[k]["ok"]
                   for k in both)}

# margin distribution (quartiles) + the uncalibrated-gate caveat.
mgs = sorted(r["margin"] for r in s1_done if "margin" in r)
if mgs:
    M["margin_quartiles"] = [round(mgs[int(q * (len(mgs) - 1))], 4)
                             for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
M["gate_caveat"] = ("trunk-score margins UNCALIBRATED; 0.50/0.30 "
                    "bands provisional (StartLux-calibrated, not "
                    "re-fit here)")

M["depth_pass"] = bool(
    (M.get("c_recovery") or 0) >= 0.5
    and not ((M.get("c_breakage") or 1) > 0.10)
    and (M.get("c_recovery") or 0) > (M.get("b_recovery") or 0))
M["think_is_depth"] = bool((M.get("b_recovery") or 0) >= 0.3)
M["conditioning_neutral"] = bool(
    0.05 > abs((M.get("c_recovery") or 0)
               - (M.get("b_recovery") or 0)))
print(json.dumps({k: v for k, v in M.items() if not k.endswith("_n")},
                 indent=1), flush=True)

json.dump(M, open("/kaggle/working/depth_metrics.json", "w"))
M["b_skipped"], M["c_skipped"] = b_skipped, c_skipped
status = {"run": "twil-depth-probe v1",
          "arms": {"s1": "done" if M["s1_n"] >= 60 else "partial",
                   "b": "done" if M.get("b_n", 0) >= 60 else "partial",
                   "c": "done" if M.get("c_n", 0) >= 60 else "partial"},
          "depth_pass": M["depth_pass"],
          "think_is_depth": M["think_is_depth"]}
json.dump(status, open("/kaggle/working/run_status.json", "w"), indent=1)
with zipfile.ZipFile("/kaggle/working/artifacts.zip", "w") as z:
    for fn in ["items.json", "s1_rows.json", "b_rows.json", "c_rows.json",
               "depth_metrics.json", "run_status.json"]:
        p = "/kaggle/working/" + fn
        if os.path.exists(p):
            z.write(p, fn)
print("artifacts.zip written; run_status:", status, flush=True)
"""),

]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT)

import ast
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        ast.parse(src)
        print(f"cell {i}: syntax OK ({len(src)} chars)")
