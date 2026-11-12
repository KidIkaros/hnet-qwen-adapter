"""Depth-recovery probe generator (v3) — does the fast pass's *decision*,
injected into the think prompt as text, let the same trunk's deliberate
mode recover what the fast pass missed?

Per docs/SPEC_TWIL_DEPTH_PROBE.md. v2 established that identity-projected
hidden-state soft tokens are destructive (Arm C: 0/51 emitted calls,
degenerate repetition on every item). v3 tests the information question
directly with Arm T: the think prompt carries S1's chosen tool as words.
If T recovers S1-wrong items above B's paired rate without anchoring
(t_s1_agree << 1.0), the fast state's information is useful and a learned
vector interface is worth building; if T ~= B, fast-state conditioning
is dead and the H-Net compressed middle becomes the S2 mechanism.

Three arms on one trunk (TwIL-8B):
A = fast scoring (s1_rows.json archived in v1 — reused, no GPU work),
T = think + text-injected fast decision (runs k=0..99, first),
B = think unconditioned (61 rows archived from v1; top-up only on
    T-completed items k>=61 to extend paired coverage).
"""
import base64
import json
import os
import zlib

OUT = "/tmp/twil_depth_probe.ipynb"
REPO = "/home/ikaaros/Coding/Gold/hnet-qwen-adapter"
V1_DIR = os.path.join(REPO, "reports/runs/twil-depth-probe-v1")
SRC_FILES = ["src/__init__.py", "src/glial.py", "src/toolcall.py"]
V1_ARTIFACTS = {
    "items.json": "items.json",
    "s1_rows.json": "s1_rows.json",
    "b_rows.json": "b_rows_v1.json",
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.split("\n")]}


def code_cell(text):
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in text.split("\n")]}


def src_writer_cell():
    parts = ["import os, base64, zlib\n",
             "os.makedirs('/kaggle/working/src', exist_ok=True)\n"]
    for rel in SRC_FILES:
        content = open(os.path.join(REPO, rel), "rb").read()
        b64 = base64.b64encode(content).decode()
        parts.append(
            f"open('/kaggle/working/src/{os.path.basename(rel)}','wb')"
            f".write(base64.b64decode('{b64}'))\n")
    for src_fn, dst_fn in V1_ARTIFACTS.items():
        raw = open(os.path.join(V1_DIR, src_fn), "rb").read()
        zb64 = base64.b64encode(zlib.compress(raw, 9)).decode()
        parts.append(
            f"open('/kaggle/working/{dst_fn}','wb')"
            f".write(zlib.decompress(base64.b64decode('{zb64}')))\n")
    parts.append("import sys; sys.path.insert(0, '/kaggle/working')\n")
    parts.append("from src.glial import glial_gate\n")
    parts.append("from src.toolcall import parse_calls, post_think\n")
    parts.append("print('src + v1 artifacts bundle OK')\n")
    return code_cell("".join(parts))


cells = [
    md_cell("""# Depth-recovery probe v3 — one trunk, two depths (2xT4)

Per `docs/SPEC_TWIL_DEPTH_PROBE.md`. v2's Arm C established that raw
hidden-state soft tokens corrupt the deliberate pass (0/51 emitted calls,
degenerate repetition). v3 tests whether the fast pass's *information*
helps deliberation: **Arm T** injects S1's chosen tool into the think
prompt as text. Runs T first (k=0..99 pairs with v1's 61 archived B rows
on k=0..60), then tops up B only on T-completed items k>=61."""),

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

    code_cell("""# Cell 2: data — load pinned ToolACE slice 560-659 from v1 bundle.
def norm_name(s):
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

items = json.load(open("/kaggle/working/items.json"))
assert len(items) == 100, f"expected 100 items, got {len(items)}"
n_ans = sum(it["answerable"] for it in items)
print(f"[data] {len(items)} pinned items (qualifying 560-659); "
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

    code_cell("""# Cell 4: S2 helpers + Leg-0 smoke gate on the ids generate path.
from src.toolcall import parse_calls, post_think

MAX_NEW = [768]
FAST_HINT = ('\\n\\n[A fast tool-selection pass chose "{pred}". '
             'Verify whether this choice is correct; if it is not, '
             'select the correct tool.]')

def twil_tools(it):
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in it["tools"]]

def chat_prompt(it, thinking):
    msgs = [{"role": "user", "content": it["user"][:3000]}]
    return tok.apply_chat_template(msgs, tools=twil_tools(it),
                                   tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=thinking)

def think_prompt(it):
    return chat_prompt(it, True)

def text_conditioned_prompt(it, s1_pred):
    msgs = [{"role": "user",
             "content": it["user"][:3000] + FAST_HINT.format(pred=s1_pred)}]
    return tok.apply_chat_template(msgs, tools=twil_tools(it),
                                   tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=True)

def _run_generate(gen_kw, max_new=None):
    t0 = time.time()
    kw = {"max_new_tokens": max_new or MAX_NEW[0], "do_sample": False,
          "pad_token_id": PAD_ID, **gen_kw}
    try:
        with torch.no_grad():
            out = twil.generate(**kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if max_new is None:
            MAX_NEW[0] = 512
        kw["max_new_tokens"] = max_new or MAX_NEW[0]
        with torch.no_grad():
            out = twil.generate(**kw)
    return out, time.time() - t0

def s2_generate_ids(input_ids, max_new=None):
    assert input_ids.dtype in (torch.int64, torch.int32), input_ids.dtype
    out, dt = _run_generate({"input_ids": input_ids}, max_new=max_new)
    return out[0, input_ids.shape[1]:], dt

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

# Leg-0 smoke gate: verify the generate path before spending GPU hours
_smoke_ids = tok(think_prompt(items[0]),
                 return_tensors="pt").input_ids[:, :32].to("cuda:0")
_tok_b, _ = s2_generate_ids(_smoke_ids, max_new=2)
assert _tok_b.ndim == 1 and _tok_b.numel() >= 1, f"bad smoke: {_tok_b.shape}"
print(f"[leg0] smoke PASS: ids->{tuple(_tok_b.shape)}", flush=True)
"""),

    code_cell("""# Cell 5: load archived S1 rows — integrity checks only (no GPU work;
# the v2 parity gate already proved this archive against live scoring).
S1_ROWS = json.load(open("/kaggle/working/s1_rows.json"))
assert len(S1_ROWS) == len(items), f"s1 len mismatch: {len(S1_ROWS)}"
for k, r in enumerate(S1_ROWS):
    assert r["i"] == k, f"s1 row {k} index mismatch: {r['i']}"
    assert r.get("pred"), f"s1 row {k} missing pred"
    assert "error" not in r, f"s1 row {k} carries error"
ok = sum(r.get("ok", False) for r in S1_ROWS)
print(f"[s1] {len(S1_ROWS)} archived rows verified; acc {ok}/{len(S1_ROWS)}",
      flush=True)
"""),

    code_cell("""# Cell 6: Arm T (RUNS FIRST) — S2 text-conditioned on the fast
# pass's decision. The fast state's information (its chosen tool) enters
# the think prompt as words. Runs k=0..99 so k=0..60 pair immediately
# with v1's 61 archived Arm B rows.
T_WALL_S = 3300                      # 55 min budget for Arm T

T_ROWS = []
tc, t_skipped = time.time(), 0
for k, it in enumerate(items):
    if time.time() - tc > T_WALL_S:
        t_skipped += 1
        continue
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"],
           "s1_pred": S1_ROWS[k].get("pred"), "s1_ok": S1_ROWS[k].get("ok")}
    try:
        ids = tok(text_conditioned_prompt(it, row["s1_pred"]),
                  return_tensors="pt").input_ids.to("cuda:0")
        new_toks, dt = s2_generate_ids(ids)
        gen = tok.decode(new_toks, skip_special_tokens=False)
        row.update({"gen_s": round(dt, 1), **judge(gen, it)})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    T_ROWS.append(row)
    if len(T_ROWS) % 10 == 0:
        json.dump(T_ROWS, open("/kaggle/working/t_rows.json", "w"))
        t_err = sum(1 for r in T_ROWS if "error" in r)
        print(f"[T] {len(T_ROWS)} done (errors={t_err}, "
              f"elapsed={time.time()-tc:.0f}s)", flush=True)

json.dump(T_ROWS, open("/kaggle/working/t_rows.json", "w"))
t_err = sum(1 for r in T_ROWS if "error" in r)
print(f"[T] done {len(T_ROWS)} (errors={t_err}), skipped {t_skipped}",
      flush=True)
"""),

    code_cell("""# Cell 7: Arm B — resume from v1's 61 archived rows (k=0..60).
# Only runs indices that Arm T also completed, so every minute spent
# extends paired B-vs-T coverage.
B_ROWS = [r for r in json.load(open("/kaggle/working/b_rows_v1.json"))
          if "error" not in r]
done_b_idx = {r["i"] for r in B_ROWS}
done_t_idx = {r["i"] for r in T_ROWS if "error" not in r}
todo_b = sorted(done_t_idx - done_b_idx)
print(f"[B] loaded {len(B_ROWS)} v1 rows; {len(todo_b)} T-matched items to run",
      flush=True)

B_WALL_S = 1200                      # up to 20 min to top up B
tb, b_skipped = time.time(), 0
for k in todo_b:
    if time.time() - tb > B_WALL_S:
        b_skipped += 1
        continue
    it = items[k]
    row = {"i": k, "gold": it["gold_tool"], "answerable": it["answerable"],
           "s1_pred": S1_ROWS[k].get("pred"), "s1_ok": S1_ROWS[k].get("ok")}
    try:
        ids = tok(think_prompt(it), return_tensors="pt").input_ids.to("cuda:0")
        new_toks, dt = s2_generate_ids(ids)
        gen = tok.decode(new_toks, skip_special_tokens=False)
        row.update({"gen_s": round(dt, 1), **judge(gen, it)})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    B_ROWS.append(row)
    if len(B_ROWS) % 10 == 0:
        json.dump(B_ROWS, open("/kaggle/working/b_rows.json", "w"))
        print(f"[B] {len(B_ROWS)} total", flush=True)

B_ROWS.sort(key=lambda r: r["i"])
json.dump(B_ROWS, open("/kaggle/working/b_rows.json", "w"))
print(f"[B] done {len(B_ROWS)}, skipped {b_skipped}", flush=True)
"""),

    code_cell("""# Cell 8: metrics, verdict, artifacts.
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
    emitted_acc, emitted_n = acc(done)
    s1_wrong = [r for r in done if r.get("s1_ok") is False]
    s1_right = [r for r in done if r.get("s1_ok") is True]
    M[tag + "_n"] = len(done)
    M[tag + "_emitted_n"] = emitted_n
    M[tag + "_acc"] = (sum(bool(r.get("ok")) for r in done)
                       / max(len(done), 1))
    M[tag + "_acc_emitted"] = emitted_acc
    M[tag + "_errors"] = sum(1 for r in rows if "error" in r)
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
    emitted = [r for r in done if r.get("pred")]
    M[tag + "_s1_agree"] = (
        sum(norm_name(r["pred"]) == norm_name(r["s1_pred"])
            for r in emitted) / max(len(emitted), 1))

arm_stats(B_ROWS, "b")
arm_stats(T_ROWS, "t")

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
M["e2e_t"], M["e2e_t_n"] = e2e(T_ROWS)

# v1-comparable e2e: answerable-only, non-emission counts wrong —
# the wiring runs' definition, so cross-session numbers are quotable.
def e2e_v1(rows):
    by_i = {r["i"]: r for r in rows if "error" not in r}
    ans_s1 = [r for r in s1_done if items[r["i"]]["answerable"]]
    correct = 0
    for r in ans_s1:
        k = r["i"]
        route = GATE.get(k, {}).get("route", "escalate")
        if route == "auto":
            correct += bool(r.get("ok"))
        elif k in by_i and by_i[k].get("pred"):
            correct += bool(by_i[k].get("ok"))
    return round(correct / max(len(ans_s1), 1), 4)

M["e2e_v1_style_b"] = e2e_v1(B_ROWS)
M["e2e_v1_style_t"] = e2e_v1(T_ROWS)
M["e2e_semantics"] = ("e2e_b/e2e_t: emitted-only denominator, all "
                      "items; e2e_v1_style_*: answerable-only, "
                      "non-emission counts wrong (wiring-run "
                      "comparable)")

# T vs B directly: paired outcome counts on S1-wrong and S1-right items
b_by_i = {r["i"]: r for r in B_ROWS if "error" not in r}
t_by_i = {r["i"]: r for r in T_ROWS if "error" not in r}
paired_idx = sorted(set(b_by_i) & set(t_by_i))
both_wrong = [k for k in paired_idx if b_by_i[k].get("s1_ok") is False]
both_right = [k for k in paired_idx if b_by_i[k].get("s1_ok") is True]
M["paired_n"] = len(paired_idx)
M["t_vs_b_on_s1_wrong"] = {
    "n": len(both_wrong),
    "b_recovery_paired": (sum(b_by_i[k]["ok"] for k in both_wrong)
                          / max(len(both_wrong), 1)) if both_wrong else None,
    "t_recovery_paired": (sum(t_by_i[k]["ok"] for k in both_wrong)
                          / max(len(both_wrong), 1)) if both_wrong else None,
    "both_recover": sum(b_by_i[k]["ok"] and t_by_i[k]["ok"]
                        for k in both_wrong),
    "t_only": sum(not b_by_i[k]["ok"] and t_by_i[k]["ok"]
                  for k in both_wrong),
    "b_only": sum(b_by_i[k]["ok"] and not t_by_i[k]["ok"]
                  for k in both_wrong),
    "neither": sum(not b_by_i[k]["ok"] and not t_by_i[k]["ok"]
                   for k in both_wrong)}
M["t_vs_b_on_s1_right"] = {
    "n": len(both_right),
    "b_breakage_paired": (sum(not b_by_i[k]["ok"] for k in both_right)
                          / max(len(both_right), 1)) if both_right else None,
    "t_breakage_paired": (sum(not t_by_i[k]["ok"] for k in both_right)
                          / max(len(both_right), 1)) if both_right else None}

mgs = sorted(r["margin"] for r in s1_done if "margin" in r)
if mgs:
    M["margin_quartiles"] = [round(mgs[int(q * (len(mgs) - 1))], 4)
                             for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
M["gate_caveat"] = ("trunk-score margins UNCALIBRATED; 0.50/0.30 "
                    "bands provisional (StartLux-calibrated, not "
                    "re-fit here)")

# Verdict flags use PAIRED recovery/breakage on the items both B and T
# completed (falls back to marginal only if paired set is empty).
_t_rec = (M["t_vs_b_on_s1_wrong"]["t_recovery_paired"]
          if M["t_vs_b_on_s1_wrong"]["n"] else M.get("t_recovery"))
_b_rec = (M["t_vs_b_on_s1_wrong"]["b_recovery_paired"]
          if M["t_vs_b_on_s1_wrong"]["n"] else M.get("b_recovery"))
_t_brk = (M["t_vs_b_on_s1_right"]["t_breakage_paired"]
          if M["t_vs_b_on_s1_right"]["n"] else M.get("t_breakage"))
_b_brk = (M["t_vs_b_on_s1_right"]["b_breakage_paired"]
          if M["t_vs_b_on_s1_right"]["n"] else M.get("b_breakage"))
M["think_is_depth"] = bool((_b_rec or 0) >= 0.3)
M["text_conditioning_helps"] = bool(
    (_t_rec or 0) >= (_b_rec or 0) + 0.10
    and (_t_brk or 1) <= (_b_brk or 0) + 0.05)
M["conditioning_info_dead"] = bool(
    abs((_t_rec or 0) - (_b_rec or 0)) < 0.05
    and abs((_t_brk or 0) - (_b_brk or 0)) < 0.05)
M["anchoring"] = bool(M["t_s1_agree"] >= 0.9)
print(json.dumps({k: v for k, v in M.items() if not k.endswith("_n")},
                 indent=1), flush=True)

M["b_skipped"], M["t_skipped"] = b_skipped, t_skipped
json.dump(M, open("/kaggle/working/depth_metrics.json", "w"))
status = {"run": "twil-depth-probe v3",
          "arms": {"s1": "done" if M["s1_n"] >= 60 else "partial",
                   "b": "done" if M.get("b_n", 0) >= 60 else "partial",
                   "t": "done" if M.get("t_n", 0) >= 60 else "partial"},
          "think_is_depth": M["think_is_depth"],
          "text_conditioning_helps": M["text_conditioning_helps"],
          "conditioning_info_dead": M["conditioning_info_dead"],
          "anchoring": M["anchoring"]}
json.dump(status, open("/kaggle/working/run_status.json", "w"), indent=1)
with zipfile.ZipFile("/kaggle/working/artifacts.zip", "w") as z:
    for fn in ["items.json", "s1_rows.json", "b_rows.json", "t_rows.json",
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
