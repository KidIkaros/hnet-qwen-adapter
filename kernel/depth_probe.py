"""Depth-recovery probe generator (v2) — does the trunk's deliberate mode,
seeded by its fast pass, recover what the fast pass missed?

Per docs/SPEC_TWIL_DEPTH_PROBE.md. Three arms on one trunk (TwIL-8B):
A = fast constrained scoring (s1_rows.json archived in v1; v2 reconstructs
    the 100 fast states in 100 single forwards + parity-checks item 0
    and saves fast_states.pt),
C = think conditioned on the fast state via inputs_embeds=..., runs FIRST,
B = think unconditioned (61 rows archived from v1; v2 resumes k=61..99
    on items C completed).

Includes a Leg-0 smoke gate that exercises BOTH s2_generate_ids and
s2_generate_embeds on the loaded trunk before any arm loop starts.
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
    md_cell("""# Depth-recovery probe v2 — one trunk, two depths (2xT4)

Per `docs/SPEC_TWIL_DEPTH_PROBE.md`. Tests whether TwIL-8B's deliberate mode,
**conditioned on its fast pass's compressed state**, recovers what the fast
pass missed. v2 fixes Arm C's `generate(inputs_embeds=...)` keyword call,
adds a Leg-0 smoke gate for both generation paths, reconstructs the 100 fast
states in 100 single forwards (~2 min) with an item-0 parity gate against
v1's archived `s1_rows.json`, runs Arm C first, and resumes Arm B from v1's
61 completed rows."""),

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

    code_cell("""# Cell 4: S2 helpers + Leg-0 smoke gate on BOTH generate paths.
from src.toolcall import parse_calls, post_think

N_SOFT = 4
MAX_NEW = [768]

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

def fast_prompt(it):
    return chat_prompt(it, False)

def think_prompt(it):
    return chat_prompt(it, True)

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

def s2_generate_embeds(inputs_embeds, attn, max_new=None):
    assert inputs_embeds.dtype == EMB.weight.dtype, inputs_embeds.dtype
    assert inputs_embeds.ndim == 3 and attn.shape == inputs_embeds.shape[:2]
    out, dt = _run_generate({"inputs_embeds": inputs_embeds,
                             "attention_mask": attn}, max_new=max_new)
    return out[0], dt

def build_conditioned_embeds(pids, state_vec):
    soft = state_vec.to(device=EMB.weight.device, dtype=EMB.weight.dtype)
    soft = soft[None, None].expand(1, N_SOFT, -1)
    inputs_embeds = torch.cat([soft, EMB(pids)], dim=1)
    attn = torch.ones(1, N_SOFT + pids.shape[1], device=pids.device,
                      dtype=torch.long)
    return inputs_embeds, attn

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

# Leg-0 smoke gate: verify BOTH generate paths before spending GPU hours
_smoke_ids = tok(think_prompt(items[0]),
                 return_tensors="pt").input_ids[:, :32].to("cuda:0")
_tok_b, _ = s2_generate_ids(_smoke_ids, max_new=2)
_dummy_state = torch.zeros(EMB.weight.shape[1], dtype=torch.float32)
_emb_c, _attn_c = build_conditioned_embeds(_smoke_ids, _dummy_state)
_tok_c, _ = s2_generate_embeds(_emb_c, _attn_c, max_new=2)
assert _tok_b.ndim == 1 and _tok_b.numel() >= 1, f"bad B smoke: {_tok_b.shape}"
assert _tok_c.ndim == 1 and _tok_c.numel() >= 1, f"bad C smoke: {_tok_c.shape}"
print(f"[leg0] smoke PASS: ids->{tuple(_tok_b.shape)}, "
      f"inputs_embeds->{tuple(_tok_c.shape)}", flush=True)
"""),

    code_cell("""# Cell 5: Arm A — verify item-0 parity with archived S1_ROWS, then
# reconstruct all 100 FAST_STATE vectors (1 forward/item, ~2 min total)
# and persist /kaggle/working/fast_states.pt.
S1_ROWS = json.load(open("/kaggle/working/s1_rows.json"))
assert len(S1_ROWS) == len(items), f"s1 len mismatch: {len(S1_ROWS)}"

# Item-0 full candidate scoring parity gate
it0 = items[0]
ctx0 = tok(fast_prompt(it0), return_tensors="pt").input_ids[0].tolist()
sc0 = {}
for t in it0["tools"]:
    cd = tok(f"{t['name']}(", add_special_tokens=False).input_ids
    ids = torch.tensor([ctx0 + cd], device="cuda:0")
    with torch.no_grad():
        out = twil(ids)
    logits = out.logits[0].float()
    s = len(ctx0) - 1
    lp = torch.log_softmax(logits[s:s + len(cd)], -1)
    sc0[t["name"]] = float(sum(lp[i, cd[i]].item()
                               for i in range(len(cd)))) / len(cd)
names0 = sorted(sc0, key=lambda n: -sc0[n])
probs0 = torch.softmax(torch.tensor([sc0[n] for n in names0]), 0)
assert names0[0] == S1_ROWS[0]["pred"], (
    f"item-0 parity fail: {names0[0]} vs {S1_ROWS[0]['pred']}")
assert abs(float(probs0[0]) - S1_ROWS[0]["conf"]) < 1e-2, (
    f"item-0 conf drift: {float(probs0[0]):.4f} vs {S1_ROWS[0]['conf']:.4f}")
print(f"[s1] item-0 parity PASS: pred={names0[0]} conf={float(probs0[0]):.4f}",
      flush=True)

FAST_STATE = {}
t_s1 = time.time()
for k, it in enumerate(items):
    pred = S1_ROWS[k].get("pred")
    if not pred or "error" in S1_ROWS[k]:
        continue
    ctx = tok(fast_prompt(it), return_tensors="pt").input_ids[0].tolist()
    cd = tok(f"{pred}(", add_special_tokens=False).input_ids
    ids = torch.tensor([ctx + cd], device="cuda:0")
    with torch.no_grad():
        out = twil(ids, output_hidden_states=True)
    FAST_STATE[k] = out.hidden_states[-1][0, -1].float().cpu()
    if (k + 1) % 25 == 0:
        print(f"[s1] fast_state {k+1}/{len(items)} ({time.time()-t_s1:.0f}s)",
              flush=True)

torch.save(FAST_STATE, "/kaggle/working/fast_states.pt")
ok = sum(r.get("ok", False) for r in S1_ROWS if "error" not in r)
print(f"[s1] done: acc {ok}/{len(S1_ROWS)}; "
      f"saved {len(FAST_STATE)} fast states in {time.time()-t_s1:.0f}s",
      flush=True)
"""),

    code_cell("""# Cell 6: Arm C (RUNS FIRST) — S2 conditioned on the fast state.
# 4 soft prefix tokens = the fast pass's final hidden state (identity
# projection, v0). Runs k=0..99 in order so k=0..60 pair immediately
# with v1's 61 completed Arm B rows.
C_WALL_S = 3900                      # 65 min budget for Arm C

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
        inputs_embeds, attn = build_conditioned_embeds(pids, FAST_STATE[k])
        new_toks, dt = s2_generate_embeds(inputs_embeds, attn)
        gen = tok.decode(new_toks, skip_special_tokens=False)
        row.update({"gen_s": round(dt, 1), **judge(gen, it)})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    C_ROWS.append(row)
    if len(C_ROWS) % 10 == 0:
        json.dump(C_ROWS, open("/kaggle/working/c_rows.json", "w"))
        c_err = sum(1 for r in C_ROWS if "error" in r)
        print(f"[C] {len(C_ROWS)} done (errors={c_err}, "
              f"elapsed={time.time()-tc:.0f}s)", flush=True)

json.dump(C_ROWS, open("/kaggle/working/c_rows.json", "w"))
c_err = sum(1 for r in C_ROWS if "error" in r)
print(f"[C] done {len(C_ROWS)} (errors={c_err}), skipped {c_skipped}",
      flush=True)
"""),

    code_cell("""# Cell 7: Arm B — resume from v1's 61 archived rows (k=0..60).
# Only runs remaining indices k>=61 that Arm C also completed, so every
# minute spent increases paired B-vs-C coverage.
B_ROWS = [r for r in json.load(open("/kaggle/working/b_rows_v1.json"))
          if "error" not in r]
done_b_idx = {r["i"] for r in B_ROWS}
done_c_idx = {r["i"] for r in C_ROWS if "error" not in r}
todo_b = sorted(done_c_idx - done_b_idx)
print(f"[B] loaded {len(B_ROWS)} v1 rows; {len(todo_b)} C-matched items to run",
      flush=True)

B_WALL_S = 1800                      # up to 30 min to top up B
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
M["e2e_v1_style_c"] = e2e_v1(C_ROWS)
M["e2e_semantics"] = ("e2e_b/e2e_c: emitted-only denominator, all "
                      "items; e2e_v1_style_*: answerable-only, "
                      "non-emission counts wrong (wiring-run "
                      "comparable)")

# C vs B directly: paired outcome counts on S1-wrong and S1-right items
b_by_i = {r["i"]: r for r in B_ROWS if "error" not in r}
c_by_i = {r["i"]: r for r in C_ROWS if "error" not in r}
paired_idx = sorted(set(b_by_i) & set(c_by_i))
both_wrong = [k for k in paired_idx if b_by_i[k].get("s1_ok") is False]
both_right = [k for k in paired_idx if b_by_i[k].get("s1_ok") is True]
M["paired_n"] = len(paired_idx)
M["c_vs_b_on_s1_wrong"] = {
    "n": len(both_wrong),
    "b_recovery_paired": (sum(b_by_i[k]["ok"] for k in both_wrong)
                          / max(len(both_wrong), 1)) if both_wrong else None,
    "c_recovery_paired": (sum(c_by_i[k]["ok"] for k in both_wrong)
                          / max(len(both_wrong), 1)) if both_wrong else None,
    "both_recover": sum(b_by_i[k]["ok"] and c_by_i[k]["ok"]
                        for k in both_wrong),
    "c_only": sum(not b_by_i[k]["ok"] and c_by_i[k]["ok"]
                  for k in both_wrong),
    "b_only": sum(b_by_i[k]["ok"] and not c_by_i[k]["ok"]
                  for k in both_wrong),
    "neither": sum(not b_by_i[k]["ok"] and not c_by_i[k]["ok"]
                   for k in both_wrong)}
M["c_vs_b_on_s1_right"] = {
    "n": len(both_right),
    "b_breakage_paired": (sum(not b_by_i[k]["ok"] for k in both_right)
                          / max(len(both_right), 1)) if both_right else None,
    "c_breakage_paired": (sum(not c_by_i[k]["ok"] for k in both_right)
                          / max(len(both_right), 1)) if both_right else None}

mgs = sorted(r["margin"] for r in s1_done if "margin" in r)
if mgs:
    M["margin_quartiles"] = [round(mgs[int(q * (len(mgs) - 1))], 4)
                             for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
M["gate_caveat"] = ("trunk-score margins UNCALIBRATED; 0.50/0.30 "
                    "bands provisional (StartLux-calibrated, not "
                    "re-fit here)")

# Verdict flags use PAIRED recovery/breakage on the items both B and C
# completed (falls back to marginal only if paired set is empty).
_c_rec = (M["c_vs_b_on_s1_wrong"]["c_recovery_paired"]
          if M["c_vs_b_on_s1_wrong"]["n"] else M.get("c_recovery"))
_b_rec = (M["c_vs_b_on_s1_wrong"]["b_recovery_paired"]
          if M["c_vs_b_on_s1_wrong"]["n"] else M.get("b_recovery"))
_c_brk = (M["c_vs_b_on_s1_right"]["c_breakage_paired"]
          if M["c_vs_b_on_s1_right"]["n"] else M.get("c_breakage"))
M["depth_pass"] = bool(
    (_c_rec or 0) >= 0.5
    and not ((_c_brk or 1) > 0.10)
    and (_c_rec or 0) > (_b_rec or 0))
M["think_is_depth"] = bool((_b_rec or 0) >= 0.3)
M["conditioning_neutral"] = bool(
    0.05 > abs((_c_rec or 0) - (_b_rec or 0)))
print(json.dumps({k: v for k, v in M.items() if not k.endswith("_n")},
                 indent=1), flush=True)

M["b_skipped"], M["c_skipped"] = b_skipped, c_skipped
json.dump(M, open("/kaggle/working/depth_metrics.json", "w"))
status = {"run": "twil-depth-probe v2",
          "arms": {"s1": "done" if M["s1_n"] >= 60 else "partial",
                   "b": "done" if M.get("b_n", 0) >= 60 else "partial",
                   "c": "done" if M.get("c_n", 0) >= 60 else "partial"},
          "depth_pass": M["depth_pass"],
          "think_is_depth": M["think_is_depth"],
          "conditioning_neutral": M["conditioning_neutral"]}
json.dump(status, open("/kaggle/working/run_status.json", "w"), indent=1)
with zipfile.ZipFile("/kaggle/working/artifacts.zip", "w") as z:
    for fn in ["items.json", "s1_rows.json", "b_rows.json", "c_rows.json",
               "fast_states.pt", "depth_metrics.json", "run_status.json"]:
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
