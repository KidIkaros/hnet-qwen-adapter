"""27B teacher ceiling probe on the pinned ToolACE exam.

Two measures per item:
  1. prefix scoring — sum logprob of `name(` after `ASSISTANT: [` (same
     protocol as the byte-level S2; comparable score matrix incl. `none`)
  2. natural generation — greedy decode after `ASSISTANT:` in the model's
     own format; first call parsed (how a real 27B does unconstrained)

    colab exec -s hrun7 -f kernel/teacher_score.py
    env: MODEL, N_ITEMS (60), SKIP (1224), GEN_LEN (120)
"""
import os, sys, subprocess, json, time

REPO = "/content/qt/hnet-qwen-adapter"
MODEL = os.environ.get(
    "MODEL", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")


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
sh("pip", "install", "-q", "bitsandbytes", "accelerate")
os.chdir(REPO)
sys.path.insert(0, REPO)

import re                                                          # noqa: E402
import torch                                                       # noqa: E402
from transformers import (AutoTokenizer, AutoModelForCausalLM,     # noqa: E402
                          BitsAndBytesConfig)
from src.probe_common import stream_toolace, RESULTS               # noqa: E402

N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
SKIP = int(os.environ.get("SKIP", "1224"))
GEN_LEN = int(os.environ.get("GEN_LEN", "120"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")

tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16)
model.eval()

items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items | teacher={MODEL} (4bit)", flush=True)


@torch.no_grad()
def prefix_lp(pid, cand_ids):
    """Summed logprob of cand_ids following prompt token ids pid."""
    ids = torch.tensor([pid + cand_ids], device="cuda")
    logits = model(ids).logits[0]
    s = len(pid) - 1
    lp = torch.log_softmax(
        logits[s:s + len(cand_ids)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(cand_ids)))


@torch.no_grad()
def gen_call(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=GEN_LEN, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = CALL_RE.search(text)
    return m.group(1) if m else None


rows = []
right_pref = right_gen = none_n = 0
for k, it in enumerate(items):
    prefix = f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: ["
    pid = tok(prefix)["input_ids"]
    sc = {n: prefix_lp(pid, tok(f"{n}(", add_special_tokens=False)
                       ["input_ids"])
          for n in it["tools"]}
    sc["none"] = prefix_lp(pid, tok("none(", add_special_tokens=False)
                           ["input_ids"])
    pred = max(sc, key=sc.get)
    gpred = gen_call(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:")
    right_pref += pred == it["gold"]
    right_gen += gpred == it["gold"]
    none_n += pred == "none"
    rows.append({"k": k, "gold": it["gold"], "tools": it["tools"],
                 "declared": it["gold"] in it["tools"], "scores": sc,
                 "pred": pred, "gen_pred": gpred})
    print(f"  item {k:2d} gold={it['gold'][:20]:20s} "
          f"pref={pred[:18]:18s} gen={str(gpred)[:18]:18s} "
          f"{'R' if pred == it['gold'] else ' '} "
          f"{'G' if gpred == it['gold'] else ''}", flush=True)

n = len(items)
dec = sum(r["declared"] for r in rows)
dr_pref = sum(r["pred"] == r["gold"] for r in rows if r["declared"])
dr_gen = sum(r["gen_pred"] == r["gold"] for r in rows if r["declared"])
json.dump(rows, open(f"{RESULTS}/teacher_scores_{int(time.time())}.json",
                     "w"))
print(f"[teacher] prefix {dr_pref}/{dec} | gen {dr_gen}/{dec} declared | "
      f"none-picks {none_n} ({n} total)", flush=True)
print("[teacher] complete", flush=True)
