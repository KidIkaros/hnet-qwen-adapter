"""StartLux-Decision-v1-0.8B as S1: score each toolace item's declared tools
via the model's native choice verdict — compare vs BGE encoder (25/40) and
byte-scorer (21/40) on the same exam.

    colab exec -s hrun7 -f kernel/startlux_baseline.py
    env: N_ITEMS (40), SKIP (2500), MODEL_DIR
"""
import os, sys, subprocess, json, re

REPO = "/content/qt/hnet-qwen-adapter"
MDIR = os.environ.get("MODEL_DIR", "/content/StartLux-0.8B")


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-3000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
if not os.path.exists(MDIR):
    sh("hf", "download", "startlux-models/StartLux-Decision-v1-0.8B",
       "--local-dir", MDIR)
if os.environ.get("STARTLUX_ALLOW_SLOW") != "1":
    sh("pip", "install", "-q", "-r", f"{MDIR}/requirements.txt")
sys.path.insert(0, MDIR)
os.chdir(REPO)

from datasets import load_dataset                                # noqa: E402
from startlux_decision import StartLuxDecision                   # noqa: E402

m = StartLuxDecision(MDIR)

N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
SKIP = int(os.environ.get("SKIP", "2500"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(
    r'"name"\s*:\s*"([\w]+)"\s*,\s*"description"\s*:\s*"([^"]{0,300})')

ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv
              if c.get("from") == "assistant"), None)
    gold = CALL_RE.findall(a) if a else []
    tools = TOOL_RE.findall(r["system"])
    if u and gold and tools:
        items.append({"sys": r["system"], "user": u,
                      "gold": gold[0], "tools": tools[:24]})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items | startlux-decision-v1-0.8B", flush=True)

right = 0
for k, it in enumerate(items):
    crit = {n: (d or n) for n, d in it["tools"]}
    ans, _ = m.decide(
        {"request": it["user"][:1500], "system": it["sys"][:2000]},
        {"tool": {"type": "choice",
                  "instructions": "Which function should be called?",
                  "criteria": crit}})
    probs = ans["tool"]["probabilities"]
    pred = max(probs, key=probs.get)
    right += pred == it["gold"]
    print(f"  item {k:2d} gold={it['gold'][:22]:22s} pred={pred[:22]:22s} "
          f"p={probs[pred]:.2f} {'RIGHT' if pred == it['gold'] else ''}",
          flush=True)

n = len(items)
json.dump({"n": n, "right": right},
          open(f"{REPO}/results/startlux_{int(__import__('time').time())}.json",
               "w"))
print(f"[startlux] gold@1 {right}/{n} "
      f"(BGE enc 25/40 | byte-scorer 21/40)", flush=True)
print("[startlux] complete", flush=True)
