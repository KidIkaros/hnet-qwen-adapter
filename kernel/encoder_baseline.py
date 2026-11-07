"""S1-candidate baseline: an off-the-shelf encoder ranks declared tools by
cosine(user, tool name+description) — zero model training. Compares against
our byte-model constrained scorer on the same items; dumps per-tool sims
for offline fusion.

    colab exec -s hrun6 -f kernel/encoder_baseline.py
    env: ENCODER (default BAAI/bge-small-en-v1.5), N_ITEMS (40), SKIP (2500)
"""
import os, sys, subprocess

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
import re, json, time                                           # noqa: E402
from datasets import load_dataset                                # noqa: E402
from transformers import AutoModel, AutoTokenizer                # noqa: E402

ENC = os.environ.get("ENCODER", "BAAI/bge-small-en-v1.5")
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
SKIP = int(os.environ.get("SKIP", "2500"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(
    r'"name"\s*:\s*"([\w]+)"\s*,\s*"description"\s*:\s*"([^"]{0,300})')

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
        items.append({"user": u, "gold": gold[0], "tools": tools[:24]})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items | encoder {ENC}", flush=True)

right1 = right3 = 0
score_rows = []
for k, it in enumerate(items):
    ue = embed([it["user"]])
    te = embed([f"{n}: {d}" for n, d in it["tools"]])
    sims = (ue * te).sum(1)
    order = sims.argsort(descending=True).cpu().tolist()
    names = [t[0] for t in it["tools"]]
    rank = order.index(names.index(it["gold"])) if it["gold"] in names else -1
    right1 += rank == 0
    right3 += 0 <= rank < 3
    score_rows.append({"tools": names, "gold": it["gold"],
                       "sims": [float(x) for x in sims]})
    print(f"  item {k:2d} gold={it['gold'][:22]:22s} rank={rank} "
          f"top={names[order[0]][:22]}", flush=True)

os.makedirs(f"{REPO}/results", exist_ok=True)
json.dump(score_rows, open(f"{REPO}/results/enc_scores.json", "w"))
n = len(items)
res = {"encoder": ENC, "n": n, "right1": right1, "right3": right3}
json.dump(res, open(f"{REPO}/results/encoder_base_{int(time.time())}.json", "w"))
print(f"[enc] gold@1 {right1}/{n} | gold@3 {right3}/{n} "
      f"(byte-scorer was 21/40, 27/40)", flush=True)
print("[enc] complete", flush=True)
