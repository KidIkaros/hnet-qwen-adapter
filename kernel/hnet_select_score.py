"""Constrained tool selection: score each declared tool by the model's own
byte logprob of emitting "[name(" after ASSISTANT: — argmax over the
DECLARED set only (hallucination impossible by construction).

    colab exec -s hrun6 -f kernel/hnet_select_score.py
    env: MODEL, CKPT_PREFIX, N_ITEMS (15), SKIP (2500)
"""
import os, sys, glob, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "/content/qt/hnet-qwen-adapter"
RESULTS = f"{REPO}/results"
P = 6


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


import torch                                                     # noqa: E402
import re, json, time                                           # noqa: E402
from datasets import load_dataset                                # noqa: E402

if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.factory import (build_hnet, fit_adapters, load_trunk,   # noqa: E402
                         periodic)

trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_adapters(hn, trunk, meta, [d[:4096] for d in all_docs[:4]], P=P)

PREFIX = os.environ.get("CKPT_PREFIX", "hnet_resume_best")
glue = enc = dec = None
for d in [RESULTS, "/content/ckpt"]:
    try:
        glue = torch.load(f"{d}/{PREFIX}_glue.pt", map_location="cuda")
        enc = torch.load(f"{d}/{PREFIX}_enc.pt", map_location="cuda")
        dec = torch.load(f"{d}/{PREFIX}_dec.pt", map_location="cuda")
        break
    except FileNotFoundError:
        continue
hn.w_sup.load_state_dict(glue["w_sup"])
for i, l in enumerate(hn.enc):
    l.load_state_dict(enc[f"enc_{i}"])
for i, l in enumerate(hn.dec):
    l.load_state_dict(dec[f"dec_{i}"])
print(f"[ckpt] loaded {PREFIX}_* from {d}", flush=True)
hn.train()

N_ITEMS = int(os.environ.get("N_ITEMS", "15"))
SKIP = int(os.environ.get("SKIP", "2500"))
CAP = 4096
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")
TOOL_RE = re.compile(r'"name"\s*:\s*"([\w]+)"')


@torch.no_grad()
def cand_score(pid, cand):
    """mean logprob of cand bytes after prompt (higher = more likely)."""
    ids = (pid + cand)[-CAP:]
    x = torch.tensor(ids, device="cuda")
    bp, _, _ = hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
                  n_ctx=1, super_k=4, head_first=trunk["head"].first)
    n0 = len(ids) - len(cand)            # cand positions start here
    lp = bp[0, n0 - 1:-1].gather(
        1, x[n0:].long().unsqueeze(1)).squeeze(1)
    return float(lp.mean())


ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
items = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    conv = r.get("conversations") or []
    u = next((c["value"] for c in conv if c.get("from") == "user"), None)
    a = next((c["value"] for c in conv if c.get("from") == "assistant"), None)
    gold = CALL_RE.findall(a) if a else []
    tools = list(dict.fromkeys(TOOL_RE.findall(r["system"])))
    if u and gold and tools:
        items.append({"sys": r["system"], "user": u,
                      "gold": gold[0], "tools": tools[:24]})
    if len(items) >= N_ITEMS:
        break
print(f"[exam] {len(items)} items", flush=True)

right_n, top3_n = 0, 0
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    scores = [(t, cand_score(pid, list(f"[{t}(".encode())))
              for t in it["tools"]]
    scores.sort(key=lambda s: -s[1])
    rank = next(i for i, (t, _) in enumerate(scores)
              if t == it["gold"]) if it["gold"] in it["tools"] else -1
    right_n += rank == 0
    top3_n += 0 <= rank < 3
    print(f"  item {k:2d} gold={it['gold'][:24]:24s} rank={rank} "
          f"top={scores[0][0][:24]} ({scores[0][1]:.2f})", flush=True)

n = len(items)
res = {"n": n, "right": right_n, "top3": top3_n}
json.dump(res, open(f"{RESULTS}/select_score_{int(time.time())}.json", "w"))
print(f"[select] gold@1 {right_n}/{n} | gold@3 {top3_n}/{n} "
      f"(free-greedy was 4/15)", flush=True)
print("[select] complete", flush=True)
