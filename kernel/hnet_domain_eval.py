"""Domain-slice eval: per-domain bpb of the current best H-Net-of-Qwen
checkpoint on UNSEEN docs (streamed past all previously-consumed prefixes).

    colab exec -s hrun3 -f kernel/hnet_domain_eval.py
    env: MODEL, CKPT_PREFIX (default hnet_resume_best), EVAL_DOCS, CAP

Domains: dolma3.5_pool configs (web/wiki/textbook/code/prose/math/multilingual)
+ ultrachat slice for conversation signal.  Output: bpb table per domain.
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
import numpy as np                                               # noqa: E402

if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.factory import (build_hnet, enwik8, fit_adapters,       # noqa: E402
                         load_trunk, periodic)
from src.hnet import HNetConfig                                  # noqa: E402

trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_docs = [d[:4096] for d in all_docs[:4]]
held_docs = [d[:4096] for d in all_docs[-16:]]
ew8 = enwik8()
ew8_test = torch.tensor(list(ew8["test"][:200_000]), device="cuda")
g = fit_adapters(hn, trunk, meta, fit_docs, P=P)
print(f"[fit] adapters fit | {g} dechunk pairs", flush=True)

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
assert glue is not None, "checkpoint not found"
hn.w_sup.load_state_dict(glue["w_sup"])
hn.dechunk.load_state_dict(glue["dechunk"])
hn.boundary.load_state_dict(glue["boundary"])
for i, l in enumerate(hn.enc):
    l.load_state_dict(enc[f"enc_{i}"])
for i, l in enumerate(hn.dec):
    l.load_state_dict(dec[f"dec_{i}"])
print(f"[ckpt] loaded {PREFIX}_* from {d}", flush=True)
import hashlib, json, time                                   # noqa: E402


def sha256_path(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


CKPT_SHA = {s: sha256_path(f"{d}/{PREFIX}_{s}.pt")
            for s in ("glue", "enc", "dec")}
print(f"[ckpt] sha256 {CKPT_SHA}", flush=True)


def doc_loss(byte_ids):
    T = len(byte_ids)
    out = hn(byte_ids, boundaries=periodic(T, P),
             cfg=HNetConfig(ctx="retrieve", n_ctx=1, super_k=4),
             head_first=trunk["head"].first)
    bp = out.byte_logprobs
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    aux = (out.boundary_p.mean() - 1.0 / P) ** 2
    return nll.mean() + 1.0 * aux


def eval_bpb(byte_ids):
    with torch.no_grad():
        return float(doc_loss(byte_ids)) / np.log(2)


CAP = int(os.environ.get("CAP", "4096"))
EVAL_DOCS = int(os.environ.get("EVAL_DOCS", "24"))
# PINNED eval slices (RESEARCH_DATA_ORDERING §7): dolma evals read a fixed,
# deterministic shard set per config — excluded from training sampling in
# hnet_resume.py — so every leg measures the SAME bytes (prior evals pulled
# the slice just past SEEN, making cross-leg deltas meaningless). The
# "eval:<cfg>" seed makes the pick identical in both scripts.
EVAL_SHARDS_PER_CFG = 4
EVAL_OFF = {"ultrachat": 0, "glaive_function_calling": 824, "toolace": 0}
_POOL_FILES = None
_DATA_EXT = (".jsonl.zst", ".jsonl.gz", ".json.zst")


def pool_files():
    global _POOL_FILES
    if _POOL_FILES is None:
        from huggingface_hub import list_repo_files
        _POOL_FILES = list_repo_files("allenai/dolma3.5_pool",
                                      repo_type="dataset")
    return _POOL_FILES


def dolma_shards(cfg):
    return sorted(f for f in pool_files()
                  if f.startswith(cfg + "/") and f.endswith(_DATA_EXT))


def eval_shards(cfg):
    import random
    return random.Random(f"eval:{cfg}").sample(dolma_shards(cfg),
                                               EVAL_SHARDS_PER_CFG)


def dolma_eval_docs(cfg, n, cap=CAP):
    from datasets import load_dataset
    picked = eval_shards(cfg)
    print(f"  [eval] {cfg}: shards {[p.split('/')[-1] for p in picked]}",
          flush=True)
    ds = load_dataset("json", data_files=[
        f"hf://datasets/allenai/dolma3.5_pool@main/{p}" for p in picked],
        split="train", streaming=True)
    buf, docs = b"", []
    for row in ds:
        text = row.get("text") or next(
            (row[k] for k in row
             if isinstance(row[k], str) and len(row[k]) > 200), "")
        buf += text.encode("utf-8")
        while len(buf) >= cap and len(docs) < n:
            docs.append(buf[:cap]); buf = buf[cap:]
        if len(docs) >= n:
            break
    return docs


def hf_eval_docs(kind, n, skip, cap=CAP):
    """Docs from non-dolma HF sets — same serialization as stream_mix."""
    from datasets import load_dataset
    if kind == "ultrachat":
        rows = ("\n".join(m["content"] for m in r.get("messages", []))
                for r in load_dataset("HuggingFaceH4/ultrachat_200k",
                                      split="train_sft", streaming=True))
    elif kind == "glaive_function_calling":
        rows = ((r.get("system") or "") + "\n" + (r.get("chat") or "")
                for r in load_dataset(
                    "glaiveai/glaive-function-calling-v2",
                    split="train", streaming=True))
    elif kind == "toolace":
        rows = ((r.get("system") or "") + "\n" + "\n".join(
            f"{t['from'].upper()}: {t.get('value') or ''}"
            for t in r.get("conversations", []))
            for r in load_dataset("Team-ACE/ToolACE", split="train",
                                  streaming=True))
    docs, buf, skipped = [], b"", 0
    for text in rows:
        buf += text.encode("utf-8")
        while len(buf) >= cap and skipped < skip:
            buf = buf[cap:]; skipped += 1
        while len(buf) >= cap and len(docs) < n:
            docs.append(buf[:cap]); buf = buf[cap:]
        if len(docs) >= n:
            break
    return docs


def bpb_of(docs):
    vals = [eval_bpb(torch.tensor(list(d), device="cuda")) for d in docs]
    del docs
    torch.cuda.empty_cache()
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


DOMAINS = [
    ("web cc_all_dressed", "cc_all_dressed"),
    ("wiki", "common_pile_wiki"),
    ("textbook", "common_pile_textbook"),
    ("code prose", "dolma_code_prose"),
    ("code pile", "common_pile_code"),
    ("code stack-v2", "the-stack-v2"),
    ("code sponge", "sponge211unified_code_prose"),
    ("math finemath", "finemath"),
    ("math proofpile2", "proofpile2"),
    ("math swallow", "swallow-math"),
    ("multilingual hplt", "hplt"),
]
print(f"[eval] {EVAL_DOCS} docs x {CAP}B per domain (pinned eval shards)",
      flush=True)
rows = []
sha_of = {}
for label, cfg in DOMAINS:
    try:
        docs = dolma_eval_docs(cfg, EVAL_DOCS)
        sha_of[label] = hashlib.sha256(b"".join(docs)).hexdigest()[:16]
        m, s, k = bpb_of(docs)
        rows.append((label, m, s, k))
        print(f"  {label:22s} bpb {m:.3f} (+/-{s:.3f}) n={k}", flush=True)
    except Exception as e:
        print(f"  {label:22s} FAILED {type(e).__name__} {str(e)[:80]}",
              flush=True)
for kind, label in [("ultrachat", "conversation ultrachat"),
                    ("glaive_function_calling", "tooluse glaive"),
                    ("toolace", "agentic toolace")]:
    try:
        docs = hf_eval_docs(kind, EVAL_DOCS, EVAL_OFF.get(kind, 0))
        sha_of[label] = hashlib.sha256(b"".join(docs)).hexdigest()[:16]
        m, s, k = bpb_of(docs)
        rows.append((label, m, s, k))
        print(f"  {label:22s} bpb {m:.3f} (+/-{s:.3f}) n={k}", flush=True)
    except Exception as e:
        print(f"  {label:22s} FAILED {type(e).__name__} {str(e)[:80]}",
              flush=True)

# reference points
m, s, k = bpb_of([d[:CAP] for d in held_docs])
print(f"  {'held decontam (16)':22s} bpb {m:.3f} (+/-{s:.3f})", flush=True)
tot, n = 0.0, 0
for off in range(0, 200000, 20000):
    seg = ew8_test[off:off + 20000]
    v = eval_bpb(seg)
    tot += float(v) * (len(seg) - 1); n += len(seg) - 1
    del seg
    torch.cuda.empty_cache()
enwik8_bpb = tot / n
print(f"  {'enwik8-test 200k':22s} bpb {enwik8_bpb:.4f}", flush=True)

# locked-exam receipt: ckpt hash + per-domain doc hashes + pre-registered
# bar, so the table can't drift from what actually ran.
BAR = {"held": 2.386, "enwik8": 1.4442}        # pre-registered leg-9 bar
receipt = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ckpt_prefix": PREFIX, "ckpt_dir": d, "ckpt_sha": CKPT_SHA,
    "eval_docs": EVAL_DOCS, "cap": CAP, "bar": BAR,
    "domains": {label: {"bpb": m, "std": s, "n": k,
                        "docs_sha": sha_of.get(label)}
                for label, m, s, k in rows},
    "enwik8_test": enwik8_bpb,
}
held_v = [eval_bpb(torch.tensor(list(hd), device="cuda")) for hd in held_docs]
receipt["held_decontam"] = float(np.mean(held_v))
receipt["pass"] = (receipt["held_decontam"] <= BAR["held"]
                   and enwik8_bpb <= BAR["enwik8"])
os.makedirs(RESULTS, exist_ok=True)
rpath = f"{RESULTS}/eval_receipt_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
json.dump(receipt, open(rpath, "w"), indent=1)
print(f"[receipt] -> {rpath}  pass={receipt['pass']}", flush=True)
print(json.dumps({k: v for k, v in receipt.items()
                  if k != "domains"}, indent=1), flush=True)
print("[eval] complete", flush=True)
