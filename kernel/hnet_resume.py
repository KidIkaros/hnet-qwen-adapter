"""Resume trainer: continue byte-NLL training of the H-Net-of-Qwen from a
checkpoint on a *different* data slice.

    colab exec -s hrun -f kernel/hnet_resume.py
    env: MODEL, CKPT_PREFIX, N_TRAIN, EPOCHS, LR, SAVE_EVERY, MIX_JSON

Two modes, picked automatically:
  live    - kernel still holds `hn`, `opt`, `held_docs`, `ew8_test` from a
            finished training cell -> resume in place (keeps AdamW state).
  rebuild - kernel restarted -> rebuild via factory, load
            results/{CKPT_PREFIX}_{glue,enc,dec}.pt, fresh AdamW.

Data: inline streamer — dolma3.5_pool configs draw a RANDOM SUBSET of shard
paths per leg (topic-partitioned layout; consumed paths logged for
SEEN_SHARDS dedup); pseudo-configs stream sequentially with per-config SKIP
of already-consumed docs.  MIX_JSON can override the mix entirely, e.g.
MIX_JSON='{"dolma4pdfs":0.5,"proofpile2":0.5}'.

Checkpoints: saved every SAVE_EVERY steps AND at the end, to
results/hnet_resume_{glue,enc,dec}.pt — never leave weights only in memory.
"""
import os, sys, json, glob, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "/content/qt/hnet-qwen-adapter"
RESULTS = f"{REPO}/results"
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "hnet_2b_v2")
OUT_PREFIX = "hnet_resume"
P = 6


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402

LIVE = all(k in dir() for k in ("hn", "opt", "held_docs", "ew8_test",
                                "eval_bpb", "doc_loss"))
print(f"[resume] mode={'LIVE (reuse kernel state)' if LIVE else 'REBUILD'}",
      flush=True)

if not LIVE:
    if os.path.exists(REPO):
        sh("git", "-C", REPO, "pull", "-q")
    else:
        sh("git", "clone", "-q",
           "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from src.factory import (build_hnet, enwik8, fit_adapters,    # noqa: E402
                             load_trunk, periodic)

    trunk = load_trunk(os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base"))
    hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)

    all_docs = [open(f, "rb").read()
                for f in sorted(glob.glob("data/decontam/*.bin"))]
    fit_docs = [d[:4096] for d in all_docs[:4]]
    held_docs = [d[:4096] for d in all_docs[-16:]]   # 16-doc held: stable stop
    ew8 = enwik8()
    ew8_test = torch.tensor(list(ew8["test"][:200_000]), device="cuda")

    g = fit_adapters(hn, trunk, meta, fit_docs, P=P)   # sets adapter buffers
    print(f"[fit] adapters fit | {g} dechunk pairs", flush=True)


CKPT_DIRS = [RESULTS, "/content/ckpt"]


def load_ckpt(prefix):
    glue = enc = dec = None
    for d in CKPT_DIRS:
        try:
            glue = torch.load(f"{d}/{prefix}_glue.pt", map_location="cuda")
            enc = torch.load(f"{d}/{prefix}_enc.pt", map_location="cuda")
            dec = torch.load(f"{d}/{prefix}_dec.pt", map_location="cuda")
            break
        except FileNotFoundError:
            continue
    if glue is None:
        raise FileNotFoundError(f"{prefix}_*.pt not in {CKPT_DIRS}")
    hn.w_sup.load_state_dict(glue["w_sup"])
    hn.dechunk.load_state_dict(glue["dechunk"])
    hn.boundary.load_state_dict(glue["boundary"])
    for i, l in enumerate(hn.enc):
        l.load_state_dict(enc[f"enc_{i}"])
    for i, l in enumerate(hn.dec):
        l.load_state_dict(dec[f"dec_{i}"])
    print(f"[ckpt] loaded {prefix}_{{glue,enc,dec}}.pt", flush=True)


if not LIVE:
    # prefix chain: resume artifacts first (crash-resume), then requested, v1
    for cand in ["hnet_resume_latest", "hnet_resume_best", CKPT_PREFIX,
                 "hnet_2b"]:
        try:
            load_ckpt(cand)
            CKPT_PREFIX = cand
            break
        except FileNotFoundError:
            continue
elif os.environ.get("FORCE_LOAD"):
    try:
        load_ckpt(CKPT_PREFIX)                     # swap live model weights
    except FileNotFoundError:
        print(f"[ckpt] {CKPT_PREFIX} files absent; keeping live weights",
              flush=True)

if not LIVE:
    for layer in list(hn.enc) + list(hn.dec):
        layer.gradient_checkpointing = True
    trainable = list(hn.enc.parameters()) + list(hn.dec.parameters()) + \
        list(hn.dechunk.parameters()) + list(hn.boundary.parameters()) + \
        list(hn.w_sup.parameters())
    for p in trainable:
        p.requires_grad_(True)
    base_lr = float(os.environ.get("LR", "1e-4"))
    opt = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=0.01)

    def doc_loss(byte_ids):
        T = len(byte_ids)
        bp, _, _ = hn(byte_ids, boundaries=periodic(T, P), ctx="retrieve",
                      n_ctx=1, super_k=4, head_first=trunk["head"].first)
        nll = -bp[0, :-1].gather(
            1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
        aux = (hn.last_p.mean() - 1.0 / P) ** 2
        return nll.mean() + 1.0 * aux

    def eval_bpb(byte_ids):
        with torch.no_grad():
            return float(doc_loss(byte_ids)) / np.log(2)
else:
    os.chdir(REPO)

# ---- data: leg-8 REDO — leg-7's winning recipe verbatim (toolace .15
# FAILED: held spiked to 3.15, early stop @625, both signals regressed).
# Keeps only the longer leg: N_TRAIN=6000, TRAIN_SHARDS_PER_CFG=8.
# Seeds from leg-7 best (re-uploaded — leg8's run overwrote remote best).
RESUME_MIX = {
    "the-stack-v2": 0.15,
    "common_pile_code": 0.15,
    "ultrachat": 0.10,                   # conversations (pseudo-config)
    "dolma_code_prose": 0.10,
    "sponge211unified_code_prose": 0.10,
    "glaive_function_calling": 0.10,     # tool calls / JSON (pseudo-config)
    "toolace": 0.10,                     # multi-turn tool calls (pseudo-config)
    "replay": 0.05,                      # experience replay (rotating buffer)
    "cc_all_dressed": 0.10,              # fresh-domain guard
    "enwik8_train": 0.05,                # anchor
    # dolma4pdfs / swallow-code excluded: fail to stream (probed 2026-11-05)
}
# pseudo-configs consume sequentially from the stream head -> doc counts.
# enwik8_train evals on the test split, so train-split consumption only.
# ultrachat/toolace pinned eval = docs [0,24) (never trained); glaive pinned
# eval = docs [824,848) (0-824 already trained) -> counts include eval pins.
SEEN = {"enwik8_train": 2700, "ultrachat": 2448,
        "glaive_function_calling": 2248, "toolace": 1824}
# dolma cfgs consume SHARD PATHS, not prefix positions (legs 6+;
# RESEARCH_DATA_ORDERING §5.3): configs are topic-partitioned dirs, so
# sequential takes covered only their first topic dirs. Path-level dedup
# starts here — legs 1-5's prefix consumption can't be mapped to shard ids,
# accepted (<4 repeats/doc is free, arXiv:2305.16264). Log prints picked
# paths; merge them in here after each leg.
SEEN_SHARDS = {'the-stack-v2': ['the-stack-v2/Common_Lisp/quality_p50/shard_00000062.jsonl.zst', 'the-stack-v2/Pascal/quality_p90/shard_00000078.jsonl.zst', 'the-stack-v2/Verilog/quality_p90/shard_00000108.jsonl.zst', 'the-stack-v2/TypeScript/quality_p15/shard_00000076.jsonl.zst', 'the-stack-v2/Swift/quality_p30/shard_00000000.jsonl.zst', 'the-stack-v2/C-Sharp/quality_p50/shard_00000006.jsonl.zst', 'the-stack-v2/Common_Lisp/quality_p50/shard_00000097.jsonl.zst', 'the-stack-v2/Pascal/quality_p90/shard_00000083.jsonl.zst', 'the-stack-v2/Verilog/quality_p95/shard_00000023.jsonl.zst', 'the-stack-v2/TypeScript/quality_p15/shard_00000135.jsonl.zst', 'the-stack-v2/Swift/quality_p30/shard_00000044.jsonl.zst', 'the-stack-v2/C-Sharp/quality_p50/shard_00000017.jsonl.zst', 'the-stack-v2/Common_Lisp/quality_p50/shard_00000108.jsonl.zst', 'the-stack-v2/Pascal/quality_p90/shard_00000119.jsonl.zst', 'the-stack-v2/Verilog/quality_p95/shard_00000105.jsonl.zst', 'the-stack-v2/TypeScript/quality_p15/shard_00000154.jsonl.zst', 'the-stack-v2/Swift/quality_p30/shard_00000083.jsonl.zst', 'the-stack-v2/C-Sharp/quality_p50/shard_00000021.jsonl.zst', 'the-stack-v2/Java/quality_p00/shard_00000776.jsonl.zst', 'the-stack-v2/Clojure/quality_p20/shard_00000092.jsonl.zst'], 'common_pile_code': ['common_pile_code/stackexchange/shard_00000052_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa25.json.zst', 'common_pile_code/github_archive/shard_00000004_upl_10.json.zst', 'common_pile_code/stackexchange/shard_00000142_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa30.json.zst', 'common_pile_code/stackexchange/shard_00000125_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/stackexchange/shard_00000133_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa30.json.zst', 'common_pile_code/stackexchange/shard_00000203_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa30.json.zst', 'common_pile_code/stackexchange/shard_00000052_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/github_archive/shard_00000004_vim.json.zst', 'common_pile_code/stackexchange/shard_00000144_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa25.json.zst', 'common_pile_code/stackexchange/shard_00000126_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/stackexchange/shard_00000135_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa25.json.zst', 'common_pile_code/stackexchange/shard_00000099_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa30.json.zst', 'common_pile_code/stackexchange/shard_00000146_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/stackexchange/shard_00000128_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa25.json.zst', 'common_pile_code/stackexchange/shard_00000136_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/stackexchange/shard_00000100_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa40.json.zst', 'common_pile_code/stackexchange/shard_00000035_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa30.json.zst', 'common_pile_code/github_archive/shard_00000003_vim.json.zst', 'common_pile_code/stackexchange/shard_00000143_creative_commons___attribution_share_alike___httpscreativecommonsorglicensesby_sa25.json.zst', 'common_pile_code/github_archive/shard_00000001_bsd_2_clause.json.zst'], 'dolma_code_prose': ['dolma_code_prose/quality_p55/shard_00000026.jsonl.zst', 'dolma_code_prose/quality_p30/shard_00000052.jsonl.zst', 'dolma_code_prose/quality_p15/shard_00000006.jsonl.zst', 'dolma_code_prose/quality_p70/shard_00000066.jsonl.zst', 'dolma_code_prose/quality_p00/shard_00000078.jsonl.zst', 'dolma_code_prose/quality_p55/shard_00000054.jsonl.zst', 'dolma_code_prose/quality_p30/shard_00000100.jsonl.zst', 'dolma_code_prose/quality_p15/shard_00000029.jsonl.zst', 'dolma_code_prose/quality_p75/shard_00000042.jsonl.zst', 'dolma_code_prose/quality_p05/shard_00000002.jsonl.zst', 'dolma_code_prose/quality_p60/shard_00000054.jsonl.zst', 'dolma_code_prose/quality_p65/shard_00000057.jsonl.zst', 'dolma_code_prose/quality_p60/shard_00000106.jsonl.zst', 'dolma_code_prose/quality_p70/shard_00000034.jsonl.zst', 'dolma_code_prose/quality_p95/shard_00000069.jsonl.zst', 'dolma_code_prose/quality_p00/shard_00000000.jsonl.zst', 'dolma_code_prose/quality_p70/shard_00000080.jsonl.zst', 'dolma_code_prose/quality_p40/shard_00000117.jsonl.zst', 'dolma_code_prose/quality_p35/shard_00000100.jsonl.zst', 'dolma_code_prose/quality_p95/shard_00000051.jsonl.zst'], 'sponge211unified_code_prose': ['sponge211unified_code_prose/quality_p60/shard_00000085.jsonl.zst', 'sponge211unified_code_prose/quality_p85/shard_00000103.jsonl.zst', 'sponge211unified_code_prose/quality_p00/shard_00000000.jsonl.zst', 'sponge211unified_code_prose/quality_p60/shard_00000110.jsonl.zst', 'sponge211unified_code_prose/quality_p35/shard_00000100.jsonl.zst', 'sponge211unified_code_prose/quality_p30/shard_00000098.jsonl.zst', 'sponge211unified_code_prose/quality_p90/shard_00000062.jsonl.zst', 'sponge211unified_code_prose/quality_p00/shard_00000004.jsonl.zst', 'sponge211unified_code_prose/quality_p65/shard_00000057.jsonl.zst', 'sponge211unified_code_prose/quality_p40/shard_00000007.jsonl.zst', 'sponge211unified_code_prose/quality_p30/shard_00000121.jsonl.zst', 'sponge211unified_code_prose/quality_p90/shard_00000050.jsonl.zst', 'sponge211unified_code_prose/quality_p15/shard_00000025.jsonl.zst', 'sponge211unified_code_prose/quality_p45/shard_00000115.jsonl.zst', 'sponge211unified_code_prose/quality_p05/shard_00000002.jsonl.zst', 'sponge211unified_code_prose/quality_p00/shard_00000078.jsonl.zst', 'sponge211unified_code_prose/quality_p00/shard_00000089.jsonl.zst', 'sponge211unified_code_prose/quality_p80/shard_00000095.jsonl.zst', 'sponge211unified_code_prose/quality_p00/shard_00000045.jsonl.zst', 'sponge211unified_code_prose/quality_p55/shard_00000054.jsonl.zst'], 'cc_all_dressed': ['cc_all_dressed/software_development/vigintile_0011/part_05_shard_00000002.jsonl.gz', 'cc_all_dressed/entertainment/vigintile_0002/part_10_shard_00000011.jsonl.gz', 'cc_all_dressed/health/vigintile_0013/part_02_shard_00000002.jsonl.gz', 'cc_all_dressed/crime_and_law/vigintile_0007/part_23_shard_00000000.jsonl.gz', 'cc_all_dressed/art_and_design/vigintile_0019/part_21_shard_00000005.jsonl.gz', 'cc_all_dressed/crime_and_law/vigintile_0002/part_13_shard_00000008.jsonl.gz', 'cc_all_dressed/entertainment/vigintile_0002/part_10_shard_00000015.jsonl.gz', 'cc_all_dressed/health/vigintile_0013/part_02_shard_00000007.jsonl.gz', 'cc_all_dressed/crime_and_law/vigintile_0007/part_23_shard_00000003.jsonl.gz', 'cc_all_dressed/art_and_design/vigintile_0019/part_21_shard_00000006.jsonl.gz', 'cc_all_dressed/crime_and_law/vigintile_0002/part_13_shard_00000010.jsonl.gz', 'cc_all_dressed/transportation/vigintile_0016/part_09_shard_00000006.jsonl.gz', 'cc_all_dressed/finance_and_business/vigintile_0014/part_22_shard_00000004.jsonl.gz', 'cc_all_dressed/literature/vigintile_0003/part_00_shard_00000011.jsonl.gz', 'cc_all_dressed/crime_and_law/vigintile_0006/part_07_shard_00000005.jsonl.gz', 'cc_all_dressed/science_math_and_technology/vigintile_0013/part_20_shard_00000001.jsonl.gz', 'cc_all_dressed/finance_and_business/vigintile_0016/part_09_shard_00000022.jsonl.gz', 'cc_all_dressed/literature/vigintile_0017/part_21_shard_00000001.jsonl.gz', 'cc_all_dressed/religion/vigintile_0010/part_18_shard_00000004.jsonl.gz', 'cc_all_dressed/social_life/vigintile_0011/part_10_shard_00000012.jsonl.gz']}

# dolma3.5_pool shard sampling (train) + pinned eval shards (see
# kernel/hnet_domain_eval.py — the "eval:"-seeded pick is identical there).
TRAIN_SHARDS_PER_CFG = 8    # random subset per leg -> topic coverage
EVAL_SHARDS_PER_CFG = 4     # pinned, never trained
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

mix = json.loads(os.environ["MIX_JSON"]) if os.environ.get("MIX_JSON") \
    else RESUME_MIX


def row_texts(cfg, rng, consumed):
    """Text-row iterator for a mix config. dolma3.5_pool configs stream a
    RANDOM SUBSET of shard paths (topic-partitioned layout — sequential takes
    covered 1-2 topic dirs); pseudo-configs stream whole datasets.
    `consumed` collects picked shard paths for SEEN_SHARDS bookkeeping.
    """
    from datasets import load_dataset
    if cfg == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                          streaming=True)
        return ("\n".join(m["content"] for m in r.get("messages", []))
                for r in ds)
    if cfg == "glaive_function_calling":
        ds = load_dataset("glaiveai/glaive-function-calling-v2",
                          split="train", streaming=True)
        return ((r.get("system") or "") + "\n" + (r.get("chat") or "")
                for r in ds)
    if cfg == "toolace":
        ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
        return ((r.get("system") or "") + "\n" + "\n".join(
            f"{t['from'].upper()}: {t.get('value') or ''}"
            for t in r.get("conversations", [])) for r in ds)
    seen = set(eval_shards(cfg)) | set(SEEN_SHARDS.get(cfg, ()))
    pool = [f for f in dolma_shards(cfg) if f not in seen]
    picked = rng.sample(pool, min(TRAIN_SHARDS_PER_CFG, len(pool)))
    if not picked:
        raise RuntimeError(f"{cfg}: shard pool exhausted")
    consumed.extend(picked)
    print(f"  [data] {cfg}: shards {picked}", flush=True)
    ds = load_dataset("json", data_files=[
        f"hf://datasets/allenai/dolma3.5_pool@main/{p}" for p in picked],
        split="train", streaming=True)
    return (row.get("text") or next(
        (row[k] for k in row
         if isinstance(row[k], str) and len(row[k]) > 200), "")
        for row in ds)


# experience-replay buffer: fixed-4096 docs concatenated in one file.
# results/ copy wins (fresh this-VM), repo data/ copy is the durable one —
# upload it at leg fire, pull results/replay_buffer.bin back after the leg.
REPLAY_CAP = 512          # buffer size in docs (uniform over all legs)
REPLAY_TOPUP = 128        # docs sampled from this leg's collection into it
REPLAY_PATHS = (f"{RESULTS}/replay_buffer.bin", "data/replay_buffer.bin")


def load_replay():
    for p in REPLAY_PATHS:
        if os.path.exists(p):
            raw = open(p, "rb").read()
            return [raw[i:i + 4096] for i in range(0, len(raw) - 4095, 4096)]
    return []


def update_replay(docs, rng):
    """Reservoir-merge a sample of this leg's docs into the buffer."""
    buf = load_replay() + rng.sample(docs, min(REPLAY_TOPUP, len(docs)))
    if len(buf) > REPLAY_CAP:
        buf = rng.sample(buf, REPLAY_CAP)
    os.makedirs(RESULTS, exist_ok=True)
    open(REPLAY_PATHS[0], "wb").write(b"".join(buf))
    print(f"  [data] replay buffer: {len(buf)} docs", flush=True)
    return buf


def stream_mix(mix, n_docs, cap=4096, seed=0, seen=None):
    """corpus_stream with per-config skip of already-consumed docs."""
    import random
    rng = random.Random(seed)
    seen = seen or {}
    consumed_shards = {}
    total_w = sum(mix.values())
    docs = []
    for cfg, w in mix.items():
        if cfg == "replay":
            continue                       # handled post-loop (needs docs)
        q = max(1, round(n_docs * w / total_w))
        skip = seen.get(cfg, 0)
        if cfg == "enwik8_train":
            try:
                from src.factory import enwik8
                raw = enwik8()["train"]                # first 90M bytes
                got = 0
                for off in range(skip * cap, len(raw) - cap, cap):
                    docs.append(bytes(raw[off:off + cap]))
                    got += 1
                    if got >= q:
                        break
                print(f"  [data] {cfg}: skipped {skip}, took {got}",
                      flush=True)
            except Exception as e:
                print(f"  [data] {cfg}: UNLOADABLE ({type(e).__name__}) "
                      f"- skipped", flush=True)
            continue
        try:
            it = row_texts(cfg, rng, consumed_shards.setdefault(cfg, []))
        except Exception as e:
            print(f"  [data] {cfg}: UNLOADABLE ({type(e).__name__}: "
                  f"{str(e)[:80]}) - skipped", flush=True)
            continue
        buf, got, skipped = b"", 0, 0
        try:
            for text in it:
                buf += text.encode("utf-8")
                while len(buf) >= cap and skipped < skip:
                    buf = buf[cap:]; skipped += 1
                while len(buf) >= cap and got < q:
                    docs.append(buf[:cap]); buf = buf[cap:]; got += 1
                if got >= q:
                    break
        except Exception as e:
            print(f"  [data] {cfg}: stream broke ({type(e).__name__}) "
                  f"after {got} docs - keeping them", flush=True)
        print(f"  [data] {cfg}: skipped {skipped}, took {got}", flush=True)
    if "replay" in mix:
        q = max(1, round(n_docs * mix["replay"] / total_w))
        buf = update_replay(docs, rng)
        take = rng.sample(buf, min(q, len(buf)))
        docs += take
        print(f"  [data] replay: took {len(take)} (buffer {len(buf)})",
              flush=True)
    if consumed_shards:
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(consumed_shards,
                  open(f"{RESULTS}/consumed_shards.json", "w"), indent=1)
        print("  [data] consumed shards -> results/consumed_shards.json "
              "(merge into SEEN_SHARDS)", flush=True)
    rng.shuffle(docs)
    return docs


N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
train_docs = stream_mix(mix, N_TRAIN, seed=int(os.environ.get("SEED", "1")),
                        seen=SEEN)
print(f"[data] {len(train_docs)} train (resume mix)", flush=True)

n_train = sum(p.numel() for p in
              (list(hn.enc.parameters()) + list(hn.dec.parameters())))
print(f"[train] {n_train/1e6:.1f}M in enc/dec slices", flush=True)


def save_ckpt(prefix, tag):
    os.makedirs(RESULTS, exist_ok=True)
    torch.save({"w_sup": hn.w_sup.state_dict(),
                "dechunk": hn.dechunk.state_dict(),
                "boundary": hn.boundary.state_dict()},
               f"{RESULTS}/{prefix}_glue.pt")
    torch.save({f"enc_{i}": l.state_dict() for i, l in enumerate(hn.enc)},
               f"{RESULTS}/{prefix}_enc.pt")
    torch.save({f"dec_{i}": l.state_dict() for i, l in enumerate(hn.dec)},
               f"{RESULTS}/{prefix}_dec.pt")
    print(f"  [ckpt] saved -> {prefix}_{{glue,enc,dec}}.pt ({tag})",
          flush=True)


def held_mean():
    """Stable early-stop signal: mean bpb over all held decontam docs."""
    return float(np.mean([eval_bpb(torch.tensor(list(hd), device="cuda"))
                          for hd in held_docs]))


EPOCHS = int(os.environ.get("EPOCHS", "1"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "250"))
# plateau early-stop: held-mean not improving for PATIENCE evals -> stop.
# Kept generous (24 evals ~= 600 steps, 0.002 improvement bar) so slow gains
# keep the run alive; it fires on flat held-mean, not on active regression.
PATIENCE = int(os.environ.get("PATIENCE", "24"))
MIN_STEPS = int(os.environ.get("MIN_STEPS", "500"))
IMPROVE = float(os.environ.get("IMPROVE", "0.002"))
WARMUP = int(os.environ.get("WARMUP", "100"))      # LR ramp-in (rewarm fix)
SWA_STALE = int(os.environ.get("SWA_STALE", "8"))  # start tail-avg at this staleness
step, stale = 0, 0
best_held, best_step = float("inf"), 0
swa_state, swa_n = None, 0
stop = False
for ep in range(EPOCHS):
    for d in train_docs:
        byte_ids = torch.tensor(list(d), device="cuda")
        loss = doc_loss(byte_ids)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g_ in opt.param_groups for p in g_["params"]
             if p.requires_grad], 1.0)
        for g in opt.param_groups:
            g["lr"] = base_lr * min(1.0, (step + 1) / WARMUP)
        opt.step()
        step += 1
        if step % 25 == 0 or step == len(train_docs):
            hb = eval_bpb(torch.tensor(
                list(held_docs[step % len(held_docs)]), device="cuda"))
            ew = eval_bpb(ew8_test[:50000])
            hm = held_mean()
            if hm < best_held - IMPROVE:
                best_held, best_step, stale = hm, step, 0
                save_ckpt("hnet_resume_best", f"best held {hm:.3f}")
            else:
                stale += 1
            if stale >= SWA_STALE:
                if swa_state is None:
                    swa_state = [p.detach().clone() for p in trainable]
                else:
                    for s, p in zip(swa_state, trainable):
                        s.add_(p.detach())
                swa_n += 1
            print(f"  ep{ep} step {step:<5} loss {float(loss):.3f} "
                  f"|gn| {float(gn):.2f} | held {hb:.3f} (mean {hm:.3f}) "
                  f"| enwik8 {ew:.3f} | best {best_held:.3f}@{best_step}",
                  flush=True)
            if step >= MIN_STEPS and stale >= PATIENCE:
                print(f"[stop] early stop: held plateaued {stale} evals; "
                      f"best {best_held:.3f} @ step {best_step}", flush=True)
                stop = True
                break
        if step % SAVE_EVERY == 0:
            save_ckpt("hnet_resume_latest", f"step {step}")
        del byte_ids, loss
        torch.cuda.empty_cache()
    if stop:
        break

if swa_state is not None:
    cur = [p.detach().clone() for p in trainable]
    with torch.no_grad():
        for p, s in zip(trainable, swa_state):
            p.copy_(s / swa_n)
    hm_swa = held_mean()
    print(f"[swa] avg of {swa_n} tail snapshots: held {hm_swa:.3f} "
          f"(best {best_held:.3f}@{best_step})", flush=True)
    if hm_swa < best_held - IMPROVE:
        best_held = hm_swa
        save_ckpt("hnet_resume_best", f"swa best held {hm_swa:.3f}")
    with torch.no_grad():
        for p, c in zip(trainable, cur):
            p.copy_(c)

save_ckpt("hnet_resume_latest", f"final step {step}")
# report final evals on the BEST checkpoint (anti-overfit artifact)
if best_step and best_step < step:
    print(f"[eval] reloading best checkpoint (step {best_step})", flush=True)
    try:
        load_ckpt("hnet_resume_best")
    except FileNotFoundError:
        print("[eval] best ckpt missing - evaluating final weights", flush=True)
tot, n = 0.0, 0
for off in range(0, 200000, 20000):
    seg = ew8_test[off:off + 20000]
    if len(seg) < 2:
        break
    v = eval_bpb(seg)
    tot += float(v) * (len(seg) - 1); n += len(seg) - 1
    print(f"  seg {off // 20000}: {v:.4f}", flush=True)
    del seg
    torch.cuda.empty_cache()
print(f"[eval] ENWIK8-TEST bpb {tot / n:.4f} ({n} positions)", flush=True)
hb_f = [eval_bpb(torch.tensor(list(hd), device="cuda")) for hd in held_docs]
print(f"[eval] held decontam {np.mean(hb_f):.4f} "
      f"({['%.3f' % v for v in hb_f]})", flush=True)
print("[resume] complete", flush=True)
