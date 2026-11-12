"""Corpus streaming — weighted, streaming-only document streams for training legs.

This is the data half of the composition root. It changes when data sources
change; ``factory.py`` changes when trunk geometry changes. Keeping them in
one module coupled two unrelated change rates.

STREAMING only — never snapshot the pool.
"""
import os


def corpus_stream(mix, n_docs, cap=4096, seed=0):
    """Weighted multi-config stream of dolma3.5_pool — the complete corpus.

    mix: {config: weight}; docs interleaved round-robin by weight, capped
    per config so the mix holds even if a config is small.
    """
    from datasets import load_dataset
    import random
    rng = random.Random(seed)
    total_w = sum(mix.values())
    quotas = {cfg: max(1, round(n_docs * w / total_w))
              for cfg, w in mix.items()}
    docs = []
    for cfg, q in quotas.items():
        ds = load_dataset("allenai/dolma3.5_pool", cfg, split="train",
                          streaming=True)
        buf, got = b"", 0
        for row in ds:
            text = row.get("text") or next(
                (row[k] for k in row
                 if isinstance(row[k], str) and len(row[k]) > 200), "")
            buf += text.encode("utf-8")
            while len(buf) >= cap and got < q:
                docs.append(buf[:cap]); buf = buf[cap:]; got += 1
            if got >= q:
                break
    rng.shuffle(docs)
    return docs


DOLMA_MIX = {   # rebalanced: code + multilingual up (domain evals were weak)
    "cc_all_dressed": 0.25,
    "common_pile_wiki": 0.10,
    "common_pile_textbook": 0.10,
    "dolma_code": 0.15,
    "the-stack-v2": 0.15,
    "finemath": 0.05,
    "hplt": 0.15,
    "cc_all_dressed_addons": 0.05,
}


# Next-leg sources: dolma3_mix-6T shard globs (the Olmo3-32B pretrain mix)
# plus agentic/tool lineages. Addressable per-source via data_files so the
# mix is weighted by *content family*, not by shard count. Layout verified
# on the Hub: data/<source>[-<subset>]/shard_*.jsonl.zst. Set DOLMA3_REPO
# to the -150B sample for smoke tests (same mix, 224 dirs vs 63.9k).
DOLMA3_REPO = os.environ.get("DOLMA3_REPO", "allenai/dolma3_mix-6T")

DOLMA3_SOURCES = {   # (repo, data_files glob, config, split)
    "common_crawl": (DOLMA3_REPO,
                     "data/common_crawl-*/*.jsonl.zst", None, "train"),
    "olmocr_science": (DOLMA3_REPO,
                       "data/olmocr_science_pdfs-*/*.jsonl.zst", None, "train"),
    "stack_edu": (DOLMA3_REPO, "data/stack_edu-*/*.jsonl.zst", None, "train"),
    "finemath_3plus": (DOLMA3_REPO,
                       "data/finemath-3plus/*.jsonl.zst", None, "train"),
    "proofpile_arxiv": (DOLMA3_REPO,
                        "data/rpj-proofpile-arxiv/*.jsonl.zst", None, "train"),
    "dolma_wiki": (DOLMA3_REPO,
                   "data/dolma1_7-wiki-en/*.jsonl.zst", None, "train"),
    # configs/splits verified on the Hub 2026-11-09
    "nemotron_terminal": ("nvidia/Nemotron-Terminal-Corpus", None,
                          "skill_based_mixed", "train"),
    "open_swe_traces": ("nvidia/Open-SWE-Traces", None, "v2.0", "claudecode"),
}

NEXT_MIX = {   # leg-2 proposal: code/science up (HumanEval gap), agentic in
    "common_crawl": 0.24,
    "olmocr_science": 0.08,
    "stack_edu": 0.14,
    "finemath_3plus": 0.06,
    "proofpile_arxiv": 0.04,
    "dolma_wiki": 0.06,
    "nemotron_terminal": 0.14,
    "open_swe_traces": 0.14,
    "common_pile_textbook": 0.05,
    "hplt": 0.05,
}


def _text_of(row):
    """Longest plausible text field — schemas differ across sources.

    Agentic corpora (Nemotron-Terminal, Open-SWE) carry turns in
    `conversations`/`messages` lists rather than a `text` column; without
    the list path those rows serialize to ~36 bytes of metadata.
    """
    t = row.get("text")
    if isinstance(t, str) and len(t) > 200:
        return t
    for k in ("conversations", "messages", "trajectory", "content"):
        v = row.get(k)
        if isinstance(v, list) and v:
            parts = []
            for m in v:
                if isinstance(m, dict):
                    parts.append(str(m.get("content")
                                     or m.get("value") or ""))
                else:
                    parts.append(str(m))
            s = "\n".join(p for p in parts if p)
            if len(s) > 200:
                return s
    best = ""
    for v in row.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    if len(best) > 200:
        return best
    import json
    return json.dumps(row, default=str)


def corpus_stream_next(mix=None, n_docs=1000, cap=4096, seed=0):
    """Weighted multi-source stream for the next training leg.

    mix: {source_key: weight}; source_key resolves through DOLMA3_SOURCES.
    Sources are loaded independently (data_files globs for dolma3_mix) so
    a source can be reweighted without touching the others. STREAMING only.
    """
    from datasets import load_dataset
    import random
    mix = mix or NEXT_MIX
    rng = random.Random(seed)
    total_w = sum(mix.values())
    quotas = {k: max(1, round(n_docs * w / total_w)) for k, w in mix.items()}
    docs = []
    for key, q in quotas.items():
        if key not in DOLMA3_SOURCES:
            # fall back to the dolma3.5_pool config of the same name
            ds = load_dataset("allenai/dolma3.5_pool", key, split="train",
                              streaming=True)
        else:
            repo, glob, cfg, split = DOLMA3_SOURCES[key]
            kw = {"split": split, "streaming": True}
            if glob:
                kw["data_files"] = glob
            ds = (load_dataset(repo, cfg, **kw) if cfg
                  else load_dataset(repo, **kw))
        buf, got = b"", 0
        for row in ds:
            buf += _text_of(row).encode("utf-8", "ignore")
            while len(buf) >= cap and got < q:
                docs.append(buf[:cap])
                buf = buf[cap:]
                got += 1
            if got >= q:
                break
        print(f"[mix] {key}: {got}/{q} docs", flush=True)
    rng.shuffle(docs)
    return docs
