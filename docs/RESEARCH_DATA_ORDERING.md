# Research — data ordering, stream sampling, and replay for leg-wise continued pretraining

Date: 2026-11-06. Question on the table: our CPT runs in "legs"
(`kernel/hnet_resume.py` → `stream_mix`). Each leg streams
`allenai/dolma3.5_pool` configs with `streaming=True`, SKIPS the docs prior
legs consumed (per-config `SEEN` dict), takes the next `q` contiguous docs,
then globally shuffles the ~4000 collected docs for a single pass (EPOCHS=1).
Per-leg domain evals (`kernel/hnet_domain_eval.py` → `dolma_eval_docs`) also
always pull the slice just past `SEEN`. Symptom: a deeper the-stack-v2 slice
scored **2.276 bpb vs 0.777** for the earlier slice — within-config stream
position correlates with content/difficulty, so the model may be learning
"the head of each stream" rather than each domain, and cross-leg eval deltas
conflate forgetting with slice drift (already flagged in RUN_STATE.md).
Constraints: streaming only (no row-level random access), ~9 corpus configs,
4096-byte docs, 16-doc held-out for early stop, SEEN-dedup preferred but
relaxable.

## 1. Stream shuffling in production pipelines — what the tools actually do

### 1.1 HuggingFace IterableDataset — buffer shuffle + shard permutation

- `IterableDataset.shuffle(buffer_size, seed)` does *approximate* shuffling
  via an in-memory buffer. Current implementation
  (`src/datasets/iterable_dataset.py` on main): fill the buffer to
  `buffer_size`, `rng.permutation`, **yield half, keep half, refill** — a
  rolling randomized buffer. (The older v1.10.2 implementation instead
  replaced the yielded slot with the next incoming item.) Docs examples use
  buffer_size 100–1000; larger = closer to uniform, bounded by memory.
  (https://huggingface.co/docs/datasets/stream,
  https://huggingface.co/docs/datasets/main/about_mapstyle_vs_iterable,
  https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py)
- The docs are explicit that the buffer alone "is not enough to provide a
  satisfactory shuffling for machine learning model training. So
  `IterableDataset.shuffle()` **also shuffles the dataset shards** if your
  dataset is made of multiple files or sources." Two-level scheme: shard
  order permuted (global coverage) + buffer for local doc mixing.
  (about_mapstyle_vs_iterable, ibid.)
- Consequence for us: calling `.shuffle(seed=leg_seed, buffer_size=B)` on the
  iterable BEFORE skip/take already converts a contiguous take into a sample
  spread across the config's shard space — a one-line mitigation (see §5.3).
- `interleave_datasets(datasets, probabilities=..., seed=...)` does doc-level
  mixture interleaving with per-source probabilities; `stopping_strategy`
  defaults to `first_exhausted` (subsampling); `all_exhausted` oversamples.
  (https://huggingface.co/docs/datasets/stream,
  https://github.com/huggingface/datasets/blob/main/src/datasets/combine.py)

### 1.2 MosaicML StreamingDataset — index-space shuffles (py1e/py1br/py1s/py2s)

- Shuffle controls: `shuffle`, `shuffle_algo` (default `'py1e'`),
  `shuffle_seed` (9176), `shuffle_block_size` (default
  `max(4_000_000/num_canonical_nodes, 1<<18)` ≈ 262k samples), and
  `num_canonical_nodes` (default = # physical nodes). Unshuffled data is
  warned to cause "divergence, loss spikes, or suboptimal training."
  (https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shuffling.html)
- Mechanics: shards are partitioned over "canonical nodes" (sample buckets);
  shard→canonical-node assignment is **fixed across epochs**; samples are
  shuffled within each canonical node; `epoch` is added to the seed so each
  epoch gets a different deterministic permutation within fixed buckets. This
  is how they get epoch-to-epoch uniqueness without re-partitioning.
  (shuffling docs + api_reference for get_shuffle_py1br/py2s)
- Algorithm tradeoffs (docs + `streaming/base/shuffle/*.py`):
  - `py1e` (default): each shard's samples are redistributed within a range
    up to `shuffle_block_size` — best cache/shuffle-quality balance, k=1
    cache factor.
  - `py1br`: shuffles in randomized blocks (0.75–1.25× block_size) — slightly
    higher quality, ~2× cache.
  - `py1s`: intra-shard shuffle only — minimal cache, lowest quality.
  - `py2s`: shard-granular shuffle.
  (https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.base.shuffle.get_shuffle_py1br.html,
  .../get_shuffle_py2s.html; DeepWiki summary of streaming/base/shuffle/)
- Production answer to "how big a buffer": the quality-critical quantity is
  the ~2^18–4M-sample *shuffle block over the global sample index*, not the
  doc buffer — shards are the downloadable unit, and block size trades cache
  limit vs shuffle breadth, not correctness.
- Multi-source mixing: `Stream(proportion | repeat | choose)` mixes at
  SAMPLE level into one global permutation; note `shuffle=True` is required
  for actual interleaving (default off).
  (https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/mixing_data_sources.html,
  https://github.com/mosaicml/streaming/issues/342)

### 1.3 WebDataset/OpenLM and the dolma toolkit — two-level shuffling is the norm

- OpenLM (`mlfoundations/open_lm`, the DCLM trainer): webdataset tar shards;
  per-source manifests, `--train-data-mix-weights` interleaves sources at
  shard granularity, `--shard-shuffle-seed` permutes shard order per epoch,
  plus the standard webdataset `.shuffle(buf)` doc buffer (~1000).
  (https://github.com/mlfoundations/open_lm/blob/main/open_lm/main.py,
  https://github.com/webdataset/webdataset)
- AI2's own dolma `tokens` tool does only "**light shuffling**": shuffle
  input file paths, then shuffle each k-document chunk read across N open
  files before writing. `dolma mix` interleaves streams at document
  granularity. ⇒ even the dataset producer treats file-level permutation +
  chunk buffer as sufficient; nobody pays for uniform-random document order.
  (https://allenai.github.io/dolma/docs/tokenize.html,
  https://allenai.github.io/dolma/docs/mixer.html)

**Synthesis**: the industry pattern is *shard/index-level randomization +
modest in-memory buffer* — never contiguous consumption of an ordered
stream. Our `stream_mix` does the exact thing nobody does: it reads each
ordered stream head-on and takes the next q docs. The faithful fix is
shard-level randomization (§5.3), which is also the cheapest.

## 2. Does document ORDER itself matter at pretraining scale?

Surprisingly little, for decoder LMs — but the literature is about order
*given full coverage*, not about which subpopulation a window covers.

- **KnoSys 2026**, "On the impact of pretraining data ordering in
  transformer encoder- and decoder-only LMs": systematic ordering ablations
  (readability curricula + inverted variants vs random baselines). Ordering
  acts as a "*selective inductive bias*" — it reshapes optimization dynamics
  and representation geometry (anisotropy↑) but gives no consistent
  performance win; encoder models more sensitive, **decoder models
  "comparatively more stable under forward curricula."**
  (https://doi.org/10.1016/j.knosys.2026.115850)
- **Fabre et al.**, "Understanding Data Temporality Impact on LLM
  Pre-training" (PMLR v306): 6B models pretrained on *temporally ordered*
  Common Crawl snapshots vs shuffled — sequential matches shuffled on
  general language understanding and common knowledge; shuffled peaks on
  older data (repetition recency), sequential keeps fresher knowledge.
  Sequential order is not catastrophic even at 6B.
  (https://proceedings.mlr.press/v306/fabre26a.html)
- Small-scale curriculum precedent: Bengio et al. 2009
  (https://doi.org/10.1145/1553374.1553380); "On the Role of Corpus
  Ordering in Language Modeling" (ACL SustainLP 2021) — doc-level
  hard-to-easy ordering beat vanilla by +1.7 GLUE at 1 epoch, 2× fewer
  steps. (https://aclanthology.org/2021.sustainlp-1.15/)
- Ordering as a lever: **In-Context Pretraining** (ICLR 2024) — sorting
  semantically related docs into the same context windows yields +8 ICL /
  +15 reading-comprehension / +9 retrieval vs random doc order. Proof that
  doc order is a real degree of freedom — but only when it changes what
  co-occurs *in context*. Our docs are independent 4KB units (no cross-doc
  context), so this lever is weak for us. (arXiv:2310.10638)

### Repetition vs coverage — the dedup question

- **Muennighoff et al.**, "Scaling Data-Constrained Language Models"
  (arXiv:2305.16264; JMLR 26:53, 2025): up to ~**4 epochs** of repeated data
  is ≈ free vs unique data at fixed compute; value decays to ~0 only by
  ~40 epochs. StarCoder2 stayed ≤5 epochs deliberately (arXiv:2402.19173).
- **Hernandez et al.** (arXiv:2205.10487, Anthropic): the dangerous pattern
  is a *small fraction repeated many times* — 0.1% of data repeated 100×
  degraded an 800M model to ~400M effective (double descent); repetition
  disproportionately damages copying/induction-head machinery.
- **Carlini et al.** (arXiv:2202.07646, ICLR23): memorization grows
  log-linearly with duplication count. **Lee et al.** (arXiv:2107.06499,
  ACL22): dedup is neutral-to-better; near-dupes drive verbatim emission.

⇒ **Exact SEEN-dedup can be relaxed.** A doc seen 2–4 times across the whole
run is fine (Muennighoff). What to avoid is a small *fixed* set replayed
every leg at high cumulative frequency (Hernandez). A rotating reservoir
buffer (§5.2) keeps per-doc repeats low.

## 3. Domain-mixture scheduling — what exists, and what it assumes

- **DoReMi** (arXiv:2305.10429, NeurIPS23): 280M proxy trained with Group
  DRO — dynamically upweight domains where the proxy's excess loss over a
  reference model is largest; resulting weights trained an 8B 2.6× faster,
  +6.5pt avg few-shot on The Pile.
- **RegMix** (arXiv:2407.01492, ICLR25): fit a regression over hundreds of
  1M-param proxy runs on varied mixes → predict the best mix; beats human
  mixtures up to 7B/100B tokens. Findings: web corpora — not "high-quality"
  wiki — correlate strongest with downstream performance; domain
  interactions are counterintuitive → automate, don't hand-tune.
- **DoGE** (arXiv:2310.15393, ICML24): bilevel "generalization estimation"
  of domain weights; better OOD-target generalization.
- **ODM** (arXiv:2312.02406): *online* mixing — Exp3 multi-armed bandit over
  per-domain training losses, weights updated during training, no proxy or
  reference model; 19% fewer steps to baseline final perplexity.
  **Best structural fit for us**: our per-domain bpb eval each leg is
  exactly the reward signal an online mixer consumes.
- **DGA** (arXiv:2410.02498): online domain-weight updates by gradient
  alignment between domain gradients and a proxy objective.
- **Data Mixing Laws** (arXiv:2403.16952, ICLR25): fitted
  performance-vs-mixture law predicts unseen mixes; "extending to continual
  training accurately predicts the critical mixture proportion that avoids
  catastrophic forgetting."
- **D-CPT Law** (arXiv:2406.01375, NeurIPS24): CPT-specific mixing law
  L(N,D,r) for the general:domain ratio, plus a cross-domain extension via a
  domain-learnability coefficient.
- Production schedules: **SmolLM2** (arXiv:2502.02737) — multi-stage
  mixtures, hold best math/code for the annealing stage, stay ≤~4 epochs;
  **OLMo 2** (arXiv:2501.00656) — two-stage, Dolmino-Mix-1124 targeted
  high-quality data in stage 2; **DCLM** (arXiv:2406.11794) — mixture-track
  ablations.

**Verdict on sequential-stream bias**: *nothing* in the mixture literature
addresses it — every method assumes uniform-random sampling within each
domain. Weights can be optimal while the sampled docs still come from a
biased window. These methods fix *how much* of each config, not *which
part*. Fix coverage first (§5), then consider ODM-style per-leg
reweighting — cheap for us since the eval harness already produces the
signal.

## 4. Replay for anti-forgetting — fractions and buffer mechanics

| Source | Setting | Replay fraction / finding |
|---|---|---|
| Scialom et al., arXiv:2205.12393 (EMNLP22) | T0 continual fine-tune | **1%** buffer ≈ full retention; 0.25% already stabilizes |
| Ibrahim et al., arXiv:2403.08763 | CPT, 100B+ tokens | **~5%** replay + LR re-warm/re-decay ≈ full retrain (incl. EN→DE) |
| Gupta et al., arXiv:2308.04014 | CPT, Pythia 410M | LR handling is the co-variable — at limited compute *no re-warm* was best |
| Abbes et al., arXiv:2508.01908 | CPT 100B tok/language, Llama | small replay rates are a better compute spend than model size; **~25% still efficient, 50% not**; +MER |
| arXiv:2510.11842 | task CPT (bAbI) | **5–10%** replay sufficient for general-knowledge retention; more buys nothing |
| Rolnick et al., arXiv:1811.11682 | continual RL | **reservoir-sampled** buffer = uniform over ALL past experience; mixing on+off-policy beats either alone |
| Wang et al., arXiv:2302.00487 (TPAMI) | CL survey | rehearsal/replay = standard method family |
| Data Mixing Laws, arXiv:2403.16952 | CPT | fitted law predicts the critical mix ratio avoiding forgetting |

**Rules of thumb**: replay fraction ~**1–25%**; ~5% is the canonical
starting point at CPT scale; the buffer should be a *uniform sample of all
prior legs* (reservoir), not last-leg-only; rotate entries so per-doc
cumulative repeats stay ≤~4 (§2 bounds). Implementation: a `"replay"`
pseudo-config at 0.05–0.10 in `RESUME_MIX` fed from a small pickle of
past-leg docs — strictly stronger than the current `cc_all_dressed 0.10
"anti-forgetting guard"` slot, which replays a *domain* with *fresh* docs
rather than replaying *experience*.

## 5. Sampling mechanics under streaming-only

Three ways to draw q docs from a sequential stream beyond "next q":

### 5.1 Strided (systematic) sampling — take every k-th doc over window [SEEN, SEEN+k·q)

- Classic result (Cochran, *Sampling Techniques*, 1977; review
  doi:10.1080/15598608.2017.1353456): when population **order correlates
  with the variable of interest**, systematic sampling beats simple random
  sampling via *implicit stratification* — the sample is spread evenly
  across the ordering. Failure case: periodicity aligned with the stride.
  Caveat: a single systematic sample can't give an unbiased variance
  estimate — matters for eval slices, not for training data.
- For us: stride k≈8 over the-stack-v2 covers ~8 topic/language partitions
  of position instead of 1–2 contiguous; fully deterministic; dedup
  bookkeeping stays trivial (`SEEN += k·q`); costs k× stream depth per leg.

### 5.2 Reservoir sampling — uniform over a window

- Vitter 1985 (doi:10.1145/3147.3165): one pass, O(q) memory, uniform
  without replacement over all positions scanned. Same coverage cost as
  strided (scan k·q, keep q) but random rather than evenly spread — for an
  *ordered/stratified* stream the survey literature favors the strided
  variant. Reservoir's real use here is the **replay buffer** (§4): maintain
  R past docs uniformly over everything ever trained (exactly Rolnick et
  al.'s ER mechanics, arXiv:1811.11682).

### 5.3 Shard-level random selection — the strong option

- dolma3.5_pool configs are directories of `.jsonl.zst` shards (§6).
  `huggingface_hub.list_repo_files("allenai/dolma3.5_pool")` → sample k
  random shard paths per config per leg → `load_dataset(..., data_files=...)`
  or per-file streaming. Near-uniform coverage across the config's topical
  space at **zero extra stream cost** — random access at shard granularity,
  streaming within shards.
- Dedup: track consumed **shard paths** in SEEN instead of doc counts —
  exact and simpler than position bookkeeping.
- Fallback if per-file loading is finicky: `ds.shuffle(seed=leg,
  buffer_size≈4096)` before skip/take (§1.1) — HF permutes shard order, so a
  q-doc take samples a random shard subset anyway; SEEN then counts
  "consumed shuffled-stream positions" and dedup becomes probabilistic —
  acceptable per §2 repetition bounds.

### 5.4 Doc-level vs token-level interleaving

All mixture machinery operates at **document level**: DoReMi/ODM/RegMix
sample domains per doc; MosaicML `proportion` mixes samples; HF
`interleave_datasets(probabilities=...)` interleaves rows. Token-level
mixing is a packing-side concern, not a weighting mechanism — no primary
source uses it for domain weighting. Our 4096-byte doc granularity is
consistent with the standard.

## 6. "Prefix bias": are dolma configs internally ordered? — YES, documented

- dolma3.5_pool card: "This is the Dolma 3.5 *pool*. It contains **no
  quality upsampling or mixing**" — ordering is whatever the pipeline
  emitted. (https://huggingface.co/datasets/allenai/dolma3.5_pool)
- **Verified repo layout — configs are topic-partitioned directories**:
  `cc_all_dressed/history_and_geography/vigintile_0000..0019/shard_*.jsonl.zst`
  (tree listing) and `hplt/history_and_geography/vigintile_0011/shard_00003028.jsonl.zst`
  (blob f2ffec1). Layout convention: `<config>/<topic>/vigintile_XXXX/shard_NNNNNNNN`,
  *topic first*. HF resolves `data_files` in path order, so sequential
  streaming walks topic directories in lexicographic order — **any
  contiguous window of a few thousand docs is one or two topic partitions**.
  This alone explains the stack-v2 0.777→2.276 eval jump.
- the-stack-v2 config: card says the code pool was re-processed for **50
  programming languages** (same partition convention expected); upstream
  `bigcode/the-stack-v2-train-*` cards state "the data is **grouped into
  repositories**" — consecutive docs share repo/language either way. (Exact
  sub-dir names under `the-stack-v2/` not independently verified — check
  `tree/main/the-stack-v2` before relying on topic-per-dir there.)
- Upstream ordering: Common Crawl's CDXJ index is **sorted by SURT urlkey**
  — domain-clustered by construction; anything built in index order
  inherits host-level clustering.
  (https://www.commoncrawl.org/cdxj-index, webrecorder CDXJ spec 0.1.0)
- The Dolma pipeline preserves per-source directory partition
  (allenai/dolma `docs/data-format.md`) and its tokenizer does only "light"
  file+chunk shuffling (`docs/tokenize.html`) — the released pool retains
  source-internal order; mixing/shuffling is explicitly the consumer's job.
  Dolma paper: arXiv:2402.00159. Precedent for domain-blocked corpora: The
  Pile = 22 weighted sub-corpora (arXiv:2101.00027).

**Corollary for evals**: `dolma_eval_docs` always evals the slice just past
SEEN, so each leg's domain bpb measures a *different* subpopulation —
cross-leg deltas conflate forgetting with slice drift (already suspected in
RUN_STATE.md leg-4 note). Fix evals before trusting them: pin a fixed eval
slice per config (e.g., a reserved `vigintile_0019` or held-out shard set)
OUTSIDE the consumed prefix.

## 7. Recommendations

| Problem | Technique | Evidence | Fit for our constraints |
|---|---|---|---|
| Contiguous-prefix sampling covers only 1–2 topic partitions per config per leg | Sample **shard paths** per leg (`list_repo_files` → random subset → `data_files`) | dolma3.5 topic-partitioned layout (§6); two-level shuffle precedent: HF/MosaicML/OpenLM/dolma-tokens (§1) | Streaming-compatible; exact dedup at shard granularity; ~free. **Primary fix.** |
| Same, one-line fallback | `.shuffle(seed=leg, buffer_size≈4096)` before skip/take | HF docs: buffer + shard permutation (§1.1) | Trivial; dedup probabilistic — fine: ≤4 repeats/doc is free (arXiv:2305.16264) |
| Same, deterministic variant | Strided take over widened window (stride k) | systematic sampling on ordered populations (Cochran 1977; doi:10.1080/15598608.2017.1353456) | Exact dedup, trivial code; costs k× stream depth |
| Forgetting across legs | `"replay"` pseudo-config 5–10% fed by rotating reservoir of past-leg docs | Scialom 1% (2205.12393); Ibrahim 5% (2403.08763); Abbes ≤25% (2508.01908); Rolnick reservoir (1811.11682) | Add to RESUME_MIX + small doc pickle; rotate so per-doc repeats ≤4 (2305.16264, 2205.10487) |
| Mixture weights hand-set per leg | ODM-style online reweighting from per-domain eval deltas | ODM Exp3 (2312.02406); DoReMi excess-loss (2305.10429); Data Mixing Laws CPT case (2403.16952) | Per-domain bpb evals already exist = the reward signal; post-leg weight update |
| Cross-leg eval drift | Pin fixed eval slices per config outside the consumed prefix | §6 layout + systematic-sampling variance caveat | Small one-time data cost; makes leg-to-leg deltas meaningful |
| Within-leg order | Keep global shuffle of collected docs (status quo) | order effects weak for decoder LMs (doi:10.1016/j.knosys.2026.115850; Fabre PMLR v306) | Already done; no change needed |
| 16-doc held-out drives early stop | Stratify held-out across configs (e.g., 32 docs, 4/cfg) | sampling lit; current held set is single-domain | One-time cost; stabilizes PATIENCE signal |

**Bottom line**: the highest-value fix is not a shuffle algorithm — it is to
stop sampling contiguous prefixes of *topic-partitioned* streams.
Shard-level random selection (or `.shuffle`-before-take as the cheap
approximation) converts each leg's draw into something close to uniform over
each config's topic space, keeps dedup exact at shard granularity, and —
once eval offsets are pinned — makes the per-domain eval slices comparable
across legs.

## 8. Honesty / evidence gaps

- No primary source studies "prefix consumption of HF streams" directly;
  §6's ordering evidence is *structural* (repo layout + upstream SURT sort
  order), not a published measurement. The fix is low-risk regardless.
- Ordering-effects papers are at from-scratch pretraining scale (≥410M
  params, ≥B tokens), not adapter-CPT at 4KB-doc / ~16MB-per-leg scale;
  treat magnitude claims as directional.
- Replay fractions were measured at large corpus scale; at 4000-doc legs a
  5–10% pseudo-config is 200–400 docs — expect the benefit mainly as eval
  stabilization, and watch per-doc repeat counts.
- The `the-stack-v2` sub-directory names (language partition assumed from
  the card's "50 languages" claim + the cc/hplt layout convention) were not
  verified — run `list_repo_files` on that config before relying on
  topic-per-dir there.
