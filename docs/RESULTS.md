# Results — a tokenizer-free adapter over frozen Qwen3.5

**Question**: can a frozen token-level LLM be driven at byte/chunk level through constructed or
cheaply-learned glue — no trunk training?

**Answer (final, 2026-11-05)**: yes, at a price — and the price is not where the research gate
predicted.  A frozen Qwen3.5 trunk + a closed-form patch map + a small trained byte decoder is a
working tokenizer-free LM that beats the trunk's own byte-level floor on held-out data.  The
measured bottleneck ranking: **byte decoder ≫ trunk choice > patch map** — the decoder is the only
component that still returns bits for effort.

Setup: the Qwen3.5 family (0.8B / 0.8B-Base / 2B(-Base) / 4B(-Base); frozen, bf16, Colab A100).
Head: the LM head marginalized to bytes (P(b) = mass of all tokens whose first byte is b; 221/256
bytes have a single-token embedding row).  Metric: **effective bpb** — total bits / total bytes,
chunk boundaries scored by the chunk model, all other positions by the byte-level fallback,
position 0 at the uniform 8 bits.  Corpora: markdown eval docs (3 held-out, 24,576 B; early sweeps
used the 8 shared docs, 65,536 B), the H-Net val split (85,796 B), and a post-cutoff
decontaminated set (`data/decontam/`, 48 arxiv-abstract + GitHub-README docs, 393,216 B,
`kernel/decontam_eval.py`).  Every number below cites its log; adapter-side numbers were always
correct — only the stock-baseline implementation was buggy (finding 1).

## Headline — the corrected picture

All stock rows use the corrected `qwen_byte_bpb`.  md = markdown eval docs, val = H-Net val
split, decontam = post-cutoff set.

| system | md | val | decontam | source |
|---|---|---|---|---|
| stock Qwen3.5-0.8B (native BPE) | 1.57 | 0.82 | 1.09 | `a100_family_sweep.log`, `a100_decontam.log` |
| stock Qwen3.5-4B-Base (native BPE) | 1.30 | 0.65 | 0.85 | `a100_family_sweep.log`, `a100_decontam.log` |
| byte-level 0.8B (period-1 floor) | 5.61 | 4.33 | 5.57 | `a100_family_sweep.log`, `a100_decontam.log` |
| byte-level 0.8B-Base | 4.92 | 3.80 | 5.03 | `a100_family_sweep.log`, `a100_decontam.log` |
| byte-level 2B-Base | 4.54 | 3.44 | — | `a100_family_sweep.log` |
| **byte-level 4B-Base — best trunk** | **4.40** | **3.35** | **4.53** | `a100_family_sweep.log`, `a100_decontam.log` |
| ridge→BPE constructed map (0.8B, zero training) | 5.45 | 4.32 | 5.47 | `a100_construct_ledger.log`, `a100_decontam.log` |
| v1 learned projection (0.8B, ~160k params) | 5.2728 ± 0.0091 | — | — | `a100_v1_seeds.log` |
| **v2 composed (0.8B, d384×6 decoder, 13.2M glue)** | **4.99** | **4.13** | — | `a100_v3_scaled_gen.log` |
| v2 composed on 4B-Base (same recipe) | 5.11–5.15 | 4.21–4.30 | — | `a100_v2_4bbase.log` |
| trained 18.6M H-Net (reference) | — | 1.4177 (published) | — | — |

## Findings

1. **Metric correction (2026-11-05) — the early stock-bpb numbers were a bug, not a
   measurement.**  The `qwen_byte_bpb` stock-baseline implementation reported **mean per-token
   surprisal**, not bits-per-byte (per-token CE × token byte-length / token count — inflated ~3×,
   the mean token length).  Every "stock BPE" figure in the early logs (4.26, 4.92, 3.76) is
   therefore wrong on its face.  Corrected stock values from `results/a100_family_sweep.log`:
   **0.8B 1.57 md / 0.82 val; 4B-Base 1.30 md / 0.65 val.**  Fixed in `src/adapter.py`
   (`qwen_byte_bpb`: summed token NLLs / bytes covered).  Two consequences: (a) the "byte-level
   nearly matches stock" claim is dead — the true token-interface gap is ~3.5 bits on val, not
   0.58; (b) corrected stock is implausibly good (< 1 bit/byte) — strong evidence the eval data
   sits inside Qwen3.5's pretraining corpus, so stock-vs-adapter comparisons are
   **contamination-confounded**; finding 4 bounds this on post-cutoff data.  All
   byte-level/adapter numbers were always correct and are unchanged.

2. **The byte-level floor is the honest baseline — and trunk choice is the cheapest win.**
   Byte-token rows in, byte-marginalized head out, trunk frozen (`a100_family_sweep.log`,
   `a100_decontam.log`): the floor drops monotonically 0.8B instruct → Base → larger Base, and
   **4B-Base is the best trunk measured** (4.40 md / 3.35 val / 4.53 decontam).  9B-Base OOMs on
   A100-40GB in this harness.  The **instruct tax is ~0.5 bits at 0.8B on clean data** (Base 5.03
   vs instruct 5.57 decontam): instruction tuning damages the byte pathway, and it is the weights,
   not the interface, that carry the damage.

3. **The instruct↔Base merge recovers byte-competence linearly — no barrier.**
   W(α) = (1−α)·instruct + α·Base at 0.8B (`a100_merge_sweep.log`, `kernel/merge_sweep.py`):

   | α | md stock | md byte | decontam stock | decontam byte | val stock | val byte |
   |---|---|---|---|---|---|---|
   | 0.00 (instruct) | 1.1405 | 4.6115 | 1.0042 | 4.5991 | 0.8158 | 4.3318 |
   | 0.25 | 1.1069 | 4.4286 | 0.9763 | 4.4103 | 0.7923 | 4.1454 |
   | 0.50 | 1.0913 | 4.2693 | 0.9609 | 4.2413 | 0.7795 | 3.9972 |
   | 0.75 | 1.0855 | 4.1376 | 0.9527 | 4.0948 | 0.7725 | 3.8765 |
   | 1.00 (Base) | 1.0859 | 4.0513 | 0.9491 | 4.0031 | 0.7692 | 3.7989 |

   Byte competence recovers ~linearly in α on every corpus; stock stays ~flat throughout.  This is
   the legal linear-mode-connectivity regime (same shape, same lineage) — the byte-capable
   directions are still inside the instruct weights, merely diluted.  **The trunk problem is
   solved two ways: pick a Base checkpoint, or interpolate toward it.**

4. **Decontam: stock stays ~1.0 on post-cutoff data — mostly real modeling, not memorization.**
   On 48 post-cutoff docs (`a100_decontam.log`): stock 0.8B 1.0890, 0.8B-Base 1.0227, 4B-Base
   0.8517 vs val 0.82 / 0.77 / 0.65 — a val-vs-decontam delta of only ~0.2 bits.  Qwen3.5's BPE
   pathway is genuinely ~1.0 bpb on technical text, and **the byte interface costs ~4 bits
   through Qwen** on both contaminated and clean data.  ridge-constructed ≈ byte-level on clean
   data — chunking is neutral, the interface is the cost.

5. **Boundary placement carries the learnable signal — and training-free rules capture most of
   it.**  At matched ~6× compression, content-aware boundaries beat periodic by 1.9–2.8
   boundary-bpb on harder targets (sim 6.39, entropy 7.32, periodic 9.19 —
   `a100_construction_sweep.log`, `a100_compression_curve.log`).  Consistent with the
   frozen-router result on the trained H-Net (+0.001 bpb for frozen boundaries).  The compression
   optimum is ~6 on markdown and drifts to ~8–12 on natural English (`a100_val_eval.log`).

6. **The patch map is closed-form solvable — construction ≈ training on the effective axis.**
   The staircase (`a100_construct_sweep.log`, `a100_construct_ledger.log`), boundary bpb:

   | step | map | periodic-6 | sim~6 |
   |---|---|---|---|
   | hand-crafted | identity (sum) | 6.5483 | 4.1574 |
   | | norm_match (median token-norm rescale) | 5.9277 | 4.8373 |
   | | cov_recolor (whiten + recolor to token Σ) | 6.9935 | 3.5747 |
   | closed-form | **ridge→BPE** (paired Procrustes) | **4.5760** | **3.7443** |
   | SGD | rank-32 residual | 3.6181 | 2.6101 |

   ridge→BPE recovers **~69%** of the identity→SGD gap under periodic boundaries and **~27%**
   under sim~6 — construction helps most where the base construction is weakest (good boundaries
   already do part of the manifold-matching work).  On the H-Net val split, periodic-6 ridge→BPE
   scores **4.32 effective — closed-form, zero training** (`a100_construct_ledger.log`), already
   at the 0.8B byte-level floor.  `cos(ΔW, construction−I) ≈ 0 for every closed-form map, under
   both chunkers`: the trained correction is a different direction in weight space than any
   statistical reshaping — but the closed-form solve lands at the same place on the effective
   axis.  `norm_match` flips sign across chunkers (patch norm carries length signal under
   variable-length chunking); `cov_recolor` likewise flips — whitening assumptions are
   chunker-dependent.

7. **Training only the glue works (the ZHA recipe) — and it lands where ridge lands.**
   A rank-32 residual on the patch sum (~160k params, 156 steps, ~2 min A100) cut boundary bpb
   4.16 → 2.56 on held-out docs (3 seeds, ±0.05) for effective **5.2728 ± 0.0091**
   (`a100_v1_seeds.log`, `a100_v1_projection.log`) — vs ridge→BPE's 5.45 on the same axis.  The
   autopsy shows a structured ~0.25‖I‖ correction (top σ ≈ 5–8 decaying), not a perturbation.
   Verdict for the patch map: **the ~160k-param problem is solved — use the closed-form ridge
   (zero training) or the rank-32 SGD, they are within ~0.2 bits of each other.**

8. **No constructed chunker beats plain byte-level on the full-byte axis** (`a100_effective.log`):
   every constructed chunker lands ~5.56–5.95 effective on markdown vs byte-level alone 5.29 —
   without a trained decoder, chunking is a compute play, not a prediction play.

9. **v2: the composed system is a real tokenizer-free LM — and the decoder is the bottleneck.**
   Frozen trunk + ridge-initialized patch map (~160k, 6 SGD epochs) + transformer byte decoder
   (pair-window cross-patch context) trained on 13 markdown docs (`a100_v2_decoder.log`,
   `a100_v2_ridge_val.log`, `a100_v3_scaled_gen.log`):

   | trunk / decoder | params | md eval | val (cross-domain) |
   |---|---|---|---|
   | 0.8B d256×4 | 4.87M | 5.0156 | 4.1651 |
   | **0.8B d384×6** | **13.2M** | **4.9881** | **4.1286** |
   | 4B-Base d256×4 | 7.23M | 5.1125 | 4.2061 |
   | 4B-Base d384×6 | 16.75M | 5.1465 | 4.3047 |

   On 0.8B, v2 beats the byte-level floor (5.61/4.33) and every constructed chunker on the
   published val split — a working tokenizer-free LM at ~13.4M glue parameters over a frozen
   trunk, ~2–3 min of A100 training.  **But the same recipe on the strictly better trunk is
   strictly worse** (`a100_v2_4bbase.log`): 5.11–5.15 / 4.21–4.30 on 4B-Base, far above that
   trunk's own byte-level floor (4.40/3.35).  A stronger trunk cannot rescue a decoder trained on
   13 documents — **the decoder is the data-bound bottleneck, not the trunk and not the patch
   map.**  Generation exists (`src/generate.py`; trunk proposes boundary bytes, decoder fills
   interiors) but output is fragmentary — the decoder's 12-byte context window can't sustain
   coherence; it is a scorer-grade artifact.  Weights: `results/byte_decoder.pt`.

10. **Learned glue is size-specific — conjugated transfer fails.**  Transporting the 0.8B-trained
    ΔW to 4B-Base through a ridge bridge R fit on 248,320 shared-vocab embedding pairs
    (R⁺·ΔWᵀ·R, `a100_conjugate.log`, `kernel/conjugate_transfer.py`): boundary bpb **5.95 vs
    identity 5.66** — the transplanted correction *hurts*.  The same trunk's own closed-form
    refit (local ridge) hits **4.37**.  Embedding-space activations bridge across sizes (the R
    fit is well-posed and massive), but the learned task-function does not ride along — the
    learned direction is a function of the trunk it was learned against.  **Per-size refit
    required** — fortunately it costs minutes (`docs/RESEARCH_WEIGHT_TRANSFER.md` §5 predicted
    exactly this).

11. **Bottleneck verdict.**  Trunk: **solved** — pick 4B-Base, or merge instruct→Base (linear
    α-dial, no barrier).  Patch map: **solved** — closed-form ridge→BPE ≈ trained rank-32 on the
    effective axis; do whichever is cheaper.  Byte decoder: **the bottleneck** — the only
    component whose scaling (4.87M→13.2M) still buys bits and whose failure mode (4B-Base
    regression) cannot be fixed by a better trunk.  The next lever is decoder data/capacity
    scaling, not interface cleverness.

12. **Open doors.**  (a) Decoder data scaling — 13 markdown docs is nothing; both the d256×4→
    d384×6 gain and the 4B-Base regression say the decoder is data-bound.  (b) dW lens-autopsy —
    read the learned map's singular directions through the workspace lenses
    (`camilablank/workspace-lenses` ships J+R lens pairs for Qwen3.5-4B/9B/27B): is the learned
    operator a readable byte→subword affinity structure?  That answers the parked WSP question.
    (c) Per-size refit table ({constructed, ridge, rank-32} × each trunk) as the transfer-control
    arm — `docs/RESEARCH_WEIGHT_TRANSFER.md` ranks the options.

## Earlier detail sections (pre-correction stock numbers inside — see finding 1)

### The composed system on the original axes

Frozen Qwen3.5-0.8B trunk + ridge-init patch map + byte decoder:

| corpus | stock BPE (buggy log value) | byte-level | best constructed | **v2 composed** |
|---|---|---|---|---|
| markdown eval docs | 4.9226 → corrected **1.57** | 5.5730 | 5.5558 | **4.99** (d384×6) |
| hnet val split (cross-domain) | 3.7560 → corrected **0.82** | 4.3310 | ~4.59 | **4.13** (d384×6) |

### Decontaminated eval detail (`a100_decontam.log`, post-cutoff arxiv+GitHub)

| model | stock | byte-level | ridge-eff | val stock | val byte | val ridge |
|---|---|---|---|---|---|---|
| 0.8B | 1.0890 | 5.5675 | 5.4656 | 0.8158 | 4.3318 | 4.3183 |
| 0.8B-Base | 1.0227 | 5.0275 | 4.9674 | 0.7692 | 3.7989 | 3.8211 |
| 4B-Base | 0.8517 | 4.5341 | 4.5775 | 0.6505 | 3.3486 | 3.4813 |

- Stock stays ~1.0 on post-cutoff data → most of the stock advantage is real modeling, not
  memorization.  The byte interface costs ~4 bits through Qwen regardless.
- The instruct tax replicates on clean data (~0.5 bits: Base 5.03 vs instruct 5.57 byte-level).
- ridge-constructed ≈ byte-level on clean data — chunking is neutral, the interface is the cost.

## Caveats

- The v1/v2 glue was trained on 13 same-domain markdown docs; its val-split transfer is measured
  (4.13) but the training corpus is tiny — the decoder bottleneck finding cuts both ways.
- All runs: single A100-40GB, bf16, unauthenticated HF downloads; 9B+ trunks OOM in this harness.
- Stock-relative claims remain contamination-confounded on the markdown/val corpora; the decontam
  set (post-cutoff) is the clean axis and it changes none of the adapter-side conclusions.
- Generation is a scorer-grade artifact (12-byte decoder context); sampled output is fragmentary.

## Logs

`results/a100_v0_period6.log` · `a100_sweep.log` · `a100_construction_sweep.log` ·
`a100_compression_curve.log` · `a100_effective.log` · `a100_v1_projection.log` ·
`a100_v1_seeds.log` · `a100_val_eval.log` · `a100_construct_sweep.log` ·
`a100_construct_ledger.log` · `a100_v2_decoder.log` · `a100_v2_ridge_val.log` ·
`a100_v3_scaled_gen.log` · `a100_v2_4bbase.log` · `a100_family_sweep.log` ·
`a100_merge_sweep.log` · `a100_decontam.log` · `a100_conjugate.log`
