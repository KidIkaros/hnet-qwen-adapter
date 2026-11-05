# Research — cross-size transfer, weight reads, and the WSP landscape for the learned glue

Date: 2026-11-05. Question on the table: the v1/v2 glue (rank-32 residual patch map
`patch = base·(I + A·Bᵀ)`, ~160k params; 13M-param interior-byte decoder) is trained on
Qwen3.5-0.8B. 4B-Base is the best byte trunk (`results/a100_family_sweep.log`). Can the glue
move across sizes — and can the learned `dW` be *read*?

Scope note: "transfer" here means small→large (0.8B glue onto 4B/9B/27B/35B-A3B trunks).
Almost all published cross-size work goes the other way (large→small downscaling), so each
section ends with an explicit applicability verdict for our direction.

## 1. Cross-size transfer

### 1.1 Wiring Beats Blending (arXiv:2608.02829, v5) — the load-bearing result

Full title: *Wiring Beats Blending: Structure-Aware Compensation for Transformer Downscaling*
(Yenugula, independent researcher; code at github.com/rsdpyenugula/ScaleOp, checkpoints at
huggingface.co/YenugulaAIML/scaleop-pythia-conversions). Pythia 1.4B→410M end-to-end, second
family Qwen3-1.7B-Base→0.6B-Base.

**The ridge numbers** (§4, App. A.2): per-layer pooled activations map large→small under a
full linear ridge (2048→1024) at **held-out R² = 0.844**; Procrustes (rotation + one global
scale) already reaches **0.717** — most of the cross-size relation is a change of basis, the
remaining ~0.13 is non-orthogonal reshaping. Raw *parameters* align weakly even in the
cleanest case (embeddings, where the two sizes literally share the vocabulary axis): ridge
R² = 0.248 input / 0.386 output. A shared operator (A,B) fitted *with* target access — the
diagnostic upper bound — leaves relative Frobenius error 0.66–0.80, i.e. at best ~30% of
weight variance is linearly explainable across sizes.
(https://arxiv.org/abs/2608.02829, https://arxiv.org/html/2608.02829v5 §4)

**Mechanism of failure** (§5): dense weight projection is destructive through *basis mixing* —
it breaks rotary (cos,sin) pairs, per-head attention boundaries, elementwise GELU
(`GELU(Bx) ≠ B·GELU(x)`), and per-feature LayerNorm gains. LayerNorm mismatch is the dominant
zero-shot collapse factor: neutral norms cut projected-model perplexity ~15×
(181,242 → 12,323). A bit-exact rebuild control (max |logit diff| = 0) isolates the loss to
the projection itself.

**The null that matters for us** (§6): residuals after the best-fit linear operator carry
**no learnable or transferable signal** — four tests against shuffle controls, p ≥ 0.99. "No
learned zero-shot correction exists at this granularity"; conversion value lives in
initialization, then continued training does the rest.

**The method** (§7, Alg. 1, "Compensated Selection"): structure-respecting selection (keep
whole rotary pairs at stride 2, whole heads, whole MLP units, strided blocks) + two
independent levers partitioned by *where normalization sits*:
- *Compensation* (function lever): closed-form ridge re-fit of norm-free read-out paths,
  `M′ = M·Σ_{:,K}·(Σ_{K,K} + λI)⁻¹` — surviving units absorb dropped units' correlated
  contribution. In Pythia exactly two paths qualify: O and MLP_down. Best zero-shot.
- *Variance-preserving rescale* `√(d_in/d_in′)` (dynamics lever): applied to norm-fronted
  read-in paths (Q, K, V, MLP_up, output embedding). Worse zero-shot, best endpoint.
- Placement rule: "compensation is well-posed exactly where no normalization sits between
  cut and read; norm-fronted paths take rescale."

Results: informed selection + compensation + rescale beats the best subcloning variant
84.6±2.1 vs 90.7±3.8 (6/6 seeds, width pair; held-out depth pair 109.3 vs 117.9, 3/3); ends
19×/23× below from-scratch at 30M tokens on Pythia/Qwen3. Limits: levers parity with plain
selection at 33× the budget; they *anti-synergize* at ~5× the donor scale. Qwen3-specific
twist: the 2×2 crossing the re-fit with the activation statistic **reverses** — on Qwen3 the
statistic carries 66 perplexity points and the ridge re-fit is a null on top (heterogeneous
MLP unit energies). The re-fit is architecture-family-dependent; the placement rule is not.

**Training needed**: construction is gradient-free; value is only realized after matched-budget
continued pretraining (30M tokens). Not a zero-shot transfer recipe — an init recipe.

**Applicability to our glue**: indirect but important. (a) The R²=0.84/0.717-vs-0.25 split
says: *transfer in activation space, never in weight space*. Our conjugation plan (§5) follows
this exactly. (b) Their post-operator residual null is a caution: a linearly-mapped dW may
capture the aligned subspace and drop precisely the task-conditional ~31% we care about —
measure, don't assume. (c) The placement rule is directly checkable in our system: our patch
map feeds inputs_embeds *upstream of the first RMSNorm*, i.e. it sits on a norm-fronted
read-in path — the Wiring rule predicts a rescale lever, not a re-fit lever, is what tunes it.
(d) Single-author preprint; numbers self-reported — treat constants as indicative, the
qualitative split (activations >> parameters) as solid.

### 1.2 APM — activation-guided cross-scale fusion (arXiv:2608.13596, v1)

*Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning*
(Fan, Chen, Hou, Zhang, Xie; USTC). Donor Qwen2.5-32B (or 14B) → recipient Qwen2.5-3B.

Method: (1) run the donor on task data, accumulate activation magnitudes for layers, hidden
dims, attention heads, MLP neurons; (2) prune donor to the *exact recipient shape* using those
saliency maps (GQA-aware head-group selection); (3) inject the donor slice into the unmodified
recipient with a micro interpolation coefficient μ ≈ 0.05. Builds on Intersection-Merge
(front-aligned truncation, Fan et al. 2026). Result: 3B average over 16 benchmarks
55.5% → 60.6% (RTE 64.3→82.3 at μ=0.05 — above the 32B donor's own 80.9). Take-away framed as
"concentration transfer": small-ratio fusion needs a concentrated task-relevant donor slice,
not neuron-wise semantic alignment. (https://arxiv.org/abs/2608.13596)

**Training needed**: none (profiling forward passes only).

**Applicability**: low for the stated goal — APM grafts *trunk* weights into a recipient of
the same shape after slicing; our trunks are frozen and our glue is the transferable
artifact. Two borrowable ideas: the μ≈0.05 micro-injection regime (a transferred adapter can
be dialed in at small strength first — our existing α-dial finding, cf. merge sweep) and the
"concentrated donor slice" principle (transfer only the top singular directions of dW rather
than all rank-32).

### 1.3 Cross-model KV cache transfer (arXiv:2608.03893, v1)

*A Closed-Form Linear Mapping for Prefill Reuse* (Heo et al., NVIDIA). Within-family
small↔large KV reuse: Qwen3, Llama 3.1, Ministral 3; matched-KV pairs (shared KV-head count
and head dim).

Method: per-head ridge regression, three ingredients — (a) each target layer fit from its
top-k most predictive source layers (concatenated), (b) keys mapped RoPE-stripped so the fit
is position-free, (c) calibration set of only 500 FineWeb-Edu sequences × 1,024 tokens.
Linear structure measured: on Qwen3 14B→32B one source layer explains 56%/32% of
key/value variance, 79%/65% with top-k (head-averaged R² peaks 0.81). Ridge λ=0.01.
Results: retains 73–98% of standalone-prefill accuracy on 4/6 pairs (Ministral pairs fail);
a nonlinear MLP adds +37 pp HellaSwag on the failures by moving error out of
attention-sensitive directions; attention-output cosine predicts retention (r = +0.57)
better than R² (−0.20). Mapper itself is 1.01–3.36B params per pair, ~47–87 min fit on an
8×H100 node. (https://arxiv.org/abs/2608.03893, §4)

**Training needed**: none — closed-form ridge on a small calibration corpus.

**Applicability**: this is the strongest *positive* existence proof for our plan: a
closed-form ridge on activations transfers function across sizes within a family, on Qwen
specifically. Their matched-KV constraint doesn't bind us — we map the *embedding* space,
which is always defined (and, better, vocabulary-aligned across the family). The R²=0.81 peak
for mid-layer maps and the "error placement > error magnitude" finding both generalize:
evaluate the transferred map by *where* its residual lands (token vs non-token directions),
not just its norm. Note their maps were fit on 47–87 min of 8×H100; ours needs ~150k paired
embedding rows — minutes on one A100.

### 1.4 Weight subcloning (arXiv:2312.09299, v1)

*Direct initialization of transformers using larger pretrained ones* (Samragh et al., Apple).
Neuron importance = mean |activation| over a small data subset (0.5% ImageNet, 0.003% Pile);
residual-connected neurons share one consistent ordering (scores summed across connected
layers); permute by importance, take the top-d′ prefix; drop blocks to match depth; plus the
variance-preserving weight rescale. Parent→smaller init only — destination then trains
normally (~4× faster to target accuracy on ViT/GPT-2).
(https://arxiv.org/abs/2312.09299)

**Training needed**: yes — it is an initialization, not a transfer of function.

**Applicability**: superseded for our purposes — Wiring paper §7 beats its best variant at
matched budget, and our trunk never changes shape anyway. The one durable idea (importance =
mean |activation|, consistent ordering along residual paths) is already the weaker version of
APM's task-conditioned maps.

### 1.5 Neural Nuggets — parametric transfer via sensitivity + LoRA (arXiv:2310.11451, ICLR 2024)

*Seeking Neural Nuggets: Knowledge Transfer in LLMs from a Parametric Perspective*
(Zhong, An, Chen, Han, He). Teacher→student parametric transfer on LLaMA/LLaMA-2.

Method: (1) compute parameter sensitivity on ~32 seed samples — first-order Taylor score
S_i ≈ |θ_i · ∂L/∂θ_i| of the loss change if the parameter were zeroed; (2) select
highest-sensitivity layers, then for dimension mismatch select the submatrix with maximal
cumulative sensitivity; (3) SVD-factorize extracted blocks and use them to *initialize the
student's LoRA* (rank 16); (4) LoRA fine-tune the student. Results: injecting teacher-derived
init lifts the LoRA-trained student on every benchmark family — LLaMA-1 13B→7B avg 27.00 →
28.22 (30B donor: 28.63) and 30B→13B 33.89 → 35.54; LLaMA-2 13B→7B 33.23 → 35.57, and 70B→7B
35.92 despite restricting transfer to FFN + embedding (0.36% of trainables) over a GQA/MHA
mismatch — larger donors transfer more. Analysis factors that
matter: teacher scale, SVD-based init >> random, ≥32 seed samples, sensitivity ordering.
(https://arxiv.org/abs/2310.11451, Tables 1 and 3; project page maszhongming.github.io/ParaKnowTransfer)

**Training needed**: yes, but LoRA-only on the student — compatible with a tiny-SGD budget.

**Applicability**: the closest published precedent to "adapter as transfer vehicle," but the
direction is opposite (large→small, extracting *task knowledge*) and dims are resolved by
crude submatrix selection. Borrowable: SVD-init of a target-side low-rank module from a
transferred matrix — if conjugation (§5) gives a full dW_t, its rank-32 SVD is exactly a
(A_t, B_t) init. Their finding "larger donor helps more" weakly suggests trying the
*reverse* of our plan too: fit the patch map at 4B-Base (the better trunk) and check whether
*it* conjugates down to 0.8B.

### 1.6 Synthesis — what this literature says about our specific transfer

- Direct weight copy is impossible (d_model differs per size) and direct *weight mapping* is
  measured to fail even where it is defined (Wiring: weight R² ≤ 0.39, residuals carry no
  signal). → Do not try to "learn a map of A,B" across sizes.
- Activation-space linear maps across sizes are the verified channel: R² 0.72–0.84 (Wiring,
  Procrustes→ridge), R² 0.79/0.65 multi-layer (KV paper), and the APM result shows even
  *shape-transplanted* donor mass transfers at small mixing coefficients. → Conjugate dW
  through a fitted embedding-space map, or better, re-express its function (§5).
- Every cheap method that works is a one-shot solve on a small calibration corpus (ridge on
  500–8k sequences; sensitivity on 32 samples; mean-|act| on 0.003% of Pile). Our byte-level
  calibration need is smaller still: the shared tokenizer gives ~150k+ *exactly paired*
  embedding rows — a massively overdetermined paired-Procrustes problem with no sampling
  error at all.
- Nothing in this literature transfers a *learned interface module* (our exact situation);
  the closest are the KV mapper (activation function transplant) and Nuggets (LoRA-init
  transplant). Both fit the §5 recipe.

## 2. Weight reading / lenses — R-lens, S-lens, and whether dW is readable

**Repo correction**: `github.com/stdlogits/rs-lenses` is 404 — the lenses live on HuggingFace
as `stdlogits/rs-lenses` (model repo, CC-BY-4.0); the code lives in
`github.com/stdlogits/neuron-lens` (MIT), catalog site neuron-lens.com. Method docs:
github.com/stdlogits/neuron-lens `docs/design.md`, `docs/findings.md`.

**What a lens is** (design.md; HF model card): a corpus-averaged Jacobian of the residual
stream under RelP linearization rules (RMSNorm `rsqrt` held constant — "LN rule"; SiLU
linearized as `sigmoid(g)·dg` — "identity rule"; SwiGLU product split β=0.5 — "half rule";
attention, q/k norms, RoPE keep true derivatives). Two directions:
- **R-lens** (reverse): `J_L = mean_t Σ_{t'≥t} ∂h_final(t')/∂h_L(t)`; read of vector `h` at
  layer L = `softmax(W_U · norm(J_L · h))` — "what this vector makes the model say."
  Descends from Anthropic's J-lens/global-workspace (transformer-circuits.pub/2026/workspace)
  and camilablank's RelP version (lesswrong.com/posts/nv8oedrnLXKRzNEL9, RelP arXiv:2508.21258).
- **S-lens** (forward): `K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)`; token v scores
  `c · K_L (W_E[v] − mean W_E)` — "which input tokens put a vector here." It is Theia Vogel's
  K-lens under the same rules; `K_L = J_Lᵀ` exactly under RelP.

Validation numbers: R-lens read of a neuron's write vs measured steering effect — median
r = 0.46–0.51 at L4–L8, 0.64–0.74 at L12–L16, ≥0.9 from L22 (Qwen3-8B); S-lens names the
driving token in top-10 for 38% of top firings at L0. Fit cost: ~11 s/doc/lens on one H100
for 8B; **0.36 s/doc for Qwen3-0.6B** — fitting a 0.8B lens on our A100 is a minutes-scale
job over ~1,000×128-token docs. Split-half stability ≤6%.

**Artifacts that already exist**: `camilablank/workspace-lenses` ships J+R lens pairs for
**Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, Qwen3.6-35B-A3B** (paper-faithful recipe, target =
penultimate layer) plus stdlogits' Qwen3-8B/Gemma-4 R+S pairs. For our 4B/9B/27B-Base trunks
the R-lens is downloadable today; for 0.8B(-Base) and for the S-lens we'd fit our own —
`neuron_lens` already composes Qwen3.5 dense and MoE (`tests/test_qwen35.py`).

**Could a lens read our dW?** A linear map has no single vector to read — the read has to be
decomposed:
- **SVD route**: dW = UΣVᵀ. Right singular vectors v_i are *input* directions in embedding
  space — read each against W_E rows directly (cosine ranking over vocab) or through the
  S-lens at L0: "which byte/token directions does the map amplify/suppress." Left singular
  vectors u_i are *output* directions added to the patch vector — read `u_i` through the
  R-lens at the embedding/L0 position: "what does the correction push the model toward
  saying." This is exactly the neuron-lens pole recipe (their `w = ±W_down[:,n]` write read
  via R-lens, `c = â±b̂` read via S-lens), transplanted to our A,B factors.
- **Row/col route**: read dW's action on each of the 221 mappable byte rows — per-byte
  correction vectors Δ_b = E[b]·dWᵀ, R-lens each — a byte-level "what does the map think this
  byte is" table.
- Caveats imported from their findings: reads are *directions, not effects* (R-lens is
  scale-free; S-lens needs unit directions); mid-layer reads go dark, but we only need L0/L1
  where reads are strongest; layer-contrastive z-scoring is needed to kill hub tokens; a
  corpus-fitted J_0 at the embedding interface is the right map — using J_L for L>0 would
  double-count the trunk.

**The "secret word" demo — not verified as described.** I found no demo in stdlogits' public
artifacts that reads trained-in content *from weight deltas*. What exists in the literature:
- Cywiński et al., *Towards eliciting latent knowledge…* (arXiv:2505.14352): "Taboo" model
  organisms fine-tuned (QLoRA r8) to know a secret word they never verbalize; the **logit
  lens on activations** at layer ~32 surfaces the secret token — a read of *activations* in a
  model with trained-in hidden content.
- Ameya-bit's z-scored J-lens linkpost: 0.805 leave-one-out secret-word accuracy on
  Gemma-2-9B-it taboo organisms (vs 0.665 for the original protocol); only non-zero readout
  on Qwen3-1.7B; offset correlates with log token frequency ρ≈0.48–0.70.
  (github.com/Ameya-bit/quantify-jlens-offset)
- Anthropic's workspace/J-lens paper: on misaligned model organisms, response-start reads
  surface *secretly, trick, fraud, hidden* vs benign baseline — again activation reads of
  trained-in content (transformer-circuits.pub/2026/workspace §"deception" figures).
- What stdlogits *does* read from weights directly: neuron read/write weight vectors
  (`W_down` columns, `γ⊙W_gate`/`γ⊙W_up` rows) — so a "read columns of a fine-tune ΔW through
  the lens" demo is a natural extension they may have shown (e.g. on X); I could not locate a
  public artifact. **Difference that matters for us**: in every verified secret demo the
  hidden content lives in *activations at inference* and the lens reads a residual vector —
  our dW is a *task-learned operator*, so the honest analogue is the SVD/row reads above:
  "which token directions does the operator amplify" is an operator-interpretability
  question, not a knowledge-elicitation one. Expect it to recover something like a
  byte→token-subword affinity structure (if the map implements part of a learned tokenizer)
  — that *would* answer the parked WSP question "is the learned A·Bᵀ interpretable/
  constructible" (DECISION.md).

## 3. Weight-space learning taxonomy

**Record correction**: `zenodo.org/records/23958070` is *not* ICML workshop slides — it is a
preprint, *"Building Foundation Models via Weight-Space Programming (WSP)"* (Quillan-Ronin
Research Team, published 2026-11-04, CC-BY). This is the WSP reference our CONTEXT.md
vocabulary ("constructed weights — the weight-space-programming post's sense") traces to, so
it is summarized here; credibility flags in §6.

**WSP contents** (the Zenodo .md): three closed-form, zero-gradient operations —
(1) *refusal-subspace abliteration*: contrastive mean difference `v` at a bottleneck layer,
then `W̃ = W − α(vvᵀ)W`, O(d·d_ffn) per layer; (2) *multi-donor grafting*: t-SVD of
truncated/tile-projected donor matrices into ternary {-1,0,+1} low-rank expert adapters,
donor singular vectors blended β=0.15 toward activation-covariance eigenvectors
(`Σ = E[xxᵀ]` eigendecomposition — the same statistic Wiring's compensation reads);
(3) *closed-form MoE folding*: uniform-routed additive low-rank experts fold exactly,
`W_folded = W_down + (γ/K)Σᵢ W₁⁽ⁱ⁾W₀⁽ⁱ⁾` — trivial linearity, but note it *is* our exact
parameterization: `A·Bᵀ` foldable into any same-shape dense map. Claimed results (1.2B model
compiled in <180 s on a GTX 1050, 48 tok/s GGUF) are unverified and the "evaluation" is three
cherry-picked prompts — the *operations* are sound, the benchmarks are not evidence.

**The actual ICML 2026 workshop**: *Workshop on Weight-Space Symmetries: from Foundations to
Practical Applications* — 2026-07-10, Seoul; organizers Ioannou, Knyazev, Lobacheva,
Mohammed, Orvieto, Theus; program on icml.cc/virtual/2026/workshop/54080 and
weightsymmetry.com. Method families on the program, ranked by relevance to interpreting/
transferring a small low-rank residual:

1. **Gauge symmetries of low-rank adapters** — *"The GL(r) Gauge Symmetry of LoRA"* (Karuturi
   et al., OpenReview): GL(r) acts freely on (A,B) leaving ΔW = B·A invariant ⇒ r² flat
   directions; linear interpolation between adapters in *raw (A,B) coordinates* suffers gauge
   ambiguity, not task incompatibility; their fix is a canonical SVD gauge section before
   merging ("gauge-aligned merging"), matching full-rank task arithmetic on LLaMA-3-8B.
   → Directly actionable: before comparing, averaging, or merging our rank-32 maps across
   seeds or sizes, canonicalize the factorization (`A′ = UΣ^{1/2}`, `B′ = VΣ^{1/2}` from
   `dW = UΣVᵀ`). Our cos(dW, construction)≈0 comparisons are on the *product* dW (gauge-
   invariant), so existing findings stand — but any (A,B)-space interpolation of the α-dial
   kind is gauge-contaminated until canonicalized. Related posters: "Low-Rank Networks Recover
   Weight and Functional Symmetry Better"; "Maximal Gauge Symmetry in Transformer
   Architectures" (full gauge group incl. RoPE commutant for LLaMA/Qwen-type models).
2. **Linear mode connectivity / permutation alignment** — Sidak Pal Singh invited talk
   ("Landscaping LMC"), "Debugging ReBasin", "LMC via Neuron Identifiability",
   "Scale-Equivariant Alignment". → The theory behind why weight-space interpolation works at
   all between same-init checkpoints (our instruct↔Base α-dial finding is an empirical
   instance) and why it cannot be assumed across separately-initialized sizes.
3. **Weight-space learning / hypernetworks** — Damian Borth invited talk ("Weight Space
   Learning: Learning Representations from Populations of Neural Network Models"), plus
   Meta-Merging by Checkpoint Nowcasting. → The family that would *predict* an adapter from
   trunk statistics; heavy machinery, but conceptually the learned version of our §5
   closed-form conjugation. Overkill at 160k params.
4. **Merging via representation/output space** — "Model Merging by Output-Space Projection",
   "Model Merging via Averaged Representational Similarity", "LS-Merge". → Same conclusion
   as §1.6 from the merging side: merge in function/representation space.
5. **Weight statistics persisting under post-training** — "Block-Level Weight-Space Structure
   Persists Under Post-Training" (Wang). → Whether byte-level competence directions survive
   instruct tuning — the weight-space twin of our α-dial result.
6. Quantization symmetry (Zechun Liu invited talk) — relevant only for GGUF/export plans.

Bottom line for a 160k-param map: families (1) and (4) give concrete cheap techniques;
family (3) is the research program our closed-form constructions already instantiate by hand;
equivariant-GNN weight encoders exist (the workshop's bread and butter) but nothing at our
size justifies them.

## 4. Local library scan

`/home/ikaaros/Documents/OBSIDIAN/IKAROS/Research Papers/ai-ml` — 40 files, scanned
2026-11-05 for weight|transfer|merge|prun(e|ing)|distill|lens|interpretab*. Two matches
(extracted via pdftotext):

- **Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer**
  (Yang, Hu et al., arXiv:2203.03466). µP: under the maximal-update parametrization the
  optimal HPs (LR, init scale, multipliers) are stable across width — tune on a small model,
  transfer zero-shot (13M→BERT-large beat published BERT-large at 1× its pretrain cost;
  40M→6.7B GPT-3 at 7% of tuning cost). Standard parametrization optima *shift* with width
  (their Fig. 1: LR optimum drifts with d_model). → For the glue: LR/rank hyperparameters
  swept at 0.8B do not transfer automatically to 4B/9B — re-sweep the rank-32 LR per size
  (it is ~2 min anyway). Not a weight-transfer result; an HP-transfer result.
- **When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource
  Languages** (Kumar, Yamaguchi, Aletras; arXiv:2608.27658). The closest published analogue
  to this whole project: hierarchical byte network over a *frozen subword LLM* (Qwen3/Gemma3
  named in text); byte embeddings initialized directly from subword representations of the
  frozen base; a **chunk-alignment loss projecting dynamically grouped byte chunks toward
  precomputed subword targets** — literally our ridge→BPE objective run as a learned loss —
  plus interleaved POS supervision for boundaries; +13.3% POS tagging over six Indic
  languages; code github.com/snjev310/ByteChunk. → Independent validation that (a) frozen
  subword trunks accept byte/chunk interfaces built from their own embeddings, and (b)
  alignment-to-subword-targets is the learnable signal. Their POS supervision is a boundary
  prior we have not tried.

Files deliberately noted as *not* matching but adjacent: H-Net/BLT/MEGABYTE/SpaceByte/
"Efficient Transformers with Dynamic Token Pooling" (architecture references, not weight
transfer); nothing in the library on lenses, merging, or weight-space learning.

## 5. Applying it: transferring 0.8B glue to 4B/9B/27B/35B-A3B

Concrete recipes, cheapest first. All are compatible with frozen trunks + closed-form +
tiny-SGD budget on one A100-40GB.

1. **Conjugated patch map (the function-transplant)** — the literature-blessed channel.
   The shared tokenizer gives ~150k+ *paired* embedding rows `E_s[v] ∈ R^{d_s}`,
   `E_t[v] ∈ R^{d_t}`. Fit ridge `R: R^{d_s}→R^{d_t}` on those pairs (or Procrustes first —
   Wiring says basis-change captures ~85% of it), then
   `dW_t = R · dW_s · R⁺` (R⁺ = pseudoinverse). Apply as
   `patch_t = base_t·(I + dW_t)`. Cost: one closed-form solve, no trunk forward passes for
   the fit itself. Prediction from §1: recovers at most the "aligned" fraction of dW's
   function — measure against ridge→BPE refit (which is itself paired-Procrustes and should
   be *inside* the transferred map's span) and against the ~31% task-conditional residual.
   The comparison answers the real scientific question: *is the learned direction
   size-invariant?*
2. **Direct re-fit per size (the control)** — ridge→BPE and rank-32 SGD are both already
   per-size cheap (~2 min). The family sweep already has per-size ridge-eff numbers; any
   transfer claim must beat "just re-fit at target" to be interesting. This is the control
   arm that makes arm 1 a measurement rather than a convenience.
3. **Decoder context transplant** — the 13M decoder sees the trunk only through
   `cond: R^{d_trunk}→R^{16·256}`. Run both trunks on the same chunked text, collect paired
   chunk-end hidden states `(h_s, h_t)`, fit `cond_t` closed-form to reproduce
   `cond_s(h_s)` from `h_t` (or chain `cond_t = cond_s ∘ R_{h}` with `R_h` an
   activation-ridge on hidden states — the KV paper's recipe, and we don't even need matched
   heads since the residual stream is one vector). The byte_emb/encoder/head internals
   (d_dec=256 throughout, no d_trunk dependence) transfer verbatim.
4. **Read before you transfer** — SVD `dW_s`, lens-read its singular vectors (§2) and its
   per-byte row actions; if the map is a structured byte→subword affinities operator, the
   *content* of the read predicts transferability better than ||dW||.
5. **Gauge-canonicalize before any (A,B)-space operation** — SVD-section the factors before
   seed-averaging, size-comparison, or α-style interpolation (GL(r) gauge, §3.1).
6. **µP caveat** — re-sweep glue LR per trunk size; standard-parameter optima shift with
   width (§4).
7. **APM-style micro-injection of the transferred map** — if conjugated dW_t hurts at full
   strength, dial `patch = base·(I + μ·dW_t)`, μ swept 0.02–1.0; their μ≈0.05 regime and our
   α-dial result both say partial transfer is the norm.

Hardware note: 9B already OOM'd in the family sweep (`a100_family_sweep.log`) on A100-40GB;
27B/35B-A3B need 4-bit trunks or reduced eval scope — the transfer fits themselves are cheap
at any size since only the glue runs training.

## 6. What I could not verify — honesty ledger

- The Zenodo record in the task (23958070) is a WSP preprint, not workshop slides — the
  workshop slides proper appear to live under different Zenodo DOIs (e.g. 10.5281/zenodo.
  21485121, "Parameter symmetries determine representational geometry…", one workshop talk).
  weightsymmetry.com/schedule.html returned 403; the talk list above is from the icml.cc
  mirror, which I did verify.
- `github.com/stdlogits/rs-lenses` does not exist publicly; the lens artifacts are on
  HuggingFace (`stdlogits/rs-lenses`, `camilablank/workspace-lenses`). The lens code is in
  `stdlogits/neuron-lens`.
- No "secret-word-from-weight-deltas" demo was found in stdlogits' public artifacts; §2 lists
  the verified activation-side analogues and the difference from our operator-read case.
- arXiv:2608.02829 is a single-author independent preprint (v5, 25 pp) — internally detailed,
  code released, but not peer-reviewed; the qualitative split (activations align, weights
  don't, residuals carry nothing) is corroborated independently by the KV paper's R²
  measurements.
- arXiv:2608.13596 (APM) is v1, no code link found; "Intersection-Merge" baseline is the same
  group's prior preprint.
- Neural Nuggets numbers are LLaMA-family only; "larger donors help more" extrapolated by me
  to suggest the 4B→0.8B reverse-conjugation test — that suggestion is mine, not theirs.
- The WSP/Quillan-Ronin document's empirical claims (180 s compile, benchmark parity, refusal
  removal) are unverified; its *math* is correctly stated but elementary.
- Poster titles at the workshop are cited as titles; full texts are on OpenReview
  (openreview.net, venue ICML 2026) — only the GL(r)-LoRA abstract was fetched and quoted.

## What to try next (ranked by feasibility on one A100-40GB)

1. **Conjugation experiment** (§5.1): fit R on shared-vocab embedding rows (closed form,
   minutes), transplant dW to 2B/4B-Base, score boundary bpb vs the per-size ridge refit
   control. This is the single experiment that answers "does the learned direction
   transfer?" and it costs less than one training run. *Feasibility: trivial — no training,
   existing eval harness.*
2. **SVD + lens read of the learned dW** (§2): decompose, read top input/output singular
   directions against W_E / the L0 lens; per-byte row-action table. Pre-built R-lenses exist
   for 4B/9B/27B; a 0.8B fit is minutes on A100 by their own bench numbers. Answers the
   parked WSP question (is A·Bᵀ interpretable?) before spending transfer effort.
   *Feasibility: high — download a .pt or fit locally; readout code is ~20 lines.*
3. **Per-size refit baseline row** (§5.2): fill the family-sweep table with
   {constructed, ridge, rank-32} × {0.8B, 2B, 4B-Base, 9B} so transfer gains are measured
   against "just re-train here," the honest null. *Feasibility: trivial compute; 9B needs the
   OOM fixed first (bf16→4-bit or chunked vocab marginalization).*
4. **Decoder context transplant** (§5.3): closed-form cond refit via paired hidden states;
   tests whether the 13M decoder's trunk interface is size-portable — the second half of the
   transfer question. *Feasibility: one forward pass per model over the corpus + one ridge.*
5. **Gauge-canonical seed/direction comparisons** (§3.1): SVD-section (A,B) before any
   weight-space comparison or interpolation; then re-run the cos(dW, construction) table
   under canonicalization to confirm the ≈0 result is content, not gauge. *Feasibility:
   trivial.*
6. **Reverse-direction check** (§1.5): fit the patch map at 4B-Base, conjugate *down* to
   0.8B; Nuggets' "larger donor transfers better" and 4B-Base being the better byte trunk
   both point this way. *Feasibility: same machinery as #1, one extra glue training run.*
7. **POS-style boundary prior** (§4, "When Tokenizers Fail"): interleave a cheap supervision
   signal for boundaries as a third arm next to sim/entropy chunkers. *Feasibility: moderate
   — new supervision head.*
8. **Skip**: APM/IM trunk grafting (violates frozen-trunk constraint), weight-space GNNs/
   hypernetworks (overkill), μP retuning of the trunk (trunk is frozen; only glue LR matters).
