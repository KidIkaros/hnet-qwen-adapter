# H-Net training run state (updated 2026-11-06)

## Lineage
- v1: 2000 dolma docs, enwik8-test 2.4662 -> checkpoints/hnet_2b_{glue,enc,dec}.pt (local)
- v2: fresh 2000-doc rebalance (worse, ~2.62, unsaved — kernel OOM'd on full-eval)
- leg2 (hrun2 exec 1): resume from v1 + new mix (enwik8_train 0.20 + fresh dolma cfgs)
  -> best@1200 held 2.778, ENWIK8-TEST 1.6155
- leg3 (hrun2 exec 2): REBUILD, loaded hnet_resume_latest@1600, N_TRAIN=4000
  -> stopped @850 (held plateaued 16 evals), best@450 held 2.717,
     ENWIK8-TEST 1.5669 -> checkpoints/resume_best/ (local) + /content/ckpt upload
- domain eval (hrun3, A100): hnet_domain_eval.py on best ckpt, EVAL_DOCS=24
  -> stack-v2 0.777 | wiki 1.389 | swallow 1.443 | code/sponge prose 1.470 |
     ultrachat 1.592 | cc 1.628 | proofpile2 1.649 | hplt 1.776 | finemath 1.810 |
     textbook 1.887 | pile_code 1.908 | held 2.717 | enwik8-test 1.5669
- leg4 (hrun4): specialist mix, seed hnet_resume_best@450 -> early stop @3150,
  best @2550 held 2.559, ENWIK8-TEST 1.5444 -> remote results/ + local
  checkpoints/resume_best/ (leg3 kept as *.leg3)
- domain eval post-leg4 (hrun4): pile_code 1.441 (-0.47), ultrachat 1.402,
  code_prose/sponge 0.453, glaive 0.831, toolace 1.365 (untrained baseline),
  finemath 1.646, wiki 1.356 | regressions: stack-v2 2.276 (slice artifact —
  deeper stream region, not forgetting), proofpile2 1.860, textbook 1.963,
  cc 1.776 | CAVEAT: dolma evals hit deeper stream slices each leg — cross-leg
  deltas conflate model change with slice drift
- leg5 (hrun4): agentic mix seeded from leg4 best@2550 -> best held 2.489,
  ENWIK8-TEST 1.5063 -> remote results/ + local checkpoints/resume_best/
  (leg4 kept as *.leg4). NB: exec stream died ~step 2650; remote cell ran to
  completion (latest@18:02Z, best@17:33Z).
- domain eval post-leg5 (FIRST pinned-shard eval — comparable going forward):
  toolace 0.853 (was 1.365 on same docs — real -0.51), ultrachat 1.326
  (same docs as leg3's 1.592), glaive 0.716, stack-v2 1.373, prose 1.453,
  sponge 1.456, pile_code 1.673, swallow 1.654, finemath 1.725, cc 1.891,
  wiki 1.887, textbook 1.727, proofpile2 2.065, hplt 2.002 | held 2.489
- legs 6+ : SHARD-SAMPLED data (RESEARCH_DATA_ORDERING §5.3/§7) — dolma cfgs
  draw rng.sample(6 shard paths) per leg instead of contiguous prefix;
  eval slices PINNED (4 deterministic shards/cfg, "eval:<cfg>" seed, same
  pick in both scripts, excluded from training). Pseudo-configs unchanged.
- leg9 (hrun4): leg-7 recipe verbatim + LR warmup(100) + SWA tail + PATIENCE
  32 -> early stop, best @50 held 2.364, ENWIK8-TEST 1.4317 — PASS (bar
  2.386/1.4442). Warmup killed the rewarm kick (opened BELOW seed — legs 8/8r
  same data all failed without it). SWA evaluated 25-tail avg 2.384, correctly
  kept step-best. ckpt -> checkpoints/resume_best/ (leg7 *.leg7)
- leg8 (FAILED both attempts): toolace .15 -> held spike 3.15, stop @625;
  redo leg-7 recipe -> stop @675 best 2.419. Diagnosis: rewarm transient
  (fresh AdamW constant-1e-4 kick) — Hagele et al. "Beyond Fixed Training
  Durations" names it; fixed in leg9 via warmup + SWA. SEEN NOT advanced for
  leg8's docs (good lineage never saw them); leg9 redrew same shards.
- leg7 (hrun4): leg6 skeleton + real replay (buffer 256 docs, drew 200) +
  cc restored .10 (stack-v2 .20->.15) -> early stop @3100, best @2500 held
  2.386, ENWIK8-TEST 1.4442 — BEST LEG EVER, cleared pre-registered bar
  (held<=2.489, enwik8<=1.5063). ckpt -> checkpoints/resume_best/ (leg5 *.leg5)
- leg6 (hrun4): first SHARD-SAMPLED leg -> early stop @975, best @375 held
  2.513, ENWIK8-TEST 1.5404 — broad-shallow regression vs leg5 (all domains
  ~+0.02 except stack-v2/sponge flat). Read: distribution bump from honest
  multi-partition draws + cc .10->.05; replay was self-referential bootstrap.
  ckpt -> checkpoints/resume_best/leg6/; replay buffer + consumed_shards ->
  data/ (SEEN_SHARDS merged in hnet_resume.py). First REAL replay next leg.
- leg6 (STAGED): leg-5 skeleton + "replay" pseudo-cfg .05 — rotating 512-doc
  reservoir of past-leg docs (cap-rotate, topup 128/leg; Ibrahim ~5%,
  Abbes <=25%). Buffer = results/replay_buffer.bin (concatenated 4KB docs);
  pull to data/replay_buffer.bin post-leg, upload at next leg fire.
  Bootstrap: leg6 tops up from its own collection (legs1-5 docs lost);
  true cross-leg replay from leg 7. cc guard .10 -> .05.
- before firing leg6 on hrun4: move results/hnet_resume_latest_* aside
  (leg5's final) so the prefix chain seeds leg5 best — same move as leg5.

## Sessions
- hrun4 ACTIVE (leg4 done, eval done). Its VM holds leg4 best in results/ —
  dies with session; local checkpoints/resume_best/ is the durable copy.
- Fresh VM needs: best ckpt parts uploaded to /content/ckpt (split -b 64m,
  remote cat-reassemble; enc=11 parts, dec=15, glue direct ~17MB).
  Chain then lands on hnet_resume_best (no latest files on fresh VM).
- ckpt download path: `scp -o "ProxyCommand=colab ssh --proxy-mode -s SESS
  -i ~/.ssh/id_ed25519" root@x:/content/.../FILE dest/`

## Script: kernel/hnet_resume.py
- prefix chain: hnet_resume_latest -> hnet_resume_best -> $CKPT_PREFIX -> hnet_2b (dirs: results/, /content/ckpt)
- saves best on held-mean improvement (16-doc held), latest every SAVE_EVERY steps
- early stop: PATIENCE=24 stale evals after MIN_STEPS=500, IMPROVE=0.002
- SEEN dict tracks per-config docs consumed (skip before collect); update after each leg
- pseudo-configs: enwik8_train (byte slices of enwik8['train']), ultrachat
  (HuggingFaceH4/ultrachat_200k train_sft, joined messages), glaive_function_calling
  (glaiveai/glaive-function-calling-v2, system+chat)
- dolma4pdfs + swallow-code are unloadable — excluded

## Leg-6 mix (RESUME_MIX, staged)
stack-v2 .20 | pile_code .15 | ultrachat .10 | code_prose .10 | sponge .10 |
glaive .10 | toolace .10 | replay .05 | cc .05 | enwik8 .05

## Consumed docs — NEW SCHEME (legs 6+)
- dolma cfgs: SEEN_SHARDS = consumed shard paths (log prints picks each leg +
  remote results/consumed_shards.json; merge into the file's SEEN_SHARDS).
  Legs 1-5's prefix consumption predates path bookkeeping (unrecoverable —
  ok, <4 repeats free). Eval shards excluded via "eval:<cfg>"-seeded pick.
- pseudo-configs (sequential still): SEEN doc counts, incl. pinned eval
  offsets — ultrachat evals [0,24), toolace [0,24), glaive [824,848).
- post-leg6 SEEN: enwik8_train 2200, ultrachat 1448, glaive 1248, toolace 824
- OBSOLETE for sampling (history only): cc 2608, wiki 1288, textbook 1288,
  stack-v2 2488, code_prose 1288, pile_code 1148, sponge 1288, finemath 748,
  hplt 1248, proofpile2 888, swallow 548, dolma_code 300, addons 100

## Colab ops
- `colab new -s hrun4 --gpu A100`, then `colab exec -s hrun4 -f kernel/hnet_resume.py --env N_TRAIN=4000`
- uploads >~20MB fail on contents API -> split -b 64m + remote cat-reassemble
- kernel restarts between execs are common -> always assume REBUILD mode
- launcher processes outlive crashed parents; local log = streamed exec output
- HF_TOKEN secret fetch fails in exec mode (unauthenticated HF requests are fine)

## 2026-11-07 — Capability + weight-space chapter (handoff for next session)

### Model artifact
- Best = **leg 9** (`checkpoints/resume_best/` local): held 2.364, enwik8-test **1.4317**. All 14 pinned domains best-ever. Receipt `results/eval_receipt_20261107_022505.json`, pass=true.
- Warmup+SWA machinery in `kernel/hnet_resume.py` is proven permanent (fixed rewarm transient that killed legs 8/8r).

### Capability findings (the big one)
- Free greedy decode on toolace exam (SKIP=2500, 15 items): right 4/15, any-call 12/15, hallucinated 8/15 — format fluency without selection.
- **Constrained scoring flips it**: score each declared tool's `[name(` bytes by model logprob, argmax → **gold@1 9/15 (60%), gold@3 11/15**. Capability is LATENT; gap was inference protocol, not weights.
- **Two-phase end-to-end (40 items)**: pick@1 21/40 (52.5%), pick@3 27/40, call-closed 30/40 — emits complete `[name(args)]` calls, args sometimes messy/duplicated.
- Scripts: `kernel/hnet_gen_eval.py`, `kernel/hnet_select_score.py`, `kernel/hnet_two_phase.py`, `kernel/selector_collect.py` (collected 60 items × 8 layers hidden states → `selector_data.npz`, pulled to /tmp).

### Weight-space experiments (all honest negatives/learnings)
- `kernel/hnet_weight_autopsy.py`: SFT delta (Qwen3.5-2B instruct − base) concentrated mid-late `linear_attn.in_proj_qkv`+`mlp.gate_proj` layers 14-22, diffuse (~rank 1500/2048). Our CPT delta ~4× larger, concentrated enc layers.
- Full delta merge @α=1.0 (`kernel/hnet_delta_graft.py`): 0/15 all metrics — broke adapter↔trunk interface. Parallel branch blend @α=0.5 (`kernel/hnet_branch_graft.py`): ~base behavior. Task vectors don't survive the calibrated interface.
- **Qwen3.8-Flash-Next = 180B Qwen4-exp MoE — not a weight donor** (dims/architecture). Big models = teachers only (and need API/hosted inference, won't fit one A100).
- `kernel/wsp_contrast.py` built but never ran (sessions died).

### Architecture decision for next session — S1+S2 / Jev layer
- User's repos in play: `qwentensor` (compile nn.Module→5-bit program running inside Qwen3.5, proven on 2B), `qwentpu` (TPU machinery underneath), `wsp-reproduction` (suppression edits). Prior art: `divagr18/qwenputer` (RISC-V CPU as 11 INSERTED layers — append-not-overwrite pattern), `muellerberndt/cadence` (equilibrium brain; arousal-gated S2, phase qualification, grow/graft).
- Reviewed JEV papers (~/Documents/Library/technical/ai-ml/): **Jev = non-generative decision layer emitting typed calibrated verdicts (Choice/Noul/Score), never text** — our constrained scorer is a primitive Jev.
- Ranked roadmap: (1) `nanocore-s1/src/decision/` verdict protocol — calibrated scores + Noul abstain (converts ~47% unsure items into honest no-call); (2) NumericJev multiway decision tree — beats flat candidate selection +~3pp even with answer in list; (3) Jev-Mem control plane for multi-step orchestration (real blindspot).
- Design: `H-Net byte-LM (S1) → Jev decision layer: constrained scoring + tree refine + calibrated abstain → emit call or Noul`. All inference-side, no retraining, receipt-able via existing eval harness.
- nanocore-s1 AGENTS.md: Kaggle-bound, read docs/ROADMAP.md first, never run model weights locally.

### Session state
- `hrun6` alive (A100) with ckpt at /content/ckpt + repo at /content/qt/. hrun4/hrun5 died earlier; all artifacts safe locally.
- Local disk was full — cleaned /tmp + uv cache (~5.6GB); old leg ckpts (.leg3/.leg4/.leg5/.leg7, leg6/) still in checkpoints/resume_best/ ~9.6GB if space needed.

### RSI design note (2026-11-07, papers reviewed)
- Recursa (arXiv:2611.04035): knowledge gap vs behavior gap — our constrained-scoring find was a BEHAVIOR gap; fix at the system layer (decision layer), not retraining. Thesis: adapt the whole system, not just the model.
- Terminal-RSI: verified trajectories -> policy update loop; verifier gates what feeds back. Our verified-replay = same loop minus RL machinery.
- MetaExp (arXiv:2610.21492): experience that transfers ACROSS episodes = the multi-step orchestration piece (Jev-Mem working trace).
- DECISION: self-improve at behavior layer, not weight layer. Verified-only replay (conformal-confirmed correct calls, capped fraction). Verdict-outcome -> gate feedback (ErrorModel-lite). NO unverified self-generation into weights (collapse). Conformal layer = the verifier that makes mild RSI safe.
- S1+S2 measured: enc 87.3% / byte 71.4% / union 92.9% / routed(margin>=.01) 89.7% @ esc~10% on n=126 clean (74/200 dropped undeclared-gold = natural Noul set). jev_decide.py = fusion+conformal+routed tier.

### emit_score correctness audit (2026-11-08, hrun8, slice skip=2500 n=40)
- New locked-exam scorer `src/emit_eval.py::score_emission` — 4 labels:
  parse_ok / tool_valid / args_valid / gold_match (+ legacy phenotype/closed).
  Driver `kernel/emit_score.py` (MODE=base|constrain). 10 unit tests.
- RESULT — the "closed" metric was inflated ~5x:
    base:      loops 19  closed 29  parse 35  tool_valid 6  args_valid 4  gold_match 1
    constrain: loops  0  closed 35  parse 35  tool_valid 6  args_valid 4  gold_match 1
  Constrain changed ZERO correctness labels — every "fixed" call is the same
  hallucinated name in a force-closed wrapper. Decode rules treat symptoms.
- Declared subset (gold in tools, n=27): tool==gold 6/27, gold_match 1/27,
  tool_valid 6/27 — and ALL 6 valid calls were the correct tool (6/6
  precision). Capability exists but is gated by grounding: model attends
  schema only ~22% of the time; failures are fuzzy-name hallucinations
  (legalClauseCategories~Library, getTeamsResponses~TeamStats).
- NEW phenotype seen in emits: post-call prompt regurgitation — after `)]`
  the model recites the system prompt instead of stopping (`)]You are an
  expert...`). Stop rules catch `]\n`/`\n\nUSER` but not `)]`+prose.
- VERDICT: bottleneck is schema-grounded tool SELECTION, not degeneration.
  Supports selector architecture (name from declared list) over free-gen.
  Glial/onset gate's real target = predict ungrounded emission
  (tool_valid failure), a semantic label no regex names.
- Receipts: results/emit_score_{base,constrain}_*.json (this slice).

### emit_score goldname decomposition (2026-11-08, hrun8, same slice)
- MODE=goldname forces `gold(` prefix — isolates arg generation from selection.
- RESULT: free-gen vs name-forced on n=40:
    tool_valid  6→27 (by construction)   args_valid 4→6   gold_match 1→1   loops 19→16
- Forcing the correct name changed NOTHING about correctness. Model invents
  arg names not in schema (angle=15×N, check_vowel="e" self-dup), argdup
  loops persist inside args; values wrong even when names valid.
- VERDICT: free-generation of tool calls is non-viable at EVERY level
  (name, args, values) — loops were a symptom of schema-space flailing.
  Architecture must make every level a closed-set problem: name from list,
  arg names from per-tool schema set, values extractive (query spans) or
  scored candidates. Model role = query understanding; structure = schema.
- Receipt: results/emit_score_goldname_1794183844.json

### call_synth constrained pipeline (2026-11-08, hrun8, skip=2500 n=40)
- kernel/call_synth.py: closed-set synthesis — name argmax over tools+none,
  arg names argmax over schema set, values argmax over query n-gram spans.
  Verified by score_emission. Receipt: results/call_synth_*.json
- RESULT vs free-gen baselines:
    tool_valid  6→39/39 answered (guaranteed by construction)
    args_valid  4→36/39 (92% — arg-name selection SOLVED)
    gold_match  1→3/40   loops 19→0   abstained 1
- Decomposition: name_acc 19/39 (49%); of name-correct, gold_match 3/19.
- VERDICT: constrained architecture validated — validity guaranteed,
  arg names solved. Remaining bottlenecks, in order:
    1. value correctness (extractive spans miss gold values — 3/19 given
       right name); candidates need richer sources (desc examples,
       numbers unquoted, noun phrases) or scored generation
    2. name selection accuracy 49% — improve via S1/StartLux ensemble,
       margin-gated escalation, or teacher distillation
- This is the System-1 shape realized: every level a scored closed-set
  decision, verifier-gated, abstention preserved.

### call_synth v2-v3 (2026-11-08, hrun8) — deterministic, ceiling measured
- Fixes: schema arg order now tuple (was frozenset → hash-order nondeterminism);
  value candidates n<=4 + bare digit forms + ".0" float variants; gold_args
  stored per row; self-healing git sync in driver preamble.
- RESULT (3 runs, identical — pinned-exam reproducibility confirmed):
    tool_valid 39/39 answered | args_valid 36/39 | gold_match 3/40
    name_acc 19/39 (49%) | abstained 1 | loops 0
- Extractive ceiling measured: 43/65 gold arg values (66%) verbatim in query;
  misses = JSON-structured args (~10, need generative composition),
  normalization (~8, partially fixed via .0 variants), absent (~4).
- PER-ARG truth: value-exact 8/36 overall, 8/33 on name-correct — the scorer
  picks plausible spans but only ~22-24% exact even when extractable.
  gold_match=3 is all-or-nothing compounding, not the capability number.
- VERDICT: name_acc 49% is now the dominant gate (everything downstream
  multiplies by it). Value selection ~22% is second. Next lever: name
  selection accuracy — teacher-ceiling run (27B prefix scoring, same slice)
  to measure achievable bound before choosing distill vs escalate.
- Receipt: results/call_synth_1794189751.json

### teacher ceiling on name selection (2026-11-08, hrun8, same slice)
- kernel/teacher_score.py SKIP=2500 N=40: 27B Heretic-abliterated 4bit,
  prefix-scoring protocol identical to call_synth phase A.
- RESULT: prefix 27/27 DECLARED (100% name-selection accuracy),
  none-picks 1 (correct abstention on undeclared), gen 11/27 (free-gen
  format noise — not comparable).
- vs call_synth 2B byte-scorer: ~19/27 declared (~70%).
- VERDICT: name selection is SOLVABLE — gap is pure scorer capability,
  not task ambiguity or pipeline design. Distillation path confirmed:
  teacher score matrices (results/teacher_scores_*.json) are soft labels
  for training the 2B scorer. synth-vs-teacher agreement 26/40.
- Full pipeline now decomposed: name selection (teacher 100%, student
  ~70% → distill), arg names (92% solved), values (~22%/arg exact,
  extractive ceiling 66%).
- Receipt: results/teacher_scores_1794190811.json (full score matrices)

### call_synth TEACHER=1 full-pipeline ceiling (2026-11-09, hrun8)
- kernel/call_synth.py TEACHER=1: same closed-set pipeline, every score()
  routed through 27B teacher-forced mean-token logprob. 27B loaded 4bit.
- RESULT (declared n=27): name_acc 27/27 (100%), args_valid 25/27,
  gold_match 6/27; overall n=40: parse 39, tool_valid 39, args_valid 36,
  gold_match 6, abstained 1, loops 0.
- Right-name misses (21) decompose: structured-JSON args ~10 (extraction
  cannot produce them — generative composition needed), normalization
  near-misses ~4 (12345 vs 12345.0, "2020" vs 2020 — candidate variants
  fixable), extractive overshoot ~6 (picked plausible n-gram, not the
  gold span — needs per-arg-conditioned value candidates).
- VERDICT: even the 27B scorer caps gold_match at 6/27 under pure
  extraction. Realistic ceiling with better value candidates ~12-15/27;
  ~10 items need generative arg composition regardless. Value selection,
  not name selection, is now the binding constraint for strict correctness.
- Receipt: results/call_synth_teacher_1794193104.json

### compress_trunk launched (2026-11-09, hrun8) — 27B bf16-offload GPTQ probe
- kernel/compress_trunk.py: evicts kernel-global GPU holders, loads
  0bserverx/Qwen3.8-27B bf16 device_map=auto (30GiB GPU / 70GiB CPU),
  measures BASE name-selection on pinned slice, then chunked Hessian
  calibration + GPTQ-4bit (ROTATE on, PRUNE=0, N_CALIB=48,
  BLOCK_LAYERS=4) over q/k/v/o/gate/up/down linears, re-measures same exam.
- First attempt OOM'd in caching_allocator_warmup (kernel-global 2B
  trunk/hn + nf4 teacher still resident); fixed by evicting all known
  GPU-holder globals + gc + empty_cache, GPU budget 37→30GiB.
- RUNNING: base exam ~25min, calib+quant ~1-2h (CPU-offloaded Cholesky
  bottleneck), quant exam ~25min. Verdict = declared name-acc pre/post.

### compress_trunk RESULT (2026-11-09, hrun9) — 27B GPTQ-4bit: ZERO capability loss
- Session history: hrun8 died mid-run (Colab kill) after base exam
  confirmed 27/27 and blocks 1-8 quantized; partial state was
  in-memory only -> full rerun on hrun9.
- hrun9 turned out to be 80GB-class GPU -> whole bf16 trunk fits on-GPU
  (max_memory 72GiB), zero meta-offloaded params, exam ~4min vs ~25min.
- Bugs fixed en route: (a) kernel-global eviction incomplete -> OOM at
  warmup; (b) CPU-offloaded Linear params are META tensors — real weights
  live in AlignDevicesHook.weights_map = PrefixedDataset (keys() gives
  full names, wm[short] fetches tensor); v1 crashed at block 9/16,
  v2 resolves all 119 meta weights up front and writes dequantized
  values back into the hook's tensor (validated live before rerun).
- METHOD: 256 trunk linears (q/k/v/o/gate/up/down x64 layers), GPTQ 4-bit
  + Hadamard rotation, Hessian calibrated on 48 ToolACE prompt blocks,
  blockwise (16 blocks x 4 layers).
- RESULT — pinned name-selection exam, same slice/protocol:
    base:  right 27/40 | declared 27/27 | decl_right 27 | none-picks 3
    quant: right 27/40 | declared 27/27 | decl_right 27 | none-picks 3
  IDENTICAL. 4-bit GPTQ + rotation + on-domain Hessian calibration
  preserved 100% of measured tool-name selection on the 27B trunk.
- CAVEATS: single 40-item exam, name-selection only (args/values/gen
  unmeasured on quantized trunk); this measures dequantized-weight
  accuracy — packed artifact + scales still needed for actual size;
  27/27 could partly reflect exam easiness (teacher ceiling was also
  27/27 — the exam may saturate; larger pinned slice needed).
- Receipts: results/compress_trunk_1794212023{,_base}.json +
  .progress.log; ledger entry gptq-trunk-27b.
- VERDICT: direct 27B->compact compression path VALIDATED at the
  behavior level. Next: (1) arg/value-level exams on quantized trunk,
  (2) packed artifact for real size measurement, (3) H-Net refit on the
  compressed host, (4) larger pinned exam (500+) before capability claims.
