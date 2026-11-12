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

### big_exam 500-item saturation check (2026-11-09, hrun9) — resident=quant
- kernel/big_exam.py: same prefix-scoring protocol, fresh slice
  SKIP=0 N=500 (283 declared-gold items — much harder than pinned 40).
- RESIDENT phase (GPTQ-4bit quantized trunk):
    right 270/500 | declared 283 | decl_right 270/283 = 95.4%
    none-abstains 49 | wrong-name picks 13 (all near-miss confusions:
    getExhibitionDetails vs getExhibition etc.) | 0 abstains on declared
- Saturation concern RESOLVED: pinned-40 27/27 was partly slice easiness;
  on 283 fresh declared items quant holds 95.4%. Phase-2 fresh-bf16
  base OOM'd (evict blacklist missed flat/blocks module refs -> 35GB
  pinned); fixed via whitelist _evict + MODE env; base rerun pending.
- INCIDENT: quantized model later evicted by crashed phase-2 before
  save_quant could persist it -> compress_trunk rerun in flight with
  SAVE_DIR to persist artifact in the same run.
- Receipt: results/big_exam_1794217042_resident.json

### Eval-gate suite built (2026-11-09) — upload gated on evidence
User directive: qwen27b-gptq4-toolace HF release happens ONLY after
evals pass; it is the BASE compressed artifact (pre-H-Net). A separate
trunk+adapter artifact follows later. Lineage stamped in manifest.

Drivers (all committed):
- kernel/trunk_call_eval.py — call_synth pipeline on TRUNK prefix_lp
  scorer, ToolACE pinned slice (skip=2500 n=40) + SKIP override; direct
  comparison vs 2B call_synth (gold_match 3/40, name ~49%). Uses
  gold_full (untruncated) — noted gold-source difference.
- kernel/bfcl_name_eval.py — Phase A: name-selection on official
  BFCL_v3 simple+multiple (gorilla-llm dataset), N_PER_CAT=100.
- kernel/bfcl_call_eval.py — Phase B: name->argnames->per-arg-conditioned
  extractive values -> AST-equivalent check vs ground_truth.
  Required-args-only v1, extra-arg penalty.
- kernel/hf_upload.py — gated upload to PRIVATE KidIkaros repo,
  README card marks pre-H-Net lineage. Needs --env HF_TOKEN (verified).

EVAL ORDER on hrun9 after compress_trunk saves artifact:
  1. trunk_call_eval (resident quant)      ~15min
  2. bfcl_name_eval   (resident quant)     ~10min
  3. bfcl_call_eval   (resident quant)     ~25min
  4. big_exam MODE=base -> fresh bf16      ~25min (one reload)
  5. bfcl_name_eval + trunk_call_eval SKIP=0 N=100 on fresh = base refs
PASS = quant≈fresh delta small AND absolute numbers respectable;
replacement-vs-2B decision is a separate later comparison.

### EVAL GATE results on resident quant (2026-11-09, hrun9)
Artifact: /content/qt/quant27b-gptq4-toolace (51G bf16-dequantized,
saved by compress_trunk SAVE_DIR; manifest written).

1. trunk_call_eval (ToolACE pinned 40, trunk scorer):
     tool_valid 34/40 answered | args_valid 34 | gold_match 3/40
     abstained 6 | declared n=27: name_acc 25/25 = 100%
   vs 2B call_synth: name ~49-70%, gold_match 3/40.
   VERDICT: trunk dominates name selection; gold_match tied at the
   extractive-value ceiling (same bottleneck as teacher's 6/27).
   Receipt: results/trunk_call_1794222836.json

2. bfcl_name_eval (official BFCL v3, quant):
     simple 100/100 = 100% | multiple 88/100 = 88% | TOTAL 188/200
   Misses are near-miss confusions (store.find_nearby vs
   library.find_nearby), zero hallucinated names.
   Receipt: results/bfcl_name_1794223782.json

3. bfcl_call_eval (Phase B full pipeline) — RUNNING.
4. big_exam MODE=base (fresh-bf16 delta) — pending.

### big_exam base-vs-quant delta (2026-11-09, hrun9) — ZERO measurable loss
- fresh bf16 on identical 500 items: decl_wrong 13/283, abstains 27
- resident quant on same:            decl_wrong 13/283, abstains 49
- Only 2 declared items differ in prediction — mirror-image swaps
  (net zero). Compression cost on name-selection at scale: NONE.
- Neighbor cost: quant abstains +22 more (slight conservatism).
- Receipts: big_exam_1794217042_resident.json + big_exam_1794225793_fresh.json
- BUGFIX: _KEEP must contain itself (self-eviction NameError) — patched
  in big_exam + compress_trunk.
- bfcl_call_eval (quant) final: simple 13/100, multiple 13/100 = 13%
  full-call — value/arg pipeline is the bottleneck (extractive ceiling),
  name-selection was 94%. Receipt: bfcl_call_1794223950.json

### bfcl_call base-vs-quant (fresh bf16 vs resident GPTQ-4bit, same 200 items)
- quant: simple 13/100, multiple 13/100 = 26/200
- base:  simple 14/100, multiple 13/100 = 27/200
- Delta: +1 item — full-call pipeline lossless under compression.
- Name-selection deltas across ALL exams: 0-1 items (same wrong answers).
- CONCLUSION: GPTQ-4bit preserves bf16 behavior on every measured axis;
  the 13% BFCL call-acc is a PIPELINE ceiling (extractive values),
  identical for bf16 and quant — not a compression defect.

### trunk_call base-vs-quant (pinned skip=2500 n=40, fresh bf16 vs GPTQ-4bit)
- IDENTICAL aggregates: parse_ok 34, tool_valid 34, args_valid 34,
  gold_match 3, abstained 6, loops 0; declared 25/25/3; name_acc 25/25.
- EVAL GATE COMPLETE — compression behaviorally lossless on every axis:
    names-40:   27/27 = 27/27
    names-500:  270/283 = 270/283 (2 mirror-swap diffs only)
    bfcl-name:  188/200 vs 189/200
    bfcl-call:  26/200 vs 27/200
    toolace-40: bit-identical aggregate
- 13% BFCL call-acc = extractive-pipeline ceiling, NOT a weight defect
  (bf16 shows the same). Name-selection layer: quant==base everywhere.
- vs 2B+H-Net call_synth (same slice): name_acc ~49% -> 100% answered;
  gold_match tied at 3 (value-composition bound on both).

### UPLOAD COMPLETE (2026-11-09)
- KidIkaros/qwen27b-gptq4-toolace — PRIVATE repo, verified:
  11 files, 53.8GB (2 safetensors shards 49.8+4.0GB, QUANT_MANIFEST.json,
  README.md with eval-gate table, tokenizer/config/template).
- hf_upload.py upload_folder: 549s. Token passed via --env only.
- TOKEN REMINDER: hf_... exposed in transcript — user must revoke at
  huggingface.co/settings/tokens.
- hrun9 still alive: fresh bf16 resident + artifact at
  /content/qt/quant27b-gptq4-toolace.
- NEXT: packed-int4 serialization driver (real size win); value-composition
  (generative arg synthesis) is the capability bottleneck, not weights.

### PACKED INT4 ARTIFACT COMPLETE (2026-11-09, Modal H200)
- KidIkaros/qwen27b-int4-toolace — PRIVATE, verified live
- 17.68GB packed int4 (5 safetensors shards), 400 linears @4bit incl
  linear_attn in_proj_qkv/in_proj_z/out_proj (~94% params covered)
- pinned name exam on dequantized resident: 27/27 declared, none=3 —
  matches bf16-dequant base artifact exactly
- loader: src/int4_load.py reconstructs packed tensors -> dequant -> model
- manifest caveats carried: rotation INERT (no pow2 in-features),
  plain GPTQ-4bit; embed+lm_head+norms remain bf16 (~5.4GB of the 17.7)
- lineage: qwen27b-gptq4-toolace (53.8GB bf16-dequant backup) vs
  qwen27b-int4-toolace (17.7GB true packed) vs future ...-hnet artifact
- SUB-10GB analysis: needs int3 linears or int2+int4-embed; int4 floor
  with bf16 embeddings ~17.7GB; <10GB enters unverified quality regime
- NEXT: 500-item big_exam on packed-dequant resident (optional deeper
  gate); H-Net refit -> separate artifact; generative arg composition

### Modal Phase A+B complete (2026-11-09) — eval gate + first 27B training
- hnet-eval general-quality gate (int4-dequant vs fresh bf16, identical
  battery, greedy):
    MMLU-5shot   int4 932/1140=81.75%   bf16 959/1140=84.12%   Δ-2.37pp
    HumanEval@1  int4 67/164=40.85%     bf16 104/164=63.41%   Δ-22.56pp
    wikitext-2   int4 ppl 5.871         bf16 ppl 5.679        Δ+0.19
    toolace40    identical both: 27/27 declared, 27 right, 3 none
  VERDICT REVISION: "GPTQ-4bit behaviorally lossless" held ONLY for
  selection/scoring metrics (names, args, MMLU-ish). Open-ended
  generation shows a real regression concentrated in code — 37/164
  pass->fail flips, ~6σ vs binomial noise. Mechanism candidates: GPTQ
  Hessian calibrated on ToolACE prompts under-weights code directions;
  pass@1 all-or-nothing sensitivity to 4-bit noise; Heretic-tune
  fragility. IMPLICATIONS: (a) HF repo card's lossless framing needs
  scoping to selection tasks; (b) H-Net trains fine on the int4 host
  (adapter learns around frozen weights) but generative downstreams
  (arg composition!) inherit the bf16-vs-int4 gap; (c) if code
  capability matters, recalibrate GPTQ Hessian on mixed-domain corpus
  or raise to int8 for lm-critical linears.
  Receipt: results/eval_gate_modal_1794249369.json
- hnet-train27b Phase B (first gradient training on the 27B trunk):
    enc4+dec4 DeltaNet slices + dechunk/boundary/w_sup = 3.119B
    trainable; ridge adapters (w_in/w_out/w_main/w_enc) frozen; int4
    trunk frozen. N_TRAIN=1000 dolma-mix docs, doc_cap 4096B, lr 1e-4
    AdamW wd .01 clip 1.0, eval every 100 steps.
    1000 steps in 2629s (~2.6s/step H200). held(bpb, 3-doc mean)
    3.407->2.794; enwik8-test(50KB) 3.641->3.142 (best 3.115@900,
    final-step uptick to 3.142@1000). Still descending at budget end —
    no plateau, more legs/data warranted.
    vs Phase A untrained enc6/dec8: enwik8 8.81->3.14, held 9.35->2.79.
    vs 2B trained ref enwik8-test 1.4317: gap remains (~2.2x) but this
    is leg-1 of a fresh lineage on a 13.5x larger trunk, 1000 docs vs
    the 2B's multi-leg corpus.
    CAVEATS: held here = mean over held_docs[:3] (Phase A used 8 docs);
    best==latest (held improved at final eval) — duplicate weights,
    ~12.4GB total on hnet-artifacts volume. Next: longer run or
    multi-leg resume; consider eval_every smaller early for curve shape.
    Receipt: results/hnet27b_train_1794249369.json

### Parallel CPU lane on Colab (2026-11-09, session `hpar`) — arg composition + next-leg mix
- Colab capacity check: A100/H100 rejected (no entitlement), T4 Service
  Unavailable, **CPU runtime READY** — so the parallel lane is a remote
  CPU VM (nothing runs on the laptop). Balance still 0.00 compute units.
- **kernel/arg_compose.py** (new): schema-aware candidate composition,
  deterministic + model-free. `arg_specs` parses per-property type/enum/
  desc/items/props/required from the tool JSON; `arg_prior` gives
  deterministic arg-name bonuses (required, lexically grounded, enum-hit);
  `value_candidates` widens the literal set (quoted literals, enum members,
  booleans, dates, numbers+".0", entity runs, spans, composed JSON for
  object/array args, ranges).
- **kernel/arg_recall.py** (new): the model-free gate. Streams the same
  pinned slice (skip=2500, n=40) and separates declared / parsed /
  recovered, with the extractive baseline recomputed on the SAME subset.
- RESULT: 65 gold args = 46 declared + **19 undeclared-gold** (gold tool
  absent from the prompt — the natural Noul set, not a composer miss).
  Schema parse: **46/46**. Candidate recall **35/46 = 76.1%** vs extractive
  **28/46 = 60.9%** → **+15.2pp**. Receipt: results/arg_recall_1794252019.json
- Parse bug found and fixed: ToolACE uses Python-style types ("int",
  "dict", "list") and tools whose properties include a field named "name"
  truncated the block splitter → 19/65 args lost. Now top-level brace
  matching + type normalization.
- Remaining 11 misses, classified: 5 nested-JSON composition (arrays of
  objects: launch_parameters, activities, submission_details,
  clickSequence, clauseCategories), 3 normalization (2023-12, 2020-2021,
  Asia/Tokyo), 1 integer form, 1 verbatim-but-dropped (careerGoal —
  ordering of span candidates).
- **src/factory.py**: added `DOLMA3_SOURCES` + `NEXT_MIX` +
  `corpus_stream_next` (weighted multi-source streaming; `DOLMA3_REPO`
  env override for smoke tests). Shard layout verified on the Hub:
  `data/<source>[-<subset>]/shard_*.jsonl.zst`. All 8 sources stream OK
  (6 dolma3_mix + Nemotron-Terminal `skill_based_mixed` + Open-SWE `v2.0`/
  `claudecode`). `_text_of` now joins `conversations`/`messages` lists —
  without it the agentic corpora yielded 36 bytes of metadata instead of
  48KB/80KB of trajectory text.
- Not yet run: the composer's GPU eval (needs the 27B scorer; Modal H200
  after the weakness probes) and a full next-leg stream (only 10-doc smoke).

### BFCL candidate-recall — the composer on real, verified golds (2026-11-09, `hpar`)
- **kernel/bfcl_recall.py** (new): same model-free gate on BFCL v3
  (gorilla-llm/Berkeley-Function-Calling-Leaderboard), categories
  simple/multiple/parallel/parallel_multiple, 40 items each.
- RESULT: 676 gold args, schema parse **674/676 = 99.7%**; candidate
  recall **522/674 = 77.4%** vs verbatim baseline **43.3%** →
  **+34.1pp**. Receipt: results/bfcl_recall_1794252306.json
- vs ToolACE (+15.2pp): the lift is 2.2× larger because BFCL golds are
  often NOT verbatim (43% vs 61%), i.e. real schemas demand exactly the
  normalization/composition the composer adds. Recall itself is
  comparable (77.4% vs 76.1%) — the composer is not overfit to ToolACE.
- Remaining 152 misses skew to integer/array in `parallel` categories
  (multi-call items where per-call value binding is ambiguous from a
  single user turn) — the next lever, not a recall-cap problem.
- **Metric correction:** BFCL's acceptable sets include `''`/defaults when
  an argument may be omitted — 54 of the "misses" were legitimate
  abstains, not composer failures. Corrected receipt
  (results/bfcl_recall_1794252374.json): required-subset recall
  **468/620 = 75.5%**, true misses **100** (string 31, array 29,
  integer 25, number 10, object 4). Verbatim baseline on the same
  required subset is 33.5% → **+42pp**.
- **Composer fixes (miss-dump driven):** the 100 misses decomposed into
  mechanical classes — `^`→`**` notation, edge-stripped periods on
  abbreviations (Washington D.C.), punctuation-bearing values (New York,
  NY), parenthesized coordinate pairs, and ranges with variable binding
  (`x=1 to x=3`). Added `_spans_raw` (punctuation-preserving n-grams),
  `_COORD`/`_RANGE2` extractors, abbreviation period preservation, and
  `^`→`**` variants. Re-measured (results/bfcl_recall_1794253868.json):
  required-subset recall **481/621 = 77.6%** (was 75.5%), true misses
  **88** (was 100). Semantic defaults (initial_velocity=0 from "started
  from rest") remain — inference, not extraction.
- **probe_int4.json WAS on the volume** (earlier "missing" was a stale
  check). Both probe arms now complete — full comparison:
  HumanEval bf16 103/164 (0.628) vs int4 67/164 (0.4085); GSM8K bf16
  95/100 vs int4 93/100 (0.93). **The int4 regression is syntax-
  specific**: fail_kinds syntax 35→78 (more than doubled) while assert
  fails stayed flat (21→16) and reasoning is preserved. Selection
  (MMLU −2.4pp) and math (GSM8K −2pp) are near-lossless; only code
  *emission precision* collapses. Weight-error map shows uniform rel-L2
  ~0.154 across all projections — no localized code-direction damage.
  Interpretation: GPTQ preserves capability but degrades fine-grained
  token emission needed for syntactically valid code; fix candidates
  are mixed-domain Hessian recalibration or int8 on emission-critical
  linears, not broad retraining.
- **Flip taxonomy** (results/probe_{bf16,int4}.json): both_pass 59,
  bf16→int4 regressions **44** (syntax 37, assert 5, ValueError 2),
  int4→bf16 improvements 8, both_fail 53. Net −36 ≈ observed −36.
  **84% of regressions are syntax**, not wrong logic — int4 makes the
  model emit malformed code, not incorrect code. Small per-layer weight
  errors compound autoregressively into broken tokens; consistent with
  uniform error map. Mitigations ranked: (a) grammar/constrained
  decoding at emit time (free, inference-side), (b) int8 on
  emission-critical linears, (c) code-weighted Hessian recalibration.
- **METRIC CORRECTION:** the old `recall_required` formula
  ((recovered−abstain_ok)/(parsed−abstain_ok)) was not a true
  required-subset recall. Both harnesses now track explicit
  req_total/req_hit/opt_total/opt_hit buckets. Corrected base-cat
  numbers (results/bfcl_recall_1794254595.json): **required recall
  472/560 = 84.3%**, optional recall 63/114 = 55.3%, parsed recall
  535/674 = 79.4%. Prior "77.6%" understated required recall.
- **BFCL breadth (Kaggle CPU kernel mauricew/bfcl-breadth, N=100/cat,
  results/bfcl_breadth_1794254424.json):** live_simple 167/278=60.1%,
  live_multiple 222/310=71.6%, live_parallel 70/109=64.2%,
  live_parallel_multiple 99/145=68.3% (recall_parsed; includes
  optionals). Schemas parse 100% on live APIs. **irrelevance**: 100/100
  schema parse, mean 45.6 candidates/arg slot — quantified
  hallucination surface for the abstain path. **multi_turn_base**:
  items carry no `function` field (schemas live in involved_classes/
  initial_config) — needs a different harness; deferred to end-to-end.
- VERDICT: **move the eval substrate to BFCL.** ToolACE's golds are
  measurably noisy (19/65 gold args name undeclared tools), its schema
  dialect needed a parser workaround, and BFCL is the leaderboard-
  comparable substrate.

### Next-leg data candidates + Modal platform notes (2026-11-09)
- Weakness probe IN FLIGHT: modal/hnet_probe27b.py — weight_map (per-tensor
  int4-vs-bf16 rel-L2 by proj/depth) + probe_arm{int4,bf16} rerunning
  HumanEval-164 w/ per-item dumps+failure classes and GSM8K-100 4-shot.
  App hnet-probe27b; receipts -> /artifacts/{weight_error_27b,probe_*}.
- DATA (recency-biased HF scan, verified ungated unless noted):
  * allenai/dolma3_mix-6T — THE Olmo3-32B pretrain mix (63.9k jsonl.zst
    shards, ODC-BY). Sources addressable via data_files globs:
    common_crawl-* (36.9k), olmocr_science_pdfs-* (24.8k),
    stack_edu-{Python,C,...} (1.98k), finemath-3plus,
    rpj-proofpile-arxiv, dolma1_7-wiki-en. Needs zstandard (have it).
    dolma3_mix-150B-1025 = same mix small-sample for smoke tests.
  * Dolmino anneal pool (moltherm1/olmo-mirror-dolma3-dolmino-mix-100b-1125,
    ODC-BY, arxiv 2512.13961) — Olmo3 stage-2 high-quality 100B: math/
    code/QA/thinking-traces/instruction/science-PDFs/STEM-crawl. Built
    for mid-training anneal = exactly our leg shape.
  * nvidia/Nemotron-Terminal-Corpus (ungated, 366k terminal-agent trajs,
    Terminal-Bench proven) — agentic/tool lineage data.
  * nvidia/Open-SWE-Traces (ungated, EMNLP'26) — SWE agent traces across
    claudecode/opencode/openhands/sweagent/minisweagent; v1.2+ includes
    qwen38_27b + qwen38_flash_next self-family rollouts.
  * nvidia/Nemotron-Pretraining-Specialized-v1 (ungated, 332 shards,
    domain-specialized CPT) ; nvidia/Nemotron-CC-Code-v1 = GATED(manual)
    100M-1B code rows — needs license acceptance before use.
  * jordangong/the-stack-v2-smollm3 (ungated, 200k dl) — permissive
    stack-v2 subset SmolLM3 trained on; cleaner than raw the-stack-v2.
- MIX IDEA for next leg: keep dolma3.5_pool backbone, swap weak configs
  toward dolma3_mix stack_edu (code — HumanEval gap) + olmocr science +
  finemath-3plus; add small Nemotron-Terminal/Open-SWE slices for agentic.
- MODAL PLATFORM (verified 2026-11-09):
  * function timeout 1s–24h (we use 4-8h); timeout is per-attempt,
    retries restart it. startup_timeout separate knob.
  * spawn/async inputs durable: survive client disconnect, results kept
    7d, ~1M queue cap, 1500/s rate — detached pattern is correct.
  * PREEMPTION IS REAL on cpu functions: weight_map container was killed
    mid-run and auto-restarted same input (no checkpoint -> redo).
    GPU functions (H200) ran 45min-1.5h w/o preemption so far, but code
    for long jobs should commit partial state (we do via art.commit()).
  * default cpu request 0.125 cores; request more explicitly. soft cpu
    limit = request+16 cores.
  * causal_conv1d NOT in image -> linear_attn falls back to slow torch
    path; adding causal_conv1d would speed every 27B forward/generate.
- **host_profile.py** (src/): per-host registry for geometry/taps/
  thresholds. qwen2b marked retired; unknown hosts raise KeyError.
  27B tap slots pending re-derivation (glial gate prerequisite).
- **bfcl_synth parser fix**: verify path now uses emit_eval.parse_call
  (quote/bracket-aware _split_top) — naive comma-split corrupted nested
  JSON/array arg values. Verified on {"x":1,"y":2}, [1,2,3], lambda.
- **bfcl_synth Modal run**: first attempt killed by client disconnect
  (modal run foreground). Redeployed as app bfcl-synth27b + spawned
  detached fc-01M70ZN1R6NH3G9RVY2SRSKT6E — survives disconnects.
  Receipt -> /artifacts/bfcl_synth_int4.json.
- **hnet_train27b_l2.py** (modal/): leg-2 driver DEPLOYED (not run).
  Resumes hnet27b_latest_* ckpts, NEXT_MIX data via corpus_stream_next,
  l2 ckpt/receipt names. Knobs: N_TRAIN=4000, LR=5e-5, RESUME,
  RESUME_PREFIX, DATA_MIX. Fire only after synth+eval gate.
- **HF checkpoint mirror**: leg-1 trained weights pushed volume->HF via
  modal/hf_checkpoint_push.py -> KidIkaros/hnet27b-leg1 (12.5GB: 6 ckpts
  best+latest x glue/enc/dec + receipts). Survives volume loss; leg-2
  resume can pull from HF instead of the volume.
- Compression artifacts already on HF: KidIkaros/qwen27b-int4-toolace
  (packed safetensors + INT4_MANIFEST) and qwen27b-gptq4-toolace
  (safetensors + QUANT_MANIFEST). No improved int4 exists yet — the
  syntax-regression fix (constrained decoding / int8 / recalibration)
  is still a pending decision.
- **bfcl_synth int4 run KILLED 2026-11-09 ~18:10 EST**: stalled ~70min on
  parallel_multiple items 21-30 (last print 20/40 @5350s). Deployed code
  predated checkpointing fix -> all in-memory rows lost; nothing on
  volume to salvage. Modal workspace then PAUSED (budget). RESUME
  TOMORROW: local kernel/bfcl_synth.py + modal/bfcl_synth27b.py now
  have incremental receipts (CKPT_EVERY=10, on_ckpt=art.commit) +
  parse_call fix — redeploy and rerun; consider N_PER_CAT=20 scope or
  OFFSET support to bound runtime. Leg-2 driver (hnet_train27b_l2)
  remains deployed/idle.
- **HF repos revamped to proper model cards** (all private):
  qwen27b-int4-toolace + int4_loader.py (self-contained dequant loader),
  qwen27b-gptq4-toolace (corrected the stale "behaviorally lossless"
  claim — now documents the HumanEval syntax regression honestly),
  hnet27b-leg1 (first README: ckpt contents, recipe, bpb results,
  resume instructions, limitations).
- **bfcl_synth hardened for resume** (2026-11-10): env knobs
  CATS/N_PER_CAT/OFFSET/MAX_CALLS/MAX_ARGS/VALUE_TOPK/CKPT_EVERY on both
  kernel and Modal entrypoint; per-item item_s + SLOW(>120s) logging;
  incremental receipts every CKPT_EVERY items w/ volume commit.
  DEPLOY BLOCKED: Modal workspace ac-xQk5rq816TPkgP8tS0M1GF PAUSED
  (billing: $43.49 metered, $30 credits, $12.47 billed — likely a
  spending-limit pause; needs owner action in Modal dashboard).
  RESUME COMMAND once unpaused:
    N_PER_CAT=20 VALUE_TOPK=64 modal run --detach modal/bfcl_synth27b.py
  (N=20 x 4 cats = 80 items; items 21-30 of parallel_multiple — the
  stall range — are excluded by the slice; OFFSET=20 can pick them up
  in a second scoped leg if wanted)

## 2026-11-12 — TRUNK/SUBSTRATE DECISION (ADR-0001, proposed)
- Trunk = TwIL-8B: the one-mind is built on it now. Route A proven (+3.35 bpb,
  leg-3 Arm A); HF-transformers native, native S1/S2 operating-point toggle,
  byte-compatible 207/256. Spark is GGUF-only -> disqualified as substrate.
- Substrate = the byte H-Net line (hnet-kaggle), endgame: owned all-256-byte
  table, S1/S2 anatomy native, learned boundaries, 1.42-1.49 bpb measured.
  No from-scratch training on the one-mind critical path.
- Path A's adapter result stands as the proven byte-input mechanism.
- Open problem on both: outcome channel (tool success/failure, user correction) —
  no run has seen one yet.
