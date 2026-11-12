
## 2026-11-10 — twil-leg1 PUSHED, RUNNING (2xT4 GPU, ~1.5-2h)
- kernel: mauricew/twil-leg1 v1 — owner-approved push
- legs: load frozen TwIL-8B + Leg1Shim (period-2, E[256,2,D] warm-start)
  -> v0 baseline on real eval docs -> ~250 train steps (byte-restricted
  CE, grad checkpointing, OOM/error-streak guards, ckpt every 50)
  -> learned eval + per-domain delta + effective_bpb sanity (2 docs,
  both modes) -> artifacts.zip
- concurrent: startlux-gate-cal still RUNNING on CPU lane (no conflict)

## 2026-11-10 — startlux-gate-cal COMPLETE (~2.5h CPU) + GATE CALIBRATED
- 160 fresh StartLux rows (pos 200-359): acc_all 70.6% (113/160),
  acc_answerable 94.2% (113/120), acc_unanswerable 0.0 (0/40) —
  StartLux is near-perfect in-domain; errors live in unanswerable +
  low-margin tail
- decide() meta: empty (str(None)) — no extra signal there
- __none__ probe: NEGATIVE — AUC 0.582 ~ chance, chose_none 0/40;
  the head ignores a synthetic none-candidate -> unbound-gold
  detection needs a different mechanism (semantic match / S2)
- GATE CALIBRATED (three-way: FIT=new[0:100]a, CAL=new[100:160]a,
  EVAL=v2-60 held-out batch):
  * margin gap found: pooled 169 answerable rows — ALL 15 wrong have
    margin<0.5 (max 0.404); 140/154 right above
  * OUT-OF-SAMPLE: 0/7 wrong above 0.5 on the new batch (never set it)
  * held-out EVAL (answerable-49): auto 28@100%, review 12@91.7%,
    escalate 9@22.2%
  * calibrated config: margin_auto=0.50, margin_escalate=0.30 ->
    src/glial.py defaults updated, 11 tests pass
  * conformal collapses to {top} at 94% acc (coverage==accuracy) —
    recorded; abstain never fires (conf always high)
- evidence: reports/runs/startlux-gate-cal-v1/ (rows + calibration);
  kernel deleted post-pull (hygiene)

## 2026-11-10 — twil-leg1 v1 ERROR (disk-full), v2 PUSHED RUNNING
- v1 died at ~132s in cell 3: snapshot_download(local_dir=/kaggle/working/
  TwIL-8B) — 16.4GB model into 19.5GB working quota + xet reconstruction
  -> "No space left on device (os error 28)". MY deviation from the v3
  pattern: v3 used from_pretrained(REPO_ID) -> HF cache on ROOT disk
- v2 fix: revert to v3-proven from_pretrained(REPO_ID) pattern (no
  local_dir), df -h diagnostic printed before load
- evidence: reports/runs/twil-leg1-v1/ (log + generator_v1)
- v2 pushed ~18:30, RUNNING — ~1.5-2h expected

## 2026-11-10 — twil-leg1 kernel BUILT + AUDITED, staged (awaiting push word)
- /tmp/twil-leg1-kernel/ — 8 code cells AST-clean, 52KB
- review pass caught BEFORE push:
  * eval readout was restricted-softmax over byte logits, but the repo
    canonical contract (v0 baseline, ChunkLM) is ByteHead.byte_probs
    full-vocab marginalization -> delta would have been confounded.
    Fixed: boundary_nll/first_byte_bpb use _lm_rows+byte_probs (v3
    shim-consistent); restricted CE kept for TRAIN objective only
  * eval docs could overlap train (same shuffle seed re-taken) ->
    single contiguous pass per source: first q*cap bytes = train, next
    4*cap = eval — disjoint BY CONSTRUCTION
  * byte_nll referenced but never defined -> implemented (v3's
    block-streamed version, embedder-parameterized)
  * dead SKIP_HEAD param, salted hash() seed -> removed/zlib.crc32
  * empty-losses IndexError guard, 5-consecutive-error abort
- dry-run validated: warm-start invariant learned==v0 EXACTLY at init
  (max diff 0.0), unmapped bytes get table entries, grad flows to
  E[256,2,D] via index_add_
- bundled adapter.py verified carrying qwen_byte_bpb fixes

## 2026-11-10 — spec written: docs/SPEC_TWIL_LEG1.md (awaiting push word)
- leg-1 = learned PatchEmbedder (E[256,period,D] table, warm-started from
  v0 sum construction) on FROZEN TwIL-8B, period-2, 2048-byte docs,
  ~200-300 steps, gradient checkpointing; 2xT4 ~1.5-2h GPU
- corpus = deployment-matched agentic mix (owner-directed): tool-use
  0.40 (nemotron_terminal/open_swe_traces/ToolACE convs), code 0.25
  (stack_edu), reasoning 0.20 (finemath/proofpile), prose 0.15 — all
  via existing DOLMA3_SOURCES + one custom ToolACE loader
- eval = real held-out non-repeated per-domain docs (v3 x30-repeat
  caveat retired); pass = effective_bpb delta >0.5 vs honest v0 baseline

## 2026-11-10 — startlux-gate-cal PUSHED, RUNNING (CPU lane, ~3h)
- kernel: mauricew/startlux-gate-cal v1 — collects 160 fresh StartLux
  rows (ToolACE stream positions 200-359, disjoint from v2 eval 140-199)
  + __none_of_these__ unbound probe on every-4th subset (~40 items) +
  decide() meta capture (str-safe). Feeds three-way glial-gate
  fit/calibrate/evaluate locally via src/glial.py
- audit fixes pre-push: dead import removed from none-cell, misleading
  `pred` store removed, `m` global guarded via globals().get + status
  check, meta serialized as str[:500] (dict contents could be
  non-JSON -> would have crashed the dump after all rows collected)
- parse code identical to s1-headtohead v2 -> stream positions align
  with archived items.json

## 2026-11-10 — glial gate v0 built + simulated on StartLux probs (local, free)
- src/glial.py: composite gate over S1 {candidate: prob} dists —
  laya/vega contract shape (route/abstain/prediction_set/route_reason/
  forced) sourced entirely from engine probs. Pure function, torch-free.
- tests/test_glial.py: 11 tests, all pass (pytest tests/test_glial.py)
- Gate-sim on the 60 StartLux rows (reports/runs/s1-headtohead-v2/
  glial_gate_sim.json):
    auto     31 items @ 90.3% acc  (margin>=0.5 band)
    review   19 items @ 57.9%     (incl all 14 forced single-candidate)
    escalate 10 items @ 20.0%     (margin<0.2 — S2 territory)
- KEY FINDING: conf=1.0/margin=1.0 is degenerate on n_candidates==1 —
  14 such items were 50% acc (gold undeclared = forced wrong pick).
  Raw-confidence gating would auto-emit them; the forced-check routes
  them to review instead. Composite contract worth +12.5pp on auto band
  vs a naive conf>=0.9 gate (77.8% -> 90.3%)
- CAVEAT: thresholds PROVISIONAL (60-item sweep, uncalibrated) — needs
  disjoint fit/calibrate/evaluate splits on a larger StartLux row set
  before coverage claims. Candidate next-leg: collect ~300 StartLux rows
  (GPU marginal leg ~minutes, or ~4h CPU)
- design note for v1: add synthetic "none_of_these" candidate to criteria
  — only way to actually DETECT unbound gold (probs can't express it)

## 2026-11-10 — RECOVERED: StartLux baseline result (was in git history)
- commit 9557c2b (2026-11-07, weekend): "startlux-0.8B S1 baseline:
  43/60 (27/40 on shared exam vs BGE 25/40)" — score lived in the commit
  message; results JSON never committed (Colab-side clone, ephemeral)
- identical substrate to laya eval: SKIP=2500, same broken CALL_RE/TOOL_RE
  parse, same qualifying predicate -> 60-item sets match laya's eval
- ZERO-SHOT head-to-head on identical 60 items (all same broken parse):
    StartLux-0.8B   43/60 = 71.7%  |  27/40 = 67.5% shared slice
    laya encoder    40/60 = 66.7%  |  25/40 = 62.5%
    laya vega       39/60 = 65.0%  |  24/40 = 60.0%
    BGE             —             |  25/40 = 62.5%
    byte-scorer     —             |  21/40 = 52.5%
- StartLux already leads zero-shot by +3 items/60, +2/40 — margins are
  suggestive not decisive (no per-item data -> no sign test); absolute
  numbers depressed equally by the parse bug for all engines
- open question unchanged: FITTED laya (v2 running) is the real test;
  v2 also re-measures StartLux on clean parse for paired stats

## 2026-11-10 — s1-headtohead v1 COMPLETE, v2 RUNNING (CPU lane)
- kernel: mauricew/s1-headtohead v1, CPU; evidence reports/runs/s1-headtohead-v1/
- v1 parse bug (inherited from startlux_baseline.py): TOOL_RE `([\w]+)`
  missed SPACED tool names ("Get WTA Tennis Rankings") and CALL_RE kept
  only last word -> 44/140 FIT + 28/60 EVAL items had gold-not-in-tools
  (auto-wrong); fit() ValueError killed whole vega leg
- v1 StartLux leg: requirements.txt causal-conv1d CUDA build failed on
  CPU image -> leg recorded unavailable (flag-gate worked as designed)
- v1 encoder zero-shot on EVAL slice (items 140-200, NOT the v1 60-item
  slice): 24/60 = 40% — different items, not comparable to 25/40 earlier
- v2 fixes: JSONDecoder.raw_decode tool parse (200/200 systems yield
  tools, 1-9 tools/item), bracket-segment gold parse w/ normalized-name
  matching, gold-tool retention in criteria, answerable flag +
  answerable-only metrics, fit_rows filtered to answerable, StartLux
  reqs filtered (causal-conv1d/flash-linear-attention/torch/torchvision
  blocked; transformers>=5.8 etc kept), torch>=2.8 ensured pre-import
  via fresh-subprocess version check + CPU wheel fallback
- v2 pushed 2026-11-10, COMPLETE ~85min — ALL LEGS PRODUCED DATA
- v2 RESULTS (60 EVAL items, 49 answerable, clean JSON-parse substrate):
    StartLux-0.8B    41/60 = 68.3% all | 41/49 = 83.7% answerable
    laya vega        31/60 = 51.7% all | 31/49 = 63.3% answerable
    laya encoder     29/60 = 48.3% all | 29/49 = 59.2% answerable
  paired sign tests: StartLux vs vega z=2.25 (p~0.02),
  StartLux vs encoder z=2.59 (p~0.01) — DECISIVE
- CRITICAL: vega fit_report serving=[] — laya's own honesty gate REFUSED
  to serve fitted heads (didn't beat zero-shot on CV rows). Fitted-head
  path produces nothing on per-item-varying tool pools; vega's 31/60 is
  its zero-shot engine number. encoder-vs-vega z=0.29 n.s. — the 0.8B
  reader adds nothing over the tiny encoder here
- glial signals measured: vega route auto 64.5% vs review 37.9% acc
  (real gate separation, even unserved); abstain 23/60, unbound 24/60;
  conformal coverage 65% (still under ~90% nominal, mean set 1.4);
  StartLux conf right=0.905 vs wrong=0.735 — usable margin signal
- StartLux CPU latency mean 47s/item (plain-torch path; GPU would be ~1-3s)
- VERDICT: StartLux confirmed as S1 — decisively, with significance.
  Fitted-laya does NOT close the gap. Adopt the ROUTE/conformal CONTRACT
  shape for glial; source the gate signal from StartLux margins or a
  fitted gate on its probs, not vega
- evidence: reports/runs/s1-headtohead-v2/ (per-item rows + calibration/)
- kernel deleted post-pull (one-off, question answered)

## 2026-11-10 — twil-probe-v1 Kaggle run FAILED at Leg B (~4min GPU)
- kernel: mauricew/twil-probe-v1 v1, 2xT4; log+postmortem in
  reports/runs/twil-probe-v1/
- passed: model load 168s, inputs_embeds+logits_to_keep OK, byte coverage
  207/256 on Kaggle; died on fp32-hs -> bf16-lm_head dtype in _lm_rows
- review found+fixed 4 more latent bugs (qwen_byte_bpb device+memory,
  byte_nll tgt device, (T,T) mask -> sdpa is_causal) + clean-code pass
- v2 pushed 2026-11-10 (owner-approved), COMPLETE ~55min
- v2 results: Leg A PASS, Leg C PASS (S1 12/12 tool-call valid+name,
  S2 11/12 w/ smart <need-more-info>), Leg B FAIL (no_grad missing in
  _lm_rows + period-1 T=8192 math-backend OOM) — evidence in
  reports/runs/twil-probe-v2/
- v3 pushed 2026-11-10 (owner-approved after clean-code/arch/functional
  review pass), COMPLETE ~15min — Leg B numbers finally landed:
  first_byte_bpb 6.74/6.56 (period-6 near-uniform, untrained embedder),
  effective_bpb 2.41 (period-1 path dominates), native bpb degenerate
  (x30 repeat docs); Leg C S1 12/12, S2 11/12 w/ real think traces,
  think_rate=1.0 — evidence in reports/runs/twil-probe-v3/
- VERDICT: Route A viable — byte pathway works end-to-end; period-1/2
  + learned local encoder is the start, period-6 chunking is the upgrade
- new spec: docs/SPEC_S1_HEADTOHEAD.md (StartLux vs fitted-laya, CPU lane)

## 2026-11-09 — hnet-int4 Modal run (hrun9 successor)
- app: ap-Wahf0QFk9CDWmirflZljdH (modal/hnet_int4.py), gpu=H200, timeout 4h
- job: packed-int4 GPTQ rerun — 400 linears incl linear_attn, ~94% params
- volumes: hnet-artifacts (receipts+shards), hnet-hf-cache (54GB model)
- secret: hf-token (HF_TOKEN) -> auto-upload KidIkaros/qwen27b-int4-toolace
- context: hrun9 (Colab A100) died post-verify; 17.7GB artifact lost;
  Colab GPU quota exhausted -> moved to Modal per AGENTS.md rules
- prior result being reproduced: 27/27 declared pinned exam, 17.7GB packed

## 2026-11-09 — hnet-int4 Modal run COMPLETED
- app: ap-Sy4RViK0CaRBV9ch7uHKkc, call: fc-01M70HV9Z9FDDG6XCDN6D27162
- gpu=H200, kernel/kernel-side save_int4.py, detach+spawn per AGENTS.md
- RESULT: quantize 400 linears in 218s (16 blocks, GPU Hessian accum),
  pack 1251 tensors -> 17.68GB packed int4, pinned exam 27/27 declared
  (none=3, n=40) — identical to Colab-verified numbers, reproduced on H200
- auto-upload: KidIkaros/qwen27b-int4-toolace PRIVATE, verified via API
  (14 files: 5 int4 shards + INT4_MANIFEST + PACKED_RECEIPT + README)
- receipt: results/int4_modal_1794241315.json
- runtime ~15min total vs ~90min Colab estimate (GPU-side X^T X was the win)

## 2026-11-09 — Modal preservation checkpoint
- ALL apps confirmed stopped (0 tasks, 0 running) — zero GPU spend
- volumes KEPT per user decision: hnet-artifacts (~18GB, verified
  duplicate of HF repo), hnet-hf-cache (~54GB base-model cache for
  fast restart ~10min saved on next run)
- restart path: modal deploy modal/hnet_int4.py + spawn, or new driver
  reusing volumes; HF repo KidIkaros/qwen27b-int4-toolace is canonical

## 2026-11-09 — hnet-eval + hnet-refit Modal runs (parallel H200s)
- eval: app ap-JoYMiTRcizOUVkJiVWYryM, call fc-01M70M4HFC177T00JND1J6T941
  GENERAL-QUALITY gate: MMLU-5shot (57subj x20) + HumanEval-164 pass@1 +
  wikitext-2 ppl + toolace40, run on BOTH int4-dequant and fresh bf16 ->
  quant-vs-base delta on general capability (not just tools)
- refit: app ap-lSE4RsVBosIEI7NQzRO1lV, call fc-01M70M8SS07DT8R9JCEWBC45E8
  H-Net Phase A on int4 trunk: HNetQwen(enc=6,dec=8) + fit_adapters
  (closed-form ridge) + byte-bpb held/enwik8 — validates path + first
  27B-trunk bpb vs 2B baseline enwik8-test 1.4317. No grad training yet.

## 2026-11-09 — hnet-train27b spawned (Phase B gradient training)
- app ap-fBUrmp7IFmtiFZr8dNO4Pu, call fc-01M70N5T985EMJ2KYE88WYF1BC
- recipe = proven 2B hnet_train ported to Modal: enc4+dec4 DeltaNet slices
  + dechunk/boundary/w_sup trainable, ridge adapters frozen, trunk frozen
- N_TRAIN=1000 dolma-mix docs, eval held+enwik8 every 100 steps,
  ckpts -> /artifacts/hnet27b_{best,latest}_{glue,enc,dec}.pt
- Phase A result (hnet-refit, complete): path works on 27B —
  enc6/dec8/main16 forward OK, 32.7k dechunk pairs; untrained bpb
  enwik8 8.81 / held 9.35 = expected pre-train regime (0.8B adapters-only
  was ~7.7; trained 2B = 1.4317). Training is the required next step.
- EVAL partial: int4 MMLU-5shot 932/1140 = 81.75% (strong; bf16+HE pending)

## 2026-11-09 — hnet-train27b crash+fix (zstd)
- first spawn fc-01M70N5T985EMJ2KYE88WYF1BC died at corpus_stream:
  datasets can't read dolma3.5_pool zstd shards w/o zstandard pkg
  (colab had it implicitly). Fit+load succeeded before crash (~6min lost).
- fix: +zstandard to image; respawned fc-01M70P6HE2KTWF7QBS007AXY7G
  app ap-ItfmIakRCc1M7PIHi0j3Rg, same params (enc4 dec4 N=1000)

## 2026-11-09 — hnet-eval + hnet-train27b COMPLETED
- eval (ap-JoYMiTRcizOUVkJiVWYryM, ~1h35m): general-quality gate done,
  both arms. int4 vs bf16: MMLU 81.75/84.12 (-2.4pp), HumanEval@1
  40.85/63.41 (-22.6pp), wikitext-2 ppl 5.871/5.679, toolace40 identical
  27/27. VERDICT: int4 NOT lossless on generative code — selection metrics
  held, pass@1 flipped 37/164. Receipt results/eval_gate_modal_*.json
- train27b (ap-ItfmIakRCc1M7PIHi0j3Rg, ~44min train): Phase B done,
  1000 steps, held 3.41->2.79, enwik8-test 3.64->3.14 (best@1000=latest,
  still descending at budget end). ckpts hnet27b_{best,latest}_*.pt on
  hnet-artifacts (~12.4GB). Receipt results/hnet27b_train_*.json

## 2026-11-09 — parallel CPU lane (Colab session `hpar`, no GPU)
- Capacity: Colab balance 0.00 units; A100/H100 not entitled, T4
  unavailable → CPU runtime only. All work ran remote (nothing local).
- arg_compose.py + arg_recall.py (new, model-free): schema-aware value
  candidate composition. Pinned slice (skip=2500, n=40):
  65 gold args = 46 declared + 19 undeclared-gold; parse 46/46;
  recall 35/46 = 76.1% vs extractive 28/46 = 60.9% (+15.2pp).
  Receipt results/arg_recall_1794252019.json
- factory.py: DOLMA3_SOURCES + NEXT_MIX + corpus_stream_next; all 8
  sources stream (dolma3_mix globs verified `data/<src>[-<subset>]/
  shard_*.jsonl.zst`; Nemotron-Terminal skill_based_mixed; Open-SWE
  v2.0/claudecode). _text_of now joins conversation lists (agentic rows
  36B → 48–80KB).
- Pending: composer GPU eval on the 27B scorer (Modal, after probes);
  full next-leg stream.

## 2026-11-10 — twil-leg1 v3 ERROR (CastError), v4 PUSHED RUNNING
- v2: tuple typo `eval_docs = {}` → dict+=list crash. Fixed, pushed v3.
- v3: `CastError: Couldn't cast` in corpus cell — first load of DOLMA3_SOURCES;
  one source's parquet schema fails datasets-library streaming cast.
- v4 fix: per-source try/except isolation in the mix loop — a bad source is
  recorded in the manifest and skipped, mix degrades gracefully (ToolACE +
  remaining DOLMA3 sources still give all four buckets).
- kernel: mauricew/twil-leg1 v4 — RUNNING (2xT4, ~1.5-2h from push)
- spec: docs/SPEC_TWIL_LEG1.md (unchanged — resilience fix, not design change)

## 2026-11-10 — twil-s1s2-wiring v1 PUSHED RUNNING (2xT4, owner-approved)
- spec: docs/SPEC_TWIL_S1S2_WIRING.md — closes the one-mind loop for the
  first time: StartLux decide -> glial_gate(calibrated) -> TwIL-S2
  (enable_thinking=True) on review+escalate -> verify -> act
- data: ToolACE qualifying items 360-459 (fresh; disjoint from v2 eval
  140-199, gate-cal 200-359, leg1 corpus 5000+); full tool parameters
  kept for the Granite chat-template tools arg
- co-load: StartLux device="cuda:0" (~1.6GB) + TwIL device_map auto
  (max_memory 12/14GiB); SL_CPU fallback flag caps at 40 items
- metrics: S1-only acc, per-route acc, S2 acc/recovery on non-auto,
  e2e policies P1/P2, accuracy-per-cost, wiring_pass verdict
  (recovery>=0.5 AND P1 > S1-only+0.02 on answerable)
- kernel: mauricew/twil-s1s2-wiring v1 — RUNNING parallel to twil-leg1 v4

## 2026-11-10 — twil-leg1 v4 COMPLETE: FLAT (+0.105 bpb) — smoke inconclusive
- honest v0 baseline on real eval docs: overall 6.9802 bpb
  (per-domain 5.92 open_swe .. 8.00 stack_edu; toolace 7.13 worst agentic)
- learned embedder after ~250 steps: 6.8751 — delta +0.105, far under
  the >0.5 pass bar. Mixed per-domain (common_crawl -0.29, dolma_wiki
  -0.31 improved; open_swe/proofpile ~flat/worse)
- finemath_3plus absent from summaries -> that source failed streaming
  (per-source isolation worked; mix degraded gracefully)
- verdict: leg-1 smoke does NOT prove Route A; cannot distinguish
  "undertrained" from "trunk lacks byte signal" — needs leg-2 (more
  steps) or byte-signal audit. adapter_leg1.pt archived.
- evidence: reports/runs/twil-leg1-v4/

## 2026-11-10 — twil-s1s2-wiring v1 COMPLETE: wiring_pass=FALSE — S2 hurts
- SL_CPU=True (StartLux cuda:0 co-load failed -> CPU fallback, 40 items,
  26 answerable; S1 still 96.2% answerable acc)
- routes: auto=20 review=17 escalate=3 (of 40)
- S2 on non-auto: only 11/20 rows completed (9 errored — likely OOM/
  timeout sharing GPUs); format_valid=0.55, think_rate=0.7,
  mean gen 81-243s/item (4-10x spec estimate)
- s2_recovery_on_s1_wrong = 0.0/12 — ZERO recovery
- s2 acc: review 30%, escalate 0% (n=1) vs S1 review-band 35%
- e2e P1 (auto->S1, rest->S2) = 76.9% vs S1-only 96.2% — escalation
  DESTROYS 19pp of accuracy
- interpretation: purpose-built StartLux-0.8B beats general TwIL-8B-S2
  at tool CHOICE even on uncertain items; naive escalate-to-bigger-model
  is architecturally wrong for this task. S2 needs a different role
  (verify/re-rank S1 candidates, or analyst-feeding-S1) not re-decide.
- caveats: n=11 S2-ok rows, escalate band n=1, CPU-mode S1
- evidence: reports/runs/twil-s1s2-wiring-v1/

## 2026-11-10 — twil-s1s2-wiring-v2 v1 PUSHED RUNNING (2xT4, owner-approved)
- spec: docs/SPEC_TWIL_S1S2_WIRING_V2.md — S2-as-VERIFIER (v1's
  escalate-to-re-decide measured 0/12 recovery; v2 flips the role)
- host: SC117/Spark-X2.6-4B-abliterated-FIT-GGUF BALANCED-3G-Q4_K_M
  (~3GB) via llama-cpp-python CUDA build — owner-selected; coexists
  with StartLux on cuda:0 (fixes v1's SL_CPU fallback)
- verifier contract: VERDICT: CONFIRM | SWITCH <name> | ABSTAIN;
  switch must norm-match a real candidate else ABSTAIN
- data: ToolACE qualifying 460-559 (fresh; v1 used 360-459)
- pass bar: e2e_P1_ans > s1_ans+0.02 AND switch_precision>=0.5 AND
  confirm_rate<=0.90 (a verifier that confirms everything adds nothing)
- metrics: verify_acc, switch_precision, recovery/breakage vs S1,
  abstain_on_unanswerable, e2e P1/P2, mean gen_s
- kernel: mauricew/twil-s1s2-wiring-v2 v1 — RUNNING (~1-1.5h;
  llama.cpp CUDA compile ~15min is the wildcard)

## 2026-11-10 — twil-s1s2-wiring-v2 v2 PUSHED RUNNING
- v1 outcome: COMPLETE but verify_n=0 — llama-cpp-python CUDA wheel
  build FAILED (nvcc/toolkit on Kaggle image) AND StartLux cuda:0
  hard-refuses without causal-conv1d+fla (RuntimeError by design) ->
  SL_CPU again, 40 items, S1 100% ans acc on this slice
- root cause found: StartLux on Kaggle = CPU-only unless we build the
  CUDA-only deps (~20min risky compile) — CPU ~11s/item is acceptable
- v2 fixes: llama.cpp CPU path only (--only-binary wheel first,
  source-build fallback; n_gpu_layers=0, n_threads=8); spark-unavailable
  guard empties nonauto instead of error-per-item; same slice 460-559
  (clean: never in gate-calibration fit)
- kernel: mauricew/twil-s1s2-wiring-v2 v2 — RUNNING

## 2026-11-10 — twil-s1s2-wiring-v2 v2 COMPLETE: wiring_v2_pass=FALSE
- Spark-4B verifier RAN (CPU llama.cpp, verify_n=19/19 non-auto):
  confirm_rate=0.74, switch_n=0 (never once switched!), abstain=0,
  verdict_none=0.26 (think-trace ate the 256-token budget),
  verify_mean_gen_s=36s
- verify_acc_final=0.42 vs s1_acc_on_verified=0.63 — verifier made the
  band WORSE; recovery_on_s1_wrong=0.0, breakage_on_s1_right=0.33
- e2e_P1=0.867 vs s1_acc_ans=1.00 (this slice: S1 perfect on answerable)
- failure anatomy (raw outputs): NONE rows = model deliberates past
  token budget; CONFIRM-on-wrong = think trace considers alternatives
  then defers to the proposal — sycophancy toward the framed question
- CROSS-KERNEL CONCLUSION (3 measurements now): escalate-to-S2-
  redecide (TwIL) 0/12, verify-with-Spark 0 recovery — a second
  same-class model pass over the SAME information does not resolve
  genuinely ambiguous items. The non-auto band is hard because the
  items are hard, not because S1 lacks a second opinion.
- architectural implication: glial gate's real value = selective
  ABSTENTION (auto band 85-100% precise); S2's role must add
  INFORMATION (tool execution, retrieval) or be a much stronger
  model — not another LM re-judging the same prompt.
- evidence: reports/runs/twil-s1s2-wiring-v2/

## 2026-11-10 — leg-1 postmortem DEEPENED: it was a failed run, not a smoke
- leg1_train_log.json: steps_done=16 of ~250 attempted; 24/40 logged
  entries oom:true; loss 8.73->11.64 bits noisy/no descent
- root cause: train_step runs one FULL 2048-byte doc/step -> T~1024
  chunk positions -> fwd+bwd through 8B trunk doesn't fit 14.5GB/GPU
- the +0.105 bpb delta = warm-start + noise; leg-1 result VOID as
  evidence about trunk capability
- spec: docs/SPEC_TWIL_LEG2.md — DOC_CAP_TRAIN 2048->512 (T~256),
  10-step smoke gate, 800 successful steps, learning-curve eval at
  {0,100,300,600,end} on fixed 8-doc subset — the curve is the answer

## 2026-11-10 — twil-leg2 v1 PUSHED RUNNING (2xT4, owner-approved)
- spec: docs/SPEC_TWIL_LEG2.md — FIRST real adapter training (leg-1
  trained only 16/250 steps due to T~1024 OOM; its +0.105 delta was void)
- changes vs leg-1: DOC_CAP_TRAIN 2048->512 (T~256), TRAIN_REGION=2048
  anchor keeps eval docs byte-identical to leg-1 + train bytes identical
  re-chunked (~4x docs), toolace convs chunked to same byte budget,
  800 SUCCESSFUL steps target / 1400 attempts, smoke gate (>=8 of first
  10 attempts + descent), learning-curve bpb at {0,100,300,600,final}
  on fixed 8-doc subset, dynamic legs.train status
- review: two-axis (standards+spec) caught eval-offset contamination
  (fixed) + kernel-name mislabel; second pass clean-code/clean-arch/
  functional-design -> Leg1Shim->AdapterShim, ms/bs->tpos/tbytes
- kernel: mauricew/twil-leg2 v1 — RUNNING (~2h)

## 2026-11-10 — twil-leg2 v1 COMPLETE (smoke false-fail), v2 PUSHED RUNNING
- v1 result: OOM FIXED (10/10 attempts at T~256, zero OOM) but the
  loss-descent smoke gate FALSE-FAILED — per-doc CE dominated by
  document entropy (different doc each step), aborted at 10 steps.
  Eval-anchor fix empirically confirmed: curve step-0 bpb 6.9798 ==
  leg-1 baseline 6.9802 (byte-identical eval docs).
- v2 patch: gate replaced with evidence-based early-bpb gate at
  ok-step 50 (>=0.05 improvement on fixed curve docs) — the actual
  metric, immune to per-doc loss noise. CURVE_AT +50.
- review round 2 (standards+spec axes): stale header comment (fixed),
  b50 positional lookup misdiagnosed eval-failure as objective bug
  (fixed: keyed on step==50 + ndone logged), legs.train said 'done'
  on post-gate partial abort (fixed: 'partial' when steps<800).
- kernel: mauricew/twil-leg2 v2 — RUNNING (~2h)

## 2026-11-10 — twil-leg2 v2 COMPLETE: PASS — ROUTE A IS ALIVE (+0.579 bpb)
- 800/800 successful steps (802 attempts, 2 skipped), smoke gate PASSED
- train CE: 10.81 -> 6.10 bits (real descent — the adapter LEARNS)
- learning curve (fixed 8 docs): 6.980 -> 6.918(50) -> 6.842(100) ->
  6.796(300) -> 6.696(600) -> 6.615(final) — monotonic, STILL DESCENDING
  at the end (~-0.0004 bpb/step) => undertrained, headroom remains
- full eval on real held-out docs: learned 6.401 vs baseline 6.980
  => delta +0.579, crosses the pre-registered >0.5 pass bar
- per-domain: dolma_wiki -1.22, common_crawl -0.83, toolace -0.61,
  proofpile -0.54, nemotron -0.48, stack_edu -0.25, open_swe -0.13
- DECISION: Route A confirmed — the frozen TwIL trunk can be taught
  byte-level input via a learned patch embedder. Leg-1's flat result
  was indeed the 16-step OOM artifact, as suspected.
- gap: effective_bpb sanity check missing from leg2_eval.json (spec
  asked for it on 2 docs) — add back in leg-3.
- evidence: reports/runs/twil-leg2-v2/ (+adapter_leg2.pt)
- next: leg-3 scale-up (curve not saturated): more steps (2-4k), period-1
  comparison, larger corpus; then H-Net++ learned boundaries

## 2026-11-10 — SPEC_TWIL_LEG3 written (scale-up + period comparison)
- two arms in one ~6h session: A period-2 @3000 steps (isolate steps),
  B period-1 @1500 steps (compression comparison, matched token budget)
- corpus 4x (N_TRAIN_TOTAL 1600), fresh split -> new eval anchor, so
  baseline re-measured in-kernel; cross-leg comparison anchors on DELTA
- carries leg-2's proven fixes (512B docs, bpb smoke gate, canonical
  eval); restores effective_bpb sanity check (leg-2 gap)
- pre-registered: Arm A slope at 3000 = scale ceiling; B-vs-A decides
  period-1 vs period-2 for the substrate config

## 2026-11-10 — twil-leg3 v1 PUSHED RUNNING (2xT4, ~6h, owner-approved)
- pre-push GPU audit: all prior kernels COMPLETE/deleted, zero quota
  burning; leg2 artifacts already archived in reports/runs/twil-leg2-v2/
- two arms one session: A period-2 @3000 steps, B period-1 @1500 steps
  (matched trunk-token budget); corpus 4x (1600 docs), fresh split,
  in-kernel v0 re-measure both periods; effective_bpb restored
- review caught a RUN-KILLER: T_START used but never defined (silent
  insert failure) -> would have NameError'd at cell 7 after ~4h with
  zero artifacts. Also fixed: empty-corpus guard in train_arm, final
  curve dedup, Arm B full eval per spec, None-safe eval, spec-faithful
  verdict (beats_leg2_delta_0.579), arm_status skipped_empty
- wall-clock guard at 5.5h (B needs ~2h; session-limit safety)
- kernel: mauricew/twil-leg3 v1 — RUNNING (~6h)

## 2026-11-10 — twil-leg3 v1 ERROR (cell-1 NameError), v2 PUSHED
- v1 died in cell 1 in ~2min: `T_START = time.time()` added but cell 1
  only imported `subprocess, sys` — NameError before any work. ~2min
  quota burned. Lesson: the review checked name USE, not the import.
- v2: cell 1 imports `time`. Verified import + definition in built cell.
- kernel: mauricew/twil-leg3 v2 — RUNNING (~6h)

## 2026-11-10 — twil-leg3 v1 COMPLETE (gate aborts), v3 PUSHED RUNNING
- v1 result (1.29h, artifacts all saved): Arm A p2 gate-failed at 50
  steps (+0.031 real progress < 0.05 bar — false negative on fresh
  eval set); Arm B p1 DIVERGED to NaN in 50 steps (no grad clipping).
- BURIED HEADLINE: period-1 v0 baseline ~3.97 bpb — better than p2
  FULLY TRAINED (6.40). One byte per position = the byte's own
  token-embedding row = near on-distribution. Period-1 is the substrate.
- v3 fixes: clip_grad_norm_([E],1.0) + nonfinite check after opt.step
  (immediate abort + 'E_nonfinite' fatal log); gate relaxed to abort
  only on NaN or worse-than-warm-start (catches non-learning, not
  slow-start). Spec updated.
- kernel: mauricew/twil-leg3 v3 — RUNNING (~6h)

## 2026-11-11 — twil-leg3 v3 COMPLETE: ARM A BREAKTHROUGH (+3.35 bpb)
- Arm A (period-2, 3000 steps): learned 3.618 vs base 6.964 => delta
  +3.35 bpb. Curve: 7.001 -> 6.763(50) -> 6.821(250) -> 6.618(750)
  -> 5.512(1500) -> 3.908(3000) — late phase transition, still falling.
  Beats leg-2's 6.40 by a mile; beats_leg2=True.
- Arm B (period-1): STILL NaN — nonfinite guard aborted before step 50.
  Grad clipping did not prevent it; root cause unknown (candidate: NaN
  loss on some doc -> NaN grads -> clip keeps NaN; or p1 lr too high).
  Needs diagnosis before period-1 can be evaluated as a substrate.
- elapsed 5.29h. evidence: reports/runs/twil-leg3-v3/

## 2026-11-11 — twil-leg4 v1 PUSHED RUNNING (~5.5h): p1 NaN fix + A resume
- p1 diagnosis from leg-3 v3 log: 43 steps, loss flat ~8.8 (no learning),
  E non-finite at step 44. Fixes: lr 3e-4->1e-4, clip 1.0->0.5,
  finite-loss pre-check (poisoned docs skipped BEFORE backward).
- Arm order: B (p1 1500 steps) FIRST, then A resume: leg-3's 3000-step
  p2 adapter (3.618 bpb) uploaded as dataset mauricew/twil-leg3-adapter,
  +3000 steps to 6000 total, gate disabled (known-good), arm-local
  curve {500,1500,3000}, guard 4.5h.
- review: guarded torch.load (mount failure would have zeroed a 5.5h
  run), stale log tags, effective_bpb skip guard. All fixed+verified.
- kernel: mauricew/twil-leg4 v1 — RUNNING

## 2026-11-11 — twil-leg4 v1 COMPLETE: both arms failed gracefully
- Arm A: ckpt_fail FileNotFoundError — dataset mauricew/twil-leg3-adapter
  not mounted (created minutes before push; likely still processing).
  Guard worked: graceful skip, artifacts produced.
- Arm B: STILL NaN before step 50 despite lr 1e-4 + clip 0.5 +
  finite-loss pre-check. Hypothesis: period-1 unmapped bytes (49/256)
  get ZERO embedding rows -> out-of-distribution trunk input -> overflow
  -> NaN. Next fix: init unmapped rows to mean of mapped rows (+ keep
  pre-check). Needs verification before rerun.
- elapsed 0.98h. evidence: reports/runs/twil-leg4-v1/

## 2026-11-11 — twil-leg4 v2 PUSHED after preflight harness (owner-directed)
- p1 NaN root cause FIXED (locally dry-run verified): unmapped bytes
  (49/256) got ZERO embedding rows -> OOD trunk input -> softmax
  overflow -> NaN grads. Fix: unmapped rows = mean of mapped rows.
  Dry run: zero rows 130 -> 0, mapped rows bit-identical.
- mount robustness: glob-resolve ckpt under dataset root + explicit
  readiness assertion; dataset confirmed "ready" before push.
- NEW scripts/kernel_preflight.py — one mechanical check per failure
  class hit so far: syntax (leg2 typo), undefined-name/import
  (leg3 T_START/time), T-budget (leg1 OOM), dataset readiness (leg4
  mount), glob fallback requirement. Must PASS before every push.
- kernel: mauricew/twil-leg4 v2 — RUNNING

## 2026-11-12 — twil-depth-probe v1 PUSHED RUNNING (2xT4, owner-approved)
- spec: docs/SPEC_TWIL_DEPTH_PROBE.md — the one-mind's core untested claim:
  the SAME trunk's deliberate mode, conditioned on the fast pass's
  compressed state, recovers what the fast pass missed
- arms: A = constrained prefix-scoring over declared tools (also captures
  the fast state = final hidden state at the predicted candidate's last
  position); B = think unconditioned; C = think + 4 soft prefix tokens =
  the fast state via inputs_embeds (identity projection v0)
- items: ToolACE qualifying 560-659 (disjoint from all prior slices)
- review: two-axis subagents + clean-code/clean-architecture/
  functional-design passes. RUN-KILLER caught: CALL_RE required </step>
  but TwIL emits the tool_call wrapper — every B/C row would have parsed
  empty. Fixed to the proven wiring-v1 pattern + extracted to
  src/toolcall.py with 6 regression tests (real-format fixtures).
  Spec-axis fixes: gate_caveat, skipped counts, margin quartiles,
  e2e_v1_style (wiring-comparable), c_vs_b_on_s1_wrong paired counts,
  None-when-empty recovery. Arm B prompt-decode slice fixed.
- preflight PASSED (0 issues); parser verified end-to-end against the
  real emission format by executing the built cell
- verdict bands (pre-registered): depth_pass = recovery_C >= 0.5 AND
  breakage_C <= 0.10 AND C > B; think_is_depth = recovery_B >= 0.3;
  both fail -> think-mode is length not depth -> H-Net middle required
- ~2h expected; per-arm wall guards 45/55min; session self-terminates
