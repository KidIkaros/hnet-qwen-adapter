
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
