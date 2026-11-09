
## 2026-11-09 — hnet-int4 Modal run (hrun9 successor)
- app: ap-Wahf0QFk9CDWmirflZljdH (modal/hnet_int4.py), gpu=H200, timeout 4h
- job: packed-int4 GPTQ rerun — 400 linears incl linear_attn, ~94% params
- volumes: hnet-artifacts (receipts+shards), hnet-hf-cache (54GB model)
- secret: hf-token (HF_TOKEN) -> auto-upload KidIkaros/qwen27b-int4-toolace
- context: hrun9 (Colab A100) died post-verify; 17.7GB artifact lost;
  Colab GPU quota exhausted -> moved to Modal per AGENTS.md rules
- prior result being reproduced: 27/27 declared pinned exam, 17.7GB packed
