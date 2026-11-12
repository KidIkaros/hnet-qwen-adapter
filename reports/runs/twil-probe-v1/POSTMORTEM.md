# twil-probe-v1 — FAILED at Leg B (dtype boundary), ~4min GPU

## What passed before the crash
- torch 2.11.0+cu130, transformers 5.20.2, 2x Tesla T4 confirmed
- TwIL-8B bf16 loaded across both T4s in 168s (~8.8GB reserved each)
- inputs_embeds + logits_to_keep verified working on GraniteForCausalLM
- byte_token_map coverage on Kaggle: 207/256, zero missing printable

## Root cause
`_hs_last` returns fp32 hidden states; `_lm_rows` fed them into bf16
lm_head -> `RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: float != c10::BFloat16` at first_byte_bpb.

## Additional latent bugs found in review (fixed for v2, unexercised in v1)
1. qwen_byte_bpb: logits land on lm_head's device (cuda:1) while targets
   stayed on cuda:0 -> cross_entropy device mismatch (src/adapter.py fixed:
   tgt -> logits.device + blocked fp32 CE for memory)
2. qwen_byte_bpb materialized full (L,V) fp32 logits + CE temps ~4-5GB
   transient -> now computed in 1024-row blocks
3. byte_nll tgt tensor created on hs.device, gathered against pb on
   lm_head device -> same mismatch class (lazy-bound to pb.device)
4. Hand-rolled (T,T) causal mask -> replaced by attention_mask=None so
   create_causal_mask returns None and sdpa uses is_causal (O(T) memory,
   flash-class path; verified modeling_granite.py:396 + sdpa_attention.py:140)

## Cleanups
- boundary_nll() added to shim (mirrors ChunkLM API); cell 4 no longer
  touches _hs_last/_lm_rows privates; triplicated span->target loop removed
- first_byte_bpb now returns repo contract (bpb, n, (icov, tcov))
- Leg C parses post-</think> text only (think-block false-positive guard)
- Leg A persists legA_interface.json + legA_coverage.json into artifacts.zip
- transformers pinned ==5.20.2; dead stores/unused imports/__import__ removed
