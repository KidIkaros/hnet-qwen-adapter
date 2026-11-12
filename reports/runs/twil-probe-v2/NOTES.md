# twil-probe-v2 — COMPLETE (~55min GPU), mixed results

## Leg A — PASS
- inputs_embeds + logits_to_keep verified on GraniteForCausalLM
- byte coverage 207/256, zero missing printable (confirmed on Kaggle)

## Leg B — FAIL (two bugs, both fixed for v3)
- All 3 docs: `Can't call numpy() on Tensor that requires grad` —
  _lm_rows applied lm_head outside no_grad -> logits carried grad_fn.
  Period-6 backbone forwards themselves ran fine.
- effective_markdown: CUDA OOM at period-1 T=8192 — sdpa math backend on
  sm_75 (no flash/mem-efficient) materialized ~(32,T,T) fp32 scores ~8GB.
  Period-6 docs (T<=2555) were fine. Fix: cap effective doc at 4096B.

## Leg C — PASS (the headline result)
- S1 (enable_thinking=False): 12/12 format_valid, 12/12 name_acc, 5.6s/gen
- S2 (enable_thinking=True): 11/12 — the miss (i=2) emitted <need-more-info>
  asking for the missing date: arguably CORRECT on underspecified input
- S2_think_rate=0.0 was a METRIC BUG: has_think required "<think>" in
  generated text, but the opening tag lives in the prompt — only "</think>"
  can appear. Fixed for v3 + full generation text now saved per row.
- Native <tool_call><function=..><parameter=..> format parses cleanly.

## v3 changes (staged /tmp/twil-kernel/)
1. _lm_rows wrapped in torch.no_grad() — unblocks all Leg B metrics
2. effective doc 8192 -> 4096 bytes — avoids math-backend OOM
3. has_think checks "</think>" only; full gen text saved to results
