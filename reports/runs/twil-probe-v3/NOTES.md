# twil-probe-v3 — COMPLETE (~15min GPU), Leg B finally produced numbers

## Leg B — byte-adapter viability (THE existential measurement)
- first_byte_bpb (period-6 chunks, first-byte-of-next-chunk):
  markdown 6.737 | prose 6.560 | code OOM (period-6 T=2250, memory creep
  across loop — 13.9GB allocated on cuda:0; 2/3 docs + effective succeeded)
- effective_bpb (hybrid: chunk preds + period-1 byte fallback):
  markdown 2.406 on 4KB
- native_token_bpb: 0.021-0.035 — DEGENERATE: docs are x30 verbatim
  repeats, model memorizes the loop; absolute values meaningless,
  only the mechanism comparison is valid
- input/target coverage 1.0 both

### Interpretation
- Period-6 chunk-level prediction is near-uniform (~6.6-6.7 vs 8.0 random):
  the untrained PatchEmbedder (sum of embedding rows) does NOT transfer LM
  ability to chunk level — expected for v0.
- The period-1 byte path (each byte = own embedding row) dominates
  effective_bpb ~2.4 — closest to native embedding distribution.
  ROUTE A IMPLICATION: start compression at period-1/2 with a learned
  local encoder; period-6 chunking is the upgrade, not the start.
- Caveat: synthetic repeated docs flatter every number; real-corpus
  bpb will be worse — mechanism proven, magnitude TBD on real text.

## Leg C — S1/S2 confirmed end-to-end
- S1: 12/12 format_valid + name_acc, 5.8s/gen
- S2: 11/12 (i=2 <need-more-info> on underspecified request — correct
  behavior), 18.8s/gen, S2_think_rate=1.0
- S2 think traces are REAL deliberation (214-2149 chars, median ~500):
  "Okay, the user is asking... let me check the tools available..."
- enable_thinking is a working S1/S2 escalation dial — glial route =
  prompt selection, exactly as designed.

## Leg A — PASS (unchanged)
inputs_embeds + logits_to_keep OK; byte coverage 207/256.

## Status: probe COMPLETE — Route A viable, needs leg-1 training
Byte pathway functions end-to-end; chunk-level needs the learned local
encoder (v0 PatchEmbedder is the bottleneck, not the trunk).
