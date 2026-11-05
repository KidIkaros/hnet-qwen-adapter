"""H-Net of Qwen — a hierarchical byte model re-sliced from a Qwen3.5 checkpoint.

Qwen3.5's backbone is already the H-Net family: a 3:1 stack of Gated DeltaNet
(SSM, byte-local work) and Gated Attention (global/compressed work) layers.
This module re-slices the checkpoint by ``layer_types``:

    bytes → [encoder: first ENC_LAYERS DeltaNet layers over byte-token embeddings]
          → learned q/k boundary routing (H-Net's dynamic chunking)
          → pool (hidden state at each chunk's last byte)
          → [main module: all full_attention layers on chunk vectors]
          → dechunk (broadcast previous chunk's global state to its bytes)
          → [decoder: last DEC_LAYERS DeltaNet layers over bytes + ctx]
          → byte head (lm_head rows marginalized to 256)

Only the boundary q/k heads and dechunk projection are new parameters —
everything else is the checkpoint's own weights, rerouted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHeads(nn.Module):
    """H-Net dynamic chunking: q/k probes on encoder states; boundary when
    cos(q_t, k_{t-1}) < threshold.  Straight-through estimator for training."""

    def __init__(self, dim, d_probe=64):
        super().__init__()
        self.q = nn.Linear(dim, d_probe, bias=False)
        self.k = nn.Linear(dim, d_probe, bias=False)

    def probs(self, hs):
        """hs (T, D) encoder states -> boundary prob p_t at every position."""
        q, k = self.q(hs), self.k(hs)
        cos = F.cosine_similarity(q[1:], k[:-1], dim=-1)      # (T-1,)
        p = (1.0 - cos) / 2.0                                # ~[0,1], high = boundary
        return F.pad(p, (1, 0), value=1.0)                   # p_0 := 1 (BOS boundary)

    def route(self, hs, tau=0.5, hard=True):
        p = self.probs(hs)
        soft = (p > tau).float()
        if hard:
            return soft                                      # inference: hard route
        return soft.detach() + p - p.detach()                # ST estimator


class HNetQwen(nn.Module):
    """The re-sliced checkpoint.  ``model`` is the HF causal LM; ``tok`` for
    the byte-token map; enc/dec sizes follow H-Net (4 local layers)."""

    def __init__(self, model, tok, byte_map, enc_layers=4, dec_layers=4,
                 d_probe=64):
        super().__init__()
        text = model.model if hasattr(model, "model") else model.language_model
        if not hasattr(text, "layers"):
            text = text.language_model
        self.layers = list(text.layers)
        self.emb = text.embed_tokens
        self.rotary = getattr(text, "rotary_emb", None)
        self.norm = text.norm
        self.lm_head = model.lm_head
        self.byte_map = byte_map                              # {byte: vocab id}
        self.register_buffer("_byte_lut", torch.tensor(
            [byte_map.get(b, 0) for b in range(256)], dtype=torch.long))
        lin = [l for l in self.layers if l.block_type == "linear_attention"]
        full = [l for l in self.layers if l.block_type == "full_attention"]
        assert len(lin) >= enc_layers + dec_layers, \
            f"need {enc_layers + dec_layers} DeltaNet layers, have {len(lin)}"
        self.enc = nn.ModuleList(lin[:enc_layers])
        self.dec = nn.ModuleList(lin[-dec_layers:])
        self.main = nn.ModuleList(full)
        self.boundary = BoundaryHeads(text.config.hidden_size, d_probe)
        self.dechunk = nn.Linear(text.config.hidden_size,
                                 text.config.hidden_size, bias=False)
        self.hidden = text.config.hidden_size
        self.w_in = None                                        # (D, D) dec input adapter
        self.w_out = None                                       # (D, D) dec output adapter
        self.w_main = None                                      # (D, D) main-input adapter
        self.w_enc = None                                       # enc-state residual (H-Net residual_proj)
        dt = self.emb.weight.dtype                              # match checkpoint dtype
        self.boundary.to(dt); self.dechunk.to(dt)

    # ---- stage helpers -----------------------------------------------------
    def causal_mask(self, T, sparse=None):
        """(1,1,T,T) additive mask.  sparse=(w, stride, sinks): sliding window w,
        every stride-th position globally visible, first `sinks` always visible —
        the LSA-style {sink + window + strided} pattern, O(T*(w + T/stride))."""
        neg = torch.finfo(torch.float32).min
        dt = self.emb.weight.dtype
        if sparse is None:
            m = torch.triu(torch.full((T, T), neg,
                                      device=self.emb.weight.device), diagonal=1)
            return m[None, None].to(dt)
        w, stride, sinks = sparse
        idx = torch.arange(T, device=self.emb.weight.device)
        vis = (idx[None, :] <= idx[:, None])                       # causal
        win = (idx[:, None] - idx[None, :]) < w                    # within window
        glob = (idx[None, :] % stride == 0) | (idx[None, :] < sinks)
        allow = vis & (win | glob)
        return torch.where(allow, 0.0, neg)[None, None].to(dt)

    def run_stack(self, layers, hs, causal=True, sparse=None):
        """Run a stack of decoder layers on (1, T, D) hidden states."""
        T = hs.shape[1]
        pos = torch.arange(T, device=hs.device).unsqueeze(0)
        pos_emb = self.rotary(hs, pos) if self.rotary is not None else None
        mask = self.causal_mask(T, sparse) if causal else torch.zeros(
            1, 1, T, T, device=hs.device, dtype=hs.dtype)
        for layer in layers:
            hs = layer(hs, position_embeddings=pos_emb, attention_mask=mask,
                       position_ids=pos)
        return hs

    def decoder_attn_pass(self, x_bytes, ctx, chunk_of, n_ctx_layers=1):
        """BLT-style chunk retrieval: one checkpoint attention layer over the
        interleaved [ctx_1..ctx_C, byte tokens] sequence — bytes attend to
        chunk states STRICTLY before their own chunk (causal, no leakage).
        chunk_of (T,) maps each byte to its chunk index."""
        T = x_bytes.shape[1]; C = ctx.shape[0]
        seq = torch.cat([ctx[None], x_bytes], dim=1)               # (1, C+T, D)
        neg = torch.finfo(torch.float32).min
        N = C + T
        allow = torch.zeros(N, N, dtype=torch.bool, device=x_bytes.device)
        allow[:C, :C] = torch.tril(torch.ones(C, C, dtype=torch.bool,
                                              device=x_bytes.device))
        ctx_pos = torch.arange(C, device=x_bytes.device)
        allow[C:, :C] = ctx_pos[None, :] < chunk_of[:, None]       # past ctx only
        allow[C:, C:] = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                              device=x_bytes.device))
        mask = torch.where(allow, 0.0, neg)[None, None].to(x_bytes.dtype)
        pos = torch.arange(N, device=x_bytes.device).unsqueeze(0)
        pos_emb = self.rotary(seq, pos) if self.rotary is not None else None
        for layer in list(self.main)[:n_ctx_layers]:
            seq = layer(seq, position_embeddings=pos_emb,
                        attention_mask=mask, position_ids=pos)
        return seq[:, C:]                                          # byte part

    def chunks(self, boundaries):
        """Group byte positions into (start, end) spans given a 0/1 boundary
        vector — boundary=1 at p means a new chunk starts at p."""
        starts = boundaries.nonzero().flatten()
        stops = torch.cat([starts[1:], torch.tensor(
            [len(boundaries)], device=starts.device)])
        return list(zip(starts.tolist(), stops.tolist()))

    # ---- the H-Net forward --------------------------------------------------
    def forward(self, byte_ids, boundaries=None, sparse=None, ctx="prev",
                head_first=None, n_ctx=1):
        """byte_ids (T,) -> (logits, boundaries, spans).

        sparse=(w, stride, sinks): LSA-style mask on the main attention stack.
        ctx: "prev" = dechunk broadcast of previous chunk (baseline);
             "retrieve" = ZHA-style — bytes attend to ALL past chunk states
             through one checkpoint attention layer."""
        T = len(byte_ids)
        hs = self.emb(self._byte_lut[byte_ids.long()])[None]  # (1, T, D)
        hs_enc = hs
        for layer in self.enc:
            hs_enc = layer(hs_enc, position_embeddings=None,
                           attention_mask=None, position_ids=None)
        enc = hs_enc[0]                                       # (T, D)
        if boundaries is None:
            boundaries = self.boundary.route(enc, hard=True)  # (T,)
        starts = boundaries.nonzero().flatten()               # chunk start indices
        stops = torch.cat([starts[1:], torch.tensor(
            [T], device=starts.device)])                      # chunk end indices
        chunk_of = boundaries.long().cumsum(0) - 1            # (T,) chunk idx
        # pool: encoder hidden state at each chunk's LAST byte (pure gather)
        patch = enc[stops - 1][None]                          # (1, C, D)
        if self.w_main is not None:
            patch = self.w_main(patch)
        glob = self.run_stack(self.main, patch, sparse=sparse)[0]  # (C, D)
        ctx_v = self.dechunk(glob)                            # (C, D)
        # encoder-state residual — H-Net residual_proj: byte-local detail
        # crosses the chunk bottleneck into the decoder input
        if ctx not in ("prev", "own", "retrieve"):
            raise ValueError(f"unknown ctx mode {ctx!r}")
        x = hs.clone() if self.w_enc is None else hs + self.w_enc(enc[None])
        if ctx == "retrieve":
            # bytes attend past-chunk ctx with their OWN queries — no broadcast
            x = x + self.decoder_attn_pass(x, ctx_v, chunk_of, n_ctx)
        else:
            src = chunk_of if ctx == "own" else (chunk_of - 1)
            valid = src >= 0
            gather = src.clamp(min=0)
            x[0, valid] += ctx_v[gather[valid]]               # vectorized dechunk
        spans = list(zip(starts.tolist(), stops.tolist()))
        hs_dec = x if self.w_in is None else self.w_in(x)
        for layer in self.dec:
            hs_dec = layer(hs_dec, position_embeddings=None,
                           attention_mask=None, position_ids=None)
        out = self.norm(hs_dec)
        if self.w_out is not None:
            out = self.w_out(out)
        if head_first is not None:                            # chunked byte log-probs
            from .learned import byte_logprobs
            bp = torch.cat([byte_logprobs(self.lm_head(
                out[:, i:i + 2048]).float()[0], head_first)
                for i in range(0, T, 2048)])
            return bp[None], boundaries, spans
        logits = self.lm_head(out)                            # (1, T, V)
        return logits, boundaries, spans
