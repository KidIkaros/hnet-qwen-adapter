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

    # ---- stage helpers -----------------------------------------------------
    def run_stack(self, layers, hs, causal=True):
        """Run a stack of decoder layers on (1, T, D) hidden states.
        DeltaNet ignores rotary/mask specifics; attention gets a causal mask."""
        T = hs.shape[1]
        pos = torch.arange(T, device=hs.device).unsqueeze(0)
        pos_emb = self.rotary(hs, pos) if self.rotary is not None else None
        mask = torch.full((1, 1, T, T), torch.finfo(hs.dtype).min,
                          device=hs.device, dtype=hs.dtype)
        if causal:
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = mask * 0
        for layer in layers:
            hs = layer(hs, position_embeddings=pos_emb, attention_mask=mask,
                       position_ids=pos)
        return hs

    def chunks(self, hs_bytes, boundaries):
        """Group byte positions into chunk spans given a 0/1 boundary vector."""
        bounds = boundaries.nonzero().flatten().tolist() + [len(boundaries)]
        return [(a, b) for a, b in zip(bounds, bounds[1:])]

    # ---- the H-Net forward --------------------------------------------------
    def forward(self, byte_ids, boundaries=None):
        """byte_ids (T,) -> byte logits (T, 256)."""
        T = len(byte_ids)
        tok_ids = torch.tensor([self.byte_map.get(int(b), 0) for b in byte_ids],
                               device=byte_ids.device)
        hs = self.emb(tok_ids)[None]                          # (1, T, D)
        hs_enc = hs
        for layer in self.enc:
            hs_enc = layer(hs_enc, position_embeddings=None,
                           attention_mask=None, position_ids=None)
        enc = hs_enc[0]                                       # (T, D)
        if boundaries is None:
            boundaries = self.boundary.route(enc, hard=True)  # (T,)
        spans = self.chunks(hs, boundaries)
        # pool: encoder hidden state at each chunk's LAST byte
        patch = torch.stack([enc[b - 1] for _, b in spans])[None]  # (1, C, D)
        glob = self.run_stack(self.main, patch)[0]            # (C, D)
        ctx = self.dechunk(glob)                              # (C, D)
        # decoder: byte emb + dechunked ctx of the PREVIOUS chunk
        x = hs.clone()
        for ci, (a, b) in enumerate(spans):
            if ci > 0:
                x[0, a:b] += ctx[ci - 1]
        hs_dec = x
        for layer in self.dec:
            hs_dec = layer(hs_dec, position_embeddings=None,
                           attention_mask=None, position_ids=None)
        out = self.norm(hs_dec)
        logits = self.lm_head(out)                            # (1, T, V)
        return logits, boundaries, spans


def byte_bpb(logits, byte_ids, head_first_rows):
    """Marginalize vocab logits to byte probabilities, then bpb."""
    lp = torch.log_softmax(logits[0].float(), -1)
    first = head_first_rows                                   # (256, V_byte1)
    probs = lp[:, first].logsumexp(-1)                        # (T, 256) byte-1
    nll = -probs[:-1, byte_ids[1:].long()]
    return float(nll.sum() / torch.log(torch.tensor(2.0)) / (len(byte_ids) - 1))
