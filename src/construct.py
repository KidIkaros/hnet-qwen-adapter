"""Closed-form patch maps — the 'no-SGD' staircase between the constructed sum and the trained
projection.  Every map here is a deterministic one-shot solve or a per-vector rule: no gradients,
no training loop.  Each returns an apply(base) -> patch callable so the eval path is identical.
"""
import numpy as np
import torch


def identity():
    return lambda base: base


def norm_match(emb, sample=None):
    """Scale each patch vector to the median token-row norm — direction unchanged, tests whether
    the correction is mostly norm matching (Qwen's first op is pre-norm, so norm shouldn't matter —
    a negative control as much as a construction)."""
    med = float(emb.norm(dim=1).median())

    def f(base):
        return base * (med / base.norm(dim=1).clamp_min(1e-8))[:, None]
    return f


def cov_recolor(patch_sums, emb, eps=1e-2):
    """W = Σ_tok^{1/2} Σ_patch^{-1/2}: whiten the patch-sum distribution, recolor to the token-row
    covariance.  One eigendecomposition each; deterministic."""
    Cp = np.cov(patch_sums.double().cpu().numpy().T) + eps * np.eye(emb.shape[1])
    Ct = np.cov(emb.double().cpu().numpy().T) + eps * np.eye(emb.shape[1])

    def half(C, inv):
        w, V = np.linalg.eigh(C)
        s = np.sqrt(np.clip(w, 1e-12, None))
        return V @ np.diag(s ** (-1.0 if inv else 1.0)) @ V.T

    W = torch.tensor(half(Cp, True) @ half(Ct, False), dtype=patch_sums.dtype,
                     device=patch_sums.device)
    return lambda base: base @ W


def ridge_to_bpe(patch_sums, targets, lam=1e-2):
    """Paired Procrustes-ridge: W = (X^T X + lam I)^{-1} X^T Y where Y = the mean embedding of the
    BPE tokens covering the same bytes.  Closed-form least squares; the OMP/tokenizer-transfer
    family reduced to a linear solve."""
    X, Y = patch_sums.double(), targets.double()
    XtX = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    W = torch.linalg.solve(XtX, X.T @ Y).to(patch_sums.dtype)
    return lambda base: base @ W
