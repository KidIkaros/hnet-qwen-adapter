"""Faithful 10%-style compression for the Qwen trunk (our circumstance).

Pipeline per the ten-percent-challenge method, reimplemented for our host:
  1. Hadamard rotation  — smooths activation outliers before quantization
  2. Hessian-aware GPTQ — layer-wise, error-feedback column quantization
  3. Verification hook  — the compressed trunk runs our pinned exams
     (prefix scoring + none-gap + emit behavior), not just perplexity

Targets our architecture: Qwen `nn.Linear` modules (q/k/v/o, gate/up/down),
frozen H-Net byte adapters refit after compression.

    from src.compress import hadamard, gptq_quantize, quantize_linear
"""
import torch


def hadamard(n):
    """Normalized Sylvester Hadamard matrix of order n (power of two)."""
    assert n & (n - 1) == 0, "n must be a power of two"
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def _round_to_grid(w, scale, zero, bits):
    qmax = 2 ** bits - 1
    return (w / scale + zero).round().clamp(0, qmax)


def gptq_quantize(W, H, bits=3, blocksize=128, damp=0.01, sym=True):
    """GPTQ on one weight matrix. W [out,in] fp; H [in,in] = X^T X (calib).

    Returns (W_hat fp, scale [out,1], zero [out,1]).
    """
    W = W.clone().float()
    out_f, in_f = W.shape
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.eye(in_f, device=H.device) * H.diag().mean()

    Hinv = torch.linalg.inv(H)
    # Cholesky of Hinv for stable error propagation (standard GPTQ trick)
    try:
        Hinv = torch.linalg.cholesky(Hinv).T
    except Exception:
        pass

    qmax = 2 ** bits - 1
    scale = torch.zeros(out_f, 1, device=W.device)
    zero = torch.zeros(out_f, 1, device=W.device)
    Wq = torch.zeros_like(W)
    for b0 in range(0, in_f, blocksize):
        b1 = min(b0 + blocksize, in_f)
        Wb = W[:, b0:b1]
        if sym:
            mx = Wb.abs().amax(1, keepdim=True).clamp(min=1e-8)
            s = mx / (qmax / 2)
            z = torch.full_like(mx, qmax / 2)
        else:
            mn = Wb.amin(1, keepdim=True)
            mx = Wb.amax(1, keepdim=True)
            s = ((mx - mn) / qmax).clamp(min=1e-8)
            z = -mn / s
        q = (_round_to_grid(Wb, s, z, bits) - z) * s
        err = (Wb - q) / Hinv[b0:b1, b0:b1].diag().clamp(min=1e-8)
        W[:, b1:] -= err @ Hinv[b0:b1, b1:]
        Wq[:, b0:b1] = q
        scale[:, 0] = s[:, 0]
        zero[:, 0] = z[:, 0]
    return Wq, scale, zero


@torch.no_grad()
def quantize_linear(lin, H, bits=3, rotate=True):
    """Rotate (optional) + GPTQ one nn.Linear; writes back dequantized fp."""
    W = lin.weight.data
    Wf, Hf = W.float(), H.float()
    if rotate and W.shape[1] & (W.shape[1] - 1) == 0:
        h = hadamard(W.shape[1]).to(W.device, torch.float32)
        Wf = Wf @ h
        Hf = h.T @ Hf @ h
    Wq, s, z = gptq_quantize(Wf, Hf, bits=bits)
    lin.weight.data.copy_(Wq.to(lin.weight.dtype))
    return {"scale": s.cpu(), "zero": z.cpu(), "bits": bits}


@torch.no_grad()
def calibrate_hessians(model, layer_names, calib_ids, forward_fn):
    """Accumulate H = X^T X for each named linear from calibration inputs."""
    H = {n: None for n in layer_names}
    hooks = []

    def mk(name):
        def f(m, inp):
            x = inp[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).cpu()
            hh = x.T @ x
            H[name] = hh if H[name] is None else H[name] + hh
        return f

    named = dict(model.named_modules())
    for n in layer_names:
        hooks.append(named[n].register_forward_pre_hook(mk(n)))
    for ids in calib_ids:
        forward_fn(ids)
    for h in hooks:
        h.remove()
    return {n: (v / max(len(calib_ids), 1)) if v is not None else None
            for n, v in H.items()}


@torch.no_grad()
def wanda_prune(lin, H, sparsity):
    """Wanda importance pruning: |W_ij| * ||X_j||, keep top-k per row."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    W = lin.weight.data.float()
    importance = W.abs() * torch.sqrt(H.diag().clamp(min=1e-8)).to(W.device)
    keep = max(1, int(W.shape[1] * (1.0 - sparsity)))
    idx = importance.argsort(dim=1, descending=True)[:, :keep]
    mask = torch.zeros_like(W)
    mask.scatter_(1, idx, 1.0)
    lin.weight.data.mul_(mask.to(lin.weight.dtype))
