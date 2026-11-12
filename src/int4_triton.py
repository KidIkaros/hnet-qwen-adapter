"""Native int4 inference path: Triton GEMM over our packed GPTQ format.

Reads the SAME artifact as src/int4_load.py (nibble-packed codes + fp16
per-(row,blocksize) scales) but keeps weights packed on-device and dequantizes
inside the GEMM — ~4.5 bits/param resident instead of bf16.

Format contract (matches src/compress.py packing):
  qweight [out, in//2] uint8 — even input col -> high nibble, odd -> low
  scales  [out, ceil(in/blocksize)] float16 — symmetric, zero-point 7.5
  W[o,k] = (code[o,k] - 7.5) * scales[o, k//blocksize]

Turing note (sm_75, T4/2070S): no bf16 tensor cores — kernel computes the
dot in fp16 with fp32 accumulation. Activations are cast to fp16 by the
caller; hidden states post-RMSNorm are small enough for fp16 range.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton optional at import time
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _int4_gemm_kernel(
        x_ptr, w_ptr, s_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wj,
        stride_sn, stride_sg,
        BK: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # BK must equal the scale-group size (blocksize): each K-chunk then
        # maps to exactly one scale column.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        j = tl.arange(0, BK // 2)           # byte columns within chunk
        xm = rm[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, K, BK):
            gidx = k0 // BK
            kb = k0 // 2 + j                # packed-byte offsets [HALF]
            wb = tl.load(
                w_ptr + rn[:, None] * stride_wn + kb[None, :] * stride_wj,
                mask=(rn[:, None] < N) & (kb[None, :] * 2 < K),
                other=0,
            )
            sc = tl.load(
                s_ptr + rn * stride_sn + gidx * stride_sg,
                mask=rn < N, other=0.0,
            ).to(tl.float32)
            wh = (((wb >> 4) & 15).to(tl.float32) - 7.5) * sc[:, None]
            wl = ((wb & 15).to(tl.float32) - 7.5) * sc[:, None]
            xh = tl.load(
                x_ptr + xm * stride_xm + (k0 + 2 * j)[None, :] * stride_xk,
                mask=(rm[:, None] < M) & ((k0 + 2 * j)[None, :] < K),
                other=0.0,
            )
            xl = tl.load(
                x_ptr + xm * stride_xm + (k0 + 2 * j + 1)[None, :] * stride_xk,
                mask=(rm[:, None] < M) & ((k0 + 2 * j + 1)[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(xh.to(tl.float16), tl.trans(wh).to(tl.float16))
            acc += tl.dot(xl.to(tl.float16), tl.trans(wl).to(tl.float16))
        tl.store(
            y_ptr + rm[:, None] * N + rn[None, :],
            acc.to(tl.float16),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )


def int4_gemm(x, qweight, scales, bits=4, blocksize=128):
    """x [..., in] x qweight [out, in//2] uint8 -> [..., out] (fp16).

    qweight/scales must be CUDA tensors (uint8 / float16).
    """
    assert _HAS_TRITON, "triton not available"
    assert bits == 4, "kernel implements 4-bit only"
    assert blocksize % 2 == 0 and blocksize >= 32, "bad blocksize"
    shp = x.shape
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = qweight.shape[0]
    assert qweight.shape[1] * 2 == K, (qweight.shape, K)
    assert K % 2 == 0
    xf = x.reshape(M, K).contiguous()
    if xf.dtype != torch.float16:
        xf = xf.to(torch.float16)
    # fp16 out: sm_75 lacks bf16 conversion hardware; caller casts as needed
    y = torch.empty(M, N, device=x.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # Triton launches on the current device — pin it to the tensors' device
    # so layers split across GPUs run their GEMMs on the right card.
    with torch.cuda.device(x.device):
        _int4_gemm_kernel[grid](
            xf, qweight, scales, y,
            M, N, K,
            xf.stride(0), xf.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            BK=blocksize, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
    return y.reshape(*shp[:-1], N)


def int4_linear_reference(x, qweight, scales, bits=4, blocksize=128):
    """Pure-torch reference: dequant then matmul (for correctness tests)."""
    from src.compress import dequant_int
    K = x.shape[-1]
    W = dequant_int(qweight, scales.float(), K, bits=bits,
                    blocksize=blocksize).to(torch.float32)
    return x.to(torch.float32) @ W.t()
