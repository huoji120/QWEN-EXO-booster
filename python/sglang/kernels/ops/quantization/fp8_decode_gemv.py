"""FP8 linears for decode-sized batches on SM120.

sgl-kernel's SM120 ``fp8_scaled_mm`` has a single 128x128x128 CUTLASS tile
config. A decode batch of 1-16 rows therefore launches only N/128 CTAs and reads
the weights at roughly half of DRAM bandwidth (e.g. 5120x17408 down_proj at
~710 GB/s on an RTX PRO 6000). Decode linears are bandwidth-bound, so both
kernels here stream every weight tile exactly once and reach ~1.45 TB/s:

- 1-2 rows: a CUDA-core GEMV with one fp32 accumulator per row.
- 3-16 rows: a skinny tensor-core GEMM that pads the rows to 16 and computes
  ``W[BLOCK_N, BLOCK_K] @ X^T[BLOCK_K, 16]`` with FP8 MMA. From 3 rows the GEMV
  loses bandwidth to its per-row FMA work; at 1-2 rows it is ~1-5% faster.

Scales follow ``fp8_scaled_mm``: per-token activation scales and per-channel
weight scales, applied after the fp32 accumulation.
"""

import torch
import triton
import triton.language as tl

FP8_DECODE_GEMV_MAX_ROWS = 16
_GEMV_MAX_ROWS = 2

# Fixed configs keep CUDA graph capture free of autotuning. Each is within ~2%
# of the per-shape best across the dense Qwen3.5 decode linears on SM120.
_GEMV_BLOCK_N = 8
_GEMV_BLOCK_K = 512
_GEMV_NUM_WARPS = 4
_GEMV_NUM_STAGES = 3
_SKINNY_BLOCK_M = 16
_SKINNY_BLOCK_N = 32
_SKINNY_BLOCK_K = 512
_SKINNY_NUM_WARPS = 4
_SKINNY_NUM_STAGES = 3


@triton.jit
def _accumulate_row(acc, x_ptr, row, stride_xm, offs_k, K, w):
    x = tl.load(x_ptr + row * stride_xm + offs_k, mask=offs_k < K, other=0.0)
    return acc + tl.sum(w * x.to(tl.float32)[None, :], axis=1)


@triton.jit
def _store_row(out_ptr, row, stride_om, offs_n, mask_n, acc, xs_ptr, ws):
    scaled = acc * tl.load(xs_ptr + row) * ws
    tl.store(out_ptr + row * stride_om + offs_n, scaled.to(tl.bfloat16), mask=mask_n)


@triton.jit
def _fp8_decode_gemv_kernel(
    x_ptr,
    w_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k_base = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_base
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None] & (offs_k[None, :] < K),
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc0 = _accumulate_row(acc0, x_ptr, 0, stride_xm, offs_k, K, w)
        if M > 1:
            acc1 = _accumulate_row(acc1, x_ptr, 1, stride_xm, offs_k, K, w)
    ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
    _store_row(out_ptr, 0, stride_om, offs_n, mask_n, acc0, xs_ptr, ws)
    if M > 1:
        _store_row(out_ptr, 1, stride_om, offs_n, mask_n, acc1, xs_ptr, ws)


@triton.jit
def _fp8_skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_m = offs_m < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    for k0 in range(0, K, BLOCK_K):
        mask_k = (k0 + offs_k) < K
        w = tl.load(
            w_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
    xs = tl.load(xs_ptr + offs_m, mask=mask_m, other=0.0)
    out = acc * ws[:, None] * xs[None, :]
    tl.store(
        out_ptr + offs_m[None, :] * stride_om + offs_n[:, None],
        out.to(tl.bfloat16),
        mask=mask_n[:, None] & mask_m[None, :],
    )


def can_use_fp8_decode_gemv(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> bool:
    """``weight`` is the ``fp8_scaled_mm`` operand: (K, N) with a row-major
    (N, K) storage, i.e. ``weight.t()`` is contiguous."""
    rows = qinput.shape[0]
    return (
        1 <= rows <= FP8_DECODE_GEMV_MAX_ROWS
        and out_dtype == torch.bfloat16
        and qinput.dim() == 2
        and qinput.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and qinput.stride(1) == 1
        and weight.t().is_contiguous()
        and x_scale.numel() == rows
        and x_scale.dtype == torch.float32
        and weight_scale.numel() == weight.shape[1]
        and weight_scale.dtype == torch.float32
    )


def fp8_decode_gemv(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """bf16 ``qinput @ weight`` with per-token and per-channel scales."""
    rows, k = qinput.shape
    n = weight.shape[1]
    weight_rows = weight.t()
    out = torch.empty((rows, n), device=qinput.device, dtype=torch.bfloat16)
    x_scale = x_scale.reshape(-1).contiguous()
    weight_scale = weight_scale.reshape(-1).contiguous()
    if rows <= _GEMV_MAX_ROWS:
        _fp8_decode_gemv_kernel[(triton.cdiv(n, _GEMV_BLOCK_N),)](
            qinput,
            weight_rows,
            x_scale,
            weight_scale,
            out,
            n,
            k,
            qinput.stride(0),
            weight_rows.stride(0),
            out.stride(0),
            M=rows,
            BLOCK_N=_GEMV_BLOCK_N,
            BLOCK_K=_GEMV_BLOCK_K,
            num_warps=_GEMV_NUM_WARPS,
            num_stages=_GEMV_NUM_STAGES,
        )
    else:
        _fp8_skinny_gemm_kernel[(triton.cdiv(n, _SKINNY_BLOCK_N),)](
            qinput,
            weight_rows,
            x_scale,
            weight_scale,
            out,
            rows,
            n,
            k,
            qinput.stride(0),
            weight_rows.stride(0),
            out.stride(0),
            BLOCK_M=_SKINNY_BLOCK_M,
            BLOCK_N=_SKINNY_BLOCK_N,
            BLOCK_K=_SKINNY_BLOCK_K,
            num_warps=_SKINNY_NUM_WARPS,
            num_stages=_SKINNY_NUM_STAGES,
        )
    return out
