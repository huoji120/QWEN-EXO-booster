"""Fixed-shape SM120 HC prefill projection and four-branch mixing.

Matches the compiled HC path's BF16 projection/activation boundaries, FP32
SiLU and sigmoid/products, and left-associated FP32 branch reduction.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=None)
def _is_sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def hc_prefill_mix_supported(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> bool:
    """Check the fixed CUDA/BF16 layout using only tensor/device metadata."""
    return (
        x.is_cuda
        and x.ndim == 2
        and x.shape[0] == 1024
        and x.shape[1] == 10240
        and down_weight.shape == (320, 10240)
        and up_weight.shape == (10240, 320)
        and x.dtype == torch.bfloat16
        and down_weight.dtype == torch.bfloat16
        and up_weight.dtype == torch.bfloat16
        and down_weight.device == x.device
        and up_weight.device == x.device
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and _is_sm120(x.device)
    )


@triton.jit
def _activate_down(Down, Activated, M: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Down + i, mask=i < M * 320, other=0).to(tl.float32) * 0.25
    activated = value / (1.0 + libdevice.exp(-value))
    tl.store(Activated + i, activated.to(tl.bfloat16), mask=i < M * 320)


@triton.jit
def _up_mix_kernel(
    X,
    Activated,
    Up,
    Out,
    M: tl.constexpr,
    BM: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    # Dot columns group the same hidden-column tile from each HC branch.
    ns = tl.arange(0, 4 * BJ)
    js = tl.program_id(1) * BJ + ns % BJ
    columns = (ns // BJ) * 2560 + js
    ks = tl.arange(0, BK)
    accum = tl.full((BM, 4 * BJ), 0, tl.float32)

    for k_start in range(tl.cdiv(320, BK)):
        rank_indices = k_start * BK + ks
        activated = tl.load(
            Activated + rows[:, None] * 320 + rank_indices[None, :],
            mask=(rows[:, None] < M) & (rank_indices[None, :] < 320),
            other=0,
        )
        weight = tl.load(
            Up + columns[None, :] * 320 + rank_indices[:, None],
            mask=(js[None, :] < 2560) & (rank_indices[:, None] < 320),
            other=0,
        )
        accum = tl.dot(activated, weight, accum)

    # Preserve the compiled up-projection rounding before FP32 gating.
    projected = accum.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(projected)
    normed = tl.load(
        X + rows[:, None] * 10240 + columns[None, :],
        mask=(rows[:, None] < M) & (js[None, :] < 2560),
        other=0,
    )
    products = gate * normed.to(tl.float32)
    branches = tl.reshape(products, (BM, 4, BJ)).permute(0, 2, 1)
    even, odd = tl.split(tl.reshape(branches, (BM, BJ, 2, 2)))
    p0, p2 = tl.split(even)
    p1, p3 = tl.split(odd)
    mixed = (((p0 + p1) + p2) + p3) * 0.25
    output_columns = tl.program_id(1) * BJ + tl.arange(0, BJ)
    tl.store(
        Out + rows[:, None] * 2560 + output_columns[None, :],
        mixed.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (output_columns[None, :] < 2560),
    )


def hc_prefill_mix(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> torch.Tensor:
    """Return BF16 [M, 2560] HC mixing for supported inference inputs.

    Requires contiguous BF16 x[M, 10240], down_weight[320, 10240], and
    up_weight[10240, 320] on one SM120 CUDA device, with M == 1024.
    Temporaries are per-call and use the current stream/capture allocator;
    no shared scratch or caller-provided output buffer is retained.
    """
    if not hc_prefill_mix_supported(x, down_weight, up_weight):
        raise ValueError(
            "hc_prefill_mix requires contiguous BF16 x[M, 10240], "
            "down_weight[320, 10240], and up_weight[10240, 320] on one "
            "SM120 CUDA device, with M == 1024"
        )

    m = x.shape[0]
    with torch.cuda.device(x.device):
        down = F.linear(x, down_weight)
        activated = torch.empty((m, 320), dtype=torch.bfloat16, device=x.device)
        out = torch.empty((m, 2560), dtype=torch.bfloat16, device=x.device)
        _activate_down[(triton.cdiv(m * 320, 256),)](
            down, activated, m, 256, enable_fp_fusion=False
        )
        _up_mix_kernel[(triton.cdiv(m, 64), triton.cdiv(2560, 32))](
            x,
            activated,
            up_weight,
            out,
            m,
            BM=64,
            BJ=32,
            BK=32,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )
    return out
