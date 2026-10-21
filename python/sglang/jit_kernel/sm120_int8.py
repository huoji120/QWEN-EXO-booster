# SPDX-License-Identifier: Apache-2.0
"""Lightweight SM120 G128 INT8 dense JIT entrypoints (no import-time GPU work)."""
from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


@cache_once
def _module(k: int):
    wrappers = [("unpack", "sm120_int8_unpack")]
    if k in (640, 2560, 6144):
        wrappers.append(("gemm", f"sm120_int8_gemm<{k}>"))
    return load_jit(
        "sm120_int8_bf16_dequant_v1", k,
        cuda_files=["quantization/sm120_int8.cuh"],
        cuda_wrappers=wrappers,
        extra_cuda_cflags=["-O3"],
    )


def prepare_sm120_int8(k: int):
    """Compile before graph capture; the caller owns the model/device lifecycle."""
    _module(0)
    if k in (640, 2560, 6144):
        _module(k)


def _check_weight(q, scales):
    if not q.is_cuda or q.dtype != torch.int8 or q.ndim != 2:
        raise ValueError("SM120 INT8 requires CUDA INT8 [N,K] weights")
    if (
        scales.dtype != torch.bfloat16 or scales.device != q.device
        or scales.shape != (q.shape[0], (q.shape[1] + 127) // 128)
        or not q.is_contiguous() or not scales.is_contiguous()
    ):
        raise ValueError("SM120 INT8 requires contiguous BF16 [N,ceil(K/128)] scales")


def sm120_int8_gemm(x, q, scales):
    _check_weight(q, scales)
    if (
        x.device != q.device or x.dtype != torch.bfloat16 or x.ndim != 2
        or x.shape[1] != q.shape[1] or not 1 <= x.shape[0] <= 8
        or q.shape[1] not in (640, 2560, 6144) or not x.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (12, 0)
    ):
        raise ValueError("SM120 INT8 MMA requires SM120, BF16 M=1..8, K=640/2560/6144")
    out = torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=x.dtype)
    _module(q.shape[1]).gemm(out, x, q, scales)
    return out


def sm120_int8_unpack(q, scales):
    _check_weight(q, scales)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    if q.numel():
        _module(0).unpack(out, q, scales)
    return out
