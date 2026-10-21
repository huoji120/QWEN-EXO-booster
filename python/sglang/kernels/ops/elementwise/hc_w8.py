from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HC = 4
_H = 2560
_K = _HC * _H
_R = 320


@cache_once
def _jit_hc_w8_module() -> Module:
    return load_jit(
        "hc_w8",
        "v2",
        cuda_files=["elementwise/hc_w8.cuh"],
        cuda_wrappers=[
            ("hc_mix_fused_w8", "sglang::HcW8Kernel::run"),
            ("hc_combine_mix_w8", "sglang::HcCombineMixW8Kernel::run"),
            ("hc_combine_w8", "sglang::HcCombineW8Kernel::run"),
        ],
    )


def _check_weight_shapes(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    wd: torch.Tensor,
    wd_s: torch.Tensor,
    w_inj: torch.Tensor | None,
    wu: torch.Tensor,
    wu_s: torch.Tensor,
) -> None:
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != _K:
        raise RuntimeError("hc_w8 requires CUDA BF16 [rows, 10240] input")
    if x.shape[0] < 1 or x.shape[0] > 8:
        raise RuntimeError("hc_w8 supports only 1..8 decode rows")
    if w_norm.shape != (_K,) or w_norm.dtype != torch.bfloat16:
        raise RuntimeError("hc_w8 norm weight must be BF16 [10240]")
    if wd.shape != (_R, _K) or wd.dtype != torch.int8:
        raise RuntimeError("hc_w8 down weight must be INT8 [320, 10240]")
    if wd_s.shape != (_R, 80) or wd_s.dtype != torch.bfloat16:
        raise RuntimeError("hc_w8 down scales must be BF16 [320, 80]")
    if wu.shape != (_K, _R) or wu.dtype != torch.int8:
        raise RuntimeError("hc_w8 up weight must be INT8 [10240, 320]")
    if wu_s.shape != (_K, 5) or wu_s.dtype != torch.bfloat16:
        raise RuntimeError("hc_w8 up scales must be BF16 [10240, 5]")
    if w_inj is not None and (
        w_inj.shape != (_HC, _K) or w_inj.dtype != torch.bfloat16
    ):
        raise RuntimeError("hc_w8 injection weight must be BF16 [4, 10240]")
    tensors = (w_norm, wd, wd_s, wu, wu_s) + (() if w_inj is None else (w_inj,))
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise RuntimeError("hc_w8 weights must be contiguous CUDA tensors")


def hc_mix_fused_w8(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    wd: torch.Tensor,
    wd_s: torch.Tensor,
    w_inj: torch.Tensor | None,
    wu: torch.Tensor,
    wu_s: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused grouped-INT8 HC norm/down/up mix for decode rows (1..8)."""
    _check_weight_shapes(x, w_norm, wd, wd_s, w_inj, wu, wu_s)
    rows = x.shape[0]
    out = torch.empty((rows, _H), dtype=torch.bfloat16, device=x.device)
    inj = torch.empty((rows, _HC), dtype=torch.float32, device=x.device)
    normed = torch.empty_like(x)
    partial = torch.empty((_HC * 8 * (_R + _HC),), dtype=torch.float32, device=x.device)
    bar = torch.zeros((2,), dtype=torch.int32, device=x.device)
    inject = (
        w_inj
        if w_inj is not None
        else torch.empty((0,), dtype=torch.bfloat16, device=x.device)
    )
    _jit_hc_w8_module().hc_mix_fused_w8(
        x, w_norm, wd, wd_s, inject, wu, wu_s, float(eps), out, inj, normed,
        partial, bar, True
    )
    return out, inj, normed


def hc_combine_w8(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject: torch.Tensor,
) -> torch.Tensor:
    """Materialize small-row BF16 HC combine with eager FP32 rounding."""
    out = torch.empty_like(residual)
    _jit_hc_w8_module().hc_combine_w8(block_output, residual, inject, out)
    return out


def hc_combine_mix_w8(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inj_prev: torch.Tensor,
    w_norm: torch.Tensor,
    wd: torch.Tensor,
    wd_s: torch.Tensor,
    w_inj: torch.Tensor | None,
    wu: torch.Tensor,
    wu_s: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse BF16 combine (with a prior injection dot) into the next HC mix."""
    _check_weight_shapes(residual, w_norm, wd, wd_s, w_inj, wu, wu_s)
    rows = residual.shape[0]
    if (
        block_output.shape != (rows, _H)
        or block_output.dtype != torch.bfloat16
        or not block_output.is_cuda
        or not block_output.is_contiguous()
    ):
        raise RuntimeError("hc_combine_mix_w8 block_output must be contiguous CUDA BF16 [rows, 2560]")
    if inj_prev.shape != (rows, _HC) or inj_prev.dtype != torch.float32:
        raise RuntimeError("hc_combine_mix_w8 inj_prev must be FP32 [rows, 4]")
    out = torch.empty((rows, _H), dtype=torch.bfloat16, device=residual.device)
    inj = torch.empty((rows, _HC), dtype=torch.float32, device=residual.device)
    normed = torch.empty_like(residual)
    partial = torch.empty(
        (_HC * 8 * (_R + _HC),), dtype=torch.float32, device=residual.device
    )
    bar = torch.zeros((2,), dtype=torch.int32, device=residual.device)
    xout = torch.empty_like(residual)
    inject = (
        w_inj
        if w_inj is not None
        else torch.empty((0,), dtype=torch.bfloat16, device=residual.device)
    )
    _jit_hc_w8_module().hc_combine_mix_w8(
        block_output,
        residual,
        inj_prev,
        w_norm,
        wd,
        wd_s,
        inject,
        wu,
        wu_s,
        float(eps),
        out,
        inj,
        normed,
        partial,
        bar,
        xout,
        True,
    )
    return out, inj, normed, xout
