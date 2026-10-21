"""Small-row Flash-Next gated RMSNorm/output projection, without global patches.

Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 at
1233bf628f42940ec8e4fb060fa35c1a30eecfc2 (qf_gdnout.cu). This version
uses SGLang JIT/FFI, ordinary stream dependencies, and explicit INT8 scales.
"""

from __future__ import annotations

import math

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _module(rows: int, int8: bool):
    args = make_cpp_args(rows, int8)
    return load_jit(
        "sm120_gdn_output",
        *args,
        cuda_files=["sm120/gdn_output.cuh"],
        cuda_wrappers=[("run", f"gdn_output<{args}>")],
    )


@cache_once
def _is_sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def fused_gdn_output(
    core: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    activation: str = "sigmoid",
    weight_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project BF16 per-head gated RMSNorm, retaining its BF16 rounding.

    ``core`` and ``gate`` are contiguous [M*48,128], M=1..8. INT8 weight
    [2560,6144] uses BF16 [2560,48] group-128 scales; q*s is rounded to
    BF16 before MMA. All scratch is CTA-local shared memory, never a cached
    Python tensor, so graphs of different M/layers cannot alias scratch.
    The producer is not required to implement PDL.
    """
    if core.ndim != 2 or core.shape[1] != 128 or core.shape[0] % 48:
        raise ValueError("GDN core must have shape [M*48,128]")
    rows = core.shape[0] // 48
    if not 1 <= rows <= 8:
        raise ValueError("GDN fused projection supports 1..8 rows")
    if activation not in ("sigmoid", "silu", "swish"):
        raise ValueError("GDN activation must be sigmoid, silu, or swish")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("GDN epsilon must be positive and finite")
    int8 = weight.dtype == torch.int8
    if weight.dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("GDN weight dtype must be BF16 or INT8")
    if int8 != (weight_scale is not None):
        raise ValueError("Only INT8 GDN weight requires group scales")
    expected = [(core, core.shape, torch.bfloat16), (gate, core.shape, torch.bfloat16),
                (norm_weight, (128,), torch.bfloat16),
                (weight, (2560, 6144), torch.int8 if int8 else torch.bfloat16)]
    if weight_scale is not None:
        expected.append((weight_scale, (2560, 48), torch.bfloat16))
    if out is not None:
        expected.append((out, (rows, 2560), torch.bfloat16))
    for tensor, shape, dtype in expected:
        if (not tensor.is_cuda or tensor.device != core.device or tensor.dtype != dtype
                or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous()
                or tensor.data_ptr() % 16):
            raise ValueError("GDN tensors must be aligned contiguous CUDA tensors with matching shape, dtype and device")
    if not _is_sm120(core.device):
        raise ValueError("GDN fused projection requires SM120")
    if out is None:
        out = torch.empty((rows, 2560), device=core.device, dtype=torch.bfloat16)
    _module(rows, int8).run(
        out, core, gate, norm_weight, weight,
        weight_scale if int8 else norm_weight, eps, activation != "sigmoid",
    )
    return out


def try_fused_gdn_output(
    core: torch.Tensor,
    gate: torch.Tensor,
    norm,
    projection,
    *,
    enabled: bool,
    tp_size: int,
    projection_is_unquantized: bool,
) -> torch.Tensor | None:
    """Decline unsupported paths so the caller uses its normal norm/projection.

    Never reinterpret a quantizer's storage as BF16. INT8 is recognized using
    the Dense backend's shared marker API, not dtype alone. No weight copy or
    dequantization allocation is performed by this dispatch.
    """
    if (not enabled or tp_size != 1 or not core.is_cuda
            or core.dtype != torch.bfloat16 or core.ndim != 2
            or core.shape[1] != 128 or not 48 <= core.shape[0] <= 384
            or core.shape[0] % 48 or gate.shape != core.shape
            or not norm.norm_before_gate or norm.group_size not in (None, 128)
            or norm.activation not in ("sigmoid", "silu", "swish")
            or norm.bias is not None or projection.bias is not None
            or not _is_sm120(core.device)):
        return None
    from sglang.srt.layers.quantization.sm120_dense import is_sm120_int8_linear

    int8 = is_sm120_int8_linear(projection)
    if not int8 and not projection_is_unquantized:
        return None
    weight = projection.weight
    scale = projection.sm120_weight_scale if int8 else None
    if (weight.shape != (2560, 6144) or norm.weight.shape != (128,)
            or (int8 and projection.sm120_group_size != 128)):
        return None
    tensors = [core, gate, norm.weight, weight]
    if int8:
        if scale.shape != (2560, 48):
            return None
        tensors.append(scale)
    if (weight.dtype != (torch.int8 if int8 else torch.bfloat16)
            or any(t.dtype != torch.bfloat16 for t in (gate, norm.weight))
            or (int8 and scale.dtype != torch.bfloat16)
            or any(not t.is_cuda or t.device != core.device or not t.is_contiguous()
                   or t.data_ptr() % 16 for t in tensors)):
        return None
    return fused_gdn_output(
        core, gate, norm.weight, weight, eps=norm.eps,
        activation=norm.activation, weight_scale=scale,
    )
