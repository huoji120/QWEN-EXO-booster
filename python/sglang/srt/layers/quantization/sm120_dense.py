# SPDX-License-Identifier: Apache-2.0
"""Model-bound G128 INT8 dense weights for the SM120 preset.

Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 qf_w8g.cu,
revision 1233bf628f42940ec8e4fb060fa35c1a30eecfc2. Modified to use native
quantization methods, bounded per-call unpacking, and identical BF16-rounded
dequantization in the small-M and general paths. No device work at import.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.srt.layers.quantization.base_config import LinearMethodBase

GROUP_SIZE = 128
_UNPACK_ROWS = 4096


def is_sm120_dense_prefix(prefix: str) -> bool:
    parts = prefix.split(".")
    if any(part in {"visual", "vision_tower", "mtp", "draft_model"} for part in parts):
        return False
    leaf = parts[-1]
    return (
        leaf == "lm_head"
        or leaf in {"in_proj_qkvz", "in_proj_ba", "qkv_proj", "o_proj"}
        or (leaf == "out_proj" and "linear_attn" in parts)
        or ("shared_expert" in parts and leaf in {"gate_up_proj", "down_proj"})
    )


def sm120_fast_enabled() -> bool:
    from sglang.srt.runtime_context import get_server_args

    try:
        args = get_server_args()
    except ValueError as exc:
        # Quantization configs are also inspected offline without an engine.
        if str(exc) != "Global server args is not set yet!":
            raise
        return False
    return bool(getattr(args, "qwen_exo_sm120_fast", False))


def sm120_dense_method(prefix: str):
    """Select only a layer already established to be originally unquantized."""
    if not is_sm120_dense_prefix(prefix):
        return None
    if sm120_fast_enabled():
        return Sm120Int8LinearMethod()
    return None


@torch.no_grad()
def quantize_sm120_int8(weight: torch.Tensor, gain=1.0):
    """Round scales to BF16 *before* symmetric quantization; pad group tails.

    An optional scalar or [N] gain is folded before quantization, not after
    dequantization. Chunking bounds transient FP32 storage for the full head.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("SM120 INT8 conversion requires a BF16 [N,K] weight")
    n, k = weight.shape
    if n == 0 or k == 0:
        raise ValueError("SM120 INT8 conversion requires nonempty weights")
    if isinstance(gain, torch.Tensor):
        if gain.numel() != 1 and gain.shape != (n,):
            raise ValueError("gain must be scalar or have shape [N]")
        gain = gain.to(device=weight.device, dtype=torch.float32)
    q = torch.empty_like(weight, dtype=torch.int8, memory_format=torch.contiguous_format)
    groups = (k + GROUP_SIZE - 1) // GROUP_SIZE
    scales = torch.empty((n, groups), dtype=torch.bfloat16, device=weight.device)
    for first in range(0, n, _UNPACK_ROWS):
        last = min(first + _UNPACK_ROWS, n)
        values = weight[first:last].float()
        chunk_gain = gain[first:last, None] if isinstance(gain, torch.Tensor) and gain.numel() != 1 else gain
        values = values * chunk_gain
        if not torch.isfinite(values).all().item():
            raise ValueError("SM120 INT8 weights and gains must be finite")
        padded = F.pad(values, (0, groups * GROUP_SIZE - k))
        grouped = padded.view(last - first, groups, GROUP_SIZE)
        scale = (grouped.abs().amax(-1) / 127).clamp_min_(1e-30).to(torch.bfloat16)
        if not torch.isfinite(scale).all().item():
            raise ValueError("SM120 INT8 scale overflow")
        quantized = (grouped / scale.float().unsqueeze(-1)).round_().clamp_(-127, 127)
        q[first:last] = quantized.view(last - first, -1)[:, :k].to(torch.int8)
        scales[first:last] = scale
    return q, scales


def dequantize_sm120_int8(weight: torch.Tensor, scales: torch.Tensor):
    """Independent eager reference/general CPU unpack, including group tails."""
    k = weight.shape[1]
    expanded = scales.float().repeat_interleave(GROUP_SIZE, dim=1)[:, :k]
    return (weight.float() * expanded).to(torch.bfloat16)


def is_sm120_int8_linear(layer) -> bool:
    weight = getattr(layer, "weight", None)
    scales = getattr(layer, "sm120_weight_scale", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.dtype == torch.int8
        and isinstance(scales, torch.Tensor)
        and scales.dtype == torch.bfloat16
        and scales.shape == (weight.shape[0], (weight.shape[1] + 127) // 128)
        and scales.device == weight.device
        and getattr(layer, "sm120_group_size", None) == GROUP_SIZE
    )


def apply_sm120_int8_linear(layer, x, bias=None):
    if not is_sm120_int8_linear(layer):
        raise ValueError("Invalid or missing SM120 INT8 linear weight/scale state")
    if x.dtype != torch.bfloat16 or x.device != layer.weight.device:
        raise ValueError("SM120 INT8 linear requires colocated BF16 activations")
    n, k = layer.weight.shape
    if x.shape[-1] != k:
        raise ValueError("SM120 INT8 linear input width does not match weight")
    shape = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if m == 0:
        return x.new_empty((*shape, n))
    if x.is_cuda:
        from sglang.jit_kernel.sm120_int8 import sm120_int8_gemm, sm120_int8_unpack

        if (
            1 <= m <= 8
            and bias is None
            and k in (640, 2560, 6144)
            and x2.is_contiguous()
            and x2.data_ptr() % 16 == 0
            and layer.weight.is_contiguous()
            and layer.weight.data_ptr() % 16 == 0
            and layer.sm120_weight_scale.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (12, 0)
        ):
            out = sm120_int8_gemm(x2, layer.weight, layer.sm120_weight_scale)
            return out.view(*shape, n)
    else:
        sm120_int8_unpack = dequantize_sm120_int8
    # Chunk by output rows: large prefills use cuBLAS, but no full-size BF16
    # resident head copy or process-global scratch shared across CUDA streams.
    out = x.new_empty((m, n))
    for first in range(0, n, _UNPACK_ROWS):
        last = min(first + _UNPACK_ROWS, n)
        unpacked = sm120_int8_unpack(layer.weight[first:last], layer.sm120_weight_scale[first:last])
        destination = out[:, first:last]
        if bias is None:
            torch.mm(x2, unpacked.T, out=destination)
        else:
            torch.addmm(bias[first:last], x2, unpacked.T, out=destination)
    return out.view(*shape, n)


class Sm120Int8FusedProjection:
    """One-shot post-load merge of two native INT8 output partitions.

    Each source remains a registered module and owns a row view of the merged
    weights/scales. Weak references avoid retaining the model through loader
    bookkeeping; no BF16 copy or second quantization is introduced.
    """

    def __init__(self, first, second):
        self._sources = (weakref.ref(first), weakref.ref(second))
        self.weight = None
        self.sm120_weight_scale = None
        self.sm120_group_size = GROUP_SIZE
        self.first_width = first.weight.shape[0]
        first._sm120_fused_projection = self
        second._sm120_fused_projection = self
        # Also support installation after the ordinary post-load pass.
        self.finalize()

    def finalize(self):
        if self.weight is not None:
            return
        first, second = (ref() for ref in self._sources)
        if first is None or second is None:
            return
        if not (is_sm120_int8_linear(first) and is_sm120_int8_linear(second)):
            return
        if (
            first.weight.shape[1] != second.weight.shape[1]
            or first.weight.device != second.weight.device
        ):
            raise ValueError("Fused SM120 projections require matching input width/device")
        weight = torch.cat((first.weight.data, second.weight.data), dim=0)
        scales = torch.cat(
            (first.sm120_weight_scale.data, second.sm120_weight_scale.data), dim=0
        )
        first.weight.data = weight[: self.first_width]
        second.weight.data = weight[self.first_width :]
        first.sm120_weight_scale.data = scales[: self.first_width]
        second.sm120_weight_scale.data = scales[self.first_width :]
        self.weight = weight
        self.sm120_weight_scale = scales
        del first._sm120_fused_projection
        del second._sm120_fused_projection
        del self._sources


class Sm120Int8LinearMethod(LinearMethodBase):
    """Load the original BF16 parameter before replacing its resident storage."""

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("SM120 INT8 preset only converts original BF16 dense weights")
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise ValueError("SM120 INT8 preset requires tensor parallel size 1")
        weight = Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype), requires_grad=False)
        # Same attributes and loader as UnquantizedLinearMethod, including
        # packed QKV/gate-up shard loading under the original checkpoint names.
        weight.input_dim = 1
        weight.output_dim = 0
        for key, value in extra_weight_attrs.items():
            setattr(weight, key, value)
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer):
        if is_sm120_int8_linear(layer):
            return
        quantized, scales = quantize_sm120_int8(layer.weight)
        layer.register_parameter("weight", Parameter(quantized, requires_grad=False))
        layer.register_parameter("sm120_weight_scale", Parameter(scales, requires_grad=False))
        layer.sm120_group_size = GROUP_SIZE
        # Compile all small-M specializations before CUDA graph capture. No
        # activations/outputs or CUDA state are allocated at module import.
        if quantized.is_cuda:
            from sglang.jit_kernel.sm120_int8 import prepare_sm120_int8

            prepare_sm120_int8(quantized.shape[1])
        fused = getattr(layer, "_sm120_fused_projection", None)
        if fused is not None:
            fused.finalize()

    def apply(self, layer, x, bias=None):
        return apply_sm120_int8_linear(layer, x, bias)
