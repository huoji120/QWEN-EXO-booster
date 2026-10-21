"""Small-row routed MoE on the *prepared* native NVFP4 Marlin payload.

Phase-four arithmetic adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000,
1233bf628f42940ec8e4fb060fa35c1a30eecfc2/serve/qwenfast/qf_a16moe.cu.
The shared expert deliberately remains in the model's existing separate path.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "sm120_persistent_nvfp4_v1",
        cuda_files=["moe/sm120_persistent_nvfp4.cuh"],
        cuda_wrappers=[("run", "persistent_moe")],
        extra_cuda_cflags=["-O3"],
    )


@dataclass
class PersistentMoeScratch:
    """One stream's fixed-size scratch; never share across concurrent streams."""

    hidden: torch.Tensor
    weighted: torch.Tensor
    partial: torch.Tensor
    control: torch.Tensor

    @classmethod
    def allocate(cls, device):
        return cls(
            torch.empty((80, 8, 640), device=device, dtype=torch.bfloat16),
            torch.empty((8, 10, 2560), device=device, dtype=torch.bfloat16),
            torch.empty((80, 10, 2, 1024), device=device, dtype=torch.float32),
            torch.empty(928, device=device, dtype=torch.int32),
        )


# A bounded eager cache, with no counters shared across streams. Unless a graph
# runner explicitly supplies an arena, capture allocates private graph scratch.
_SCRATCH = OrderedDict()
_CAPTURE_SCRATCH: ContextVar[PersistentMoeScratch | None] = ContextVar(
    "persistent_moe_capture_scratch", default=None
)


@contextmanager
def use_persistent_moe_capture_scratch(scratch: PersistentMoeScratch):
    """Scope a runner-owned arena to sequential MoE launches during capture.

    The owner must retain the arena as long as the captured graph and must not
    reuse it across concurrently replayed graphs or streams. These kernels do
    not use PDL: initialization and persistent launches on one stream finish
    before the next routed layer can overwrite the arena. Nested scopes restore
    the outer arena, including when capture raises.
    """
    if not isinstance(scratch, PersistentMoeScratch):
        raise TypeError("capture arena must be PersistentMoeScratch")
    token = _CAPTURE_SCRATCH.set(scratch)
    try:
        yield
    finally:
        _CAPTURE_SCRATCH.reset(token)


def _scratch(device):
    scoped = _CAPTURE_SCRATCH.get()
    if scoped is not None:
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        if any(
            tensor.device != requested_device
            for tensor in (
                scoped.hidden, scoped.weighted, scoped.partial, scoped.control
            )
        ):
            raise ValueError("persistent MoE capture arena device mismatch")
        return scoped
    if torch.cuda.is_current_stream_capturing():
        return PersistentMoeScratch.allocate(device)
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    scratch = _SCRATCH.get(key)
    if scratch is None:
        scratch = PersistentMoeScratch.allocate(device)
        _SCRATCH[key] = scratch
        if len(_SCRATCH) > 16:
            _SCRATCH.popitem(last=False)
    else:
        _SCRATCH.move_to_end(key)
    return scratch


def supports_persistent_moe(hidden_states, ids, weights, quant, config):
    """Metadata-only guard; every declined shape retains native W4A16 Marlin."""
    if not (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and 1 <= hidden_states.shape[0] <= 8
        and hidden_states.shape[1] == 2560
        and config.is_gated
        and config.activation == "silu"
        and config.gemm1_alpha is None
        and not config.apply_router_weight_on_input
        and config.gemm1_clamp_limit is None
        and config.swiglu_limit is None
        and config.routed_scaling_factor in (None, 1.0)
        and quant.weight_bits == 4
        and quant.is_k_full
        and quant.expert_map is None
        and quant.global_num_experts in (-1, 512)
        and quant.w13_qzeros is None
        and quant.w2_qzeros is None
        and quant.w13_bias is None
        and quant.w2_bias is None
    ):
        return False
    for value in (
        quant.w13_g_idx, quant.w2_g_idx,
        quant.w13_g_idx_sort_indices, quant.w2_g_idx_sort_indices,
    ):
        if value is not None and value.numel():
            return False
    m = hidden_states.shape[0]
    tensors = (
        (hidden_states, (m, 2560), torch.bfloat16),
        (ids, (m, 10), torch.int32),
        (weights, (m, 10), torch.float32),
        (quant.w13_qweight, (512, 160, 2560), torch.int32),
        (quant.w2_qweight, (512, 40, 5120), torch.int32),
        (quant.w13_scales, (512, 160, 1280), torch.float8_e4m3fn),
        (quant.w2_scales, (512, 40, 2560), torch.float8_e4m3fn),
        (quant.w13_global_scale, (512,), torch.bfloat16),
        (quant.w2_global_scale, (512,), torch.bfloat16),
    )
    return all(
        t is not None and t.shape == shape and t.dtype == dtype
        and t.device == hidden_states.device and t.is_contiguous()
        for t, shape, dtype in tensors
    ) and torch.cuda.get_device_capability(hidden_states.device) == (12, 0)


def persistent_moe(hidden_states, ids, weights, quant, *, scratch=None, out=None):
    """Compute routed output; IDs outside [0, 512) are zero/padding slots.

    Consumes processed Marlin block scales and global scales, not checkpoint
    scales. Callers must qualify activation/bias/layout with the runner guard.
    Duplicate expert selections are weighted independently in top-k order.
    Explicit scratch is useful for component/graph-isolation tests; it must not
    be used concurrently by multiple streams or overlapping graphs.
    """
    if scratch is None:
        scratch = _scratch(hidden_states.device)
    if out is None:
        out = torch.empty_like(hidden_states)
    _module().run(
        out, hidden_states, ids, weights,
        quant.w13_qweight, quant.w13_scales, quant.w13_global_scale,
        quant.w2_qweight, quant.w2_scales, quant.w2_global_scale,
        scratch.hidden, scratch.weighted, scratch.partial, scratch.control,
    )
    return out
