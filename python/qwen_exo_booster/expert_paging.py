"""Router-selected ModelOpt NVFP4 expert paging, without changing Top-K.

The checkpoint bank stays pageable on the host. Each layer has at most the
checkpoint's native Top-K expert slots on CUDA. A prefill is partitioned by its
route sets, not treated as though ten experts could cover the entire batch.
"""

from __future__ import annotations

import copy
import math
import threading
import weakref
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn


_HOST_RESERVATIONS = weakref.WeakKeyDictionary()
_HOST_BASELINE = None
_HOST_HEADROOM = 8 * 1024**3


def _reserve_host_storage(layer, size: int) -> None:
    """Account for empty, not-yet-faulted banks against the real cgroup limit."""
    global _HOST_BASELINE
    limit_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.exists() and current_path.exists():
        limit_text = limit_path.read_text().strip()
        if limit_text != "max":
            limit = int(limit_text)
            current = int(current_path.read_text())
            stats_path = Path("/sys/fs/cgroup/memory.stat")
            if stats_path.exists():
                stats = {
                    name: int(value)
                    for name, value in (line.split() for line in stats_path.read_text().splitlines())
                }
                # Clean file-backed pages, including active mmap pages, are
                # reclaimable. Shared, dirty, writeback and locked pages are not
                # capacity we can promise to an anonymous expert bank.
                reclaimable = max(
                    0,
                    stats.get("file", 0)
                    - sum(stats.get(name, 0) for name in (
                        "shmem", "file_dirty", "file_writeback", "unevictable"
                    )),
                )
                current -= min(current, reclaimable)
            if _HOST_BASELINE is None or not _HOST_RESERVATIONS:
                _HOST_BASELINE = current
            projected = max(current, _HOST_BASELINE + sum(_HOST_RESERVATIONS.values()))
            if projected + size + _HOST_HEADROOM > limit:
                raise MemoryError(
                    "NVFP4 CPU expert bank exceeds cgroup memory.max with 8 GiB "
                    f"headroom: committed={projected}, requested={size}, limit={limit}. "
                    "Do not load alongside the production model."
                )
    _HOST_RESERVATIONS[layer] = size


def plan_route_groups(route_rows, capacity: int, num_experts: int):
    """Stable buckets of identical expert sets; original row/order is retained.

    No probability is recomputed. Repeated ids remain repeated kernel routes.
    Negative ids are rejected: this eager single-rank path has no padded ranks.
    """
    if capacity <= 0:
        raise ValueError("Expert residency must be positive")
    groups = {}
    for row_index, row in enumerate(route_rows):
        if any(expert < 0 or expert >= num_experts for expert in row):
            raise ValueError("Router selected an out-of-range expert")
        experts = tuple(sorted(set(row)))
        if len(experts) > capacity:
            raise ValueError("A token's native route set exceeds expert residency")
        if experts not in groups:
            groups[experts] = []
        groups[experts].append(row_index)
    return [(rows, experts) for experts, rows in groups.items()]


def remap_route_ids(global_ids: torch.Tensor, experts):
    """Map sorted resident slots without reordering a token's expert routes."""
    expert_ids = torch.tensor(experts, dtype=global_ids.dtype, device=global_ids.device)
    return torch.searchsorted(expert_ids, global_ids.contiguous()).to(global_ids.dtype)


def create_nvfp4_host_weights(
    method,
    layer,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype,
    **extra_weight_attrs,
):
    """ModelOpt create_weights hook: allocate ONLY checkpoint-shaped CPU tensors.

    These are the same parameter types/shard loader as ordinary ModelOpt. GPU
    swizzling/alignment is deferred until a bounded set is selected by routing.
    """
    from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter

    if (
        not method.quant_config.is_checkpoint_nvfp4_serialized
        or getattr(method.quant_config, "is_nvfp4_online", False)
        or method.quant_config.group_size != 16
    ):
        raise ValueError("Expert paging requires a serialized ModelOpt NVFP4 group-16 checkpoint")
    if extra_weight_attrs.get("with_bias", False):
        raise ValueError("ModelOpt NVFP4 expert paging does not support biased expert GEMMs")
    intermediate = intermediate_size_per_partition
    if hidden_size % 16 or intermediate_size_per_partition % 16:
        raise ValueError("NVFP4 expert dimensions must be aligned to group size 16")
    shards = 2 if layer.moe_runner_config.is_gated else 1
    specifications = {
        "w13_weight": ((num_experts, shards * intermediate, hidden_size // 2), torch.uint8),
        "w2_weight": ((num_experts, hidden_size, intermediate // 2), torch.uint8),
        "w13_weight_scale": ((num_experts, shards * intermediate, hidden_size // 16), torch.float8_e4m3fn),
        "w2_weight_scale": ((num_experts, hidden_size, intermediate // 16), torch.float8_e4m3fn),
        "w13_weight_scale_2": ((num_experts, shards) if shards == 2 else (num_experts,), torch.float32),
        "w2_weight_scale_2": ((num_experts,), torch.float32),
        "w13_input_scale": ((num_experts, shards), torch.float32),
        "w2_input_scale": ((num_experts,), torch.float32),
    }
    size = sum(math.prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
               for shape, dtype in specifications.values())
    _reserve_host_storage(layer, size)
    loader = extra_weight_attrs["weight_loader"]
    for name, (shape, dtype) in specifications.items():
        data = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=False)
        if name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            param = ModelWeightParameter(data=data, input_dim=1, output_dim=2, weight_loader=loader)
        else:
            param = PerTensorScaleParameter(data=data, weight_loader=loader)
        if "input_scale" in name:
            param._sglang_require_global_experts = True
        layer.register_parameter(name, param)
    layer.w13_blockscale_swizzled = None
    layer.w2_blockscale_swizzled = None
    layer.intermediate_size_per_partition = intermediate
    layer.params_dtype = params_dtype
    layer.quant_config = method.quant_config


class NVFP4ExpertBank:
    """Pageable host bank and a per-layer, native-Top-K CUDA working set.

    Eager, single-rank dispatch deliberately synchronizes before eviction. It
    never pins the full bank, nor temporarily puts all experts on CUDA. Reusing
    the last route set skips copies and quantization-layout processing.
    """

    _WEIGHTS = (
        "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
        "w13_weight_scale_2", "w2_weight_scale_2",
    )

    def __init__(self, layer):
        self.layer = weakref.proxy(layer)
        self.capacity = int(layer.top_k)
        self.working = None
        self.resident = None
        self.completion = None
        self.activation_scales = None
        self._dispatch_lock = threading.Lock()

    def finalize(self):
        """Postload hook; validate host scales without materializing GPU weights."""
        layer = self.layer
        for name in self._WEIGHTS + ("w13_input_scale", "w2_input_scale"):
            tensor = getattr(layer, name)
            if tensor.device.type != "cpu" or tensor.is_pinned():
                raise RuntimeError(f"{name} must remain in pageable CPU storage")
        # Preserve the full-bank activation maxima used by the native CUTLASS /
        # TRTLLM quantization path. A selected-subset maximum would change FP4
        # rounding even though route ids and routing probabilities stayed equal.
        scales = []
        for name in ("w13_input_scale", "w2_input_scale"):
            tensor = getattr(layer, name)
            if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
                raise ValueError(f"Invalid ModelOpt checkpoint activation scale: {name}")
            scales.append(tensor.max().item())
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            tensor = getattr(layer, name)
            if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
                raise ValueError(f"Invalid ModelOpt checkpoint weight scale: {name}")
        self.activation_scales = tuple(scales)
        if self.completion is not None:
            self.completion.synchronize()
        self.working = None
        self.resident = None
        self.completion = None

    def _new_working_layer(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher

        layer = self.layer
        working = copy.copy(layer)
        # Do not alias the host parameter registry, or register the working
        # layer as a child that a generic model.to()/postload traversal visits.
        working._parameters = {}
        working._buffers = {}
        working._modules = {}
        working.qwen_exo_moe_cpu_offload = False
        working.expert_bank = None
        working.num_experts = self.capacity
        working.num_local_experts = self.capacity
        working._num_global_routed = self.capacity
        working._num_local_routed = self.capacity
        working.supports_deferred_finalize = False
        working.moe_runner_config = replace(
            layer.moe_runner_config,
            num_experts=self.capacity,
            num_local_experts=self.capacity,
            inplace=False,
        )
        working.quant_method = copy.copy(layer.quant_method)
        if hasattr(working.quant_method, "_cache_permute_indices"):
            working.quant_method._cache_permute_indices = {}
        working.quant_method.create_moe_runner(working, working.moe_runner_config)
        working.dispatcher = create_moe_dispatcher(working.moe_runner_config)
        working.runner = working.quant_method.runner
        return working

    def _stage(self, experts, device):
        if self.resident == experts and self.working is not None:
            if self.working.w13_weight.device != device:
                raise RuntimeError("Expert paging cannot change CUDA devices after initialization")
            if self.completion is not None:
                torch.cuda.current_stream(device).wait_event(self.completion)
            return self.working
        if self.completion is not None:
            # The previous group may still read its weights on a different
            # stream. Eviction is not safe until that kernel has completed.
            self.completion.synchronize()
        self.working = None
        self.resident = None
        working = self._new_working_layer()
        layer = self.layer
        for name in self._WEIGHTS:
            source = getattr(layer, name)
            shape = (self.capacity,) + tuple(source.shape[1:])
            target = torch.empty(shape, dtype=source.dtype, device=device)
            if len(experts) < self.capacity:
                target.zero_() if "scale_2" not in name else target.fill_(1)
            for local_id, global_id in enumerate(experts):
                target[local_id].copy_(source[global_id], non_blocking=False)
            working.register_parameter(name, nn.Parameter(target, requires_grad=False))
        for name, maximum in zip(("w13_input_scale", "w2_input_scale"), self.activation_scales):
            source = getattr(layer, name)
            shape = (self.capacity,) + tuple(source.shape[1:])
            working.register_parameter(name, nn.Parameter(
                torch.full(shape, maximum, dtype=source.dtype, device=device), requires_grad=False
            ))
        working.w13_blockscale_swizzled = None
        working.w2_blockscale_swizzled = None
        # Existing ModelOpt processing handles exact packed bytes, gate/up
        # ordering, FP8 blockscale swizzles, and FP32 per-tensor scales. Only
        # the selected experts are present during this call.
        working.quant_method.process_weights_after_loading(working)
        self.working = working
        self.resident = experts
        return working

    def forward(self, hidden_states, topk_output):
        # Serialize residency mutation across callers on different streams.
        with self._dispatch_lock:
            return self._forward(hidden_states, topk_output)

    def _forward(self, hidden_states, topk_output):
        from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker

        if self.activation_scales is None:
            raise RuntimeError("Expert bank must be finalized after checkpoint loading")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("Router expert paging requires CUDA computation")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Router expert paging is incompatible with CUDA graph capture")
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError("Expert paging requires native StandardTopKOutput; fused rerouting is unsupported")
        if topk_output.topk_ids.shape != (hidden_states.shape[0], self.capacity):
            raise ValueError("Expert paging must retain the checkpoint-native Top-K width")
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        groups = plan_route_groups(
            topk_output.topk_ids.detach().to(device="cpu").tolist(),
            self.capacity,
            self.layer.num_experts,
        )
        output = torch.empty_like(hidden_states)
        for rows, experts in groups:
            working = self._stage(experts, hidden_states.device)
            row_ids = torch.tensor(rows, dtype=torch.long, device=hidden_states.device)
            global_ids = topk_output.topk_ids.index_select(0, row_ids)
            # Sorted expert slots give a deterministic global-to-local mapping;
            # neither route order, probabilities, nor normalization is changed.
            local_ids = remap_route_ids(global_ids, experts)
            local_topk = StandardTopKOutput(
                topk_weights=topk_output.topk_weights.index_select(0, row_ids),
                topk_ids=local_ids,
                router_logits=(
                    topk_output.router_logits.index_select(0, row_ids)
                    if topk_output.router_logits is not None else None
                ),
            )
            result = working.forward_impl(hidden_states.index_select(0, row_ids), local_topk)
            output.index_copy_(0, row_ids, result)
            self.completion = torch.cuda.Event()
            self.completion.record(torch.cuda.current_stream(hidden_states.device))
            # Do not let this loop-local reference keep the evicted working
            # layer alive while the next set's CUDA tensors are allocated.
            del working
        return output
