"""Single-token, no-grad packed NVFP4 experts; no serving-runtime dependency.

The caller supplies unique experts in ascending expert-ID order and owns scale
value validation. Only pointer metadata is copied: packed weights stay in place.
Triton is imported and kernels constructed only after CUDA metadata validation.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def project(table, inputs, output, N: tl.constexpr, K: tl.constexpr,
                TABLE_OFFSET: tl.constexpr, COMMON_INPUT: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
        projection = tl.program_id(1)
        base = (TABLE_OFFSET + projection) * 3
        packed = tl.load(table + base).to(tl.pointer_type(tl.uint8))
        blocks = tl.load(table + base + 1).to(tl.pointer_type(tl.float8e4nv))
        global_scale = tl.load(table + base + 2).to(tl.pointer_type(tl.float32))
        scale_global = tl.load(global_scale)
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        columns = tl.arange(0, BK)
        acc = tl.full((BN,), 0, tl.float32)
        input_base = 0 if COMMON_INPUT else projection * K
        for start in range(tl.cdiv(K, BK)):
            cols = start * BK + columns
            mask = (rows[:, None] < N) & (cols[None, :] < K)
            byte = tl.load(packed + rows[:, None] * (K // 2) + cols[None, :] // 2,
                           mask=mask, other=0)
            code = (byte >> ((cols[None, :] % 2) * 4)) & 15
            magnitude = code & 7
            value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5,
                    tl.where(magnitude < 4, 1.0 + (magnitude - 2).to(tl.float32) * 0.5,
                    tl.where(magnitude < 6, 2.0 + (magnitude - 4).to(tl.float32),
                             4.0 + (magnitude - 6).to(tl.float32) * 2.0)))
            # Bitwise sign insertion retains the E2M1 negative-zero code.
            value = (value.to(tl.uint32, bitcast=True) |
                     ((code.to(tl.uint32) & 8) << 28)).to(tl.float32, bitcast=True)
            block_scale = tl.load(blocks + rows[:, None] * (K // 16) + cols[None, :] // 16,
                                  mask=mask, other=0.0).to(tl.float32)
            scale = block_scale * scale_global
            weight = (value * scale).to(tl.bfloat16).to(tl.float32)
            x = tl.load(inputs + input_base + cols, mask=cols < K, other=0).to(tl.float32)
            acc += tl.sum(weight * x[None, :], axis=1)
        tl.store(output + projection * N + rows, acc.to(tl.bfloat16), mask=rows < N)

    @triton.jit
    def reduce_ordered(values, routes, slots, output, H: tl.constexpr,
                       EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
        columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        result = tl.full((BLOCK,), 0, tl.float32)
        for expert in range(EXPERTS):
            slot = tl.load(slots + expert)
            route = tl.load(routes + slot).to(tl.bfloat16).to(tl.float32)
            value = tl.load(values + expert * H + columns,
                            mask=columns < H, other=0).to(tl.float32)
            weighted = (value * route).to(tl.bfloat16).to(tl.float32)
            result = (result + weighted).to(tl.bfloat16).to(tl.float32)
        tl.store(output + columns, result.to(tl.bfloat16), mask=columns < H)

    return project, reduce_ordered


def grouped_nvfp4_decode(hidden, route_weights, projections):
    """Compute ordered packed experts for contiguous CUDA BF16 ``[1,H]``.

    ``projections`` contains ``(slot, gate, up, down)`` in ascending unique
    expert-ID order. Each projection is ``(uint8 packed, FP8 E4M3 blocks,
    FP32 global scalar)``. Slots must permute all top-K positions. No tensor
    values are read on the host. Unsupported inputs raise, never fall back.

    FP32 reductions are tiled rather than cuBLAS reductions: numerical parity
    must be measured with a bounded tolerance, not assumed bitwise identical.
    Per-call scratch and current-stream launches allow concurrent callers;
    pointer-table host upload makes this wrapper unsuitable for graph capture.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Packed expert decode requires no_grad or inference_mode")
    if (not isinstance(hidden, torch.Tensor) or hidden.device.type != "cuda"
            or hidden.dtype != torch.bfloat16 or hidden.ndim != 2
            or hidden.shape[0] != 1 or not hidden.is_contiguous()):
        raise ValueError("hidden must be contiguous CUDA BF16 [1,H]")
    h = hidden.shape[1]
    if h <= 0 or h % 16:
        raise ValueError("hidden size must be positive and divisible by 16")
    if (not isinstance(route_weights, torch.Tensor)
            or route_weights.device != hidden.device or route_weights.ndim != 2
            or route_weights.shape[0] != 1 or not route_weights.is_contiguous()
            or route_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError("route_weights must be contiguous CUDA floating [1,top_k]")
    entries = tuple(projections)
    count = len(entries)
    if count == 0 or route_weights.shape[1] != count:
        raise ValueError("Exactly one projection entry is required per route slot")
    slots = []
    groups = [[], [], []]
    intermediate = None
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError("Projection entry must be (slot, gate, up, down)")
        slot = entry[0]
        if type(slot) is not int or not 0 <= slot < count:
            raise ValueError("Route slot must be an in-range Python integer")
        slots.append(slot)
        for index, triplet in enumerate(entry[1:]):
            if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
                raise ValueError("Projection must contain packed, block and global tensors")
            packed, block, global_scale = triplet
            if any(not isinstance(t, torch.Tensor) or t.device != hidden.device
                   or not t.is_contiguous() for t in triplet):
                raise ValueError("All projection tensors must be contiguous on the hidden device")
            if (packed.dtype != torch.uint8 or packed.ndim != 2
                    or block.dtype != torch.float8_e4m3fn or block.ndim != 2
                    or global_scale.dtype != torch.float32 or global_scale.numel() != 1):
                raise ValueError("Projection requires U8 packed, FP8 E4M3 blocks and scalar FP32 global")
            n, half_k = packed.shape
            k = half_k * 2
            if n <= 0 or k <= 0 or k % 16 or block.shape != (n, k // 16):
                raise ValueError("Invalid packed projection/block-scale geometry")
            if intermediate is None:
                intermediate = n
                if intermediate % 16:
                    raise ValueError("Intermediate size must be divisible by 16")
            expected = (intermediate, h) if index < 2 else (h, intermediate)
            if (n, k) != expected:
                raise ValueError("All experts must share gate/up/down geometry")
            groups[index].extend(t.data_ptr() for t in triplet)
    if len(set(slots)) != count:
        raise ValueError("Route slots must be unique")
    with torch.cuda.device(hidden.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Packed expert pointer upload is not CUDA graph capture safe")
    project, reduce_ordered = _kernels()
    # Tiny metadata only; references in entries retain the packed allocations.
    table = torch.tensor(groups[0] + groups[1] + groups[2], dtype=torch.int64, device=hidden.device)
    slot_table = torch.tensor(slots, dtype=torch.int64, device=hidden.device)
    gate_up = torch.empty((2 * count, intermediate), dtype=torch.bfloat16, device=hidden.device)
    down = torch.empty((count, h), dtype=torch.bfloat16, device=hidden.device)
    output = torch.empty_like(hidden)
    with torch.cuda.device(hidden.device):
        project[((intermediate + 3) // 4, 2 * count)](
            table, hidden, gate_up, intermediate, h, 0, True, 4, 256,
            num_warps=4, enable_fp_fusion=False)
        # Keep eager PyTorch activation boundaries identical to the reference.
        activated = F.silu(gate_up[:count]) * gate_up[count:]
        project[((h + 3) // 4, count)](
            table, activated, down, h, intermediate, 2 * count, False, 4, 256,
            num_warps=4, enable_fp_fusion=False)
        reduce_ordered[((h + 255) // 256,)](
            down, route_weights, slot_table, output, h, count, 256,
            num_warps=4, enable_fp_fusion=False)
    return output
