"""Opt-in exhaustive QSA prefill, adapted from upstream PR #47878.

Source: fdd4371ba2991bd25068f7ab1bcb231117c5b2a1. The sparse selector
still runs and owns compressed-key state. Only attention is replaced when the
selected causal key set is complete; floating-point summation order can differ.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled


def exhaustive_prefill_eligible(staged_lengths, query_rows: int, budget: int) -> bool:
    """Decide from existing host metadata, never from a device tensor."""
    if budget <= 0 or query_rows <= 0 or not staged_lengths:
        return False
    return sum(query for _, query in staged_lengths) == query_rows and all(
        0 <= query <= total <= budget for total, query in staged_lengths
    )


def causal_prefix_token_count(staged_lengths) -> int:
    return sum(
        query * (total - query) + query * (query + 1) // 2
        for total, query in staged_lengths
    )


@lru_cache(maxsize=8)
def _resolve_dense_varlen(device_index: int) -> Optional[Callable]:
    """Resolve only architectures with a native dense FlashAttention path."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability in ((10, 0), (10, 3), (12, 0)):
        try:
            from flash_attn.cute.interface import flash_attn_varlen_func
        except ImportError:
            return None

        def run(**kwargs):
            result = flash_attn_varlen_func(**kwargs)
            return result[0] if isinstance(result, tuple) else result

        return run
    if capability not in ((8, 0), (8, 6), (8, 9), (9, 0)):
        return None
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        return None
    # Cached-prefix causality must be bottom-right aligned (FA2 >= 2.1).
    try:
        version = tuple(int(part) for part in flash_attn.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return None
    return flash_attn_varlen_func if version >= (2, 1) else None


def try_dense_causal_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    staged_lengths: tuple,
    budget: int,
    scale: float,
) -> Optional[torch.Tensor]:
    """Use already-materialized KV; None preserves the existing sparse path.

    The caller gates this experimental path with SGLANG_ENABLE_QSA_DENSE_PREFILL
    and supplies ordinary-prefill metadata. KV must already have the baseline's
    dequantization/scaling applied. No gathering or quantization is done here.
    QSA top-k expansion guarantees distinct causal positions; its cardinality is
    checked asynchronously before dense attention ignores their ordering.
    """
    if (
        not q.is_cuda
        or torch.version.hip is not None
        or q.dtype not in (torch.bfloat16, torch.float16)
        or q.ndim != 3
        or is_batch_invariant_mode_enabled()
        or not exhaustive_prefill_eligible(staged_lengths, q.shape[0], budget)
    ):
        return None
    total_keys = sum(total for total, _ in staged_lengths)
    if (
        total_keys > 2**31 - 1
        or q.shape[0] > 2**31 - 1
        or k.ndim != 3
        or v.shape != k.shape
        or k.shape[0] != total_keys
        or k.shape[1] <= 0
        or q.shape[1] % k.shape[1]
        or q.shape[2] != k.shape[2]
        or any(t.device != q.device or t.dtype != q.dtype for t in (k, v))
        or topk_indices.ndim != 2
        or topk_indices.shape[0] != q.shape[0]
        or topk_indices.device != q.device
        or topk_indices.dtype not in (torch.int32, torch.int64)
        or any(
            t.device != q.device
            or t.dtype not in (torch.int32, torch.int64)
            or t.shape != (len(staged_lengths) + 1,)
            or not t.is_contiguous()
            for t in (cu_seqlens_q, cu_seqlens_k)
        )
    ):
        return None
    varlen = _resolve_dense_varlen(q.device.index)
    if varlen is None:
        return None
    torch._assert_async(
        (topk_indices >= 0).sum() == causal_prefix_token_count(staged_lengths),
        "QSA dense prefill requires an exhaustive causal selection",
    )
    return varlen(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        cu_seqlens_q=cu_seqlens_q.to(dtype=torch.int32),
        cu_seqlens_k=cu_seqlens_k.to(dtype=torch.int32),
        max_seqlen_q=max(query for _, query in staged_lengths),
        max_seqlen_k=max(total for total, _ in staged_lengths),
        softmax_scale=scale,
        causal=True,
    )
