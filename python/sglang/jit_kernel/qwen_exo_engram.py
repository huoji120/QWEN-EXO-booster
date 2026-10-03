from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_qwen_exo_engram_module(*, num_heads: int, head_dim: int) -> Module:
    args = make_cpp_args(num_heads, head_dim)
    return load_jit(
        "qwen_exo_engram",
        *args,
        cuda_files=["qwen_exo_engram.cuh"],
        cuda_wrappers=[("gather_dequant", f"&QwenExoEngramKernel<{args}>::gather_dequant")],
    )


@debug_kernel_api
def engram_gather_dequant(
    *,
    table_data: torch.Tensor,
    table_scale: torch.Tensor,
    rows: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
) -> None:
    """Gather Engram table rows from pinned host memory into ``out``.

    Args:
        table_data: uint8 [table_rows, head_dim] FP8 E4M3 bytes (pinned host).
        table_scale: float32 [table_rows] per-row scale (pinned host).
        rows: int64 [tokens * num_heads] global row ids (CUDA).
        out: bf16 [tokens * num_heads, head_dim] (CUDA), dequantized rows.
    """
    module = _jit_qwen_exo_engram_module(num_heads=num_heads, head_dim=table_data.shape[1])
    module.gather_dequant(table_data, table_scale, rows, out)
