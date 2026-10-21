from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi import Module


@cache_once
def _chain_target_only_module(vec_size: int) -> Module:
    args = make_cpp_args(vec_size)
    return load_jit(
        "chain_target_only",
        *args,
        cuda_files=["speculative/chain_target_only.cuh"],
        cuda_wrappers=[("sample", f"chain_target_only<{args}>")],
        extra_dependencies=["flashinfer"],
        # Match the AOT sgl-kernel build's architecture-specific math policy.
        extra_cuda_cflags=(
            ["--use_fast_math"] if get_jit_cuda_arch().major < 10 else []
        ),
    )


def chain_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
) -> None:
    """Deterministic CUDA target-only sampling for a statically known topk=1 chain.

    Mutates the same three outputs as the generic tree sampler, with the same
    provided coins and acceptance/sampling arithmetic. The caller must provide
    valid chain links, candidate token IDs and flattened retrieve indices. No
    sibling proposals are supported; the production caller selects this path
    from tree_topk, never by reading GPU metadata. Rejected proposal mass is
    represented in registers instead of a zero-initialized [B, W, V] tensor.
    """
    # Shape/device/dtype/contiguity and integer-offset limits are checked by the
    # FFI TensorMatcher before launching; no tensor contents move to the host.
    if target_probs.ndim != 3:
        raise ValueError("target_probs must have shape [batch, width, vocab]")
    if not target_probs.is_cuda or torch.version.hip is not None:
        raise ValueError("chain target-only sampling requires CUDA")
    vec_size = math.gcd(4, target_probs.shape[-1])
    _chain_target_only_module(vec_size).sample(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        threshold_single,
        threshold_acc,
    )
