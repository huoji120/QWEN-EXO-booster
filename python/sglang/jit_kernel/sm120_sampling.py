"""Model-bound SM120 proposal and exact complete-tie verification kernels.

The proposal has at most 64 *full-vocabulary* entries, not a static token map.
The verifier's sparse representation is used only when the device proves that
it includes every target cutoff tie. Ineligible requests compute a full dense
p and use the existing exact rejection kernel, without CPU synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "sm120_exact_sampling",
        *args,
        cuda_files=["sampling/sm120_distribution.cuh"],
        cuda_wrappers=[
            ("topk", f"sm120_topk<{args}>"),
            ("distribution", "sm120_distribution"),
            ("chain", "sm120_sparse_chain"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def sorted_topk(logits: torch.Tensor, k: int = 64):
    """Reusable sorted top-k; cutoff tie membership is arbitrary, never assumed complete."""
    if logits.ndim != 2 or not logits.is_cuda:
        raise ValueError("sorted_topk requires a two-dimensional CUDA tensor")
    if not 1 <= k <= min(64, logits.shape[1]):
        raise ValueError("sorted_topk requires 1 <= k <= min(64, vocab)")
    rows, vocab = logits.shape
    segment = (vocab + 63) // 64
    if (
        logits.dtype not in (torch.float32, torch.bfloat16)
        or not k <= segment <= 4096
    ):
        values, ids = torch.topk(logits.float(), k, dim=-1, sorted=True)
        return values, ids
    logits = logits.contiguous()
    values = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    keys = torch.empty((rows, 64, k), dtype=torch.int32, device=logits.device)
    candidate_ids = torch.empty_like(keys)
    _module(logits.dtype).topk(logits, values, ids, keys, candidate_ids)
    return values, ids


@dataclass
class TargetDistribution:
    dense: torch.Tensor
    sparse: torch.Tensor
    ids: torch.Tensor
    eligible: torch.Tensor


def _distribution(logits, temperatures, top_ks, top_ps, depth, proposal, expo=None):
    logits = logits.float().contiguous()
    rows, vocab = logits.shape
    if vocab < 64 or rows % depth or not 1 <= depth <= 8:
        raise ValueError("sampling requires vocab >= 64 and request width 1..8")
    batch = rows // depth
    temperatures = temperatures.reshape(-1).float().contiguous()
    top_ks = top_ks.reshape(-1).to(torch.int32).contiguous()
    top_ps = top_ps.reshape(-1).float().contiguous()
    if any(t.numel() != batch for t in (temperatures, top_ks, top_ps)):
        raise ValueError("one temperature/top-k/top-p value is required per request")
    values, ids = sorted_topk(logits)
    sparse = torch.empty_like(values)
    dense = torch.zeros_like(logits) if proposal else torch.empty_like(logits)
    eligible = torch.empty((batch,), dtype=torch.int32, device=logits.device)
    # The target branch never writes proposal outputs; pass contiguous views
    # of existing workspace rather than allocate two unused result tensors.
    px = (
        torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
        if proposal else values.reshape(-1)[:rows].view(rows, 1)
    )
    tokens = (
        torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
        if proposal else ids.reshape(-1)[:rows].view(rows, 1)
    )
    # Proposal coins belong to the graph's registered generator. Never generate
    # them on a host stream or reuse captured tensor contents between replays.
    if expo is None:
        expo = torch.empty_like(values).exponential_() if proposal else values
    _module(torch.float32).distribution(
        logits, values, ids, temperatures, top_ks, top_ps, sparse, dense,
        eligible, expo, px, tokens, depth, proposal,
    )
    return TargetDistribution(dense, sparse.view(batch, depth, 64),
                              ids.view(batch, depth, 64), eligible), px, tokens


def sample_truncated_proposal(logits, temperatures, top_ks, top_ps, expo=None):
    """Return the actual dense q, q(X), and X drawn from bounded draft top-k/p.

    top_k <= 0 means the draft's own 64-candidate cap. Target truncation is
    independent and never capped. The returned q owns its storage.
    """
    distribution, px, tokens = _distribution(
        logits, temperatures, top_ks, top_ps, 1, True, expo
    )
    return distribution.dense, px, tokens


def prepare_target_distribution(logits, sampling_info, depth):
    distribution, _, _ = _distribution(
        logits, sampling_info.temperatures, sampling_info.top_ks,
        sampling_info.top_ps, depth, False,
    )
    return distribution


def exact_chain_sampling(
    distribution, *, predicts, accept_index, accept_token_num, candidates,
    retrieve_index, retrieve_next_token, retrieve_next_sibling,
    uniform_samples, uniform_samples_for_final_sampling, draft_probs,
):
    """Sparse verification plus device-gated full-dense exact fallback."""
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )

    batch, depth = candidates.shape
    vocab = distribution.dense.shape[-1]
    if (
        draft_probs is None or draft_probs.ndim != 3
        or draft_probs.shape[0] != batch or draft_probs.shape[1] < depth - 1
        or draft_probs.shape[-1] != vocab
    ):
        raise ValueError("exact verification requires the actual full-vocabulary q")
    # Both kernels write disjoint requests. Neither reads a row from the other
    # representation: eligible rows of dense and ineligible sparse are undefined.
    _module(torch.float32).chain(
        distribution.sparse, distribution.ids, draft_probs.contiguous(),
        distribution.eligible, candidates.to(torch.int64).contiguous(),
        retrieve_index.to(torch.int64).contiguous(), uniform_samples.contiguous(),
        uniform_samples_for_final_sampling, predicts, accept_index, accept_token_num,
    )
    chain_speculative_sampling_triton(
        predicts=predicts, accept_index=accept_index, accept_token_num=accept_token_num,
        candidates=candidates, retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token, retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=distribution.dense.view(batch, depth, vocab),
        draft_probs=draft_probs, threshold_single=1.0, threshold_acc=1.0,
        deterministic=True, skip_requests=distribution.eligible,
    )
