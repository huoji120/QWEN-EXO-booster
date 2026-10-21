"""Offline component benchmark; does not load or launch a model."""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.sm120_sampling import (
    prepare_target_distribution,
    sample_truncated_proposal,
    sorted_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def torch_proposal(logits, temperatures, ks, ps):
    values, ids = torch.topk(logits, 64, dim=-1, sorted=True)
    ranks = torch.arange(64, device=logits.device)
    values = values / temperatures
    values = values.masked_fill(ranks[None, :] >= ks[:, None], -float("inf"))
    weights = values.softmax(-1)
    cdf = weights.cumsum(-1)
    crossing = (cdf >= ps[:, None]).int().argmax(-1)
    pivot = weights.gather(1, crossing[:, None])
    weights = weights.masked_fill(weights < pivot, 0)
    weights /= weights.sum(-1, keepdim=True)
    q = torch.zeros_like(logits).scatter_(1, ids, weights)
    noise = torch.empty_like(weights).exponential_().clamp_min_(torch.finfo(torch.float32).tiny)
    tokens = ids.gather(1, (weights / noise).argmax(-1, keepdim=True))
    return q, q.gather(1, tokens), tokens


def torch_target(logits, info, depth):
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    p = (logits / info.temperatures.repeat_interleave(depth, 0)).softmax(-1)
    p = top_k_renorm_prob(p, info.top_ks.repeat_interleave(depth))
    return top_p_renorm_prob(p, info.top_ps.repeat_interleave(depth))


@marker.parametrize("batch", [1, 2, 8, 10], [1, 10])
@marker.parametrize("phase", ["topk", "proposal", "target"])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(batch: int, phase: str, impl: str):
    from types import SimpleNamespace

    depth = 4 if phase == "target" else 1
    logits = torch.randn((batch * depth, 248320), device="cuda")
    temps = torch.full((batch, 1), 0.8, device="cuda")
    ks = torch.full((batch,), 40, dtype=torch.int32, device="cuda")
    ps = torch.full((batch,), 0.9, device="cuda")
    info = SimpleNamespace(temperatures=temps, top_ks=ks, top_ps=ps)
    if phase == "topk":
        fn = sorted_topk if impl == "jit" else lambda x: torch.topk(x, 64, dim=-1)
        args = (logits,)
    elif phase == "proposal":
        fn = sample_truncated_proposal if impl == "jit" else torch_proposal
        args = (logits, temps, ks, ps)
    else:
        fn = prepare_target_distribution if impl == "jit" else torch_target
        args = (logits, info, depth)
    return marker.do_bench(
        fn, input_args=args, memory_output=None, graph_clone_args=(0,),
        memory_args=(logits,),
    )


if __name__ == "__main__":
    benchmark.run()
