"""Compare the full old dense-buffer path against register-only chain sampling."""

import torch
import sgl_kernel  # noqa: F401  # Registers the dense reference torch operator.

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.speculative.chain_target_only import (
    chain_speculative_sampling_target_only,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def _sample(
    predicts,
    accept_index,
    counts,
    candidates,
    indices,
    next_token,
    sibling,
    coins,
    final_coins,
    probs,
    impl,
):
    if impl == "compact":
        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            counts,
            candidates,
            indices,
            next_token,
            coins,
            final_coins,
            probs,
            1.0,
            1.0,
        )
    else:
        # Include the allocation/zero fill removed from production. Never reuse
        # a dirty dense buffer: the reference kernel mutates rejected entries.
        draft_probs = torch.zeros_like(probs)
        args = (
            predicts,
            accept_index,
            counts,
            candidates,
            indices,
            next_token,
            sibling,
            coins,
            final_coins,
            probs,
            draft_probs,
            1.0,
            1.0,
            True,
        )
        op = torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default
        if len(op._schema.arguments) >= 15:
            args += (torch.cuda.current_stream().cuda_stream,)
        op(*args)


@marker.parametrize("bs", [1, 10])
@marker.parametrize("width", [2, 4, 8], [4])
@marker.parametrize("vocab", [32000, 248320], [248320])
@marker.parametrize("outcome", ["all_accept", "first_reject", "last_reject"])
@marker.benchmark("impl", ["old_dense", "compact"])
def benchmark(bs: int, width: int, vocab: int, outcome: str, impl: str):
    # CPU seed and provided coins ensure both implementations see identical
    # inputs across marker columns, independent of benchmark execution order.
    generator = torch.Generator().manual_seed(1729)
    probs = torch.rand(bs, width, vocab, generator=generator) + 0.01
    probs /= probs.sum(dim=-1, keepdim=True)
    probs = probs.cuda()
    candidates = torch.randint(vocab, (bs, width), generator=generator).cuda()
    indices = torch.arange(bs * width, device="cuda").reshape(bs, width)
    next_token = torch.arange(1, width + 1, device="cuda").repeat(bs, 1)
    next_token[:, -1] = -1
    sibling = torch.full_like(next_token, -1)
    coins = torch.zeros(bs, width, device="cuda")
    if outcome == "first_reject":
        coins[:, 0] = 1.0
    elif outcome == "last_reject":
        coins[:, -2] = 1.0
    final_coins = torch.full((bs,), 0.53125, device="cuda")
    predicts = torch.zeros(bs * width, dtype=torch.int32, device="cuda")
    accept_index = torch.full((bs, width), -1, dtype=torch.int32, device="cuda")
    counts = torch.empty(bs, dtype=torch.int32, device="cuda")
    return marker.do_bench(
        _sample,
        input_args=(
            predicts,
            accept_index,
            counts,
            candidates,
            indices,
            next_token,
            sibling,
            coins,
            final_coins,
            probs,
            impl,
        ),
        graph_clone_args=(3, 4, 5, 6, 7, 8, 9),
        memory_output=(predicts, accept_index, counts),
    )


if __name__ == "__main__":
    benchmark.run()
