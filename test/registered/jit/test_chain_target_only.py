"""Provided-coin equivalence to the dense CUDA target-only tree sampler."""

import pytest
import torch
import sgl_kernel  # noqa: F401  # Registers the dense reference torch operator.

from sglang.kernels.ops.speculative.chain_target_only import (
    chain_speculative_sampling_target_only,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _inputs(bs, width, vocab, case):
    generator = torch.Generator().manual_seed(1729 + bs + width + vocab)
    probs = torch.rand(bs, width, vocab, generator=generator) + 0.01
    probs /= probs.sum(dim=-1, keepdim=True)
    candidates = torch.randint(vocab, (bs, width), generator=generator)
    if case in ("duplicate", "duplicate_last_reject"):
        candidates[:] = candidates[:, :1]
    if case == "zero_mass":
        probs.zero_()
    elif case == "sparse_tail":
        probs.zero_()
        probs[..., -1] = 0.75
        probs[..., 0] += 0.25
    # Traverse nontrivial local node order and write nontrivial flat indices.
    order = [0, *range(width - 1, 0, -1)]
    next_token = torch.full((bs, width), -1, dtype=torch.int64)
    for node, successor in zip(order, order[1:]):
        next_token[:, node] = successor
    retrieve_index = torch.arange(bs * width).reshape(bs, width).flip(1)
    sibling = torch.full_like(next_token, -1)
    coins = torch.zeros(bs, width)
    if case in ("first_reject", "zero_mass", "sparse_tail"):
        coins[:, 0] = 1.0
    elif case in ("last_reject", "duplicate_last_reject") and width > 1:
        coins[:, order[-2]] = 1.0
    elif case == "random":
        coins = torch.rand(bs, width, generator=generator)
    # Exercise both the zero-CDF and near-one fallback boundary without draws
    # inside either implementation.
    final_coins = torch.linspace(0, 1, bs) if bs > 1 else torch.tensor([0.99999994])
    tensors = (
        candidates,
        retrieve_index,
        next_token,
        sibling,
        coins,
        final_coins,
        probs,
    )
    return tuple(t.cuda() for t in tensors)


def _outputs(bs, width):
    return (
        torch.full((bs * width,), -123, dtype=torch.int32, device="cuda"),
        torch.full((bs, width), -1, dtype=torch.int32, device="cuda"),
        torch.empty(bs, dtype=torch.int32, device="cuda"),
    )


def _dense(outputs, inputs, threshold_single, threshold_acc):
    candidates, indices, next_token, sibling, coins, final_coins, probs = inputs
    args = (
        *outputs,
        candidates,
        indices,
        next_token,
        sibling,
        coins,
        final_coins,
        probs,
        torch.zeros_like(probs),
        threshold_single,
        threshold_acc,
        True,
    )
    op = torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default
    if len(op._schema.arguments) >= 15:
        args += (torch.cuda.current_stream().cuda_stream,)
    op(*args)


def _compact(outputs, inputs, threshold_single, threshold_acc):
    candidates, indices, next_token, _, coins, final_coins, probs = inputs
    chain_speculative_sampling_target_only(
        *outputs,
        candidates,
        indices,
        next_token,
        coins,
        final_coins,
        probs,
        threshold_single,
        threshold_acc,
    )


@pytest.mark.parametrize("bs", [1, 10])
@pytest.mark.parametrize("width", [1, 2, 4, 7])
@pytest.mark.parametrize("vocab", [1, 31, 130, 1024, 4100, 4101])
def test_provided_coins_match_dense(bs, width, vocab):
    for case in (
        "all_accept",
        "first_reject",
        "last_reject",
        "duplicate",
        "duplicate_last_reject",
        "zero_mass",
        "sparse_tail",
        "random",
    ):
        inputs = _inputs(bs, width, vocab, case)
        for single, acc in ((1.0, 1.0), (0.2, 0.37), (0.0, 0.0), (1.0, 0.0)):
            dense, compact = _outputs(bs, width), _outputs(bs, width)
            _dense(dense, inputs, single, acc)
            _compact(compact, inputs, single, acc)
            for actual, expected in zip(compact, dense):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if case in ("all_accept", "duplicate") or single == 0:
                torch.testing.assert_close(
                    compact[2], torch.full_like(compact[2], width - 1)
                )
            if case == "first_reject" and vocab > 1 and (single, acc) == (1, 1):
                torch.testing.assert_close(compact[2], torch.zeros_like(compact[2]))
            if (
                case in ("last_reject", "duplicate_last_reject")
                and vocab > 1
                and (single, acc) == (1, 1)
            ):
                torch.testing.assert_close(
                    compact[2], torch.full_like(compact[2], max(width - 2, 0))
                )


@pytest.mark.parametrize("bs,width,vocab", [(1, 1, 31), (1, 4, 4101), (10, 4, 4100)])
def test_graph_replay_and_rng_preservation(bs, width, vocab):
    inputs = _inputs(bs, width, vocab, "random")
    compact = _outputs(bs, width)
    # Warm up JIT outside capture, as in the server's graph warmup.
    _compact(compact, inputs, 1.0, 0.37)
    torch.cuda.synchronize()
    rng = torch.cuda.get_rng_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compact(compact, inputs, 1.0, 0.37)
    for coin in (0.0, 1.0, 0.53125):
        inputs[4].fill_(coin)
        inputs[5].fill_(1 - coin)
        dense = _outputs(bs, width)
        _dense(dense, inputs, 1.0, 0.37)
        # Reset outputs too: graph replay must not retain prior accepted slots.
        compact[0].fill_(-123)
        compact[1].fill_(-1)
        graph.replay()
        for actual, expected in zip(compact, dense):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@pytest.mark.parametrize("acc", [0.0, 0.37, 1.0])
def test_acceptance_coin_boundaries(acc):
    inputs = _inputs(10, 4, 130, "random")
    candidates, _, _, _, coins, _, probs = inputs
    boundaries = torch.zeros_like(coins)
    for node, successor in ((0, 3), (3, 2), (2, 1)):
        token = candidates[:, successor : successor + 1]
        boundaries[:, node] = (
            probs[:, node].gather(1, token).squeeze(1) / max(acc, 1e-9)
        ).clamp(max=1)
    for direction in (-float("inf"), None, float("inf")):
        if direction is None:
            coins.copy_(boundaries)
        else:
            coins.copy_(
                torch.nextafter(boundaries, torch.full_like(boundaries, direction))
            )
        dense, compact = _outputs(10, 4), _outputs(10, 4)
        _dense(dense, inputs, 1.0, acc)
        _compact(compact, inputs, 1.0, acc)
        for actual, expected in zip(compact, dense):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["prob_dtype", "index_dtype", "output_shape", "stride", "threshold"]
)
def test_metadata_validation(invalid):
    inputs = list(_inputs(1, 4, 130, "random"))
    outputs = list(_outputs(1, 4))
    single = 1.0
    if invalid == "prob_dtype":
        inputs[-1] = inputs[-1].half()
    elif invalid == "index_dtype":
        inputs[1] = inputs[1].int()
    elif invalid == "output_shape":
        outputs[0] = outputs[0][:-1]
    elif invalid == "stride":
        inputs[-1] = inputs[-1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        single = -0.1
    with pytest.raises((RuntimeError, ValueError)):
        _compact(outputs, inputs, single, 1.0)


def test_cpu_input_rejected():
    inputs = _inputs(1, 4, 31, "random")
    inputs = (*inputs[:-1], inputs[-1].cpu())
    with pytest.raises(ValueError, match="CUDA"):
        _compact(_outputs(1, 4), inputs, 1.0, 1.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
