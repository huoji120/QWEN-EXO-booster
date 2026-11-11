"""Packed versus padded Qwen4 PLE prefill (SiLU excluded from both paths)."""

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.qwen4_ple import fused_qwen4_varlen_conv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def _padded(x, weight, state, indices, starts, requests, offsets, lengths, width):
    selected = state.index_select(0, indices).to(x.dtype)
    padded = x.new_zeros((lengths.numel(), width, x.shape[1]))
    padded[requests, offsets] = x
    conv_input = torch.cat([selected, padded.transpose(1, 2)], dim=-1)
    output = F.conv1d(conv_input, weight, dilation=3, groups=x.shape[1]).transpose(1, 2)
    columns = torch.arange(state.shape[2], device=x.device)
    next_state = conv_input.gather(
        2,
        (lengths[:, None] + columns[None, :])[:, None, :].expand(-1, x.shape[1], -1),
    )
    state[indices] = next_state.to(state.dtype)
    return output[requests, offsets]


def _packed(x, weight, state, indices, starts, requests, offsets, lengths, width):
    return fused_qwen4_varlen_conv(x, weight, state, indices, starts, requests, offsets, 3)


@marker.parametrize(
    "lengths",
    [(8192, 1, 1, 1), (4096, 2048, 1024, 512), (2048, 2048, 2048, 2048)],
    [(2048, 1, 1, 1)],
)
@marker.benchmark("impl", ["packed", "padded"])
def benchmark(lengths, impl):
    channels = 10240
    x = torch.randn((sum(lengths), channels), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((channels, 1, 4), device="cuda", dtype=torch.bfloat16)
    state = torch.randn((len(lengths) + 1, channels, 9), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, len(lengths) + 1, device="cuda")
    lens = torch.tensor(lengths, device="cuda", dtype=torch.long)
    starts = torch.cat([lens.new_zeros(1), lens.cumsum(0)])
    requests = torch.repeat_interleave(torch.arange(len(lengths), device="cuda"), lens)
    offsets = torch.arange(sum(lengths), device="cuda") - starts[requests]
    return marker.do_bench(
        _packed if impl == "packed" else _padded,
        input_args=(x, weight, state, indices, starts, requests, offsets, lens, max(lengths)),
        # State is both read and written; clone it as well as activation inputs.
        graph_clone_args=(0, 1, 2, 3, 4, 5, 6, 7),
        # Both paths also write state; omit bandwidth rather than undercount it.
        memory_output=None,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
