"""Warm-weight, eager-dispatch full-HC comparison on SM120; not serving throughput."""

from functools import lru_cache

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.layers.hc_prefill_triton import hc_prefill_mix
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@lru_cache(maxsize=1)
def original_mix():
    return GatedResidual(
        HyperConnectionConfig(hc_count=4, hidden_size=2560, hc_lowrank=320),
        use_mix=False,
        use_combine=False,
    )._mix_compute


def baseline(x, down, up):
    return original_mix()(x, down, up, 4, 2560)


@marker.parametrize("rows", [1024], [1024])
@marker.benchmark("implementation", ["compiled", "fused"])
def benchmark(rows, implementation):
    if torch.cuda.get_device_capability() != (12, 0):
        marker.skip("HC prefill specialization requires SM120")
    torch.manual_seed(271027)
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(320, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(10240, 320, device="cuda", dtype=torch.bfloat16) * 0.02
    with torch.inference_mode():
        expected = baseline(x, down, up)
        actual = hc_prefill_mix(x, down, up)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
        return marker.do_bench(
            baseline if implementation == "compiled" else hc_prefill_mix,
            input_args=(x, down, up), use_cuda_graph=False,
            graph_clone_args=None, graph_clone_kwargs=None,
        )


if __name__ == "__main__":
    benchmark.run()
