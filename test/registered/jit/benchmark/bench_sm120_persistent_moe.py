"""Component latency only: prepared native W4A16 vs persistent routed MoE.

Does not load or launch a model and makes no end-to-end decode-speed claim.
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.sm120_moe import (
    make_inputs,
    make_native_payload,
    native_moe,
)
from sglang.jit_kernel.sm120_persistent_moe import persistent_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@marker.parametrize("rows", [1, 2, 4, 8])
@marker.parametrize("routing", ["unique", "duplicates"])
@marker.benchmark("impl", ["native-marlin", "persistent"], unit="us")
def benchmark(rows, routing, impl):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        return marker.skip("requires SM120")
    quant = make_native_payload()
    x, ids, weights = make_inputs(rows, routing)
    operation = persistent_moe if impl == "persistent" else native_moe

    def run(x, ids, weights):
        return operation(x, ids, weights, quant)

    return marker.do_bench(
        run,
        input_args=(x, ids, weights),
        graph_clone_args=(0, 1, 2),
        memory_args=(
            x, ids, weights, quant.w13_qweight, quant.w2_qweight,
            quant.w13_scales, quant.w2_scales,
        ),
        # Full expert storage is not read per call, so its total allocation is
        # not a valid bandwidth numerator; report latency, not synthetic GB/s.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
