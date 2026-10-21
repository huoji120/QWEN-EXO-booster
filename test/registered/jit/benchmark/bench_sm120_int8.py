# SPDX-License-Identifier: Apache-2.0
"""Isolated SM120 dense-kernel latency, not an end-to-end speed claim."""
import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.sm120_int8 import prepare_sm120_int8, sm120_int8_gemm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@marker.parametrize("m", [1, 2, 4, 8], [1, 4])
@marker.parametrize("n,k", [(16480, 2560), (13312, 2560), (2560, 6144), (1280, 2560), (2560, 640), (248320, 2560)], [(16480, 2560), (2560, 640)])
@marker.benchmark("impl", ["int8-mma", "bf16-cublas"])
def benchmark(m, n, k, impl):
    if torch.cuda.get_device_capability() != (12, 0):
        return marker.skip("SM120-only preset")
    prepare_sm120_int8(k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.full((n, k // 128), 0.001, device="cuda", dtype=torch.bfloat16)
    if impl == "int8-mma":
        return marker.do_bench(sm120_int8_gemm, input_args=(x, q, scales))
    weight = (q.float() * scales.float().repeat_interleave(128, 1)).bfloat16()
    return marker.do_bench(F.linear, input_args=(x, weight))


if __name__ == "__main__":
    benchmark.run()
