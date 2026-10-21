"""Component benchmark only; does not load or start a model."""

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.sm120_gdn_output import fused_gdn_output
from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def separate(core, gate, norm, weight):
    normalized = layernorm_fn(
        core, norm, None, z=gate, eps=1e-6, group_size=None,
        norm_before_gate=True, is_rms_norm=True, activation="sigmoid",
    )
    return F.linear(normalized.reshape(-1, 6144), weight)


@marker.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8], [1, 4, 8])
@marker.parametrize("int8", [False, True])
@marker.benchmark("impl", ["fused", "norm+linear"])
def benchmark(rows: int, int8: bool, impl: str):
    if torch.cuda.get_device_capability() != (12, 0):
        return marker.skip("Requires SM120")
    core = torch.randn(rows * 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(core)
    norm = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    if int8:
        weight = torch.randint(-127, 128, (2560, 6144), device="cuda", dtype=torch.int8)
        scale = (torch.rand(2560, 48, device="cuda") * 0.0003 + 0.0001).to(torch.bfloat16)
    else:
        weight = torch.randn(2560, 6144, device="cuda", dtype=torch.bfloat16)
        scale = None
    if impl == "fused":
        return marker.do_bench(
            fused_gdn_output,
            input_args=(core, gate, norm, weight),
            input_kwargs={"eps": 1e-6, "weight_scale": scale},
        )
    # Deliberately exclude dequantization from the baseline timing: it measures
    # the original two-operator pipeline on the identical new quantized target.
    if int8:
        weight = (
            weight.reshape(2560, 48, 128).float() * scale.float().unsqueeze(-1)
        ).to(torch.bfloat16).reshape(2560, 6144)
    return marker.do_bench(separate, input_args=(core, gate, norm, weight))


if __name__ == "__main__":
    benchmark.run()
