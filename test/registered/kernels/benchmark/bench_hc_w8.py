import torch

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.elementwise.hc_w8 import (
    hc_combine_mix_w8,
    hc_combine_w8,
    hc_mix_fused_w8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

HC, H, K, R = 4, 2560, 10240, 320


def _w8(weight, group):
    grouped = weight.float().reshape(weight.shape[0], -1, group)
    scale = (grouped.abs().amax(-1) / 127).clamp_min(1e-30).to(torch.bfloat16)
    q = torch.round(grouped / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape_as(weight).contiguous(), scale.contiguous()


def _make(rows):
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
    ip = torch.randn(rows, HC, device="cuda", dtype=torch.float32)
    wn = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    wd, wds = _w8(torch.randn(R, K, device="cuda", dtype=torch.bfloat16) * 0.02, 128)
    wu, wus = _w8(torch.randn(K, R, device="cuda", dtype=torch.bfloat16) * 0.02, 64)
    wi = torch.randn(HC, K, device="cuda", dtype=torch.bfloat16) * 0.02
    return x, y, ip, wn, wd, wds, wi, wu, wus


@marker.parametrize("rows", [1, 2, 4, 8], [1, 4])
@marker.benchmark("path", ["mix", "combine_mix"])
def benchmark(rows: int, path: str):
    x, y, ip, wn, wd, wds, wi, wu, wus = _make(rows)
    if path == "mix":
        fn = hc_mix_fused_w8
        args = (x, wn, wd, wds, wi, wu, wus, 1e-6)
    else:
        fn = hc_combine_mix_w8
        args = (y, x, ip, wn, wd, wds, wi, wu, wus, 1e-6)
    return marker.do_bench(fn, input_args=args, graph_clone_args=tuple(range(len(args) - 1)))


def _eager_combine(y, x, ip):
    gates = 2.0 * torch.sigmoid(ip / HC)
    return (
        x.unflatten(-1, (HC, H)) + y.unsqueeze(-2) * gates.unsqueeze(-1)
    ).flatten(-2).to(torch.bfloat16)


@marker.parametrize("rows", list(range(1, 9)), [1, 4, 8])
@marker.benchmark("path", ["combine_eager", "combine_native"])
def benchmark_combine(rows: int, path: str):
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
    ip = torch.randn(rows, HC, device="cuda", dtype=torch.float32)
    fn = hc_combine_w8 if path == "combine_native" else _eager_combine
    return marker.do_bench(fn, input_args=(y, x, ip), graph_clone_args=(0, 1, 2))


if __name__ == "__main__":
    benchmark.run()
    benchmark_combine.run()
