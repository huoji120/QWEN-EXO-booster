"""Numerical and graph checks for the model-bound SM120 GDN output kernel."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.sm120_gdn_output import (
    fused_gdn_output,
    try_fused_gdn_output,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires SM120; no model weights or running inference service needed",
)


@pytest.fixture(scope="module", params=[False, True], ids=["bf16", "int8-g128"])
def weights(request):
    generator = torch.Generator(device="cuda").manual_seed(207)
    norm = torch.randn(128, generator=generator, device="cuda", dtype=torch.bfloat16)
    if request.param:
        weight = torch.randint(
            -127, 128, (2560, 6144), generator=generator, device="cuda", dtype=torch.int8
        )
        scale = (
            torch.rand(2560, 48, generator=generator, device="cuda") * 0.0003 + 0.0001
        ).to(torch.bfloat16)
        dequantized = (
            weight.reshape(2560, 48, 128).float() * scale.float().unsqueeze(-1)
        ).to(torch.bfloat16).reshape(2560, 6144)
    else:
        weight = (
            torch.randn(2560, 6144, generator=generator, device="cuda") * 0.02
        ).to(torch.bfloat16)
        scale, dequantized = None, weight
    return norm, weight, scale, dequantized


def reference(core, gate, norm, weight, eps, activation):
    # Independent per-head FP32 normalization and gate, then the same BF16
    # materialization boundary as RMSNormGated. Do not use the fused wrapper.
    x, z = core.float(), gate.float()
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    gate_value = torch.sigmoid(z) if activation == "sigmoid" else F.silu(z)
    normalized = (normalized * norm.float() * gate_value).to(torch.bfloat16)
    return F.linear(normalized.reshape(-1, 6144), weight)


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_gdn_output_numerics(weights, rows, activation, eps):
    norm, weight, scale, dequantized = weights
    core = torch.randn(rows * 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(core)
    # Includes zeros representing the existing DP-attention padding path.
    core[-17:] = 0
    expected = reference(core, gate, norm, dequantized, eps, activation)
    actual = fused_gdn_output(
        core, gate, norm, weight, eps=eps, activation=activation, weight_scale=scale
    )
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)
    assert actual.shape == (rows, 2560)
    assert actual.dtype == torch.bfloat16


def test_graph_shared_scratch_lifetime(weights):
    norm, weight, scale, dequantized = weights
    inputs = [
        (torch.randn(m * 48, 128, device="cuda", dtype=torch.bfloat16),
         torch.randn(m * 48, 128, device="cuda", dtype=torch.bfloat16))
        for m in (1, 8)
    ]
    outputs = [torch.empty(m, 2560, device="cuda", dtype=torch.bfloat16) for m in (1, 8)]

    def run():
        for (core, gate), out in zip(inputs, outputs):
            assert fused_gdn_output(
                core, gate, norm, weight, eps=1e-6, weight_scale=scale, out=out
            ) is out

    # Compile/configure outside capture. Capture both row counts and replay with
    # changed inputs, catching scratch/state accidentally shared across kernels.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        for core, gate in inputs:
            core.normal_()
            gate.normal_()
        graph.replay()
        for (core, gate), actual in zip(inputs, outputs):
            expected = reference(core, gate, norm, dequantized, 1e-6, "sigmoid")
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)


@pytest.mark.parametrize(
    "rows,overrides",
    [(9, {}), (10, {}), (1, {"enabled": False}), (1, {"tp_size": 2}),
     (1, {"norm_before_gate": False}), (1, {"group_size": 64}),
     (1, {"activation": "relu"}), (1, {"norm_bias": True}),
     (1, {"projection_bias": True})],
)
def test_unsupported_dispatch_preserves_normal_fallback(weights, rows, overrides):
    norm_weight, weight, scale, dequantized = weights
    core = torch.randn(rows * 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(core)
    norm = SimpleNamespace(
        weight=norm_weight, bias=None, eps=1e-6, group_size=None,
        norm_before_gate=True, activation="sigmoid",
    )
    projection = SimpleNamespace(weight=weight, bias=None)
    if scale is not None:
        # Large/unsupported rows decline before querying the shared marker API.
        projection.sm120_weight_scale = scale
        projection.sm120_group_size = 128
    for key in ("norm_before_gate", "group_size", "activation"):
        if key in overrides:
            setattr(norm, key, overrides[key])
    if overrides.get("norm_bias"):
        norm.bias = torch.zeros_like(norm_weight)
    if overrides.get("projection_bias"):
        projection.bias = torch.zeros(2560, device="cuda", dtype=torch.bfloat16)
    actual = try_fused_gdn_output(
        core, gate, norm, projection,
        enabled=overrides.get("enabled", True),
        tp_size=overrides.get("tp_size", 1),
        projection_is_unquantized=scale is None,
    )
    assert actual is None
    # The fallback target is the new dequantized INT8 weight, not saved BF16.
    assert torch.isfinite(reference(core, gate, norm_weight, dequantized, 1e-6, "sigmoid")).all()


def test_invalid_scale_and_rows(weights):
    norm, weight, scale, _ = weights
    core = torch.randn(48, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires group scales"):
        fused_gdn_output(
            core, core, norm, weight, eps=1e-6,
            weight_scale=None if scale is not None else norm,
        )
    with pytest.raises(ValueError, match="1..8 rows"):
        fused_gdn_output(
            core.repeat(9, 1), core.repeat(9, 1), norm, weight,
            eps=1e-6, weight_scale=scale,
        )


@pytest.mark.parametrize("rows", [1, 8, 9, 10])
def test_native_gdn_forward_and_padding(weights, rows):
    from unittest.mock import patch

    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    norm_weight, weight, scale, dequantized = weights
    core = torch.randn(rows * 48 - 17, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.bias = None
            self.quant_method = UnquantizedLinearMethod()
            self.calls = 0
            if scale is not None:
                self.register_buffer("sm120_weight_scale", scale)
                self.sm120_group_size = 128

        def forward(self, x):
            self.calls += 1
            return F.linear(x, dequantized), None

    class Producer(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return core

    # Exercise the real native forward method, but do not construct a model or
    # allocate a recurrent cache. Only its input/recurrence producers are fixed.
    layer = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.num_v_heads, layer.num_k_heads = 48, 16
    layer.attn_tp_size, layer.hidden_size = 1, 2560
    layer._sm120_fused_output = True
    layer.head_k_dim = layer.head_v_dim = 128
    projected_qkvz = torch.zeros(rows, 16384, device="cuda", dtype=torch.bfloat16)
    projected_qkvz[:, 10240:] = gate.flatten(1)
    projected_ba = torch.zeros(rows, 96, device="cuda", dtype=torch.bfloat16)
    layer._forward_input_proj = lambda hidden: (projected_qkvz, projected_ba)
    layer.attn = Producer()
    layer.norm = RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.bfloat16, activation="sigmoid")
    layer.norm.weight.data.copy_(norm_weight)
    layer.out_proj = Projection()
    with patch(
        "sglang.srt.models.qwen3_5.try_fused_gdn_output", wraps=try_fused_gdn_output
    ) as dispatch:
        actual = layer(torch.empty(rows, 2560, device="cuda", dtype=torch.bfloat16), None)
    dispatch.assert_called_once()
    assert layer.out_proj.calls == (0 if rows <= 8 else 1)
    padded = torch.zeros(rows * 48, 128, device="cuda", dtype=torch.bfloat16)
    padded[:core.shape[0]].copy_(core)
    expected = reference(padded, gate.reshape(-1, 128), norm_weight, dequantized, 1e-6, "sigmoid")
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
