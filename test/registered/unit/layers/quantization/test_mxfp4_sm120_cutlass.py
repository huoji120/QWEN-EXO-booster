"""SM120 FlashInfer MXFP8-by-MXFP4 MoE integration test."""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


def _random_weights(num_experts: int, hidden: int, intermediate: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randint(
        -128,
        128,
        (num_experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        -128,
        128,
        (num_experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w13_scale_u8 = torch.randint(
        125,
        130,
        (num_experts, 2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2_scale_u8 = torch.randint(
        125,
        130,
        (num_experts, hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    return (
        w13,
        w2,
        w13_scale_u8.view(torch.float8_e8m0fnu),
        w2_scale_u8.view(torch.float8_e8m0fnu),
    )


def test_cutlass_adapter_import_does_not_require_flashinfer(monkeypatch):
    module_name = "sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe"
    # Load the package before blocking FlashInfer so this test isolates the
    # adapter import exercised by non-CUDA backends.
    importlib.import_module("sglang.srt.layers.quantization")
    cached_module = sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def import_without_flashinfer(name, *args, **kwargs):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ModuleNotFoundError("No module named 'flashinfer'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flashinfer)
    try:
        module = importlib.import_module(module_name)
        assert hasattr(module, "Mxfp4FlashinferCutlassMoEMethod")
    finally:
        sys.modules.pop(module_name, None)
        if cached_module is not None:
            sys.modules[module_name] = cached_module


@pytest.mark.parametrize(
    "mode,chunked,max_prefill,dp_size,pp_size,dynamic,num_tokens,expected",
    [
        ("null", 1024, 16384, 1, 1, False, 1, 1024),
        ("null", 1024, 16384, 1, 1, False, 777, 1024),
        ("null", 1024, 16384, 1, 1, False, 1024, 1024),
        ("prefill", 1024, 16384, 4, 1, False, 33, 4096),
        ("null", 4096, 16384, 4, 1, False, 33, 8192),
        ("null", 0, 6144, 1, 1, False, 33, 6144),
        ("null", -1, 16384, 1, 1, False, 33, 8192),
        ("null", 1024, 4096, 1, 2, True, 33, 4096),
        ("null", 1024, 1024, 1, 2, True, 33, 1280),
        ("null", 1024, 4096, 1, 1, True, 33, 1024),
        ("null", 16384, 16384, 1, 1, False, 8192, 8192),
        ("null", 16384, 16384, 1, 1, False, 8193, 8193),
        ("null", 1024, 16384, 1, 1, False, 1025, 1025),
        ("decode", 1024, 16384, 4, 1, False, 33, 64),
        ("decode", 1024, 16384, 4, 1, False, 8193, 16384),
    ],
)
def test_cutlass_tuning_ceiling(
    monkeypatch,
    mode,
    chunked,
    max_prefill,
    dp_size,
    pp_size,
    dynamic,
    num_tokens,
    expected,
):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as runner_module
    from sglang.srt.server_args import ServerArgs

    # Use the real buffer sizing policy without model/device initialization.
    args = ServerArgs.__new__(ServerArgs)
    args.disaggregation_mode = mode
    args.chunked_prefill_size = chunked
    args.max_prefill_tokens = max_prefill
    args.pp_size = pp_size
    args.enable_dynamic_chunking = dynamic
    monkeypatch.setattr(runner_module, "get_server_args", lambda: args)

    def attention_dp_size():
        assert mode != "decode", "PD decode must not consult prefill topology"
        return dp_size

    monkeypatch.setattr(runner_module, "get_attention_dp_size", attention_dp_size)
    ceiling = runner_module._cutlass_moe_tune_max_num_tokens(num_tokens)
    assert ceiling == expected
    assert ceiling >= num_tokens


def test_dsv4_sm120_matches_direct_flashinfer(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 required")
    pytest.importorskip("flashinfer.fused_moe")

    from flashinfer import block_scale_interleave, mxfp8_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as runner_module
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )

    monkeypatch.setattr(
        runner_module, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(runner_module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "get_server_args",
        lambda: SimpleNamespace(
            disaggregation_mode="null", max_prefill_buffer_tokens=lambda: 8192
        ),
    )
    monkeypatch.setattr(runner_module, "get_attention_dp_size", lambda: 1)

    num_experts, hidden, intermediate = 4, 256, 256
    w13, w2, w13_scale, w2_scale = _random_weights(num_experts, hidden, intermediate)
    w1, w3 = w13.chunk(2, dim=1)
    w1_scale, w3_scale = w13_scale.chunk(2, dim=1)
    # Simulate FusedMoE's ``load_up_proj_weight_first`` loader contract.
    w31 = torch.cat((w3, w1), dim=1)
    w31_scale = torch.cat(
        (w3_scale.view(torch.uint8), w1_scale.view(torch.uint8)),
        dim=1,
    ).view(torch.float8_e8m0fnu)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w31.clone(), requires_grad=False),
        w2_weight=torch.nn.Parameter(w2.clone(), requires_grad=False),
        w13_weight_scale_inv=torch.nn.Parameter(w31_scale.clone(), requires_grad=False),
        w2_weight_scale_inv=torch.nn.Parameter(w2_scale.clone(), requires_grad=False),
        num_local_experts=num_experts,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )

    method = Mxfp4FlashinferCutlassMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        activation="silu",
        is_gated=True,
        swiglu_limit=10,
    )
    method.create_moe_runner(layer, config)

    w13_parameter = layer.w13_weight
    w2_parameter = layer.w2_weight
    w13_scale_parameter = layer.w13_weight_scale_inv
    w2_scale_parameter = layer.w2_weight_scale_inv
    method.process_weights_after_loading(layer)

    expected_w13_scale = block_scale_interleave(w31_scale.view(torch.uint8)).reshape_as(
        w31_scale
    )
    expected_w2_scale = block_scale_interleave(w2_scale.view(torch.uint8)).reshape_as(
        w2_scale
    )
    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    assert layer.w13_weight_scale_inv is w13_scale_parameter
    assert layer.w2_weight_scale_inv is w2_scale_parameter
    assert torch.equal(layer.w13_weight_scale_inv.view(torch.uint8), expected_w13_scale)
    assert torch.equal(layer.w2_weight_scale_inv.view(torch.uint8), expected_w2_scale)

    generator = torch.Generator(device="cuda").manual_seed(1)
    x = (
        torch.randn(
            8,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    logits = torch.randn(
        8,
        num_experts,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk = StandardTopKOutput(topk_weights, topk_ids.to(torch.int32), logits)
    dispatch_output = StandardDispatchOutput(x, None, topk)

    actual = method.apply(layer, dispatch_output).hidden_states

    x_quant, x_scale = mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
    )
    global_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    swiglu_limit = torch.full((num_experts,), 10.0, dtype=torch.float32, device="cuda")
    expected = torch.empty_like(x)
    cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=layer.w13_weight.view(torch.int64),
        fc2_expert_weights=layer.w2_weight.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[
            layer.w13_weight_scale_inv.view(torch.int32),
            global_scale,
            layer.w2_weight_scale_inv.view(torch.int32),
            global_scale,
        ],
        input_sf=x_scale,
        # Compare the adapter's implicit defaults against the old explicit
        # alpha=1/beta=0 representation.
        swiglu_alpha=torch.ones(num_experts, dtype=torch.float32, device="cuda"),
        swiglu_beta=torch.zeros(num_experts, dtype=torch.float32, device="cuda"),
        swiglu_limit=swiglu_limit,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=8,
        output=expected,
    )

    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
