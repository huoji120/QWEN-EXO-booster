"""Synthetic native-loader fixture shared by SM120 MoE tests and marker bench."""

from types import SimpleNamespace

import torch


def make_native_payload():
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    # Prepare a small set through the actual ModelOpt->Marlin implementation,
    # then replicate its *prepared* layout to the model's physical expert count.
    # No compressed-tensors scale convention enters this fixture.
    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = MoeRunnerConfig(is_gated=True)
    layer.intermediate_size_per_partition = 640
    layer.params_dtype = torch.bfloat16
    shapes = {
        "w13_weight": (8, 1280, 1280),
        "w2_weight": (8, 2560, 320),
    }
    for name, shape in shapes.items():
        layer.register_parameter(name, torch.nn.Parameter(
            torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8),
            requires_grad=False,
        ))
    for name, shape in (
        ("w13_weight_scale", (8, 1280, 160)),
        ("w2_weight_scale", (8, 2560, 40)),
    ):
        # Include zero block scales as well as nonuniform exact FP8 values.
        value = torch.randint(0, 16, shape, device="cuda").float().mul_(1 / 16)
        layer.register_parameter(name, torch.nn.Parameter(
            value.to(torch.float8_e4m3fn), requires_grad=False,
        ))
    for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
        layer.register_parameter(name, torch.nn.Parameter(
            torch.linspace(0.001, 0.008, 8, device="cuda"), requires_grad=False,
        ))
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight.detach().repeat(64, 1, 1),
        w2_qweight=layer.w2_weight.detach().repeat(64, 1, 1),
        w13_scales=layer.w13_weight_scale.detach().repeat(64, 1, 1),
        w2_scales=layer.w2_weight_scale.detach().repeat(64, 1, 1),
        w13_global_scale=layer.w13_weight_scale_2.detach().repeat(64),
        w2_global_scale=layer.w2_weight_scale_2.detach().repeat(64),
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        global_num_experts=512,
    )


def native_moe(x, ids, weights, quant):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

    # Native Marlin's expert alignment need not understand arbitrary invalid
    # IDs. Its equivalent zero-weight valid slots are the padding oracle.
    valid = (ids >= 0) & (ids < 512)
    return fused_marlin_moe(
        hidden_states=x,
        w1=quant.w13_qweight, w2=quant.w2_qweight,
        w1_scale=quant.w13_scales, w2_scale=quant.w2_scales,
        w1_global_scale=quant.w13_global_scale,
        w2_global_scale=quant.w2_global_scale,
        gating_output=torch.empty((x.shape[0], 512), device=x.device),
        topk_ids=ids.masked_fill(~valid, 0),
        topk_weights=weights.masked_fill(~valid, 0),
        global_num_experts=512,
        num_bits=4,
        inplace=False,
    )


def make_inputs(rows, pattern):
    x = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, 512, (rows, 10), device="cuda", dtype=torch.int32)
    if pattern == "duplicates":
        ids[:] = torch.tensor([511, 0, 511, 7, 0, 8, 511, 8, 7, 0], device="cuda")
    elif pattern == "padding":
        ids[:, ::2] = -1
        ids[:, 1] = 512
        ids[-1] = -1
    elif pattern == "all_padding":
        ids[:] = -1
    elif pattern == "unique":
        ids[:] = torch.arange(rows * 10, device="cuda", dtype=torch.int32).view(rows, 10)
    weights = torch.rand((rows, 10), device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    # Include precisely zero and tiny nonzero weights with nonuniform globals.
    weights[:, 0] = 0
    weights[:, -1] = 2**-20
    return x, ids, weights
