# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the model-bound dense conversion and loader lifecycle."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.sm120_dense import (
    Sm120Int8FusedProjection,
    Sm120Int8LinearMethod,
    apply_sm120_int8_linear,
    dequantize_sm120_int8,
    is_sm120_dense_prefix,
    is_sm120_int8_linear,
    quantize_sm120_int8,
    sm120_dense_method,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSm120Dense(CustomTestCase):
    def test_group_tail_gain_and_scale_rounding(self):
        for k in (1, 127, 128, 129, 257):
            torch.manual_seed(k)
            weight = torch.randn(5, k, dtype=torch.bfloat16)
            weight[0] = 0
            gain = torch.tensor([0, 0.5, 1, -2, 4.0])
            q, scales = quantize_sm120_int8(weight, gain)
            self.assertEqual(q.shape, weight.shape)
            self.assertEqual(scales.shape, (5, (k + 127) // 128))
            expected = torch.empty_like(weight)
            for row in range(5):
                for first in range(0, k, 128):
                    v = weight[row, first:first + 128].float() * gain[row]
                    s = (v.abs().max() / 127).clamp_min(1e-30).bfloat16()
                    expected_q = (v / s.float()).round().clamp(-127, 127).to(torch.int8)
                    self.assertTrue(torch.equal(q[row, first:first + 128], expected_q))
                    self.assertTrue(torch.equal(scales[row, first // 128], s))
                    expected[row, first:first + 128] = (expected_q.float() * s.float()).bfloat16()
            self.assertTrue(torch.equal(dequantize_sm120_int8(q, scales), expected))

    def test_original_loader_then_conversion_and_bias(self):
        layer = torch.nn.Module()
        method = Sm120Int8LinearMethod()

        def loader(param, loaded, shard_id):
            # Mimic fused checkpoint gate/up loading without changing the
            # parameter name, dtype, input_dim/output_dim or loader identity.
            offset = shard_id * 3
            param.data[offset:offset + 3].copy_(loaded)

        method.create_weights(layer, 129, [3, 3], 129, 6, torch.bfloat16, weight_loader=loader)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertIs(layer.weight.weight_loader, loader)
        self.assertEqual((layer.weight.input_dim, layer.weight.output_dim), (1, 0))
        source = torch.randn(6, 129, dtype=torch.bfloat16)
        loader(layer.weight, source[:3], 0)
        loader(layer.weight, source[3:], 1)
        method.process_weights_after_loading(layer)
        self.assertTrue(is_sm120_int8_linear(layer))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertNotIn("original_weight", dict(layer.named_parameters()))
        old_weight = layer.weight
        method.process_weights_after_loading(layer)
        self.assertIs(layer.weight, old_weight)
        x = torch.randn(9, 129, dtype=torch.bfloat16)
        bias = torch.randn(6, dtype=torch.bfloat16)
        expected = F.linear(x, dequantize_sm120_int8(layer.weight, layer.sm120_weight_scale), bias)
        self.assertTrue(torch.equal(method.apply(layer, x, bias), expected))
        layer.sm120_group_size = 64
        self.assertFalse(is_sm120_int8_linear(layer))
        with self.assertRaisesRegex(ValueError, "weight/scale"):
            method.apply(layer, x)

    def test_scope_and_off_default(self):
        for prefix in ("lm_head", "model.layers.0.self_attn.qkv_proj", "model.layers.1.linear_attn.in_proj_qkvz", "model.layers.1.linear_attn.in_proj_ba", "model.layers.1.linear_attn.out_proj", "model.layers.0.mlp.shared_expert.gate_up_proj"):
            self.assertTrue(is_sm120_dense_prefix(prefix))
        for prefix in ("model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.gate", "model.hyper_connection_mixer.input_mix_weight_down", "mtp.layers.0.self_attn.qkv_proj", "visual.lm_head", "model.ple.key_proj"):
            self.assertFalse(is_sm120_dense_prefix(prefix))
        with patch("sglang.srt.layers.quantization.sm120_dense.sm120_fast_enabled", return_value=False):
            self.assertIsNone(sm120_dense_method("lm_head"))
        with patch("sglang.srt.layers.quantization.sm120_dense.sm120_fast_enabled", return_value=True):
            self.assertIsInstance(sm120_dense_method("lm_head"), Sm120Int8LinearMethod)
            self.assertIsNone(sm120_dense_method("mtp.lm_head"))

    def test_invalid_conversion_and_parallelism(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                quantize_sm120_int8(torch.full((1, 129), value, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "BF16"):
            quantize_sm120_int8(torch.ones(2, 128))
        with self.assertRaisesRegex(ValueError, "gain"):
            quantize_sm120_int8(torch.ones(2, 128, dtype=torch.bfloat16), torch.ones(3))
        with self.assertRaisesRegex(ValueError, "parallel"):
            Sm120Int8LinearMethod().create_weights(torch.nn.Module(), 64, [3], 128, 6, torch.bfloat16)

    def test_shared_head_marker_not_stale_method(self):
        q, s = quantize_sm120_int8(torch.randn(7, 129, dtype=torch.bfloat16))
        layer = SimpleNamespace(weight=q, sm120_weight_scale=s, sm120_group_size=128, quant_method=object())
        x = torch.randn(2, 129, dtype=torch.bfloat16)
        self.assertTrue(is_sm120_int8_linear(layer))
        expected = F.linear(x, dequantize_sm120_int8(q, s))
        self.assertTrue(torch.equal(apply_sm120_int8_linear(layer, x), expected))

    def test_native_fused_checkpoint_loader(self):
        from sglang.srt.layers.linear import MergedColumnParallelLinear

        config = SimpleNamespace(get_quant_method=lambda layer, prefix: Sm120Int8LinearMethod())
        layer = MergedColumnParallelLinear(
            input_size=129, output_sizes=[3, 3], bias=False,
            params_dtype=torch.bfloat16, quant_config=config, tp_rank=0, tp_size=1,
        )
        source = torch.randn(6, 129, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, source[:3], 0)
        layer.weight.weight_loader(layer.weight, source[3:], 1)
        self.assertTrue(torch.equal(layer.weight, source))
        layer.quant_method.process_weights_after_loading(layer)
        self.assertTrue(is_sm120_int8_linear(layer))

    def test_fused_projection_postprocess_storage_and_numerics(self):
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                layers = [torch.nn.Module(), torch.nn.Module()]
                method = Sm120Int8LinearMethod()
                originals = []
                for layer, n in zip(layers, (17, 6)):
                    method.create_weights(layer, 129, [n], 129, n, torch.bfloat16)
                    source = torch.randn(n, 129, dtype=torch.bfloat16)
                    layer.weight.data.copy_(source)
                    originals.append(quantize_sm120_int8(source))
                fused = Sm120Int8FusedProjection(*layers)
                self.assertIsNone(fused.weight)
                method.process_weights_after_loading(layers[order[0]])
                self.assertIsNone(fused.weight)
                method.process_weights_after_loading(layers[order[1]])
                self.assertTrue(is_sm120_int8_linear(fused))
                for layer, (q, scales) in zip(layers, originals):
                    self.assertTrue(torch.equal(layer.weight, q))
                    self.assertTrue(torch.equal(layer.sm120_weight_scale, scales))
                    self.assertEqual(
                        layer.weight.untyped_storage().data_ptr(),
                        fused.weight.untyped_storage().data_ptr(),
                    )
                    self.assertEqual(
                        layer.sm120_weight_scale.untyped_storage().data_ptr(),
                        fused.sm120_weight_scale.untyped_storage().data_ptr(),
                    )
                    self.assertFalse(hasattr(layer, "_sm120_fused_projection"))
                self.assertEqual(layers[1].weight.storage_offset(), 17 * 129)
                self.assertEqual(layers[1].sm120_weight_scale.storage_offset(), 17 * 2)
                self.assertEqual(fused.weight.untyped_storage().nbytes(), 23 * 129)
                self.assertEqual(fused.sm120_weight_scale.untyped_storage().nbytes(), 23 * 2 * 2)
                expected_weight = torch.cat([
                    dequantize_sm120_int8(q, scales) for q, scales in originals
                ])
                weight_ptr = fused.weight.data_ptr()
                with patch("torch.cat", side_effect=AssertionError("Repeated merge")), patch(
                    "sglang.srt.layers.quantization.sm120_dense.quantize_sm120_int8",
                    side_effect=AssertionError("Repeated quantization"),
                ):
                    fused.finalize()
                    for layer in layers:
                        method.process_weights_after_loading(layer)
                    for rows in (1, 8, 9):
                        x = torch.randn(rows, 129, dtype=torch.bfloat16)
                        actual = apply_sm120_int8_linear(fused, x)
                        self.assertTrue(torch.equal(actual, F.linear(x, expected_weight)))
                self.assertEqual(fused.weight.data_ptr(), weight_ptr)

    def test_fused_projection_installed_after_conversion(self):
        layers = [torch.nn.Module(), torch.nn.Module()]
        method = Sm120Int8LinearMethod()
        for layer, n in zip(layers, (5, 3)):
            method.create_weights(layer, 128, [n], 128, n, torch.bfloat16)
            layer.weight.data.normal_()
            method.process_weights_after_loading(layer)
        before = [(layer.weight.clone(), layer.sm120_weight_scale.clone()) for layer in layers]
        with patch(
            "sglang.srt.layers.quantization.sm120_dense.quantize_sm120_int8",
            side_effect=AssertionError("Requantized existing INT8 weights"),
        ):
            fused = Sm120Int8FusedProjection(*layers)
        self.assertTrue(is_sm120_int8_linear(fused))
        for layer, (weight, scale) in zip(layers, before):
            self.assertTrue(torch.equal(layer.weight, weight))
            self.assertTrue(torch.equal(layer.sm120_weight_scale, scale))

    def test_actual_logits_processor_with_shared_stale_method(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        q, scales = quantize_sm120_int8(torch.randn(17, 129, dtype=torch.bfloat16))
        head = SimpleNamespace(weight=q, sm120_weight_scale=scales, sm120_group_size=128, quant_method=object())
        x = torch.randn(4, 129, dtype=torch.bfloat16)
        processor = object.__new__(LogitsProcessor)
        actual = processor._compute_lm_head(x, head)
        expected = F.linear(x, dequantize_sm120_int8(q, scales))
        self.assertTrue(torch.equal(actual, expected))


    def test_modelopt_preserves_native_fp8_and_nvfp4(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization import modelopt_quant as modelopt

        layer = object.__new__(LinearBase)
        config = SimpleNamespace(
            exclude_modules=[], packed_modules_mapping={},
            is_layer_excluded=lambda prefix: False,
            fp8_config=object(), fp8_block_config=object(), nvfp4_config=object(),
            nvfp4a16_config=object(),
        )
        with patch("sglang.srt.layers.quantization.sm120_dense.sm120_fast_enabled", return_value=True):
            for algo, constructor, native_config in (
                ("FP8", "ModelOptFp8LinearMethod", config.fp8_config),
                ("FP8_BLOCK_SCALES", "Fp8LinearMethod", config.fp8_block_config),
                ("NVFP4", "ModelOptFp4LinearMethod", config.nvfp4_config),
                ("W4A16_NVFP4", "ModelOptNvFp4A16LinearMethod", config.nvfp4a16_config),
            ):
                config._resolve_quant_algo = lambda prefix, algo=algo: algo
                with patch.object(modelopt, constructor) as native:
                    actual = modelopt.ModelOptMixedPrecisionConfig.get_quant_method(config, layer, "model.layers.0.self_attn.qkv_proj")
                    self.assertIs(actual, native.return_value)
                    native.assert_called_once_with(native_config)
            config._resolve_quant_algo = lambda prefix: None
            actual = modelopt.ModelOptMixedPrecisionConfig.get_quant_method(config, layer, "model.layers.0.self_attn.qkv_proj")
            self.assertIsInstance(actual, Sm120Int8LinearMethod)

    def test_marlin_profile_rejects_close_but_unequal_scales(self):
        from sglang.srt.layers.quantization import modelopt_quant as modelopt

        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        scales = torch.tensor([[1.0, 1.0 + 1e-6], [0.25, 0.25]])
        self.assertTrue(torch.allclose(scales[:, 0], scales[:, 1]))
        layer.register_parameter("w13_weight_scale_2", torch.nn.Parameter(scales, requires_grad=False))
        backend = SimpleNamespace(is_marlin=lambda: True)
        method = SimpleNamespace(_moe_runner_backend=backend)
        with patch.object(modelopt, "get_moe_runner_backend", return_value=backend), patch.object(modelopt, "sm120_fast_enabled", return_value=True), patch.object(modelopt, "prepare_moe_nvfp4_layer_for_marlin") as prepare:
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                modelopt.ModelOptNvFp4FusedMoEMethod.process_weights_after_loading(method, layer)
            prepare.assert_not_called()
            self.assertEqual(layer.w13_weight_scale_2.shape, (2, 2))
            layer.w13_weight_scale_2.data[:, 1] = layer.w13_weight_scale_2.data[:, 0]
            modelopt.ModelOptNvFp4FusedMoEMethod.process_weights_after_loading(method, layer)
            prepare.assert_called_once_with(layer)
            self.assertEqual(layer.w13_weight_scale_2.shape, (2,))


if __name__ == "__main__":
    unittest.main()
