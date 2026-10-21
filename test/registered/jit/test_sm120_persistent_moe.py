import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.benchmark.sm120_moe import (
    make_inputs,
    make_native_payload,
    native_moe,
)
from sglang.jit_kernel.sm120_persistent_moe import (
    PersistentMoeScratch,
    _scratch,
    persistent_moe,
    supports_persistent_moe,
    use_persistent_moe_capture_scratch,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "persistent NVFP4 MoE is an SM120 component",
)
class TestSm120PersistentMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(120)
        cls.quant = make_native_payload()
        cls.config = MoeRunnerConfig()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "quant"):
            del cls.quant

    def assert_native(self, x, ids, weights, **kwargs):
        actual = persistent_moe(x, ids, weights, self.quant, **kwargs)
        expected = native_moe(x, ids, weights, self.quant)
        # Split-K reduction order differs from general Marlin; both use the
        # same prepared target weights, BF16 boundaries and FP32 accumulation.
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-2)
        self.assertTrue(torch.isfinite(actual).all().item())
        return actual

    def test_loader_repack_reuses_storage_without_changing_expert_layout(self):
        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales, prepare_moe_nvfp4_layer_for_marlin,
        )

        generator = torch.Generator(device="cuda").manual_seed(121)
        for gated in (True, False):
            layer = torch.nn.Module()
            layer.quant_config = SimpleNamespace(group_size=16)
            layer.moe_runner_config = MoeRunnerConfig(is_gated=gated)
            layer.intermediate_size_per_partition = 256
            layer.params_dtype = torch.bfloat16
            expected, pointers = {}, {}
            for name, n, k in (
                ("w13", 512 if gated else 256, 128),
                ("w2", 128, 256),
            ):
                packed = torch.randint(0, 256, (3, n, k // 2),
                    device="cuda", dtype=torch.uint8, generator=generator)
                scales = torch.randint(0, 16, (3, n, k // 16),
                    device="cuda", generator=generator).float().mul_(1 / 16).to(torch.float8_e4m3fn)
                layer.register_parameter(name + "_weight", torch.nn.Parameter(packed, requires_grad=False))
                layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scales, requires_grad=False))
                layer.register_parameter(name + "_weight_scale_2", torch.nn.Parameter(
                    torch.full((3,), 0.001, device="cuda"), requires_grad=False))
                pointers[name] = packed.data_ptr(), scales.data_ptr()
                perm = torch.empty(0, dtype=torch.int, device="cuda")
                expected[name] = torch.stack([
                    gptq_marlin_repack(packed[i].view(torch.int32).T.contiguous(),
                        perm, k, n, 4) for i in range(3)
                ])
                expected[name + "_scale"] = torch.stack([
                    nvfp4_marlin_process_scales(marlin_permute_scales(
                        scales[i].T.contiguous().bfloat16(), k, n, 16))
                    for i in range(3)
                ])
            prepare_moe_nvfp4_layer_for_marlin(layer)
            for name in ("w13", "w2"):
                weight, scale = getattr(layer, name + "_weight"), getattr(layer, name + "_weight_scale")
                self.assertEqual((weight.data_ptr(), scale.data_ptr()), pointers[name])
                torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)
                self.assertTrue(torch.equal(scale.view(torch.uint8), expected[name + "_scale"].view(torch.uint8)))

    def test_native_loader_rowcounts_duplicates_padding_scales(self):
        for rows in range(1, 9):
            for pattern in ("unique", "duplicates", "padding", "all_padding"):
                with self.subTest(rows=rows, pattern=pattern):
                    x, ids, weights = make_inputs(rows, pattern)
                    self.assertTrue(supports_persistent_moe(
                        x, ids, weights, self.quant, self.config,
                    ))
                    self.assert_native(x, ids, weights)

    def test_dirty_scratch_reset_and_deterministic_sum(self):
        scratch = PersistentMoeScratch.allocate(torch.device("cuda"))
        x, ids, weights = make_inputs(8, "duplicates")
        scratch.control.fill_(123456)
        scratch.partial.fill_(float("nan"))
        scratch.hidden.fill_(float("nan"))
        scratch.weighted.fill_(float("nan"))
        first = self.assert_native(x, ids, weights, scratch=scratch)
        for _ in range(5):
            other = persistent_moe(x, ids, weights, self.quant, scratch=scratch)
            torch.testing.assert_close(other, first, atol=0, rtol=0)
        # Reusing the same scratch for a wholly padded row must not return stale
        # output or count an invalid expert toward the completion barrier.
        x, ids, weights = make_inputs(1, "all_padding")
        actual = self.assert_native(x, ids, weights, scratch=scratch)
        self.assertEqual(actual.count_nonzero().item(), 0)

    def test_concurrent_graph_scratch_isolation(self):
        inputs = [make_inputs(4, "duplicates"), make_inputs(8, "padding")]
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        expected = [native_moe(*args, self.quant) for args in inputs]
        # Compile and populate eager cache before capturing either graph.
        for args in inputs:
            persistent_moe(*args, self.quant)
        torch.cuda.synchronize()
        outputs = []
        for graph, stream, args in zip(graphs, streams, inputs):
            with torch.cuda.graph(graph, stream=stream):
                outputs.append(persistent_moe(*args, self.quant))
        for _ in range(5):
            for graph, stream in zip(graphs, streams):
                with torch.cuda.stream(stream):
                    graph.replay()
        torch.cuda.synchronize()
        for actual, oracle in zip(outputs, expected):
            torch.testing.assert_close(actual, oracle, atol=2e-3, rtol=3e-2)

    def test_capture_arena_scope_restoration_and_device(self):
        device = torch.device("cuda", torch.cuda.current_device())
        outer = PersistentMoeScratch.allocate(device)
        inner = PersistentMoeScratch.allocate(device)
        eager = _scratch(device)
        with use_persistent_moe_capture_scratch(outer):
            self.assertIs(_scratch(device), outer)
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with use_persistent_moe_capture_scratch(inner):
                    self.assertIs(_scratch(device), inner)
                    raise RuntimeError("capture failed")
            self.assertIs(_scratch(device), outer)
            wrong_device = dataclasses.replace(outer, control=torch.empty(928))
            with use_persistent_moe_capture_scratch(wrong_device):
                with self.assertRaisesRegex(ValueError, "device mismatch"):
                    _scratch(device)
            self.assertIs(_scratch(device), outer)
        self.assertIs(_scratch(device), eager)

    def test_graph_arena_reused_by_sequential_layers(self):
        inputs = [make_inputs(4, "duplicates"), make_inputs(4, "padding")]
        expected = [native_moe(*args, self.quant) for args in inputs]
        arena = PersistentMoeScratch.allocate(inputs[0][0].device)
        for args in inputs:
            persistent_moe(*args, self.quant)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with use_persistent_moe_capture_scratch(arena):
            with torch.cuda.graph(graph):
                outputs = [persistent_moe(*args, self.quant) for args in inputs]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        for actual, oracle in zip(outputs, expected):
            torch.testing.assert_close(actual, oracle, atol=2e-3, rtol=3e-2)

    def test_runner_profile_and_large_row_fallback(self):
        from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin

        for enabled, rows in ((True, 4), (True, 9), (False, 4)):
            with self.subTest(enabled=enabled, rows=rows):
                x, ids, weights = make_inputs(rows, "duplicates")
                dispatch = SimpleNamespace(
                    hidden_states=x,
                    topk_output=SimpleNamespace(
                        topk_ids=ids, topk_weights=weights,
                        router_logits=torch.empty((rows, 512), device="cuda"),
                    ),
                )
                with patch(
                    "sglang.srt.runtime_context.get_server_args",
                    return_value=SimpleNamespace(qwen_exo_sm120_fast=enabled),
                ):
                    actual = fused_experts_none_to_marlin(
                        dispatch, self.quant, self.config,
                    ).hidden_states
                torch.testing.assert_close(
                    actual, native_moe(x, ids, weights, self.quant),
                    atol=2e-3, rtol=3e-2,
                )

    def test_metadata_qualification(self):
        x, ids, weights = make_inputs(4, "duplicates")
        for kwargs in (
            {"activation": "relu2"}, {"is_gated": False},
            {"gemm1_alpha": 1.7}, {"gemm1_clamp_limit": 7},
            {"swiglu_limit": 7}, {"routed_scaling_factor": 0.5},
            {"apply_router_weight_on_input": True},
        ):
            self.assertFalse(supports_persistent_moe(
                x, ids, weights, self.quant,
                dataclasses.replace(self.config, **kwargs),
            ))
        for kwargs in (
            {"w13_global_scale": None}, {"weight_bits": 8},
            {"w13_bias": torch.zeros(512, 1280, device="cuda")},
            {"w13_scales": self.quant.w13_scales[:, :, :1279]},
            {"expert_map": torch.arange(512, device="cuda")},
        ):
            self.assertFalse(supports_persistent_moe(
                x, ids, weights, dataclasses.replace(self.quant, **kwargs), self.config,
            ))
        self.assertFalse(supports_persistent_moe(
            x, ids.long(), weights, self.quant, self.config,
        ))
        x, ids, weights = make_inputs(9, "duplicates")
        self.assertFalse(supports_persistent_moe(
            x, ids, weights, self.quant, self.config,
        ))


if __name__ == "__main__":
    unittest.main()
