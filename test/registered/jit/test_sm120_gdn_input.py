"""Native INT8 GDN input projection, strided split, and CUDA graph regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.srt.layers.quantization.sm120_dense import (
    Sm120Int8FusedProjection,
    Sm120Int8LinearMethod,
    dequantize_sm120_int8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


class _Projection(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.quant_method = Sm120Int8LinearMethod()
        self.bias = None
        self.calls = 0
        with torch.device("cuda"):
            self.quant_method.create_weights(
                self, 2560, [width], 2560, width, torch.bfloat16
            )
        self.weight.data.normal_(std=0.02)

    def forward(self, x):
        self.calls += 1
        return self.quant_method.apply(self, x), None


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "Requires SM120; no model checkpoint or server needed",
)
class TestSm120GdnInput(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        torch.manual_seed(3305)
        cls.model = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(cls.model)
        cls.model.in_proj_qkvz = _Projection(16384)
        cls.model.in_proj_ba = _Projection(96)
        cls.model._fused_in_proj_weight = None
        cls.model._fused_input_proj_cpu_enabled = SimpleNamespace(value=False)
        cls.model.alt_stream = None
        cls.model._sm120_fused_in_proj = Sm120Int8FusedProjection(
            cls.model.in_proj_qkvz, cls.model.in_proj_ba
        )
        for layer in (cls.model.in_proj_qkvz, cls.model.in_proj_ba):
            layer.quant_method.process_weights_after_loading(layer)
        cls.reference_weights = [
            dequantize_sm120_int8(layer.weight, layer.sm120_weight_scale)
            for layer in (cls.model.in_proj_qkvz, cls.model.in_proj_ba)
        ]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "reference_weights"):
            del cls.reference_weights
        if hasattr(cls, "model"):
            del cls.model
        super().tearDownClass()

    def _reference_split(self, qkvz, ba):
        q, k, v, z = qkvz.split((2048, 2048, 6144, 6144), dim=-1)
        b, a = ba.split((48, 48), dim=-1)
        return torch.cat((q, k, v), dim=-1), z.reshape(-1, 48, 128), b, a

    def _project_and_split(self, x):
        qkvz, ba = self.model._forward_input_proj(x)
        return fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, 16, 48, 128, 128)

    def test_small_rows_exact_native_parity_and_strides(self):
        for rows in range(1, 9):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                qkvz, ba = self.model._forward_input_proj(x)
                self.assertEqual(qkvz.stride(), (16480, 1))
                self.assertEqual(ba.stride(), (16480, 1))
                self.assertEqual(ba.storage_offset(), 16384)
                expected = [
                    layer.quant_method.apply(layer, x)
                    for layer in (self.model.in_proj_qkvz, self.model.in_proj_ba)
                ]
                for actual, separate, weight in zip((qkvz, ba), expected, self.reference_weights):
                    torch.testing.assert_close(actual, separate, atol=0, rtol=0)
                    torch.testing.assert_close(actual, F.linear(x, weight), atol=0.015, rtol=0.02)
                actual_split = fused_qkvzba_split_reshape_cat_contiguous(
                    qkvz, ba, 16, 48, 128, 128
                )
                for actual, reference in zip(actual_split, self._reference_split(*expected)):
                    self.assertTrue(actual.is_contiguous())
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_large_rows_keep_individual_projections(self):
        from sglang.srt.models import qwen3_5

        for rows in (9, 10, 64):
            with self.subTest(rows=rows), patch.object(
                qwen3_5, "check_cuda_graph_backend", return_value=False
            ), patch.object(
                qwen3_5, "apply_sm120_int8_linear",
                side_effect=AssertionError("Large-row combined GEMM"),
            ):
                x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                before = (self.model.in_proj_qkvz.calls, self.model.in_proj_ba.calls)
                actual = self.model._forward_input_proj(x)
                self.assertEqual(
                    (self.model.in_proj_qkvz.calls, self.model.in_proj_ba.calls),
                    tuple(n + 1 for n in before),
                )
                for output, weight in zip(actual, self.reference_weights):
                    torch.testing.assert_close(output, F.linear(x, weight), atol=0.015, rtol=0.02)

    def test_ratio_three_native_forward_numeric_dispatch(self):
        from sglang.srt.models import qwen3_5

        model = self.model
        model.num_k_heads = 16
        model.num_v_heads = 48
        model.attn_tp_size = 1
        model.head_k_dim = model.head_v_dim = 128
        model._sm120_fused_output = False
        # Isolate input projection/splitting from recurrent attention and output
        # projection, while checking the actual native forward call's values.
        observed = {}

        def attention(forward_batch, *, mixed_qkv, a, b):
            observed.update(qkv=mixed_qkv, a=a, b=b)
            return mixed_qkv[:, 4096:].reshape(-1, 128)

        model.attn = attention
        model.norm = lambda core, z: core + z
        model.out_proj = lambda x: (x, None)
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        projected = model._forward_input_proj(x)
        qkv, z, b, a = self._reference_split(*projected)
        with patch.object(
            qwen3_5, "fused_qkvzba_split_reshape_cat_contiguous",
            wraps=fused_qkvzba_split_reshape_cat_contiguous,
        ) as splitter:
            output = model.forward(x, None)
        splitter.assert_called_once()
        for key, expected in (("qkv", qkv), ("a", a), ("b", b)):
            torch.testing.assert_close(observed[key], expected, atol=0, rtol=0)
        torch.testing.assert_close(output, qkv[:, 4096:] + z.flatten(1), atol=0, rtol=0)

    def test_graph_replay_uses_changed_inputs_without_remerge(self):
        inputs = [
            torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            for rows in (1, 8)
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for x in inputs:
                self._project_and_split(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with patch("torch.cat", side_effect=AssertionError("Hot-path merge")):
            with torch.cuda.graph(graph):
                outputs = [self._project_and_split(x) for x in inputs]
        for _ in range(3):
            for x in inputs:
                x.normal_()
            graph.replay()
            for x, actual_split in zip(inputs, outputs):
                separate = [
                    layer.quant_method.apply(layer, x)
                    for layer in (self.model.in_proj_qkvz, self.model.in_proj_ba)
                ]
                for actual, reference in zip(actual_split, self._reference_split(*separate)):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
