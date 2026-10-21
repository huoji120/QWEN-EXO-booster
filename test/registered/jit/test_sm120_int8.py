# SPDX-License-Identifier: Apache-2.0
"""SM120 MMA/unpack oracle and CUDA-graph replay checks; never loads a model."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.jit_kernel.sm120_int8 import (
    prepare_sm120_int8,
    sm120_int8_gemm,
    sm120_int8_unpack,
)
from sglang.srt.layers.quantization.sm120_dense import apply_sm120_int8_linear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def independent_weight(q, s):
    k = q.shape[1]
    indices = torch.arange(k, device=q.device) // 128
    return (q.float() * s.float().index_select(1, indices)).bfloat16()


def dense_reference_diagnostics(actual, x, weight):
    """Compare the BF16 operands using three accumulation configurations.

    FP32 input GEMM with TF32 disabled is the independent accumulation oracle;
    ordinary BF16 cuBLAS can truncate intermediate reductions on this device.
    Backend controls are restored, including when an operation raises.
    """
    default_bf16 = F.linear(x, weight)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        fp32 = F.linear(x.float(), weight.float())
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        full_acc_bf16 = F.linear(x, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_reduced

    def errors(value, reference):
        difference = (value.float() - reference.float()).abs()
        relative = difference / reference.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
        return {"max_abs": difference.max().item(), "max_rel": relative.max().item()}

    diagnostics = {
        "M,N,K": (x.shape[0], weight.shape[0], weight.shape[1]),
        "bf16_reduced_precision_reduction": previous_reduced,
        "kernel_vs_fp32": errors(actual, fp32),
        "default_bf16_vs_fp32": errors(default_bf16, fp32),
        "full_acc_bf16_vs_fp32": errors(full_acc_bf16, fp32),
        "kernel_vs_default_bf16": errors(actual, default_bf16),
        "kernel_vs_full_acc_bf16": errors(actual, full_acc_bf16),
        "kernel_oracle_failures": (~torch.isclose(actual, fp32.bfloat16(), rtol=0.016, atol=0.016)).sum().item(),
        "default_bf16_oracle_failures": (~torch.isclose(default_bf16, fp32.bfloat16(), rtol=0.016, atol=0.016)).sum().item(),
        "full_acc_bf16_oracle_failures": (~torch.isclose(full_acc_bf16, fp32.bfloat16(), rtol=0.016, atol=0.016)).sum().item(),
    }
    print(f"SM120_DENSE_NUMERICS {diagnostics}", flush=True)
    # The kernel contract includes final BF16 rounding after FP32 accumulation.
    return fp32.bfloat16(), default_bf16, diagnostics


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "requires SM120")
class TestSm120Int8(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.generator = torch.Generator(device="cuda").manual_seed(120)

    def test_small_m_and_n_tails(self):
        for k in (640, 2560, 6144):
            prepare_sm120_int8(k)
            for n in (1, 15, 16, 33, 257):
                q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=self.generator)
                scales = (torch.rand(n, k // 128, device="cuda", generator=self.generator) * 0.002 + 0.0001).bfloat16()
                reference_weight = independent_weight(q, scales)
                self.assertTrue(torch.equal(sm120_int8_unpack(q, scales), reference_weight))
                for m in range(1, 9):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=self.generator)
                    actual = sm120_int8_gemm(x, q, scales)
                    expected, _, diagnostics = dense_reference_diagnostics(actual, x, reference_weight)
                    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016, msg=str(diagnostics))

    def test_unpack_group_tails_and_general_fallback(self):
        for k in (1, 127, 129, 257, 640):
            q = torch.randint(-127, 128, (33, k), device="cuda", dtype=torch.int8, generator=self.generator)
            s = (torch.rand(33, (k + 127) // 128, device="cuda", generator=self.generator) * 0.002 + 0.0001).bfloat16()
            expected_weight = independent_weight(q, s)
            self.assertTrue(torch.equal(sm120_int8_unpack(q, s), expected_weight))
            layer = SimpleNamespace(weight=q, sm120_weight_scale=s, sm120_group_size=128)
            for m in (4, 10):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=self.generator)
                bias = torch.randn(33, device="cuda", dtype=torch.bfloat16, generator=self.generator)
                torch.testing.assert_close(apply_sm120_int8_linear(layer, x, bias), F.linear(x, expected_weight, bias), rtol=0, atol=0)

    def test_cuda_graph_replays_updated_input(self):
        for k in (640, 2560, 6144):
            prepare_sm120_int8(k)
            q = torch.randint(-127, 128, (65, k), device="cuda", dtype=torch.int8, generator=self.generator)
            s = torch.full((65, k // 128), 0.001, device="cuda", dtype=torch.bfloat16)
            layer = SimpleNamespace(weight=q, sm120_weight_scale=s, sm120_group_size=128)
            for m in (1, 4, 8, 10):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=self.generator)
                apply_sm120_int8_linear(layer, x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = apply_sm120_int8_linear(layer, x)
                for _ in range(3):
                    x.normal_(generator=self.generator)
                    graph.replay()
                    expected, default_bf16, diagnostics = dense_reference_diagnostics(output, x, independent_weight(q, s))
                    if m <= 8:
                        torch.testing.assert_close(output, expected, rtol=0.016, atol=0.016, msg=str(diagnostics))
                    else:
                        # >8 rows intentionally use the ordinary BF16 cuBLAS
                        # fallback, including its configured reduction policy.
                        torch.testing.assert_close(output, default_bf16, rtol=0, atol=0, msg=str(diagnostics))


if __name__ == "__main__":
    unittest.main()
