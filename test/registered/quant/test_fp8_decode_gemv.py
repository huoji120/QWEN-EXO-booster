import unittest

import torch

from sglang.kernels.ops.quantization.fp8_decode_gemv import (
    FP8_DECODE_GEMV_MAX_ROWS,
    can_use_fp8_decode_gemv,
    fp8_decode_gemv,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def _operands(rows, n, k):
    """``weight`` is the fp8_scaled_mm operand: (K, N) viewing (N, K) storage."""
    weight = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn).t()
    weight_scale = (torch.rand(n, 1, device="cuda") * 0.01 + 1e-3).float()
    qinput = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    x_scale = (torch.rand(rows, 1, device="cuda") * 0.01 + 1e-3).float()
    return qinput, weight, x_scale, weight_scale


@unittest.skipUnless(torch.cuda.is_available() and is_sm120_supported(), "SM120 only")
class TestFp8DecodeGemv(CustomTestCase):
    def test_matches_cutlass_scaled_mm_including_tails(self):
        """Both decode kernels (GEMV for 1-2 rows, skinny MMA for 3-16 rows,
        whose rows are padded to 16) must be drop-ins for fp8_scaled_mm,
        including N and K that are not multiples of the kernel tiles (masked
        tails). fp8_scaled_mm itself requires N and K to be multiples of 16."""
        from sgl_kernel import fp8_scaled_mm

        torch.manual_seed(0)
        for n, k in [(5120, 17408), (16384, 5120), (5120 + 48, 6144 + 272)]:
            for rows in range(1, FP8_DECODE_GEMV_MAX_ROWS + 1):
                qinput, weight, x_scale, weight_scale = _operands(rows, n, k)
                self.assertTrue(
                    can_use_fp8_decode_gemv(
                        qinput, weight, x_scale, weight_scale, torch.bfloat16
                    )
                )
                expected = fp8_scaled_mm(
                    qinput, weight, x_scale, weight_scale, torch.bfloat16, None
                )
                actual = fp8_decode_gemv(qinput, weight, x_scale, weight_scale)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)

    def test_rejects_operands_outside_the_kernel_contract(self):
        qinput, weight, x_scale, weight_scale = _operands(2, 512, 512)
        wide, wide_weight, wide_scale, _ = _operands(
            FP8_DECODE_GEMV_MAX_ROWS + 1, 512, 512
        )
        cases = {
            "more rows than accumulators": (
                wide,
                wide_weight,
                wide_scale,
                weight_scale,
                torch.bfloat16,
            ),
            "fp16 output": (qinput, weight, x_scale, weight_scale, torch.float16),
            "bf16 activations": (
                qinput.to(torch.bfloat16),
                weight,
                x_scale,
                weight_scale,
                torch.bfloat16,
            ),
            "per-tensor activation scale": (
                qinput,
                weight,
                x_scale[:1],
                weight_scale,
                torch.bfloat16,
            ),
            "row-major (K, N) weight storage": (
                qinput,
                weight.contiguous(),
                x_scale,
                weight_scale,
                torch.bfloat16,
            ),
            "per-tensor weight scale": (
                qinput,
                weight,
                x_scale,
                weight_scale[:1],
                torch.bfloat16,
            ),
        }
        for name, args in cases.items():
            with self.subTest(name):
                self.assertFalse(can_use_fp8_decode_gemv(*args))


if __name__ == "__main__":
    unittest.main()
