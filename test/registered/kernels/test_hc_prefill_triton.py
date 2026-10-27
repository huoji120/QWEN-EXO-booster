"""SM120 HC prefill fusion against the original compiled BF16 computation."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.hc_prefill_triton import (
    hc_prefill_mix,
    hc_prefill_mix_supported,
)
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

HC_COUNT = 4
HIDDEN_SIZE = 2560
WIDTH = HC_COUNT * HIDDEN_SIZE
LOWRANK = 320
ROWS = (1024,)


class TestHCPrefillTriton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("HC prefill fusion requires CUDA SM120")
        if torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("HC prefill fusion requires CUDA SM120")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        # Use the real compiled fallback, not a copy of its source or the fused
        # dispatch. Disabling parameter construction avoids unused GPU weights.
        cls.original = GatedResidual(
            HyperConnectionConfig(
                hc_count=HC_COUNT,
                hidden_size=HIDDEN_SIZE,
                hc_lowrank=LOWRANK,
                params_dtype=torch.bfloat16,
            ),
            use_mix=False,
            use_combine=False,
        )
        # All captures share one stream: cycling through the stream pool can
        # invalidate cuBLAS workspaces still referenced by earlier graphs.
        cls.capture_stream = torch.cuda.Stream(device=cls.device)

    def _inputs(self, rows, scale=1.0, seed=1234):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(
            (rows, HC_COUNT, HIDDEN_SIZE),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        x = (x.flatten(1) * scale).to(torch.bfloat16)
        down = (
            torch.randn(
                (LOWRANK, WIDTH), generator=generator, device=self.device
            )
            * 0.02
        ).to(torch.bfloat16)
        up = (
            torch.randn(
                (WIDTH, LOWRANK), generator=generator, device=self.device
            )
            * 0.02
        ).to(torch.bfloat16)
        return x, down, up

    def _compiled_original(self, x, down, up):
        return self.original._mix_compute(x, down, up, HC_COUNT, HIDDEN_SIZE)

    @staticmethod
    def _fp32_reference(x, down, up):
        # Same BF16 input values, but no intermediate BF16 rounding or TF32.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            xf = x.float()
            gate = torch.sigmoid(
                F.linear(F.silu(F.linear(xf, down.float()) / HC_COUNT), up.float())
            )
            return (
                gate.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
                * xf.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
            ).mean(-2)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def _assert_numerics(self, actual, baseline, reference, rows):
        self.assertEqual(actual.shape, (rows, HIDDEN_SIZE))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.device, self.device)
        for tensor in (actual, baseline, reference):
            self.assertTrue(torch.isfinite(tensor).all().item())
        torch.testing.assert_close(actual, baseline, rtol=0.01, atol=0.005)
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
        candidate_error = (
            torch.linalg.vector_norm(actual.float() - reference) / denominator
        ).item()
        baseline_error = (
            torch.linalg.vector_norm(baseline.float() - reference) / denominator
        ).item()
        self.assertLessEqual(candidate_error, baseline_error * 1.05 + 1e-5)

    def _assert_unchanged(self, tensors, snapshots):
        for tensor, snapshot in zip(tensors, snapshots):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)

    def _capture(self, function):
        stream = self.capture_stream
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = function()
        torch.cuda.synchronize(self.device)
        return graph, output

    def test_compiled_original_and_fp32_accuracy(self):
        for rows in ROWS:
            scales = (1.0, 1e-3, 32.0)
            for case, scale in enumerate(scales):
                with self.subTest(rows=rows, scale=scale):
                    inputs = self._inputs(rows, scale, seed=rows + case)
                    snapshots = tuple(tensor.clone() for tensor in inputs)
                    self.assertTrue(hc_prefill_mix_supported(*inputs))
                    baseline = self._compiled_original(*inputs)
                    reference = self._fp32_reference(*inputs)
                    actual = hc_prefill_mix(*inputs)
                    self._assert_numerics(actual, baseline, reference, rows)
                    self._assert_unchanged(inputs, snapshots)

    @torch.no_grad()
    def test_gated_residual_mix_preserves_residuals(self):
        rows = 1024
        module = GatedResidual(
            HyperConnectionConfig(
                hc_count=HC_COUNT,
                hidden_size=HIDDEN_SIZE,
                hc_lowrank=LOWRANK,
                params_dtype=torch.bfloat16,
            )
        ).to(device=self.device, dtype=torch.bfloat16)
        x, down, up = self._inputs(rows, seed=4257)
        module.input_mix_weight_down.weight.copy_(down)
        module.input_mix_weight_up.weight.copy_(up)
        input_snapshot = x.clone()
        parameters = tuple(module.parameters())
        parameter_snapshots = tuple(parameter.clone() for parameter in parameters)
        normalized = module.hc_norm(
            x.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        ).flatten(-2)
        baseline = module._mix_compute(
            normalized,
            module.input_mix_weight_down.weight,
            module.input_mix_weight_up.weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )
        reference = self._fp32_reference(normalized, down, up)
        actual, residuals = module.mix(x)
        self._assert_numerics(actual, baseline, reference, rows)
        self.assertEqual(len(residuals), 2)
        self.assertIs(residuals[0], x)
        torch.testing.assert_close(residuals[1], normalized, rtol=0, atol=0)
        self._assert_unchanged((x, *parameters), (input_snapshot, *parameter_snapshots))

    def test_changed_input_cuda_graph_replay(self):
        for rows in ROWS:
            with self.subTest(rows=rows):
                x, down, up = self._inputs(rows, seed=rows + 100)
                weight_snapshots = (down.clone(), up.clone())
                baseline_graph, baseline_out = self._capture(
                    lambda: self._compiled_original(x, down, up)
                )
                candidate_graph, candidate_out = self._capture(
                    lambda: hc_prefill_mix(x, down, up)
                )
                previous = None
                for replay_index, scale in enumerate((1.0, 1e-3, 32.0)):
                    with self.subTest(replay=replay_index, scale=scale):
                        changed_x, _, _ = self._inputs(
                            rows, scale, seed=rows + 200 + replay_index
                        )
                        x.copy_(changed_x)
                        baseline_graph.replay()
                        candidate_graph.replay()
                        reference = self._fp32_reference(x, down, up)
                        self._assert_numerics(
                            candidate_out, baseline_out, reference, rows
                        )
                        self._assert_unchanged(
                            (x, down, up), (changed_x, *weight_snapshots)
                        )
                        if previous is not None:
                            self.assertFalse(torch.equal(candidate_out, previous))
                        previous = candidate_out.clone()
                # Finish all uses before graph-owned outputs and pools expire.
                torch.cuda.synchronize(self.device)

    def _assert_unsupported(self, inputs):
        self.assertFalse(hc_prefill_mix_supported(*inputs))
        with self.assertRaises(ValueError):
            hc_prefill_mix(*inputs)

    def test_unsupported_devices_dtypes_and_layouts(self):
        inputs = self._inputs(1024)
        self._assert_unsupported(tuple(tensor.cpu() for tensor in inputs))
        for index in range(3):
            with self.subTest(argument=index, device="cpu"):
                changed = list(inputs)
                changed[index] = changed[index].cpu()
                self._assert_unsupported(changed)
            for dtype in (torch.float16, torch.float32):
                with self.subTest(argument=index, dtype=dtype):
                    changed = list(inputs)
                    changed[index] = changed[index].to(dtype)
                    self._assert_unsupported(changed)
            with self.subTest(argument=index, layout="noncontiguous"):
                changed = list(inputs)
                changed[index] = changed[index].t().contiguous().t()
                self.assertFalse(changed[index].is_contiguous())
                self._assert_unsupported(changed)

    def test_unsupported_shapes_and_row_bounds(self):
        inputs = self._inputs(1024)
        for rows in (0, 1, 16, 64, 255, 256, 257, 511, 512, 1023, 1025):
            with self.subTest(rows=rows):
                changed = list(inputs)
                changed[0] = torch.empty(
                    (rows, WIDTH), dtype=torch.bfloat16, device=self.device
                )
                self._assert_unsupported(changed)
        invalid_shapes = (
            (0, (1024, WIDTH - 1)),
            (0, (1024, WIDTH + 1)),
            (1, (LOWRANK - 1, WIDTH)),
            (1, (LOWRANK, WIDTH - 1)),
            (2, (WIDTH - 1, LOWRANK)),
            (2, (WIDTH, LOWRANK - 1)),
        )
        for index, shape in invalid_shapes:
            with self.subTest(argument=index, shape=shape):
                changed = list(inputs)
                changed[index] = torch.empty(
                    shape, dtype=torch.bfloat16, device=self.device
                )
                self._assert_unsupported(changed)
        for index in range(3):
            for ndim in (1, 3):
                with self.subTest(argument=index, ndim=ndim):
                    changed = list(inputs)
                    changed[index] = (
                        changed[index].flatten()
                        if ndim == 1
                        else changed[index].unsqueeze(0)
                    )
                    self._assert_unsupported(changed)


if __name__ == "__main__":
    unittest.main()
