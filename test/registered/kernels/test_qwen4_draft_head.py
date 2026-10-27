"""GPU acceptance for draft-owned BF16 vocabulary rows; never enables serving."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers.qwen4_draft_head import DraftVocabLogitsProcessor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestQwen4DraftHeadCUDA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA acceptance requires an idle GPU")
        cls.stream = torch.cuda.Stream()

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        args = SimpleNamespace(enable_dp_lm_head=False, enable_fp32_lm_head=False,
                               enable_mis=False, rl_on_policy_target=None)
        parallel = SimpleNamespace(tp_size=1, attn_tp_size=1, attn_dp_size=1)
        # Only process topology is supplied by the fixture; matmul, scatter,
        # Graph execution and the selected-head implementation are real.
        stack.enter_context(patch.object(logits_module, "get_server_args", return_value=args))
        stack.enter_context(patch.object(logits_module, "get_parallel", return_value=parallel))
        stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "MultimemAllGatherer"))
        stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "recommended_max_tokens", return_value=128))
        torch.manual_seed(271027)
        self.head = torch.nn.Linear(256, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
        self.head.requires_grad_(False)
        self.ids = torch.cat((torch.arange(0, 4096, 3), torch.tensor([4095]))).unique().long()
        self.processor = DraftVocabLogitsProcessor(
            SimpleNamespace(vocab_size=4096, hidden_size=256), self.head, self.ids
        )

    def check_output(self, x, actual):
        full = torch.nn.functional.linear(x, self.head.weight)
        ids = self.ids.cuda()
        torch.testing.assert_close(actual[:, ids], full[:, ids], rtol=0.02, atol=0.02)
        outside = torch.ones(4096, device="cuda", dtype=torch.bool)
        outside[ids] = False
        self.assertTrue(torch.isneginf(actual[:, outside]).all())
        self.assertTrue(torch.isfinite(full[:, outside]).all())
        self.assertFalse(torch.isnan(actual).any())
        probabilities = actual.float().softmax(-1)
        self.assertEqual(torch.count_nonzero(probabilities[:, outside]).item(), 0)

    @torch.no_grad()
    def test_full_target_support_and_private_selected_weights(self):
        original = self.head.weight.clone()
        for rows in (1, 2, 4, 8):
            x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
            before = x.clone()
            actual = self.processor._compute_lm_head(x, self.head)
            self.check_output(x, actual)
            torch.testing.assert_close(x, before, rtol=0, atol=0)
        torch.testing.assert_close(self.head.weight, original, rtol=0, atol=0)
        self.assertNotEqual(self.processor.selected_weight.data_ptr(), self.head.weight.data_ptr())

    @torch.no_grad()
    def test_graph_replay_updates_full_width_output_without_stale_rows(self):
        for rows in (1, 4, 8):
            x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
            operation = lambda: self.processor._compute_lm_head(x, self.head)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(3): operation()
            torch.cuda.current_stream().wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                output = operation()
            for scale in (1.0, 0.01, 5.0):
                x.normal_().mul_(scale)
                expected = operation().clone()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.check_output(x, output)

    def test_target_only_rejection_can_emit_outside_draft_map(self):
        from sglang.kernels.ops.speculative.chain_target_only import chain_speculative_sampling_target_only
        # Draft proposes ID0. Target has all probability on ID4094, which is
        # absent from this draft's every-third-row map. Forced rejection must
        # still emit the full target's ID, not a compressed row index.
        self.assertNotIn(4094, self.ids.tolist())
        probs = torch.zeros((1, 2, 4096), device="cuda", dtype=torch.float32)
        probs[..., 4094] = 1
        predicts = torch.full((2,), -1, device="cuda", dtype=torch.int32)
        accepted = torch.full((1, 2), -1, device="cuda", dtype=torch.int32)
        count = torch.empty((1,), device="cuda", dtype=torch.int32)
        candidates = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        indices = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
        links = torch.tensor([[1, -1]], device="cuda", dtype=torch.int64)
        coins = torch.ones((1, 2), device="cuda", dtype=torch.float32)
        final_coin = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        chain_speculative_sampling_target_only(
            predicts, accepted, count, candidates, indices, links, coins,
            final_coin, probs, 1.0, 1.0,
        )
        self.assertEqual(count.item(), 0)
        self.assertEqual(predicts[0].item(), 4094)


if __name__ == "__main__":
    unittest.main()
