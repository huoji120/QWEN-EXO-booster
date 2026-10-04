import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpeculativeTargetOnlyCudaGraph(CustomTestCase):
    def test_target_only_batch_rejects_wider_speculative_graph(self):
        forward_batch = SimpleNamespace(
            replace_embeds=None,
            spec_algorithm=MagicMock(is_none=MagicMock(return_value=True)),
        )
        for width in (4, 8):
            with self.subTest(captured_req_width=width):
                runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
                runner.captured_req_width = width
                self.assertFalse(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
