"""Unit tests for the isolated QWEN-EXO Qwen3.5 MoE Top-K override."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.models.qwen2_moe import _resolve_qwen_exo_moe_top_k
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenExoMoeTopK(CustomTestCase):
    @staticmethod
    def _config(model_type="qwen3_5_moe_text", top_k=8, experts=256):
        return SimpleNamespace(
            model_type=model_type,
            num_experts_per_tok=top_k,
            num_experts=experts,
        )

    def test_unset_override_preserves_checkpoint(self):
        with patch(
            "sglang.srt.models.qwen2_moe.get_server_args",
            return_value=SimpleNamespace(qwen_exo_moe_top_k=None),
        ):
            self.assertEqual(
                _resolve_qwen_exo_moe_top_k(self._config()),
                8,
            )

    def test_qwen35_override_selects_requested_top_k(self):
        with patch(
            "sglang.srt.models.qwen2_moe.get_server_args",
            return_value=SimpleNamespace(qwen_exo_moe_top_k=32),
        ):
            self.assertEqual(
                _resolve_qwen_exo_moe_top_k(self._config()),
                32,
            )

    def test_override_does_not_affect_other_moe_models(self):
        with patch(
            "sglang.srt.models.qwen2_moe.get_server_args",
            return_value=SimpleNamespace(qwen_exo_moe_top_k=32),
        ):
            self.assertEqual(
                _resolve_qwen_exo_moe_top_k(self._config(model_type="qwen2_moe")),
                8,
            )

    @pytest.mark.parametrize("value", [0, -1, 257])
    def test_override_rejects_out_of_range_values(self, value):
        with patch(
            "sglang.srt.models.qwen2_moe.get_server_args",
            return_value=SimpleNamespace(qwen_exo_moe_top_k=value),
        ):
            with self.assertRaisesRegex(ValueError, "between 1"):
                _resolve_qwen_exo_moe_top_k(self._config())


if __name__ == "__main__":
    import unittest

    unittest.main()
