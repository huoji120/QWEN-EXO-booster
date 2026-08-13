"""Unit tests for MoE WNA16 quantization configuration."""

import unittest

from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.utils import get_dynamic_override
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoeWNA16Config(CustomTestCase):
    def test_gptq_dynamic_negative_rules_are_preserved(self):
        config = MoeWNA16Config.from_config(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
                "dynamic": {
                    "-:.*attn.*": {},
                    "-:.*shared_expert.*": {},
                },
            }
        )

        self.assertIs(
            get_dynamic_override(config, layer_name="model.layers.0.self_attn.q_proj"),
            False,
        )
        self.assertIs(
            get_dynamic_override(
                config, layer_name="model.layers.0.mlp.shared_expert.gate_up_proj"
            ),
            False,
        )
        self.assertIsNone(
            get_dynamic_override(config, layer_name="model.layers.0.mlp.experts")
        )


if __name__ == "__main__":
    unittest.main()
