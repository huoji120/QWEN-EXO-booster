"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig, get_hybrid_layer_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestQuantizationCompatibility(CustomTestCase):
    def test_moe_wna16_accepts_gptq_checkpoint_metadata(self):
        config = object.__new__(ModelConfig)
        config.quantization = "moe_wna16"
        config.hf_config = SimpleNamespace(
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
            }
        )
        config._find_quant_modelslim_config = lambda: None
        config.is_draft_model = False

        config._verify_quantization()

        self.assertEqual(config.quantization, "moe_wna16")


if __name__ == "__main__":
    unittest.main()
