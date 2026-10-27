"""Native draft worker keeps full target IDs and target head ownership intact."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers.qwen4_draft_head import DraftVocabLogitsProcessor
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4DraftVocabWorker(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.args = SimpleNamespace(
            speculative_algorithm="EAGLE", speculative_token_map="bound-map.json",
            model_path="model", tokenizer_path="tokenizer", enable_dp_lm_head=False,
            enable_fp32_lm_head=False, enable_mis=False, rl_on_policy_target=None,
            enable_lora=False, speculative_eagle_topk=1,
            speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0,
        )
        self.parallel = SimpleNamespace(tp_size=1, pp_size=1, attn_tp_size=1, attn_dp_size=1)
        self.stack.enter_context(patch.object(logits_module, "get_server_args", return_value=self.args))
        self.stack.enter_context(patch.object(logits_module, "get_parallel", return_value=self.parallel))
        self.stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "MultimemAllGatherer"))
        self.stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "recommended_max_tokens", return_value=128))
        self.config = SimpleNamespace(vocab_size=32, hidden_size=16)
        torch.manual_seed(17)
        self.head = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
        self.embedding = torch.nn.Parameter(torch.randn(32, 16).bfloat16(), requires_grad=False)
        self.ids = torch.tensor([0, 3, 7, 19, 31], dtype=torch.int64)

    def worker(self, draft):
        worker = object.__new__(EagleDraftWorker)
        worker.server_args = self.args
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.draft_runner = SimpleNamespace(model=draft)
        worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(
            lm_head=self.head, get_embed_and_head=lambda: (self.embedding, self.head.weight)
        )))
        return worker

    def test_worker_retains_full_target_coordinates_and_independent_draft_rows(self):
        # CPU fixture uses real selected-row processor; worker initialization is
        # unmodified. No CUDA/model construction is needed to test ownership.
        owner = self
        class Draft:
            config = owner.config
            def set_embed_and_head(self, embed, head):
                self.embedding = embed
                self.lm_head = SimpleNamespace(weight=head)
            def set_lm_head_from_target(self, head):
                self.lm_head = head
            def configure_draft_head_vocab(self, path, *, model_path, tokenizer_path):
                self.logits_processor = DraftVocabLogitsProcessor(self.config, self.lm_head, owner.ids)
        draft = Draft()
        worker = self.worker(draft)
        before_weight = self.head.weight.clone()
        hidden = torch.randn(3, 16).bfloat16()
        original_target = hidden @ self.head.weight.T
        worker.init_token_map()
        worker.init_lm_head()
        self.assertIsNone(worker.hot_token_id)
        self.assertIs(draft.lm_head, self.head)
        self.assertEqual(draft.config.vocab_size, 32)
        self.assertNotEqual(draft.logits_processor.selected_weight.data_ptr(), self.head.weight.data_ptr())
        logits = draft.logits_processor._compute_lm_head(hidden, draft.lm_head)
        self.assertEqual(logits.shape, (3, 32))
        torch.testing.assert_close(logits[:, self.ids], original_target[:, self.ids], rtol=0.02, atol=0.02)
        outside = torch.ones(32, dtype=torch.bool)
        outside[self.ids] = False
        self.assertTrue(torch.isneginf(logits[:, outside]).all())
        torch.testing.assert_close(self.head.weight, before_weight, rtol=0, atol=0)
        torch.testing.assert_close(hidden @ self.head.weight.T, original_target, rtol=0, atol=0)
        # Native target argmax can select tokens never offered by this draft.
        self.assertTrue(torch.isfinite(original_target[:, outside]).all())

    def test_model_specific_map_rejects_other_worker_algorithms(self):
        draft = SimpleNamespace(configure_draft_head_vocab=lambda *args, **kwargs: None)
        worker = self.worker(draft)
        for algorithm in (SpeculativeAlgorithm.STANDALONE, SpeculativeAlgorithm.FROZEN_KV_MTP):
            with self.subTest(algorithm=algorithm):
                worker.speculative_algorithm = algorithm
                with self.assertRaisesRegex(ValueError, "native EAGLE"):
                    worker.init_token_map()

    def test_disabled_map_retains_original_shared_head_without_processor(self):
        self.args.speculative_token_map = None
        class Draft:
            def set_embed_and_head(self, embed, head):
                self.embedding = embed
                self.lm_head = SimpleNamespace(weight=head)
            def set_lm_head_from_target(self, head):
                self.lm_head = head
            def configure_draft_head_vocab(self, *args, **kwargs):
                raise AssertionError("disabled map must not install a shortlist")
        draft = Draft()
        worker = self.worker(draft)
        worker.init_token_map()
        worker.init_lm_head()
        self.assertIs(draft.lm_head, self.head)
        self.assertIsNone(worker.hot_token_id)

    def test_native_install_rejects_incompatible_topology_and_sampling(self):
        from sglang.srt.models import qwen4_exp_mtp as model_module
        model = object.__new__(Qwen4ExpForCausalLMMTP)
        torch.nn.Module.__init__(model)
        model.tp_size = 1
        model.lm_head = self.head
        with patch.object(model_module, "get_server_args", return_value=self.args), patch.object(
            model_module, "get_parallel", return_value=self.parallel
        ):
            for field, value in (("enable_lora", True), ("enable_dp_lm_head", True),
                                 ("speculative_eagle_topk", 2), ("speculative_algorithm", "STANDALONE"),
                                 ("speculative_accept_threshold_single", 0.5),
                                 ("speculative_accept_threshold_acc", 0.5)):
                original = getattr(self.args, field)
                with self.subTest(field=field):
                    setattr(self.args, field, value)
                    with self.assertRaises(ValueError):
                        model.configure_draft_head_vocab("absent", model_path="absent", tokenizer_path="absent")
                    setattr(self.args, field, original)
            for field in ("pp_size", "attn_dp_size"):
                setattr(self.parallel, field, 2)
                with self.assertRaises(ValueError):
                    model.configure_draft_head_vocab("absent", model_path="absent", tokenizer_path="absent")
                setattr(self.parallel, field, 1)
            with self.assertRaisesRegex(ValueError, "CUDA"):
                model.configure_draft_head_vocab("absent", model_path="absent", tokenizer_path="absent")


if __name__ == "__main__":
    unittest.main()
