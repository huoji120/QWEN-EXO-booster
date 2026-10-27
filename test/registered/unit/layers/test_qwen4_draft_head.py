"""CPU contracts for the opt-in Qwen4 selected-vocabulary MTP head."""

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.qwen4_draft_head import (
    DraftVocabLogitsProcessor,
    load_draft_vocab_map,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftVocabMap(CustomTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {
            "model_type": "qwen4_exp",
            "bos_token_id": 0,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "vocab_size": 16,
                "hidden_size": 8,
                "eos_token_id": [1, 2],
            },
        }
        self.tokenizer = {"added_tokens": [{"id": 3, "special": True}, {"id": 9, "special": False}]}
        self.map = {
            "schema": "qwen4-draft-vocab-v1",
            "vocab_size": 16,
            "hidden_size": 8,
            "token_ids": [0, 1, 2, 3, 7, 15],
            "provenance": {"source": "https://example.org/token-counts", "revision": "test"},
        }
        self.write_assets()

    def write_assets(self):
        for name, value, hash_key in (
            ("config.json", self.config, "model_config_sha256"),
            ("tokenizer.json", self.tokenizer, "tokenizer_sha256"),
        ):
            raw = json.dumps(value).encode("utf-8")
            (self.root / name).write_bytes(raw)
            self.map[hash_key] = hashlib.sha256(raw).hexdigest()

    def load(self, mapping=None):
        path = self.root / "map.json"
        path.write_text(json.dumps(self.map if mapping is None else mapping), encoding="utf-8")
        return load_draft_vocab_map(path, model_path=self.root, tokenizer_path=self.root)

    def test_identity_bound_cpu_ids_and_flat_config(self):
        ids = self.load()
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(ids.device.type, "cpu")
        self.assertEqual(ids.tolist(), self.map["token_ids"])
        self.config = {**self.config["text_config"], "bos_token_id": 0}
        self.write_assets()
        torch.testing.assert_close(self.load(), ids)
        path = self.root / "map.json"
        torch.testing.assert_close(
            load_draft_vocab_map(path, model_path=self.root / "config.json", tokenizer_path=self.root / "tokenizer.json"),
            ids,
        )

    def test_missing_every_kind_of_required_special(self):
        for token in (0, 1, 2, 3):
            with self.subTest(token=token):
                mapping = copy.deepcopy(self.map)
                mapping["token_ids"].remove(token)
                with self.assertRaisesRegex(ValueError, "missing required special"):
                    self.load(mapping)

    def test_invalid_map_fields(self):
        cases = [
            ("schema", "unknown"), ("provenance", []),
            ("model_config_sha256", "bad"), ("tokenizer_sha256", "g" * 64),
            ("model_config_sha256", "0" * 64), ("tokenizer_sha256", "0" * 64),
            ("vocab_size", True), ("vocab_size", 15), ("hidden_size", 0),
            ("hidden_size", 9), ("token_ids", []), ("token_ids", list(range(16))),
            ("token_ids", [0, 1, 2, 3, 3]), ("token_ids", [0, 2, 1, 3]),
            ("token_ids", [False, 1, 2, 3]), ("token_ids", [0, 1, 2, 3, 16]),
            ("token_ids", [-1, 0, 1, 2, 3]), ("token_ids", [0, 1, 2, 3, 7.0]),
            ("token_ids", "0,1,2,3"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                mapping = {**self.map, field: value}
                with self.assertRaises(ValueError):
                    self.load(mapping)
        for key in self.map:
            with self.subTest(missing=key):
                mapping = dict(self.map)
                del mapping[key]
                with self.assertRaises(ValueError):
                    self.load(mapping)

    def test_changed_asset_bytes_fail_identity_binding(self):
        for name in ("config.json", "tokenizer.json"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.load()
                path.write_bytes(original)

    def test_invalid_model_and_special_metadata(self):
        original_config, original_tokenizer = copy.deepcopy(self.config), copy.deepcopy(self.tokenizer)
        cases = [
            ("model_type", "qwen3"), ("vocab_size", True), ("hidden_size", 0),
            ("bos_token_id", False), ("eos_token_id", [1, 16]),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                self.config = copy.deepcopy(original_config)
                self.config["text_config"][key] = value
                self.write_assets()
                with self.assertRaises(ValueError):
                    self.load()
        self.config = original_config
        for added in ({}, [None], [{"special": 1, "id": 3}], [{"special": True, "id": True}], [{"special": True, "id": 16}]):
            with self.subTest(added=added):
                self.tokenizer = {**original_tokenizer, "added_tokens": added}
                self.write_assets()
                with self.assertRaises(ValueError):
                    self.load()

    def test_bounded_json_only_and_duplicate_keys(self):
        path = self.root / "map.json"
        for raw in (b"[]", b"not json", b"\xff", b'{"schema":"a","schema":"b"}', b" " * (8 * 1024 * 1024 + 1)):
            with self.subTest(size=len(raw)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_draft_vocab_map(path, model_path=self.root, tokenizer_path=self.root)


class TestDraftVocabLogits(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.args = SimpleNamespace(enable_dp_lm_head=False, enable_fp32_lm_head=False, enable_mis=False, rl_on_policy_target=None)
        self.parallel = SimpleNamespace(tp_size=1, attn_tp_size=1, attn_dp_size=1)
        self.stack.enter_context(patch.object(logits_module, "get_server_args", return_value=self.args))
        self.stack.enter_context(patch.object(logits_module, "get_parallel", return_value=self.parallel))
        self.stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "MultimemAllGatherer"))
        self.stack.enter_context(patch.object(logits_module.triton_symm_mem_ag, "recommended_max_tokens", return_value=128))
        self.config = SimpleNamespace(vocab_size=64, hidden_size=32)
        generator = torch.Generator().manual_seed(41)
        self.head = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
        self.head.weight = torch.nn.Parameter(torch.randn(64, 32, generator=generator).bfloat16(), requires_grad=False)
        self.hidden = torch.randn(5, 32, generator=generator).bfloat16()
        self.ids = torch.tensor([0, 1, 5, 11, 31, 63], dtype=torch.int64)

    def make_processor(self, **kwargs):
        return DraftVocabLogitsProcessor(self.config, self.head, self.ids, **kwargs)

    def assert_selected(self, actual, expected):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        torch.testing.assert_close(actual[:, self.ids], expected[:, self.ids], rtol=0.02, atol=0.02)
        outside = torch.ones(self.config.vocab_size, dtype=torch.bool)
        outside[self.ids] = False
        self.assertTrue(torch.isneginf(actual[:, outside]).all().item())
        self.assertFalse(torch.isnan(actual).any().item())

    def test_reduced_matmul_preserves_target_and_has_no_shared_scratch(self):
        baseline = LogitsProcessor(self.config)
        before_weight = self.head.weight.clone()
        before_logits = baseline._compute_lm_head(self.hidden, self.head)
        processor = self.make_processor()
        with patch("torch.matmul", wraps=torch.matmul) as matmul:
            actual = processor._compute_lm_head(self.hidden, self.head)
            self.assertEqual(matmul.call_count, 1)
            self.assertEqual(tuple(matmul.call_args.args[1].shape), (32, len(self.ids)))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.device, self.hidden.device)
        self.assert_selected(actual, before_logits)
        torch.testing.assert_close(self.head.weight, before_weight, rtol=0, atol=0)
        torch.testing.assert_close(baseline._compute_lm_head(self.hidden, self.head), before_logits, rtol=0, atol=0)
        self.assertEqual(list(processor.parameters()), [])
        self.assertEqual(processor.state_dict(), {})
        self.assertEqual(set(dict(processor.named_buffers())), {"token_ids", "selected_weight"})
        self.assertNotEqual(processor.selected_weight.data_ptr(), self.head.weight.data_ptr())
        snapshot = actual.clone()
        later = processor._compute_lm_head(self.hidden + 1, self.head)
        self.assertNotEqual(actual.data_ptr(), later.data_ptr())
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
        self.ids.fill_(0)
        self.assertEqual(processor.token_ids.tolist(), [0, 1, 5, 11, 31, 63])

    def test_inherited_pruning_scaling_and_buffer_copy(self):
        processor = self.make_processor(logit_scale=1.5)
        buffer = torch.empty(2, 64, dtype=torch.float32)
        metadata = LogitsMetadata(forward_mode=ForwardMode.DRAFT_EXTEND_V2, draft_extend_select_index=torch.tensor([1, 4]), next_token_logits_buffer=buffer)
        with patch.object(processor, "_compute_lm_head", wraps=processor._compute_lm_head) as compute:
            output = processor(torch.arange(5), self.hidden, self.head, metadata)
            self.assertEqual(tuple(compute.call_args.args[0].shape), (2, 32))
        expected = (self.hidden[[1, 4]] @ self.head.weight.T).mul_(1.5).float()
        self.assert_selected(output.next_token_logits, expected)
        self.assertEqual(output.next_token_logits.data_ptr(), buffer.data_ptr())
        self.assertEqual(output.next_token_logits.dtype, torch.float32)
        self.assertIs(DraftVocabLogitsProcessor.forward, LogitsProcessor.forward)
        self.assertIs(DraftVocabLogitsProcessor._get_logits, LogitsProcessor._get_logits)

    def test_mutation_replacement_and_storage_rebinding_fail_closed(self):
        for mutation in ("inplace", "parameter", "storage", "module"):
            with self.subTest(mutation=mutation):
                processor = self.make_processor()
                head = self.head
                if mutation == "inplace":
                    with torch.no_grad():
                        head.weight.add_(1)
                elif mutation == "parameter":
                    head.weight = torch.nn.Parameter(head.weight.clone(), requires_grad=False)
                elif mutation == "storage":
                    head.weight.data = head.weight.clone()
                else:
                    head = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
                    head.weight = self.head.weight
                with self.assertRaisesRegex(RuntimeError, "source head changed"):
                    processor._compute_lm_head(self.hidden, head)

    def test_inference_tensor_without_version_is_supported(self):
        with torch.inference_mode():
            self.head.weight = torch.nn.Parameter(self.head.weight.clone(), requires_grad=False)
            processor = self.make_processor()
            self.assert_selected(processor._compute_lm_head(self.hidden, self.head), self.hidden @ self.head.weight.T)
            self.head.weight = torch.nn.Parameter(self.head.weight.clone(), requires_grad=False)
            with self.assertRaisesRegex(RuntimeError, "source head changed"):
                processor._compute_lm_head(self.hidden, self.head)

    def test_stale_modelopt_methods_do_not_reject_actual_bf16(self):
        for name in ("ModelOptFp8LinearMethod", "ModelOptFp4LinearMethod", "ModelOptNvFp4A16LinearMethod"):
            with self.subTest(name=name):
                def unexpected_apply(*args):
                    raise AssertionError("Stale quantization method must never run")
                self.head.quant_method = type(name, (), {"apply": unexpected_apply})()
                processor = self.make_processor()
                self.assert_selected(processor._compute_lm_head(self.hidden, self.head), self.hidden @ self.head.weight.T)

    def test_invalid_ids(self):
        for ids in (torch.tensor([], dtype=torch.int64), torch.arange(64), torch.tensor([0, 0]), torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([0, 64]), torch.tensor([0, 1], dtype=torch.int32), torch.tensor([[0, 1]]), [0, 1]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    DraftVocabLogitsProcessor(self.config, self.head, ids)

    def test_incompatible_head_features_and_states(self):
        original = self.head.weight
        for weight in (original.float(), original.to(torch.int8), original[:32], original[:, :16], original.T.contiguous().T):
            with self.subTest(dtype=weight.dtype, shape=weight.shape):
                self.head.weight = torch.nn.Parameter(weight, requires_grad=False)
                with self.assertRaisesRegex(ValueError, "full-width contiguous resident BF16"):
                    self.make_processor()
        self.head.weight = original
        for attr, value in (("quant_method", SimpleNamespace(apply=lambda *args: None)), ("set_lora", lambda: None), ("apply_lora", lambda: None), ("bias", torch.nn.Parameter(torch.zeros(64)))):
            with self.subTest(attr=attr):
                setattr(self.head, attr, value)
                with self.assertRaisesRegex(ValueError, "quantization, LoRA, or bias"):
                    self.make_processor()
                delattr(self.head, attr)
        processor = self.make_processor()
        for hidden in (self.hidden.float(), self.hidden[:, :16], self.hidden[0]):
            with self.assertRaisesRegex(ValueError, "colocated BF16"):
                processor._compute_lm_head(hidden, self.head)
        with self.assertRaisesRegex(ValueError, "embedding_bias"):
            processor._compute_lm_head(self.hidden, self.head, torch.zeros(64))
        processor.float()
        with self.assertRaisesRegex(ValueError, "colocated BF16"):
            processor._compute_lm_head(self.hidden, self.head)

    def test_incompatible_processor_settings(self):
        for field, value in (("enable_fp32_lm_head", True), ("enable_dp_lm_head", True), ("rl_on_policy_target", "target")):
            with self.subTest(field=field):
                original = getattr(self.args, field)
                setattr(self.args, field, value)
                with self.assertRaises(ValueError):
                    self.make_processor()
                setattr(self.args, field, original)
        self.parallel.tp_size = 2
        with self.assertRaises(ValueError):
            self.make_processor()
        self.parallel.tp_size = 1
        for softcap in (0, 10):
            self.config.final_logit_softcapping = softcap
            with self.assertRaises(ValueError):
                self.make_processor()
        self.config.final_logit_softcapping = None
        for scale in (0, -1, float("inf"), float("nan"), True, "1"):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.make_processor(logit_scale=scale)

    def test_hot_weight_updates_reject_before_target_mutation(self):
        from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager

        processor = self.make_processor()
        original = self.head.weight.clone()
        def mutate(_request):
            self.head.weight.data.zero_()
            return True, "updated"
        target = SimpleNamespace(**{
            name: mutate for name in (
                "update_weights_from_disk", "update_weights_from_distributed",
                "update_weights_from_tensor", "update_weights_from_ipc"
            )
        })
        model = SimpleNamespace(logits_processor=processor)
        draft = SimpleNamespace(_draft_worker=SimpleNamespace(draft_runner=SimpleNamespace(model=model)))
        manager = SchedulerWeightUpdaterManager(
            tp_worker=target, draft_worker=draft, tp_cpu_group=None,
            memory_saver_adapter=None, flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda: True,
        )
        request = SimpleNamespace(flush_cache=False, disable_draft_model=True)
        for name in ("update_weights_from_disk", "update_weights_from_distributed",
                     "update_weights_from_tensor", "update_weights_from_ipc"):
            with self.subTest(source=name):
                response = getattr(manager, name)(request)
                self.assertFalse(response.success)
                self.assertIn("restart", response.message)
                torch.testing.assert_close(self.head.weight, original, rtol=0, atol=0)
        model.logits_processor = LogitsProcessor(self.config)
        response = manager.update_weights_from_distributed(request)
        self.assertTrue(response.success)
        self.assertEqual(torch.count_nonzero(self.head.weight).item(), 0)


if __name__ == "__main__":
    unittest.main()
