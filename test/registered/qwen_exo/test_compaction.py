import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from collections import OrderedDict

from qwen_exo_booster.compaction import (
    CompactionSummary,
    ResponseCompactionService,
)
from qwen_exo_booster.runtime import QwenExoRuntime

try:
    from sglang.srt.entrypoints.openai.serving_responses import (
        OpenAIServingResponses,
    )
except ModuleNotFoundError as exc:
    if exc.name != "resource":
        raise
    OpenAIServingResponses = None

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError as exc:
    if exc.name != "resource":
        raise
    from unittest import TestCase as CustomTestCase

    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Tokenizer:
    def __init__(self):
        self.last_enable_thinking = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(str(text))))

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del tokenize, add_generation_prompt
        self.last_enable_thinking = enable_thinking
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


class _Telemetry:
    def __init__(self):
        self.events = []

    def emit(self, request_id, event_type, payload):
        self.events.append((request_id, event_type, payload))


class _Runner:
    async def run_batch(self, jobs, prompts, sampling_params):
        del prompts, sampling_params
        job = tuple(jobs)[0]
        from qwen_exo_booster.internal_jobs import InternalJobResult

        return (
            InternalJobResult(
                job=job,
                text="GOAL: preserve the verified task state\nNEXT ACTION: continue",
                prompt_tokens=12,
                completion_tokens=10,
                finish_reason="stop",
                latency_seconds=0.001,
            ),
        )


class TestResponseCompaction(CustomTestCase):
    def test_compaction_drops_only_safe_trajectory_items(self):
        runtime = object.__new__(QwenExoRuntime)
        runtime.config = SimpleNamespace(
            response_compaction_max_history_tokens=100,
            response_compaction_max_dropped_items=1,
        )
        items = [
            {"type": "message", "role": "user", "content": "keep this goal"},
            {"type": "message", "role": "assistant", "content": "x" * 400},
        ]

        kept, dropped, source = runtime._prepare_compaction_source(items, _Tokenizer())

        assert kept == [items[0]]
        assert len(dropped) == 1
        assert dropped[0]["role"] == "assistant"
        assert "keep this goal" in source

    def test_compaction_context_round_trip_is_opaque_to_request_parser(self):
        summary = CompactionSummary(
            summary="GOAL: exact task\nNEXT ACTION: run the smoke",
            input_tokens=10,
            output_tokens=8,
            reasoning_tokens=2,
            source_digest="source-digest",
        )
        encoded = summary.encrypted_content(
            response_id="resp_compact_test", memory={"selected_document_ids": []}
        )

        assert (
            QwenExoRuntime._response_compaction_context(
                [{"type": "compaction", "encrypted_content": encoded}]
            )
            == summary.summary
        )
        assert (
            QwenExoRuntime._response_compaction_context(
                [{"type": "compaction", "encrypted_content": "invalid"}]
            )
            == ""
        )

    def test_compaction_discards_reasoning_before_orphaned_close_tag(self):
        summary = ResponseCompactionService._clean_summary(
            "Thinking Process:\ninspect the prompt\n</think>\n"
            "GOAL: preserve task state\nNEXT ACTION: continue"
        )

        assert summary == "GOAL: preserve task state\nNEXT ACTION: continue"

    def test_compaction_prompt_disables_thinking(self):
        tokenizer = _Tokenizer()
        service = ResponseCompactionService(
            _Runner(),
            tokenizer,
            _Telemetry(),
            model_fingerprint="model-fingerprint",
            max_output_tokens=256,
        )

        service._prompt(source_text="goal", memory={}, dropped_items=())

        assert tokenizer.last_enable_thinking is False

    def test_compaction_service_uses_internal_job_and_reports_usage(self):
        telemetry = _Telemetry()
        service = ResponseCompactionService(
            _Runner(),
            _Tokenizer(),
            telemetry,
            model_fingerprint="model-fingerprint",
            max_output_tokens=256,
        )

        result = asyncio.run(
            service.summarize(
                parent_request_id="request-1",
                source_digest="source-digest",
                source_text="verified task state",
                memory={
                    "selected_document_ids": ["knowledge:doc"],
                    "native_prefix_restore": {"active": True},
                    "hybrid_restoration_mode": "full",
                },
                dropped_items=(),
            )
        )

        assert result.summary.startswith("GOAL:")
        assert result.output_tokens == len(result.summary)
        assert [event[1] for event in telemetry.events] == [
            "response_compaction.started",
            "response_compaction.completed",
        ]
        assert telemetry.events[0][2]["previous_native_memory_active"] is True
        assert telemetry.events[0][2]["deltanet_state_active"] is True

    def test_known_compaction_item_normalizes_to_replayable_assistant_context(self):
        summary = CompactionSummary(
            summary="GOAL: retain EXO-COMPACT-ALPHA",
            input_tokens=10,
            output_tokens=6,
            reasoning_tokens=1,
            source_digest="source-digest",
        )
        encoded = summary.encrypted_content(
            response_id="resp_compact_known",
            memory={},
            model_fingerprint="model-fingerprint",
        )
        runtime = object.__new__(QwenExoRuntime)
        runtime._compaction_summaries = OrderedDict(
            {
                "resp_compact_known": {
                    "summary": summary.summary,
                    "source_digest": summary.source_digest,
                    "model_fingerprint": "model-fingerprint",
                }
            }
        )
        input_items = [
            {"type": "message", "role": "user", "content": "original task"},
            {"type": "compaction", "encrypted_content": encoded},
            {"type": "message", "role": "user", "content": "continue"},
        ]

        payload = runtime._verified_response_compaction_envelope(input_items)
        normalized = runtime._normalize_response_compaction_input(input_items, payload)

        assert payload["response_id"] == "resp_compact_known"
        assert [item["type"] for item in normalized] == [
            "message",
            "message",
            "message",
        ]
        assert normalized[1]["role"] == "assistant"
        assert "EXO-COMPACT-ALPHA" in normalized[1]["content"]

    def test_compaction_response_registers_as_replayable_previous_response(self):
        if OpenAIServingResponses is None:
            self.skipTest("OpenAI serving imports require POSIX resource module")

        async def exercise():
            serving = object.__new__(OpenAIServingResponses)
            serving.tokenizer_manager = SimpleNamespace(served_model_name="model")
            serving.response_store = {}
            serving.response_store_lock = asyncio.Lock()
            serving.msg_store = {}
            response = await serving.register_compaction_response(
                response_id="resp_compact_replay",
                model_name="model",
                user_items=(
                    {
                        "type": "message",
                        "role": "user",
                        "content": "original task",
                    },
                ),
                summary="GOAL: preserve the task",
            )
            replay = serving._construct_input_messages(
                SimpleNamespace(instructions=None, input="continue"), response
            )
            return serving, response, replay

        serving, response, replay = asyncio.run(exercise())

        assert response.status == "completed"
        assert serving.response_store[response.id] is response
        assert response.id in serving.msg_store
        assert [message["role"] for message in replay] == [
            "user",
            "assistant",
            "user",
        ]
        assert replay[1]["content"] == "GOAL: preserve the task"

    def test_runtime_rejects_empty_compaction_input(self):
        runtime = object.__new__(QwenExoRuntime)
        runtime.config = SimpleNamespace(response_compaction_mode="active")
        runtime.compaction_service = object()
        runtime.tokenizer_manager = SimpleNamespace(tokenizer=_Tokenizer())

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            asyncio.run(
                runtime.compact_responses(
                    SimpleNamespace(
                        request_id="empty", input=[], previous_response_id=None
                    )
                )
            )

    def test_forced_reasoning_boundary_adds_stop_overthinking_instruction(self):
        if OpenAIServingResponses is None:
            self.skipTest("OpenAI serving imports require POSIX resource module")

        class BoundaryTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return list(str(text).encode("utf-8"))

            @staticmethod
            def decode(_values, skip_special_tokens=False):
                del skip_special_tokens
                return "</think>"

        serving = object.__new__(OpenAIServingResponses)
        serving.tokenizer_manager = SimpleNamespace(tokenizer=BoundaryTokenizer())

        text, token_ids = serving._reasoning_boundary_tokens(999)

        assert text == "\nlet me do this now stop over thinking\n</think>"
        assert token_ids[-1] == 999


if __name__ == "__main__":
    import unittest

    unittest.main()
