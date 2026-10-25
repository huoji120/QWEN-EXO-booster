import asyncio
import unittest
from unittest.mock import Mock, patch

from utils import (
    collect_stream_events,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StreamFixture:
    def __init__(self, serving, request):
        self.serving = serving
        self.request = request
        self.request_metadata = RequestResponseMetadata(request_id=request.request_id)

    def run(self, chunks):
        async def gen():
            for ch in chunks:
                yield ch

        async def collect():
            return await collect_stream_events(
                self.serving.responses_stream_generator_non_harmony(
                    self.request,
                    sampling_params={},
                    result_generator=gen(),
                    model_name="x",
                    tokenizer=Mock(),
                    request_metadata=self.request_metadata,
                )
            )

        return asyncio.run(collect())


def _engine_chunk(text, completion_tokens, *, finish=False, finish_reason=None):
    if finish_reason is None and finish:
        finish_reason = {"type": "stop"}
    return {
        "text": text,
        "meta_info": {
            "id": "rid",
            "prompt_tokens": 5,
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "finish_reason": finish_reason,
        },
    }


class NonHarmonyStreamTestCase(unittest.TestCase):
    def test_terminal_usage_matches_json_and_internal_accounting(self):
        from openai.types.responses import (
            Response,
            ResponseCompletedEvent,
            ResponseIncompleteEvent,
        )
        from sglang.srt.entrypoints.context import SimpleContext

        for details_enabled in (False, True):
            for cached_tokens in (0, 3):
                for finish_type in ("stop", "length"):
                    with self.subTest(
                        details_enabled=details_enabled,
                        cached_tokens=cached_tokens,
                        finish_type=finish_type,
                    ):
                        serving = make_serving()
                        serving.enable_prompt_tokens_details = details_enabled
                        serving.reasoning_parser = None
                        serving.tool_call_parser = None
                        chunk = _engine_chunk(
                            "done", 7, finish_reason={"type": finish_type}
                        )
                        chunk["meta_info"].update(
                            prompt_tokens=11, cached_tokens=cached_tokens,
                            reasoning_tokens=2,
                        )
                        request = ResponsesRequest(
                            model="x", input="hi", stream=True, store=True,
                        )
                        fixture = _StreamFixture(serving, request)
                        payloads = event_payloads(fixture.run([chunk]))
                        self.assertIsNone(payloads[0]["response"]["usage"])
                        self.assertIsNone(payloads[1]["response"]["usage"])
                        event_type = (
                            ResponseIncompleteEvent if finish_type == "length"
                            else ResponseCompletedEvent
                        )
                        terminal = event_type.model_validate(payloads[-1])
                        stored_stream_usage = serving.response_store[
                            request.request_id
                        ].model_dump()["usage"]

                        context = SimpleContext()
                        context.last_output = chunk
                        metadata = RequestResponseMetadata(request_id=request.request_id)

                        async def empty_generator():
                            for _ in ():
                                yield None

                        response = asyncio.run(serving.responses_full_generator(
                            request.model_copy(update={"stream": False}),
                            sampling_params={}, result_generator=empty_generator(),
                            context=context, model_name="x",
                            tokenizer=serving.tokenizer_manager.tokenizer,
                            request_metadata=metadata,
                        ))
                        parsed = Response.model_validate_json(response.model_dump_json())
                        expected = {
                            "input_tokens": 11,
                            "input_tokens_details": {"cached_tokens": cached_tokens},
                            "output_tokens": 7,
                            "output_tokens_details": {"reasoning_tokens": 2},
                            "total_tokens": 18,
                        }
                        self.assertEqual(terminal.response.usage.model_dump(), expected)
                        self.assertEqual(parsed.usage.model_dump(), expected)
                        self.assertEqual(stored_stream_usage, expected)
                        self.assertEqual(
                            serving.response_store[request.request_id].model_dump()["usage"],
                            expected,
                        )
                        self.assertEqual(
                            fixture.request_metadata.final_usage_info,
                            metadata.final_usage_info,
                        )
                        self.assertEqual(metadata.final_usage_info.prompt_tokens, 11)
                        self.assertEqual(metadata.final_usage_info.completion_tokens, 7)

    def test_harmony_terminal_usage_preserves_nested_cached_tokens(self):
        from openai.types.responses import Response, ResponseCompletedEvent
        from sglang.srt.entrypoints.context import StreamingHarmonyContext

        for details_enabled in (False, True):
            with self.subTest(details_enabled=details_enabled):
                serving = make_serving()
                serving.use_harmony = True
                serving.enable_prompt_tokens_details = details_enabled
                # No generated content is needed to exercise terminal accounting;
                # avoid loading a Harmony tokenizer for this CPU-only boundary test.
                with patch(
                    "sglang.srt.entrypoints.context.get_streamable_parser_for_assistant",
                    return_value=Mock(current_content="", messages=[]),
                ):
                    context = StreamingHarmonyContext(messages=[], tool_sessions={})
                context.num_prompt_tokens = 11
                context.num_output_tokens = 7
                context.num_cached_tokens = 3
                context.num_reasoning_tokens = 2
                request = ResponsesRequest(model="x", input="hi", store=False)
                metadata = RequestResponseMetadata(request_id=request.request_id)

                async def empty_generator():
                    for _ in ():
                        yield None

                async def collect():
                    return await collect_stream_events(serving.responses_stream_generator(
                        request, {}, empty_generator(), context, "x",
                        serving.tokenizer_manager.tokenizer, metadata,
                    ))

                events = asyncio.run(collect())
                terminal = ResponseCompletedEvent.model_validate(find_completed_event(events))
                response = asyncio.run(serving.responses_full_generator(
                    request, {}, empty_generator(), context, "x",
                    serving.tokenizer_manager.tokenizer, metadata,
                ))
                parsed = Response.model_validate_json(response.model_dump_json())
                expected = {
                    "input_tokens": 11,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 18,
                }
                self.assertEqual(terminal.response.usage.model_dump(), expected)
                self.assertEqual(parsed.usage.model_dump(), expected)
                self.assertEqual(metadata.final_usage_info.prompt_tokens_details.cached_tokens, 3)

    def test_emits_typed_sse_events_in_order(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        fixture = _StreamFixture(serving, request)
        events = fixture.run(
            [
                _engine_chunk("Hel", 1),
                _engine_chunk("Hello", 2),
                _engine_chunk("Hello world", 4, finish=True),
            ]
        )

        types = event_types(events)
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[1], "response.in_progress")
        for ev in (
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            self.assertIn(ev, types)
        self.assertEqual(types[-1], "response.completed")

        seqs = [p["sequence_number"] for p in event_payloads(events)]
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_length_finish_emits_incomplete_event(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        events = _StreamFixture(serving, request).run(
            [
                _engine_chunk(
                    "partial",
                    16,
                    finish_reason={"type": "length", "length": 16},
                )
            ]
        )
        payloads = event_payloads(events)
        terminal = payloads[-1]

        self.assertEqual(event_types(events)[-1], "response.incomplete")
        self.assertNotIn("response.completed", event_types(events))
        self.assertEqual(terminal["response"]["status"], "incomplete")
        self.assertEqual(
            terminal["response"]["incomplete_details"],
            {"reason": "max_output_tokens"},
        )

    def test_reasoning_effort_none_streams_with_null_response_effort(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            reasoning={"effort": "none"},
        )
        events = _StreamFixture(serving, request).run(
            [_engine_chunk("OK", 1, finish=True)]
        )
        payloads = event_payloads(events)
        created = next(item for item in payloads if item["type"] == "response.created")
        completed = find_completed_event(events)

        self.assertIsNone(created["response"]["reasoning"]["effort"])
        self.assertIsNone(completed["response"]["reasoning"]["effort"])
        self.assertFalse(
            any(item["type"].startswith("response.reasoning_") for item in payloads)
        )

    def test_required_tool_choice_emits_function_call_events(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        payload = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        chunks = []
        sent = 0
        while sent < len(payload):
            sent += min(8, len(payload) - sent)
            chunks.append(
                _engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        types = event_types(events)

        self.assertIn("response.function_call_arguments.delta", types)
        self.assertIn("response.function_call_arguments.done", types)
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.output_item.done", types)
        self.assertNotIn("response.output_text.delta", types)

        added_kinds = [
            payload["item"]["type"]
            for payload in event_payloads(events)
            if payload.get("type") == "response.output_item.added"
        ]
        self.assertIn("function_call", added_kinds)

    def test_final_output_preserves_text_tool_text_order(self):
        from sglang.srt.function_call.core_types import (
            StreamingParseResult,
            ToolCallItem,
        )

        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        scripted = [
            StreamingParseResult(normal_text="I'll check.", calls=[]),
            StreamingParseResult(
                normal_text="",
                calls=[
                    ToolCallItem(
                        tool_index=0,
                        name="get_weather",
                        parameters='{"city": "Beijing"}',
                    )
                ],
            ),
            StreamingParseResult(normal_text="It's sunny.", calls=[]),
        ]
        chunks = [
            _engine_chunk(" " * 3, 3),
            _engine_chunk(" " * 10, 10),
            _engine_chunk(" " * 14, 14, finish=True),
        ]

        script_iter = iter(scripted)

        def fake_parse_stream_chunk(delta):
            sp = next(script_iter)
            return sp.normal_text, sp.calls

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            parser_cls.return_value.detector.supports_structural_tag.return_value = True
            parser_cls.return_value.parse_stream_chunk.side_effect = (
                fake_parse_stream_chunk
            )
            fixture = _StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")

    def test_multi_token_chunk_finishes_open_tool_call_before_text(self):
        """Speculative decoding streams several tokens per chunk, so one chunk
        can carry a call's closing arguments and the newline after
        </tool_call>. The parser returns that newline as normal text, which
        closed the open call before its closing arguments were applied: the
        real call lost its closing brace and a nameless call with arguments
        "}" followed, so agents saw malformed tool calls."""
        import json

        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "qwen3_coder"
        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "run_shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
        )
        head = "<tool_call>\n<function=run_shell>\n<parameter=command>\nls -la"
        full = head + "\n</parameter>\n</function>\n</tool_call>\n"

        events = _StreamFixture(serving, request).run(
            [_engine_chunk(head, 8), _engine_chunk(full, 16, finish=True)]
        )

        output = find_completed_event(events)["response"]["output"]
        calls = [item for item in output if item["type"] == "function_call"]
        self.assertEqual([call["name"] for call in calls], ["run_shell"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"command": "ls -la"})


if __name__ == "__main__":
    unittest.main()
