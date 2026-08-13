import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from utils import collect_stream_events, event_payloads, make_serving

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class InputMessageConstructionTestCase(unittest.TestCase):
    def test_previous_response_replays_assistant_text_not_instructions(self):
        serving = make_serving()
        prev_response = Mock(id="resp_prev")
        prev_response.output = [
            ResponseReasoningItem(
                id="rs_prev", summary=[], type="reasoning", content=None, status=None
            ),
            ResponseOutputMessage(
                id="msg_prev",
                content=[
                    ResponseOutputText(
                        text="first answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                    ResponseOutputText(
                        text="second answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        serving.msg_store["resp_prev"] = [{"role": "user", "content": "old input"}]

        request = ResponsesRequest(
            model="x",
            instructions="Be brief",
            previous_response_id="resp_prev",
            input="new input",
            store=False,
        )

        messages = serving._construct_input_messages(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "old input"},
                {
                    "role": "assistant",
                    "content": "first answer part\nsecond answer part",
                },
                {"role": "user", "content": "new input"},
            ],
        )

    def test_replayed_reasoning_strips_qwen_exo_self_check(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "reasoning",
                "content": [
                    {
                        "text": (
                            "Observed the failing wrapper.\n"
                            "<qwen_exo_self_check>\n"
                            "Self-question: Which boundary failed?\n"
                            "Self-answer: The generated path is stale.\n"
                            "</qwen_exo_self_check>\n"
                            "Inspect the generator next."
                        )
                    }
                ],
            }
        )

        self.assertEqual(
            normalized,
            {
                "role": "assistant",
                "reasoning_content": (
                    "Observed the failing wrapper.\nInspect the generator next."
                ),
            },
        )
        self.assertIsNone(
            OpenAIServingResponses._normalize_response_message_for_chat(
                {
                    "type": "reasoning",
                    "content": [
                        {"text": ("<qwen_exo_self_check>private</qwen_exo_self_check>")}
                    ],
                }
            )
        )

    def test_input_parts_normalized_for_chat_templates(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        messages = serving._construct_input_messages(request)

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "http://example.com/cat.png",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        )

    def test_previous_multimodal_tool_history_is_normalized(self):
        serving = make_serving()
        previous = Mock(id="resp_prev")
        previous.output = []
        serving.msg_store["resp_prev"] = [
            {
                "role": "tool",
                "tool_call_id": "call_screenshot",
                "content": [
                    {"type": "input_text", "text": "Took a screenshot."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    },
                ],
            }
        ]
        request = ResponsesRequest(
            model="x",
            previous_response_id="resp_prev",
            input="Continue.",
            store=False,
        )

        messages = serving._construct_input_messages(request, previous)

        self.assertEqual(messages[0]["role"], "tool")
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "text", "text": "Took a screenshot."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,AAAA",
                        "detail": "auto",
                    },
                },
            ],
        )
        self.assertEqual(messages[-1], {"role": "user", "content": "Continue."})

    def test_response_output_items_can_be_replayed_as_next_input(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "id": "msg_previous",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I will inspect the repository.",
                            "annotations": [],
                            "logprobs": None,
                        }
                    ],
                },
                {
                    "id": "fc_previous",
                    "type": "function_call",
                    "call_id": "call_previous",
                    "name": "bash",
                    "arguments": '{"command":"pwd"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_previous",
                    "output": "/workspace",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "bash",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )

        messages = serving._construct_input_messages(request)

        self.assertEqual(
            messages,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I will inspect the repository.",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call_previous",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_previous",
                    "content": "/workspace",
                },
            ],
        )

    def test_previous_response_id_input_list_does_not_call_copy_module(self):
        serving = make_serving()
        serving.use_harmony = True
        prev = Mock(id="resp_prev")
        prev.output = [
            ResponseFunctionToolCall(
                arguments="{}",
                call_id="call_x",
                name="t",
                type="function_call",
                id="fc_x",
                status="completed",
            )
        ]
        request = ResponsesRequest(
            model="x",
            input=[{"role": "user", "content": "hi"}],
            previous_response_id="resp_prev",
            store=False,
        )
        try:
            serving._construct_input_messages_with_harmony(request, prev)
        except TypeError as exc:
            self.fail(f"copy() module-call regression: {exc}")
        except Exception:
            pass


class ChatToolForwardingTestCase(unittest.TestCase):
    def test_make_request_passes_function_tools_to_chat_processing(self):
        serving = make_serving()
        seen = {}

        def fake_process(chat_request, is_multimodal):
            seen["tools"] = chat_request.tools
            seen["tool_choice"] = chat_request.tool_choice
            seen["parallel_tool_calls"] = chat_request.parallel_tool_calls
            return MessageProcessingResult(
                prompt="prompt",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=["</s>"],
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

        serving._process_messages = Mock(side_effect=fake_process)
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            store=False,
        )

        messages, request_prompts, engine_prompts, processed = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        self.assertEqual(messages, [{"role": "user", "content": "call the tool"}])
        self.assertEqual(request_prompts, [[1, 2, 3]])
        self.assertEqual(engine_prompts, [[1, 2, 3]])
        self.assertEqual(seen["tools"][0].function.name, "lookup")
        self.assertEqual(seen["tool_choice"], "required")
        self.assertFalse(seen["parallel_tool_calls"])
        self.assertEqual(processed.tool_call_constraint[0], "json_schema")

    def test_required_tool_choice_without_function_tool_returns_400(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[{"type": "web_search"}, {"type": "mcp"}],
            store=False,
        )
        result = asyncio.run(serving.create_responses(request, raw_request=None))
        self.assertEqual(getattr(result, "status_code", None), 400)


class InputItemNormalizationTestCase(unittest.TestCase):
    def test_function_call_becomes_assistant_tool_call(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "lookup",
                "arguments": '{"key": "val"}',
                "status": "completed",
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "val"}',
                        },
                    }
                ],
            },
        )

    def test_developer_role_becomes_system(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {"role": "developer", "content": "Be terse."}
        )
        self.assertEqual(normalized, {"role": "system", "content": "Be terse."})

    def test_function_call_output_becomes_tool_message(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42",
            }
        )
        self.assertEqual(
            normalized,
            {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        )

    def test_empty_function_call_output_gets_neutral_text(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_empty",
                "output": "",
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "tool",
                "tool_call_id": "call_empty",
                "content": "(tool returned no textual output)",
            },
        )

    def test_multimodal_function_call_output_is_normalized(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_screenshot",
                "output": [
                    {"type": "input_text", "text": "Took a screenshot."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    },
                ],
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "tool",
                "tool_call_id": "call_screenshot",
                "content": [
                    {"type": "text", "text": "Took a screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": "auto",
                        },
                    },
                ],
            },
        )

    def test_unknown_input_item_type_raises(self):
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_message_for_chat(
                {"type": "web_search_call", "id": "ws_1"}
            )


class FullResponseUsageTestCase(unittest.TestCase):
    def test_full_response_uses_dict_meta_info_for_usage(self):
        serving = make_serving()
        context = SimpleContext()
        context.last_output = {
            "text": "done",
            "meta_info": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "cached_tokens": 3,
                "reasoning_tokens": 2,
            },
        }
        request = ResponsesRequest(
            model="x", input="hello", request_id="resp_usage", store=False
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)

        async def empty_generator():
            for _ in ():
                yield None

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                sampling_params={},
                result_generator=empty_generator(),
                context=context,
                model_name="x",
                tokenizer=serving.tokenizer_manager.tokenizer,
                request_metadata=metadata,
                created_time=123,
            )
        )

        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(metadata.final_usage_info, response.usage)


class MultimodalRequestTestCase(unittest.TestCase):
    def test_multimodal_create_responses_sends_text_and_media_to_engine(self):
        serving = make_serving(is_multimodal=True)
        captured = {}

        serving._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="rendered multimodal prompt",
                prompt_ids=[9, 9, 9],
                image_data=["http://example.com/cat.png"],
                audio_data=None,
                video_data=None,
                modalities=["image"],
                stop=[],
            )
        )

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["request_prompt"] = request_prompt
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "looks like a cat",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            request_id="resp_mm",
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertEqual(captured["request_prompt"], "rendered multimodal prompt")
        self.assertEqual(captured["adapted_request"].text, "rendered multimodal prompt")
        self.assertIsNone(captured["adapted_request"].input_ids)
        self.assertEqual(
            captured["adapted_request"].image_data, ["http://example.com/cat.png"]
        )
        self.assertEqual(captured["adapted_request"].modalities, ["image"])


class OutputItemsTestCase(unittest.TestCase):
    def _function_tool_request(self):
        return ResponsesRequest(
            model="x",
            input="weather?",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

    def test_function_tool_call_extracted_via_parser(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = ("trailing text", [fake_call])
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output with <tool_call>",
                tokenizer=Mock(),
            )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')

        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, "trailing text")

    def test_prose_emitted_before_tool_call_item(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = (
                "I'll check the weather.",
                [fake_call],
            )
            output_items = serving._make_response_output_items(
                self._function_tool_request(), "raw model output", tokenizer=Mock()
            )

        types = [type(item).__name__ for item in output_items]
        self.assertEqual(types, ["ResponseOutputMessage", "ResponseFunctionToolCall"])

    def test_required_tool_choice_parses_json_array_without_native_parser(self):
        serving = make_serving()
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock()
        )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')
        self.assertEqual(
            [item for item in output_items if isinstance(item, ResponseOutputMessage)],
            [],
        )

    def test_no_tool_call_extraction_when_tool_choice_none(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            output_items = serving._make_response_output_items(
                request, "just a plain answer", tokenizer=Mock()
            )
            parser_cls.assert_not_called()

        self.assertEqual(len(output_items), 1)
        self.assertIsInstance(output_items[0], ResponseOutputMessage)


class HarmonyResponsesTestCase(unittest.TestCase):
    def test_public_store_keeps_actions_and_tools_without_private_thinking(self):
        from openai_harmony import Role

        serving = make_serving()
        request = ResponsesRequest(
            model="x", input="hello", request_id="resp_history", store=True
        )
        serving.msg_store[request.request_id] = ["public-system", "user"]
        analysis = Mock(
            author=Mock(role=Role.ASSISTANT),
            recipient=None,
            channel="analysis",
        )
        action = Mock(
            author=Mock(role=Role.ASSISTANT),
            recipient="browser.search",
            channel="analysis",
        )
        tool_result = Mock(
            author=Mock(role=Role.TOOL), recipient="assistant", channel="analysis"
        )
        private_developer = Mock(
            author=Mock(role=Role.DEVELOPER),
            recipient=None,
            channel="analysis",
        )
        final = Mock(
            author=Mock(role=Role.ASSISTANT),
            recipient=None,
            channel="final",
        )
        context_messages = [
            "private-system",
            "user",
            analysis,
            action,
            tool_result,
            private_developer,
            final,
        ]
        context = Mock(
            messages=context_messages,
            _messages=context_messages,
            num_init_messages=2,
        )

        serving._store_public_harmony_messages(request, context)

        self.assertEqual(
            serving.msg_store[request.request_id],
            ["public-system", "user", action, tool_result, final],
        )
        self.assertNotIn(analysis, serving.msg_store[request.request_id])
        self.assertNotIn(private_developer, serving.msg_store[request.request_id])

    def test_streaming_harmony_store_uses_parser_messages_without_initial_slice(self):
        from openai_harmony import Role

        serving = make_serving()
        request = ResponsesRequest(
            model="x", input="hello", request_id="resp_stream_history", store=True
        )
        serving.msg_store[request.request_id] = ["public-system", "user"]
        final = Mock(
            author=Mock(role=Role.ASSISTANT),
            recipient=None,
            channel="final",
        )
        context = Mock(
            messages=[final],
            _messages=["private-system", "user"],
            num_init_messages=2,
        )

        serving._store_public_harmony_messages(request, context)

        self.assertEqual(
            serving.msg_store[request.request_id],
            ["public-system", "user", final],
        )

    def test_cancelled_previous_response_is_not_replayable(self):
        serving = make_serving()
        serving.response_store["resp_cancelled"] = Mock(status="cancelled")
        request = ResponsesRequest(
            model="x",
            input="continue",
            previous_response_id="resp_cancelled",
        )

        response = asyncio.run(serving.prevalidate_qwen_exo_request(request))

        self.assertEqual(response.status_code, 409)
        self.assertIn(b"not a completed, replayable response", response.body)

    def test_pending_cancel_returns_schema_complete_response(self):
        from openai.types.responses import Response

        serving = make_serving()
        serving.tokenizer_manager.served_model_name = "x"

        response = asyncio.run(
            serving.register_pending_cancelled_response("resp_pending_cancel")
        )
        sdk_response = Response.model_validate(response.model_dump())

        self.assertEqual(sdk_response.id, "resp_pending_cancel")
        self.assertEqual(sdk_response.status, "cancelled")
        self.assertEqual(sdk_response.model, "x")
        self.assertEqual(sdk_response.output, [])

    def test_synchronous_stream_is_cancellable_while_in_progress(self):
        serving = make_serving()
        serving.tokenizer_manager.abort_request = Mock()
        request = ResponsesRequest(
            model="x",
            input="keep generating",
            request_id="resp_sync_stream",
            stream=True,
            store=True,
        )

        pending = asyncio.run(
            serving.register_in_progress_response(
                request,
                {"max_new_tokens": 8192},
                model_name="x",
                created_time=1,
            )
        )
        cancelled = asyncio.run(serving.cancel_responses(request.request_id))
        repeated = asyncio.run(serving.cancel_responses(request.request_id))

        self.assertEqual(pending.status, "cancelled")
        self.assertIs(cancelled, pending)
        self.assertIs(repeated, pending)
        serving.tokenizer_manager.abort_request.assert_called_once_with(
            rid=request.request_id
        )

    def test_pending_cancellation_wins_registration_race(self):
        serving = make_serving()
        serving.tokenizer_manager.served_model_name = "x"
        request = ResponsesRequest(
            model="x",
            input="keep generating",
            request_id="resp_cancel_race",
            stream=True,
            store=True,
        )

        cancelled = asyncio.run(
            serving.register_pending_cancelled_response(request.request_id)
        )
        registered = asyncio.run(
            serving.register_in_progress_response(
                request,
                {"max_new_tokens": 8192},
                model_name="x",
                created_time=1,
            )
        )

        self.assertIs(registered, cancelled)
        self.assertEqual(registered.status, "cancelled")

    def test_generation_abort_accepts_null_status_code(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as raised:
            OpenAIServingResponses._raise_for_generation_abort(
                {
                    "type": "abort",
                    "status_code": None,
                    "message": "Request cancelled",
                }
            )

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail["message"], "Request cancelled")

    def test_scheduler_overload_maps_to_sdk_rate_limit_error(self):
        from fastapi import HTTPException
        from openai.types.responses import Response
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        error, diagnostics = OpenAIServingResponses._public_response_error(
            HTTPException(
                status_code=429,
                detail={
                    "message": "capacity exhausted",
                    "code": "qwen_exo_capacity_exhausted",
                    "retry_after": 1,
                },
            )
        )
        response = ResponsesResponse(
            id="resp_overload",
            model="x",
            status="failed",
            error=error,
            metadata=diagnostics,
        )

        sdk_response = Response.model_validate(response.model_dump())

        self.assertEqual(sdk_response.error.code, "rate_limit_exceeded")
        self.assertEqual(
            sdk_response.metadata["qwen_exo_error_code"],
            "qwen_exo_capacity_exhausted",
        )

    def test_nested_response_metadata_is_json_encoded_as_strings(self):
        metadata = OpenAIServingResponses._stringify_response_metadata(
            {"trajectory_id": "trace-1", "memory": {"status": "prepared"}}
        )

        self.assertEqual(metadata["trajectory_id"], "trace-1")
        self.assertEqual(metadata["memory"], '{"status":"prepared"}')
        self.assertTrue(all(isinstance(value, str) for value in metadata.values()))

    def test_qwen_non_harmony_rejects_unexecuted_builtin_tools(self):
        serving = make_serving()
        serving.use_harmony = False
        request = ResponsesRequest(
            model="x",
            input="search",
            tools=[{"type": "web_search"}],
        )

        response = asyncio.run(serving.prevalidate_qwen_exo_request(request))

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not supported", response.body)

    def test_developer_message_skips_unsupported_tool_types(self):
        from sglang.srt.entrypoints.harmony_utils import get_developer_message
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        tools = [
            ResponseTool(
                type="function",
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object"},
            ),
            ResponseTool(type="web_search"),
            ResponseTool(type="namespace", name="codex"),
            ResponseTool(type="mcp"),
        ]
        msg = get_developer_message(instructions="be helpful", tools=tools)
        self.assertIsNotNone(msg)


class ResponsesThinkingDefaultTestCase(unittest.TestCase):
    def test_default_template_disables_thinking_when_reasoning_is_omitted(self):
        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        serving.default_chat_template_kwargs = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
        serving.template_manager.reasoning_config = SimpleNamespace(
            toggle_param="enable_thinking",
            default_enabled=True,
            special_case=None,
        )

        self.assertFalse(
            serving._is_thinking_enabled_for_request(
                ResponsesRequest(model="x", input="inspect", store=False)
            )
        )

    def test_explicit_reasoning_overrides_disabled_template_default(self):
        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        serving.default_chat_template_kwargs = {"enable_thinking": False}
        serving.template_manager.reasoning_config = SimpleNamespace(
            toggle_param="enable_thinking",
            default_enabled=True,
            special_case=None,
        )

        self.assertTrue(
            serving._is_thinking_enabled_for_request(
                ResponsesRequest(
                    model="x",
                    input="inspect",
                    reasoning={"effort": "high"},
                    store=False,
                )
            )
        )


class NativeThinkContinuationTestCase(unittest.TestCase):
    def test_phase_two_self_ask_spill_is_partitioned_from_final_text(self):
        spill = (
            "\n\nSelf-question: What does the challenge say?\n"
            "Self-answer: Padding Oracle FTW.\n"
            "</think>\n\n"
            "Self-question: What is the likely type?\n"
            "Self-answer: Padding Oracle.\n"
            "</think>\n\nFinal answer"
        )

        reasoning, normal = OpenAIServingResponses._partition_qwen_exo_self_ask_spill(
            spill
        )

        self.assertIn("Self-question: What does the challenge say?", reasoning)
        self.assertIn("Self-question: What is the likely type?", reasoning)
        self.assertNotIn("</think>", reasoning)
        self.assertEqual(normal, "\n\nFinal answer")

    def test_complete_response_moves_phase_two_self_ask_into_reasoning(self):
        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        request = ResponsesRequest(model="x", input="inspect", store=False)
        parser = Mock()
        parser.parse_non_stream.return_value = (
            "Initial reasoning.\nSelf-question: Official?\nSelf-answer: Yes.\n",
            (
                "\n\nSelf-question: Follow-up?\n"
                "Self-answer: More evidence.\n</think>\n\nFinal answer"
            ),
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser",
            return_value=parser,
        ):
            items = serving._make_response_output_items(
                request,
                "ignored",
                serving.tokenizer_manager.tokenizer,
            )

        self.assertEqual([item.type for item in items], ["reasoning", "message"])
        reasoning = items[0].content[0].text
        answer = items[1].content[0].text
        self.assertIn("Self-question: Official?", reasoning)
        self.assertIn("Self-question: Follow-up?", reasoning)
        self.assertNotIn("</think>", reasoning)
        self.assertEqual(answer, "\n\nFinal answer")

    def test_streamed_phase_two_self_ask_stays_in_reasoning_and_complete(self):
        class FakeReasoningParser:
            def __init__(self, **_kwargs):
                self.in_reasoning = True

            def parse_stream_chunk(self, text):
                if not self.in_reasoning:
                    return None, text
                before, boundary, after = text.partition("</think>")
                before = before.replace("<think>", "", 1)
                if boundary:
                    self.in_reasoning = False
                    return before, after
                return before, ""

        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        serving.tokenizer_manager.tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: list(range(len(str(text).split())))
        )
        request = ResponsesRequest(
            model="x",
            input="inspect",
            request_id="resp-self-ask-spill",
            stream=True,
            store=True,
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)
        prefix = (
            "<think>Initial reasoning.\n\n"
            "Self-question: Official?\nSelf-answer: Yes.\n</think>"
        )
        spill = (
            "\n\nSelf-question: Follow-up one?\n"
            "Self-answer: Evidence one.\n</think>\n\n"
            "Self-question: Follow-up two?\n"
            "Self-answer: Evidence two.\n</think>"
        )
        final_text = "\n\nFinal answer"

        async def results():
            yield {
                "text": prefix,
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "reasoning_tokens": 8,
                    "qwen_exo_self_ask_boundary": True,
                },
            }
            yield {
                "text": prefix + spill + final_text,
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 30,
                    "reasoning_tokens": 8,
                    "finish_reason": {"type": "stop", "matched": 0},
                },
            }

        async def collect():
            with patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser",
                FakeReasoningParser,
            ):
                return await collect_stream_events(
                    serving.responses_stream_generator_non_harmony(
                        request,
                        {},
                        results(),
                        "x",
                        serving.tokenizer_manager.tokenizer,
                        metadata,
                    )
                )

        payloads = event_payloads(asyncio.run(collect()))
        reasoning_deltas = "".join(
            payload.get("delta", "")
            for payload in payloads
            if payload.get("type") == "response.reasoning_text.delta"
        )
        answer_deltas = "".join(
            payload.get("delta", "")
            for payload in payloads
            if payload.get("type") == "response.output_text.delta"
        )
        completed = next(
            payload["response"]
            for payload in payloads
            if payload.get("type") == "response.completed"
        )

        self.assertIn("Self-question: Official?", reasoning_deltas)
        self.assertIn("Self-question: Follow-up one?", reasoning_deltas)
        self.assertIn("Self-question: Follow-up two?", reasoning_deltas)
        self.assertNotIn("</think>", reasoning_deltas)
        self.assertEqual(answer_deltas, final_text)
        self.assertEqual(
            [item["type"] for item in completed["output"]],
            ["reasoning", "message"],
        )
        self.assertNotIn("Self-question:", completed["output"][1]["content"][0]["text"])
        self.assertGreater(
            completed["usage"]["output_tokens_details"]["reasoning_tokens"], 8
        )

    def test_score_bias_logprobs_start_at_earliest_capture_span(self):
        prompt = list(range(1000))

        self.assertEqual(
            OpenAIServingResponses._score_bias_logprob_start_len(
                prompt,
                ({"start": 720, "end": 848}, {"start": 600, "end": 700}),
            ),
            599,
        )
        self.assertEqual(
            OpenAIServingResponses._score_bias_logprob_start_len(prompt, ()),
            len(prompt),
        )
        self.assertEqual(
            OpenAIServingResponses._score_bias_logprob_start_len(
                prompt, ({"start": -1, "end": 10}, {"start": 900, "end": 1200})
            ),
            len(prompt),
        )

    def test_self_ask_is_committed_inside_think_before_final_answer(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [200 + index for index, _ in enumerate(text.split())]

            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "</think>" if token_ids == [99] else ""

        class FakeTokenizerManager:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.server_args = SimpleNamespace(incremental_streaming_output=False)
                self.model_config = SimpleNamespace(context_len=1024)
                self.num_reserved_tokens = 0
                self.requests = []

            def generate_request(self, request, _raw_request):
                self.requests.append(request)
                request_index = len(self.requests)

                async def generate():
                    if request_index == 1:
                        yield {
                            "text": "<think>draft",
                            "output_ids": [10, 11],
                            "meta_info": {
                                "prompt_tokens": 2,
                                "cached_tokens": 0,
                                "completion_tokens": 2,
                                "finish_reason": {"type": "stop", "matched": 99},
                            },
                        }
                    else:
                        yield {
                            "text": "final",
                            "output_ids": [20],
                            "meta_info": {
                                "prompt_tokens": len(request.input_ids),
                                "completion_tokens": 1,
                                "output_token_logprobs": [-0.1],
                            },
                        }
                        yield {
                            "text": "final answer",
                            "output_ids": [20, 21],
                            "meta_info": {
                                "prompt_tokens": len(request.input_ids),
                                "completion_tokens": 2,
                                "output_token_logprobs": [-0.1, -0.2],
                                "finish_reason": {"type": "stop", "matched": 0},
                            },
                        }

                return generate()

        class FakeRuntime:
            reasoning_end_token_id = 99
            think_context_enabled = True

            def __init__(self):
                self.observed = []
                self.boundaries = []

            def register_generation_prompt(self, *_args, **_kwargs):
                pass

            def observe_generation_result(self, _request_id, result, **kwargs):
                self.observed.append((result, kwargs))

            async def await_think_context(self, _request_id):
                return SimpleNamespace(
                    text="\n\nSelf-question: Which invariant failed?\n"
                    "Self-answer: The wrapper omitted the final callsite.\n"
                )

            def record_reasoning_boundary(self, request_id, **kwargs):
                self.boundaries.append((request_id, kwargs))

        serving = object.__new__(OpenAIServingResponses)
        serving.tokenizer_manager = FakeTokenizerManager()
        runtime = FakeRuntime()
        raw_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(qwen_exo_runtime=runtime))
        )
        adapted_request = GenerateReqInput(
            input_ids=[1, 2],
            sampling_params={"max_new_tokens": 32},
            stream=False,
            rid="resp-native-think",
        )
        context = SimpleContext()

        async def collect():
            snapshots = []
            async for current in serving._generate_with_builtin_tools(
                "resp-native-think",
                [1, 2],
                adapted_request,
                {"max_new_tokens": 32},
                context,
                raw_request=raw_request,
            ):
                snapshots.append(dict(current.last_output))
            return snapshots

        snapshots = asyncio.run(collect())

        self.assertEqual(
            serving.tokenizer_manager.requests[0].sampling_params["stop_token_ids"],
            [99],
        )
        self.assertEqual(serving.tokenizer_manager.requests[1].input_ids[-1], 99)
        self.assertNotIn("finish_reason", snapshots[0]["meta_info"])
        self.assertIn("Self-question: Which invariant failed?", snapshots[1]["text"])
        self.assertIn("Self-answer: The wrapper omitted", snapshots[1]["text"])
        self.assertTrue(snapshots[1]["text"].endswith("</think>"))
        self.assertTrue(snapshots[1]["meta_info"]["qwen_exo_self_ask_boundary"])
        self.assertEqual(
            snapshots[-1]["text"],
            snapshots[1]["text"] + "final answer",
        )
        self.assertEqual(runtime.boundaries[0][0], "resp-native-think")
        self.assertEqual(runtime.boundaries[0][1]["token_ids"][-1], 99)
        self.assertEqual(runtime.observed[-2][0]["output_ids"], [20])
        self.assertEqual(runtime.observed[-1][0]["output_ids"], [21])
        self.assertEqual(
            runtime.observed[-1][0]["meta_info"]["output_token_logprobs"],
            [-0.2],
        )

    def test_reasoning_budget_forces_boundary_without_self_ask(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [200 + index for index, _ in enumerate(text.split())]

            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "</think>" if token_ids == [99] else ""

        class FakeTokenizerManager:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.server_args = SimpleNamespace(incremental_streaming_output=False)
                self.model_config = SimpleNamespace(context_len=1024)
                self.num_reserved_tokens = 0
                self.requests = []

            def generate_request(self, request, _raw_request):
                self.requests.append(request)
                request_index = len(self.requests)

                async def generate():
                    if request_index == 1:
                        yield {
                            "text": "<think>bounded reasoning",
                            "output_ids": [10, 11, 12, 13],
                            "meta_info": {
                                "prompt_tokens": 2,
                                "completion_tokens": 4,
                                "finish_reason": {"type": "length"},
                            },
                        }
                    else:
                        yield {
                            "text": "final action",
                            "output_ids": [20, 21],
                            "meta_info": {
                                "prompt_tokens": len(request.input_ids),
                                "completion_tokens": 2,
                                "finish_reason": {"type": "stop", "matched": 0},
                            },
                        }

                return generate()

        class FakeRuntime:
            reasoning_end_token_id = 99
            think_context_enabled = True
            max_reasoning_tokens = 4

            def __init__(self):
                self.discarded = []
                self.boundaries = []

            def register_generation_prompt(self, *_args, **_kwargs):
                pass

            def observe_generation_result(self, *_args, **_kwargs):
                pass

            async def await_think_context(self, _request_id):
                raise AssertionError("Forced reasoning boundary must skip Self-Ask")

            async def discard_think_context_for_reasoning_budget(
                self, request_id, **kwargs
            ):
                self.discarded.append((request_id, kwargs))

            def record_reasoning_boundary(self, request_id, **kwargs):
                self.boundaries.append((request_id, kwargs))

        serving = object.__new__(OpenAIServingResponses)
        serving.tokenizer_manager = FakeTokenizerManager()
        runtime = FakeRuntime()
        raw_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(qwen_exo_runtime=runtime))
        )
        adapted_request = GenerateReqInput(
            input_ids=[1, 2],
            sampling_params={"max_new_tokens": 10},
            stream=False,
            rid="resp-reasoning-budget",
        )
        context = SimpleContext()

        async def collect():
            snapshots = []
            async for current in serving._generate_with_builtin_tools(
                "resp-reasoning-budget",
                [1, 2],
                adapted_request,
                {"max_new_tokens": 10},
                context,
                raw_request=raw_request,
            ):
                snapshots.append(dict(current.last_output))
            return snapshots

        snapshots = asyncio.run(collect())

        self.assertEqual(
            serving.tokenizer_manager.requests[0].sampling_params["max_new_tokens"],
            4,
        )
        self.assertNotIn("finish_reason", snapshots[0]["meta_info"])
        self.assertTrue(snapshots[1]["text"].endswith("</think>"))
        self.assertNotIn("Self-question", snapshots[1]["text"])
        self.assertFalse(snapshots[1]["meta_info"]["qwen_exo_self_ask_boundary"])
        self.assertEqual(serving.tokenizer_manager.requests[1].input_ids[-1], 99)
        self.assertEqual(
            serving.tokenizer_manager.requests[1].sampling_params["max_new_tokens"],
            5,
        )
        self.assertEqual(runtime.discarded[0][1]["observed_tokens"], 4)
        self.assertIsNone(runtime.boundaries[0][1]["injection"])


if __name__ == "__main__":
    unittest.main()
