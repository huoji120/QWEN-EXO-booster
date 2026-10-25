# isort: skip_file
import unittest

# Import the CPU bootstrap before SGLang; it installs Windows test stubs.
from utils import make_serving  # noqa: F401
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, UsageInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class ResponsesRequestTestCase(unittest.TestCase):
    def test_prompt_cache_key_is_preserved(self):
        request = ResponsesRequest.model_validate(
            {
                "model": "x",
                "input": [{"role": "user", "content": "task"}],
                "prompt_cache_key": "opencode-session-123",
            }
        )

        self.assertEqual(request.prompt_cache_key, "opencode-session-123")

    def test_function_tool_accepted(self):
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    "strict": True,
                }
            ],
            store=False,
        )

        self.assertEqual(request.tools[0].type, "function")
        self.assertEqual(request.tools[0].name, "lookup")
        self.assertTrue(request.tools[0].strict)

    def test_function_tool_requires_name(self):
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x", input="hi", tools=[{"type": "function"}], store=False
            )
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function", "name": ""}],
                store=False,
            )

    def test_extended_tool_types_accepted(self):
        for tool_type in (
            "web_search",
            "web_search_preview",
            "code_interpreter",
            "file_search",
            "image_generation",
            "computer_use_preview",
            "local_shell",
            "mcp",
            "custom",
            "namespace",
        ):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": tool_type}],
                store=False,
            )
            self.assertEqual(request.tools[0].type, tool_type)

    def test_namespace_tool_carries_inner_tools_list(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            tools=[
                {
                    "type": "namespace",
                    "name": "codex",
                    "tools": [
                        {"type": "function", "name": "apply_patch"},
                        {"type": "function", "name": "shell"},
                    ],
                }
            ],
            store=False,
        )
        self.assertEqual(request.tools[0].type, "namespace")
        self.assertEqual(len(request.tools[0].tools), 2)
        self.assertEqual(request.tools[0].tools[0]["name"], "apply_patch")


class ResponsesSamplingParamsTestCase(unittest.TestCase):
    def test_processed_stop_and_tool_constraint_propagate(self):
        request = ResponsesRequest(model="x", input="call the tool", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            stop=["</s>"],
            tool_call_constraint=("json_schema", {"type": "object"}),
        )
        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_constraint_conflict_raises(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={"json_schema": '{"type": "object"}'},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

    def test_structural_tag_with_model_dump(self):
        class _FakeStructuralTag:
            def model_dump(self, by_alias=False):
                return {"type": "structural_tag"}

        request = ResponsesRequest(model="x", input="hi", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            tool_call_constraint=("structural_tag", _FakeStructuralTag()),
        )
        self.assertEqual(params["structural_tag"], '{"type": "structural_tag"}')


class ResponsesResponseFromRequestTestCase(unittest.TestCase):
    def test_usage_json_is_sdk_compatible_and_roundtrips_without_loss(self):
        from openai.types.responses import Response
        from sglang.srt.entrypoints.openai.protocol import (
            PromptTokensDetails,
            ResponsesResponse,
        )

        for cached_tokens in (0, 9):
            with self.subTest(cached_tokens=cached_tokens):
                usage = UsageInfo(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
                    reasoning_tokens=2,
                )
                response = ResponsesResponse.from_request(
                    ResponsesRequest(model="x", input="hi", store=False),
                    {}, "x", 123, [], "completed", usage,
                )
                expected = {
                    "input_tokens": 11,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                    "output_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 18,
                }
                self.assertEqual(response.model_dump()["usage"], expected)
                parsed = Response.model_validate_json(response.model_dump_json())
                self.assertEqual(parsed.usage.model_dump(), expected)
                restored = ResponsesResponse.model_validate_json(response.model_dump_json())
                self.assertEqual(restored.model_dump()["usage"], expected)
                self.assertEqual(usage.prompt_tokens, 11)
                self.assertNotIn("input_tokens", usage.model_dump())

    def test_absent_optional_usage_details_become_zero_not_null(self):
        from openai.types.responses import Response
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        response = ResponsesResponse.from_request(
            ResponsesRequest(model="x", input="hi", store=False),
            {}, "x", 123, [], "completed",
            UsageInfo(prompt_tokens=11, completion_tokens=None, reasoning_tokens=None,
                      total_tokens=11),
        )
        parsed = Response.model_validate_json(response.model_dump_json())
        self.assertEqual(parsed.usage.input_tokens_details.cached_tokens, 0)
        self.assertEqual(parsed.usage.output_tokens, 0)
        self.assertEqual(parsed.usage.output_tokens_details.reasoning_tokens, 0)

    def test_unfinished_response_usage_stays_null(self):
        from openai.types.responses import Response
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        for status in ("queued", "in_progress", "cancelled", "failed"):
            with self.subTest(status=status):
                response = ResponsesResponse.from_request(
                    ResponsesRequest(model="x", input="hi", store=False),
                    {}, "x", 123, [], status, None,
                )
                self.assertIsNone(response.model_dump()["usage"])
                self.assertIsNone(Response.model_validate_json(response.model_dump_json()).usage)

    def test_openapi_exposes_responses_usage_not_chat_usage(self):
        from fastapi import FastAPI
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        app = FastAPI()

        @app.get("/responses/{response_id}", response_model=ResponsesResponse)
        def retrieve_response(response_id: str):
            raise NotImplementedError("Schema-only route")

        schemas = app.openapi()["components"]["schemas"]
        usage_ref = next(
            item["$ref"]
            for item in schemas["ResponsesResponse"]["properties"]["usage"]["anyOf"]
            if "$ref" in item
        )
        usage_schema = schemas[usage_ref.rsplit("/", 1)[-1]]
        fields = {
            "input_tokens", "input_tokens_details", "output_tokens",
            "output_tokens_details", "total_tokens",
        }
        self.assertEqual(set(usage_schema["properties"]), fields)
        self.assertEqual(set(usage_schema["required"]), fields)
        for name, detail in (
            ("input_tokens_details", "cached_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ):
            detail_ref = usage_schema["properties"][name]["$ref"]
            self.assertIn(detail, schemas[detail_ref.rsplit("/", 1)[-1]]["required"])

    def test_parallel_tool_calls_false_preserved(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        request = ResponsesRequest(
            model="x", input="hi", parallel_tool_calls=False, store=False
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertFalse(response.parallel_tool_calls)

    def test_native_xhigh_serializes_as_openai_high(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        for effort in ("xhigh", "max"):
            with self.subTest(effort=effort):
                request = ResponsesRequest(
                    model="x",
                    input="hi",
                    reasoning={"effort": effort},
                    store=False,
                )
                response = ResponsesResponse.from_request(
                    request,
                    sampling_params={},
                    model_name="x",
                    created_time=0,
                    output=[],
                    status="in_progress",
                    usage=None,
                )
                self.assertEqual(response.reasoning["effort"], "high")


if __name__ == "__main__":
    unittest.main()
