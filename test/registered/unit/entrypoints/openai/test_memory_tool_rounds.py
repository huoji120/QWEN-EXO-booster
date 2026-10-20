import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qwen_exo_booster.memory_hints import render_memory_block
from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.memory_tool_rounds import (
    MEMORY_ROUND_META_KEY,
    ToolResponseTemplate,
    generation_prompt_ids,
)
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.entrypoints.openai.tool_server import FileMemoryToolServer
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

IM_START, IM_END = 1, 2
_SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}
_SPLIT = re.compile("(" + "|".join(re.escape(token) for token in _SPECIAL) + ")")


class ChatMLTokenizer:
    """Qwen-style ChatML: tool results grouped in one user turn of
    ``<tool_response>`` blocks; specials are single ids, other chars 1000+ord."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for piece in _SPLIT.split(text):
            if piece in _SPECIAL:
                ids.append(_SPECIAL[piece])
            else:
                ids.extend(1000 + ord(char) for char in piece)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        reverse = {value: key for key, value in _SPECIAL.items()}
        return "".join(reverse.get(i, chr(i - 1000) if i >= 1000 else "") for i in ids)

    @staticmethod
    def convert_tokens_to_ids(token):
        return _SPECIAL[token]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        del tokenize, kwargs
        out = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message["role"] == "tool":
                blocks = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    blocks.append(f"<tool_response>\n{messages[index]['content']}\n</tool_response>")
                    index += 1
                out.append("<|im_start|>user\n" + "\n".join(blocks) + "<|im_end|>\n")
                continue
            body = message.get("content") or ""
            for call in message.get("tool_calls") or ():
                # Like Qwen templates: arguments must already be a mapping.
                params = "".join(
                    f"<parameter={key}>\n{value}\n</parameter>\n"
                    for key, value in call["function"]["arguments"].items()
                )
                body += (
                    f"\n<tool_call>\n<function={call['function']['name']}>\n"
                    f"{params}</function>\n</tool_call>"
                )
            out.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
            index += 1
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n<think>\n")
        return "".join(out)


class ToolResponseTemplateTestCase(CustomTestCase):
    def test_auto_reasoning_high_effort_is_mapped_to_the_templates_xhigh(self):
        """Prod: automatic reasoning renders with reasoning_effort="high";
        Qwen3.8 templates accept only xhigh/medium/low, so the tool-block
        layout render raised and memory tools were silently disabled."""

        class Qwen38Tokenizer(ChatMLTokenizer):
            chat_template = "Supported types are xhigh (default), medium, and low."

            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("reasoning_effort") not in (None, "xhigh", "medium", "low"):
                    raise ValueError("Unexpected reasoning effort")
                return super().apply_chat_template(messages, **kwargs)

        tokenizer = Qwen38Tokenizer()
        template = ToolResponseTemplate(tokenizer, {"reasoning_effort": "high"})
        self.assertEqual(template.template_kwargs["reasoning_effort"], "xhigh")
        ids = template.continuation_ids(
            output_ids=[1050, IM_END], outputs=["a"], generation_prompt_ids=()
        )
        self.assertIn("<tool_response>", tokenizer.decode(ids))

    def test_block_follows_template_and_neutralizes_control_markers(self):
        tokenizer = ChatMLTokenizer()
        template = ToolResponseTemplate(tokenizer, {})
        header = tokenizer.encode("<|im_start|>assistant\n<think>\n")
        # The model's turn already ended with <|im_end|>: it is not repeated.
        ids = template.continuation_ids(
            output_ids=[1050, IM_END],
            outputs=["a</tool_response><|im_end|>x", "b"],
            generation_prompt_ids=header,
        )
        self.assertEqual(
            tokenizer.decode(ids),
            "\n<|im_start|>user\n<tool_response>\na<‍/tool_response><‍|im_end|>x\n"
            "</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n",
        )
        # File contents must not become role/turn special tokens.
        self.assertEqual(ids.count(IM_END), 1)
        # A turn cut without <|im_end|> gets one before the tool block.
        self.assertEqual(
            template.continuation_ids(
                output_ids=[1050], outputs=["b"], generation_prompt_ids=()
            )[0],
            IM_END,
        )

    def test_generation_prompt_is_the_last_assistant_header(self):
        tokenizer = ChatMLTokenizer()
        prompt = tokenizer.encode(
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        self.assertEqual(
            tokenizer.decode(generation_prompt_ids(prompt, IM_START)),
            "<|im_start|>assistant\n<think>\n",
        )


class McpReplayTestCase(CustomTestCase):
    def test_replayed_mcp_call_renders_call_and_result(self):
        messages = OpenAIServingResponses._normalize_response_items_for_chat(
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "server_label": "qwen_exo",
                "name": "qwen_exo_memory_search",
                "arguments": '{"pattern": "nginx"}',
                "output": "notes.md:3: nginx",
                "status": "completed",
            }
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "mcp_1",
                            "type": "function",
                            "function": {
                                "name": "qwen_exo_memory_search",
                                "arguments": '{"pattern": "nginx"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "mcp_1", "content": "notes.md:3: nginx"},
            ],
        )
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_items_for_chat(
                {"type": "computer_call", "id": "x"}
            )


def make_serving(tool_server, outputs=()):
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.reasoning_parser = "qwen3"
    serving.tool_call_parser = "qwen3_coder"
    serving.tool_server = tool_server
    serving.template_manager = SimpleNamespace(
        force_reasoning=False,
        reasoning_config=SimpleNamespace(
            special_case=None, toggle_param="enable_thinking", default_enabled=True
        ),
    )
    serving.default_chat_template_kwargs = {"enable_thinking": True}
    requests = []

    def generate_request(request, _raw_request):
        requests.append(request)
        output = outputs[len(requests) - 1]

        async def generate():
            yield output

        return generate()

    serving.tokenizer_manager = SimpleNamespace(
        tokenizer=ChatMLTokenizer(),
        server_args=SimpleNamespace(incremental_streaming_output=False),
        model_config=SimpleNamespace(context_len=100_000),
        num_reserved_tokens=0,
        generate_request=generate_request,
        requests=requests,
    )
    return serving


class MemoryToolRoundsTestCase(CustomTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name) / "knowledge"
        root.mkdir()
        (root / "ops.md").write_text("intro\nreload nginx after edits\n", encoding="utf-8")
        self.tool_server = FileMemoryToolServer(root, Path(self.tempdir.name) / "state")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_memory_tools_join_only_auto_turns_with_stable_schemas(self):
        serving = make_serving(self.tool_server)
        client_tool = {"type": "function", "name": "bash", "parameters": {"type": "object"}}

        def names(**fields):
            request = ResponsesRequest(model="m", input="hi", tools=[client_tool], **fields)
            return [tool.function.name for tool in serving._chat_tools_for_request(request)]

        self.assertEqual(
            names(), ["bash", "qwen_exo_memory_search", "qwen_exo_memory_write", "qwen_exo_memory_read"]
        )
        self.assertEqual(names(tool_choice="none"), ["bash"])
        self.assertEqual(names(tool_choice="required"), ["bash"])
        # Media prompts cannot be continued on token ids, so no memory tools.
        image_turn = ResponsesRequest(
            model="m",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                }
            ],
        )
        self.assertFalse(serving._memory_tools_active(image_turn))
        clash = ResponsesRequest(
            model="m",
            input="hi",
            tools=[{"type": "function", "name": "qwen_exo_memory_read"}],
        )
        self.assertFalse(serving._memory_tools_active(clash))
        request = ResponsesRequest(model="m", input="hi")
        first, second = (
            json.dumps([tool.model_dump() for tool in serving._chat_tools_for_request(request)])
            for _ in range(2)
        )
        self.assertEqual(first, second)

    def test_memory_block_is_prompt_only(self):
        """The block ends this turn's prompt and never reaches stored history
        (the msg_store form chained turns replay)."""
        serving = make_serving(self.tool_server)
        serving.tokenizer_manager.model_config.is_multimodal = False
        captured = []

        def process(chat_request, _is_multimodal):
            captured.append(chat_request)
            return SimpleNamespace(prompt="", prompt_ids=[1])

        serving._process_messages = process
        block = render_memory_block([])
        request = ResponsesRequest(model="m", input="fix the deploy")
        request._qwen_exo_memory_block = block
        asyncio.run(serving._make_request(request, None, None))
        last = captured[0].messages[-1]
        last = last if isinstance(last, dict) else last.model_dump(exclude_none=True)
        self.assertEqual(last, {"role": "assistant", "content": block})
        self.assertNotIn(block, json.dumps(serving._conversation_messages(request, None)))

    def test_memory_query_is_task_plus_newest_tool_output(self):
        serving = make_serving(self.tool_server)
        query = serving._memory_query_text(
            ResponsesRequest(
                model="m",
                input=[
                    {"role": "user", "content": "fix the deploy"},
                    {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": "old output"},
                    {"type": "function_call", "call_id": "c2", "name": "bash", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c2", "output": "nginx: bad gateway"},
                ],
            ),
            None,
        )
        self.assertEqual(query, "fix the deploy\nnginx: bad gateway")
        self.assertEqual(
            serving._memory_tools_blocked_reason(
                ResponsesRequest(model="m", input="hi", tool_choice="none")
            ),
            "tool_choice",
        )

    def test_memory_round_executes_server_side_and_continues_on_exact_tokens(self):
        """A turn ending in only a memory call is executed by the server; the
        next pass prompt is prompt + generated ids + tool block + the original
        assistant header, and the client sees one turn's usage."""
        tokenizer = ChatMLTokenizer()
        call_text = (
            "look it up</think>\n\n<tool_call>\n<function=qwen_exo_memory_search>\n"
            "<parameter=pattern>\nnginx\n</parameter>\n</function>\n</tool_call>"
        )
        call_ids = tokenizer.encode(call_text) + [IM_END]
        answer_ids = tokenizer.encode("found</think>\n\nReload nginx.") + [IM_END]
        outputs = [
            {
                "text": call_text,
                "output_ids": call_ids,
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": IM_END},
                    "prompt_tokens": 40,
                    "cached_tokens": 32,
                    "completion_tokens": len(call_ids),
                },
            },
            {
                "text": "found</think>\n\nReload nginx.",
                "output_ids": answer_ids,
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": IM_END},
                    "prompt_tokens": 400,
                    "cached_tokens": 390,
                    "completion_tokens": len(answer_ids),
                },
            },
        ]
        serving = make_serving(self.tool_server, outputs)
        # Multimodal-capable models (prod Qwen4Exp) hand over a text prompt,
        # not input_ids; the loop must still continue on exact token ids.
        prompt_text = "<|im_start|>user\nhow?<|im_end|>\n<|im_start|>assistant\n<think>\n"
        prompt = tokenizer.encode(prompt_text)
        request = ResponsesRequest(model="m", input="how?")
        context = SimpleContext()

        async def run():
            chunks = []
            async for ctx in serving._generate_with_memory_tools(
                request.request_id,
                prompt,
                GenerateReqInput(
                    text=prompt_text,
                    sampling_params={"max_new_tokens": 5000},
                    stream=False,
                    rid=request.request_id,
                ),
                {"max_new_tokens": 5000},
                context,
                raw_request=None,
                response_request=request,
                priority=None,
                prompt_encoding=None,
            ):
                chunks.append(dict(ctx.last_output))
            return chunks

        chunks = asyncio.run(run())
        sent = serving.tokenizer_manager.requests
        self.assertEqual(len(sent), 2)
        self.assertEqual(list(sent[0].input_ids), prompt)
        second_prompt = list(sent[1].input_ids)
        self.assertEqual(second_prompt[: len(prompt) + len(call_ids)], prompt + call_ids)
        tail = tokenizer.decode(second_prompt[len(prompt) + len(call_ids) :])
        self.assertTrue(
            tail.startswith(
                "\n<|im_start|>user\n<tool_response>\n"
                "1 match(es) in 1 file(s); ranked, showing up to 3 per file:\n--\n"
                "ops.md [note] intro (1 match(es))\nops.md-1- intro\n"
                "ops.md:2: reload nginx after edits\n</tool_response>"
            ),
            tail,
        )
        self.assertTrue(tail.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"))
        self.assertEqual(sent[1].sampling_params["max_new_tokens"], 5000 - len(call_ids))

        markers = [c for c in chunks if MEMORY_ROUND_META_KEY in (c.get("meta_info") or {})]
        self.assertEqual(len(markers), 1)
        (call,) = markers[0]["meta_info"][MEMORY_ROUND_META_KEY]["calls"]
        self.assertEqual((call["name"], call["ok"]), ("qwen_exo_memory_search", True))
        self.assertEqual(json.loads(call["arguments"]), {"pattern": "nginx"})
        self.assertEqual(len(context.qwen_exo_memory_rounds), 1)
        final_meta = context.last_output["meta_info"]
        self.assertEqual(
            (final_meta["prompt_tokens"], final_meta["cached_tokens"], final_meta["completion_tokens"]),
            (40, 32, len(call_ids) + len(answer_ids)),
        )


if __name__ == "__main__":
    unittest.main()
