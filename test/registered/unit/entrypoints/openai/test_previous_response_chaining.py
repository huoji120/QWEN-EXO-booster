import unittest

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from sglang.srt.entrypoints.openai.bounded_store import BoundedStore
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponsesResponse
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

INSTRUCTIONS = "You are an agent."


def make_serving():
    serving = object.__new__(OpenAIServingResponses)
    serving.msg_store = BoundedStore(max_entries=16, idle_ttl_seconds=3600)
    return serving


def turn_output(turn):
    """One agent turn: reasoning, then a tool call (turns 1-2) or an answer."""
    reasoning = ResponseReasoningItem(
        id=f"rs_{turn}",
        type="reasoning",
        summary=[],
        content=[ResponseReasoningTextContent(type="reasoning_text", text=f"think {turn}")],
    )
    if turn < 3:
        tail = ResponseFunctionToolCall(
            id=f"fc_{turn}",
            call_id=f"call_{turn}",
            type="function_call",
            name="bash",
            arguments='{"command": "ls"}',
            status="completed",
        )
    else:
        tail = ResponseOutputMessage(
            id="msg_3",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
        )
    return [reasoning, tail]


def tool_result(turn):
    return {"type": "function_call_output", "call_id": f"call_{turn}", "output": f"out {turn}"}


class PreviousResponseChainingTestCase(CustomTestCase):
    def test_chained_turns_render_like_full_history(self):
        """Gateways chain every turn (store=true + previous_response_id + only
        new items). The chained prompt must equal the full-history prompt: the
        parent's tool calls and reasoning replayed, instructions not repeated."""
        serving = make_serving()
        full_history = [{"role": "user", "content": "list files"}]
        chained_input = list(full_history)
        prev = None
        for turn in (1, 2, 3):
            chained = ResponsesRequest(
                model="m",
                instructions=INSTRUCTIONS,
                input=chained_input,
                previous_response_id=prev.id if prev else None,
                request_id=f"resp_{turn}",
            )
            full = ResponsesRequest(model="m", instructions=INSTRUCTIONS, input=full_history)
            self.assertEqual(
                serving._construct_input_messages(chained, prev),
                serving._construct_input_messages(full, None),
                f"turn {turn}",
            )
            serving.msg_store[chained.request_id] = serving._conversation_messages(
                chained, prev
            )
            output = turn_output(turn)
            prev = ResponsesResponse(
                id=chained.request_id, model="m", output=output, status="completed"
            )
            full_history = full_history + [item.model_dump() for item in output]
            if turn < 3:
                full_history.append(tool_result(turn))
            chained_input = [tool_result(turn)]
        system_messages = [
            m for m in serving._construct_input_messages(full, None) if m["role"] == "system"
        ]
        self.assertEqual(system_messages, [{"role": "system", "content": INSTRUCTIONS}])


class BoundedStoreTestCase(CustomTestCase):
    def test_lru_count_bytes_and_idle_ttl(self):
        now = [0.0]
        store = BoundedStore(
            max_entries=3,
            idle_ttl_seconds=10,
            max_bytes=10,
            sizer=len,
            clock=lambda: now[0],
        )
        store["a"], store["b"], store["c"] = "x", "y", "z"
        self.assertEqual(store["a"], "x")  # touch: b is now least recent
        store["d"] = "w"
        self.assertEqual(sorted(store), ["a", "c", "d"])
        store["big"] = "0123456789"  # 13 bytes total > 10: oldest go first
        self.assertEqual(sorted(store), ["big"])
        self.assertEqual(store.total_bytes, 10)
        now[0] = 11
        self.assertNotIn("big", store)
        self.assertEqual((len(store), store.total_bytes), (0, 0))


if __name__ == "__main__":
    unittest.main()
