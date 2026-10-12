import asyncio
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.sampling.sampling_params import QWEN_EXO_REASONING_BUDGET_KEY
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

END = 99


def budget_req(output_ids, max_tokens=4):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={
                QWEN_EXO_REASONING_BUDGET_KEY: {"end_token_id": END, "max_tokens": max_tokens}
            }
        ),
        output_ids=array("q", output_ids),
        finished_reason=None,
        finished_len=None,
    )


class ReasoningBudgetFinishTestCase(CustomTestCase):
    def test_open_reasoning_finishes_at_budget_trimming_speculative_overshoot(self):
        # A speculative step accepted 2 tokens past the 4-token budget.
        req = budget_req([10, 11, 12, 13, 14, 15])
        self.assertTrue(Req._reasoning_budget_exhausted(req))
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 4)

    def test_natural_reasoning_end_within_budget_keeps_generating(self):
        req = budget_req([10, END, 20, 21, 22, 23, 24])
        self.assertFalse(Req._reasoning_budget_exhausted(req))
        self.assertIsNone(req.finished_reason)

    def test_reasoning_end_past_budget_is_still_a_budget_finish(self):
        # </think> accepted in the overshoot beyond the budget does not count.
        req = budget_req([10, 11, 12, 13, END])
        self.assertTrue(Req._reasoning_budget_exhausted(req))
        self.assertEqual(req.finished_len, 4)

    def test_requests_without_budget_are_untouched(self):
        req = budget_req([10, 11, 12, 13, 14])
        req.sampling_params.custom_params = None
        self.assertFalse(Req._reasoning_budget_exhausted(req))


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [200 + index for index, _ in enumerate(str(text).split())]

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "</think>" if list(token_ids) == [END] else ""


class FakeTokenizerManager:
    def __init__(self, outputs):
        self.tokenizer = FakeTokenizer()
        self.server_args = SimpleNamespace(incremental_streaming_output=False)
        self.model_config = SimpleNamespace(context_len=1024)
        self.num_reserved_tokens = 0
        self.outputs = list(outputs)
        self.requests = []

    def generate_request(self, request, _raw_request):
        self.requests.append(request)
        output = self.outputs[len(self.requests) - 1]

        async def generate():
            yield output

        return generate()


class FakeRuntime:
    reasoning_end_token_id = END
    think_context_enabled = False
    max_reasoning_tokens = 4

    def reasoning_budget_for_effort(self, effort):
        return self.max_reasoning_tokens

    def __init__(self):
        self.boundaries = []
        self.cutoffs = []

    def register_generation_prompt(self, *_args, **_kwargs):
        pass

    def observe_generation_result(self, *_args, **_kwargs):
        pass

    async def await_think_context(self, _request_id):
        raise AssertionError("single-phase reasoning has no think-context stop")

    async def build_reasoning_cutoff_injection(self, request_id, **kwargs):
        self.cutoffs.append((request_id, kwargs))
        return None

    def record_reasoning_boundary(self, request_id, **kwargs):
        self.boundaries.append((request_id, kwargs))


def run_generation(outputs, rid, *, runtime=None, response_request=None, output_limit=10):
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.reasoning_parser = "qwen3"
    serving.template_manager = SimpleNamespace(
        force_reasoning=False,
        reasoning_config=SimpleNamespace(
            special_case=None, toggle_param="enable_thinking", default_enabled=True,
        ),
    )
    serving.tokenizer_manager = FakeTokenizerManager(outputs)
    runtime = runtime or FakeRuntime()
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(qwen_exo_runtime=runtime))
    )
    adapted_request = GenerateReqInput(
        input_ids=[1, 2], sampling_params={"max_new_tokens": output_limit}, stream=False, rid=rid
    )
    context = SimpleContext()

    async def collect():
        snapshots = []
        async for current in serving._generate_with_builtin_tools(
            rid, [1, 2], adapted_request, {"max_new_tokens": output_limit}, context,
            raw_request=raw_request,
            response_request=response_request,
        ):
            snapshots.append(dict(current.last_output))
        return snapshots

    return serving, runtime, context, asyncio.run(collect())


def tier_config(**overrides):
    return QwenExoConfig(
        state_directory=Path("."), knowledge_directory=Path("."),
        max_internal_fanout=8, max_internal_tokens=4096, max_candidates=8,
        max_memory_tokens=256, observer_mode="off",
        feature_flags=QwenExoFeatureFlags(
            hybrid_prefix=True, external_memory=False, reference_judge=False,
            capsule=False, observer=False, adaptive_refresh=False,
        ),
        model_path="model", tp_size=1, **overrides,
    )


class ReasoningEffortTierTestCase(CustomTestCase):
    def test_requested_tiers_control_scheduler_budget_without_alias_collapse(self):
        config = tier_config(reasoning_budget_high=5000)
        for effort, expected in (
            ("min", 512), ("minimal", 512), ("low", 1024), ("medium", 2048),
            ("high", 5000), ("xhigh", 8192), ("max", 8192),
        ):
            with self.subTest(effort=effort):
                runtime = FakeRuntime()
                runtime.reasoning_budget_for_effort = config.reasoning_budget_for_effort
                request = ResponsesRequest(input="task", reasoning={"effort": effort})
                serving, _, _, _ = run_generation(
                    [{
                        "text": "<think>done</think>answer",
                        "output_ids": [10, END, 20],
                        "meta_info": {"finish_reason": {"type": "stop", "matched": 0}},
                    }],
                    "resp-tier", runtime=runtime, response_request=request, output_limit=10000,
                )
                params = serving.tokenizer_manager.requests[0].sampling_params
                self.assertEqual(params["max_new_tokens"], 10000)
                budget = params["custom_params"][QWEN_EXO_REASONING_BUDGET_KEY]
                self.assertEqual(budget["max_tokens"], expected)
                accepted = budget_req([10] * (expected + 2), max_tokens=budget["max_tokens"])
                self.assertTrue(Req._reasoning_budget_exhausted(accepted))
                self.assertEqual(accepted.finished_len, expected)
                self.assertEqual(request.reasoning.effort, effort)

    def test_smaller_request_output_limit_is_not_expanded_by_tier(self):
        runtime = FakeRuntime()
        runtime.reasoning_budget_for_effort = tier_config().reasoning_budget_for_effort
        serving, _, _, _ = run_generation(
            [{"text": "bounded", "output_ids": [10, 11],
              "meta_info": {"finish_reason": {"type": "length", "length": 2}}}],
            "resp-output-limit", runtime=runtime,
            response_request=ResponsesRequest(input="task", reasoning={"effort": "xhigh"}),
            output_limit=2,
        )
        params = serving.tokenizer_manager.requests[0].sampling_params
        self.assertEqual(params["max_new_tokens"], 2)
        self.assertNotIn(QWEN_EXO_REASONING_BUDGET_KEY, params.get("custom_params") or {})
        self.assertFalse(runtime.cutoffs)


class SinglePhaseReasoningTestCase(CustomTestCase):
    def test_natural_reasoning_end_answers_in_the_same_request(self):
        serving, runtime, context, snapshots = run_generation(
            [
                {
                    "text": "<think>short</think>answer",
                    "output_ids": [10, END, 20, 21],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 4,
                        "finish_reason": {"type": "stop", "matched": 0},
                    },
                }
            ],
            "resp-single-phase",
        )
        self.assertEqual(len(serving.tokenizer_manager.requests), 1)
        params = serving.tokenizer_manager.requests[0].sampling_params
        # The whole answer budget stays available; reasoning is capped by the scheduler.
        self.assertEqual(params["max_new_tokens"], 10)
        self.assertNotIn(END, params.get("stop_token_ids") or [])
        self.assertEqual(
            params["custom_params"][QWEN_EXO_REASONING_BUDGET_KEY],
            {"end_token_id": END, "max_tokens": 4},
        )
        self.assertEqual(snapshots[-1]["meta_info"]["finish_reason"]["type"], "stop")
        self.assertEqual(context.num_reasoning_tokens, 2)
        self.assertEqual(len(runtime.boundaries), 1)
        self.assertIsNone(runtime.boundaries[0][1]["injection"])
        self.assertFalse(runtime.cutoffs)

    def test_budget_finish_still_forces_the_cutoff_continuation(self):
        serving, runtime, _context, snapshots = run_generation(
            [
                {
                    "text": "<think>bounded reasoning",
                    "output_ids": [10, 11, 12, 13],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 4,
                        "finish_reason": {"type": "length", "length": 4},
                    },
                },
                {
                    "text": "final action",
                    "output_ids": [20, 21],
                    "meta_info": {
                        "prompt_tokens": 9,
                        "completion_tokens": 2,
                        "finish_reason": {"type": "stop", "matched": 0},
                    },
                },
            ],
            "resp-single-phase-budget",
        )
        self.assertEqual(len(serving.tokenizer_manager.requests), 2)
        self.assertNotIn("finish_reason", snapshots[0]["meta_info"])
        self.assertIn("stop over thinking", snapshots[1]["text"])
        self.assertEqual(serving.tokenizer_manager.requests[1].input_ids[-1], END)
        self.assertEqual(runtime.cutoffs[0][1]["observed_tokens"], 4)

    def test_answer_length_finish_after_reasoning_end_is_not_a_cutoff(self):
        # max_new_tokens exhausted by the answer: no forced boundary or retry.
        serving, runtime, _context, snapshots = run_generation(
            [
                {
                    "text": "<think>r</think>very long answer",
                    "output_ids": [10, END, 20, 21, 22, 23, 24, 25, 26, 27],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 10,
                        "finish_reason": {"type": "length", "length": 10},
                    },
                }
            ],
            "resp-single-phase-answer-cap",
        )
        self.assertEqual(len(serving.tokenizer_manager.requests), 1)
        self.assertEqual(snapshots[-1]["meta_info"]["finish_reason"]["type"], "length")
        self.assertFalse(runtime.cutoffs)


if __name__ == "__main__":
    unittest.main()
