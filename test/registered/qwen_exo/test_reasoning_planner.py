import json
import asyncio
from types import SimpleNamespace

import pytest

from qwen_exo_booster.reasoning_planner import DynamicReasoningPlanner, reasoning_context
from qwen_exo_booster.system_one import SystemOneEvaluator


class TextTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class ScoreRunner:
    max_fanout = 1
    max_tokens_per_parent = 32

    def __init__(self, winner=3, fail=False):
        self.winner = winner
        self.fail = fail
        self.parents_finished = []
        self.prompts = []

    async def run_option_score_batch(self, jobs, prompts, labels):
        self.prompts.extend(prompts)
        if self.fail:
            raise RuntimeError("score backend unavailable")
        return [SimpleNamespace(option_logprobs=tuple(
            0.0 if index == self.winner else -10.0 for index in range(len(labels))
        )) for _ in jobs]

    async def finish_parent(self, parent):
        self.parents_finished.append(parent)


def test_context_preserves_latest_roles_without_media_or_full_history():
    tokenizer = TextTokenizer()
    messages = [
        {"role": "user", "content": "USER_GOAL"},
        {"role": "assistant", "content": "ASSISTANT_ACTION"},
        *[{"role": "tool", "content": "old-" + str(i)} for i in range(20)],
        {"role": "tool", "content": "x" * 100000 + "CURRENT_RESULT"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET_IMAGE"}},
            {"type": "text", "text": "CURRENT_USER"},
        ]},
    ]
    state, count = reasoning_context(messages, tokenizer, 2048)
    rendered = json.dumps(state)
    assert count <= 2048
    assert "CURRENT_USER" in rendered
    assert "ASSISTANT_ACTION" in rendered
    assert "CURRENT_RESULT" in rendered
    assert "SECRET_IMAGE" not in rendered
    assert "USER_GOAL" not in rendered
    assert state["partial_context"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("winner,effort", list(enumerate(("min", "low", "medium", "high", "xhigh"))))
async def test_dynamic_effort_is_selected_by_internal_option_scores(winner, effort):
    runner = ScoreRunner(winner)
    evaluator = SystemOneEvaluator(runner, TextTokenizer(), max_prompt_tokens=10000)
    planner = DynamicReasoningPlanner(evaluator, context_tokens=512)
    budgets = dict(zip(("min", "low", "medium", "high", "xhigh"), (512, 1024, 2048, 4090, 8192)))
    selection = await planner.choose([
        {"role": "user", "content": "Diagnose the failing request"},
        {"role": "assistant", "content": "Checked parser"},
        {"role": "tool", "content": "parser passed; timeout persists"},
    ], budgets)
    assert selection["effort"] == effort
    assert selection["max_reasoning_tokens"] == budgets[effort]
    assert selection["context_tokens"] <= 512
    assert selection["probabilities"][effort] > .99
    assert len(runner.parents_finished) == 1


@pytest.mark.asyncio
async def test_failed_scoring_is_not_replaced_with_a_fabricated_tier():
    runner = ScoreRunner(fail=True)
    planner = DynamicReasoningPlanner(
        SystemOneEvaluator(runner, TextTokenizer(), max_prompt_tokens=10000), context_tokens=512
    )
    with pytest.raises(RuntimeError, match="backend unavailable"):
        await planner.choose([{"role": "user", "content": "task"}],
                             dict.fromkeys(("min", "low", "medium", "high", "xhigh"), 512))
    assert len(runner.parents_finished) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [None, {"summary": "concise"}])
async def test_responses_dynamic_choice_preserves_controls_but_excludes_tool_history(reasoning):
    from qwen_exo_booster.runtime import QwenExoRuntime
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
    from sglang.srt.sampling.sampling_params import QWEN_EXO_REASONING_BUDGET_KEY

    runner = ScoreRunner(winner=3)
    runtime = object.__new__(QwenExoRuntime)
    budgets = dict(zip(("min", "low", "medium", "high", "xhigh"), (512, 1024, 2048, 4090, 8192)))
    runtime.config = SimpleNamespace(
        reasoning_mode="dynamic", force_auto_reasoning=False,
        reasoning_session_once=False,
        reasoning_budget_for_effort=lambda effort: budgets[effort]
    )
    runtime.reasoning_planner = DynamicReasoningPlanner(
        SystemOneEvaluator(runner, TextTokenizer(), max_prompt_tokens=10000), context_tokens=1024
    )
    runtime.telemetry = SimpleNamespace(emit=lambda *args: None)
    serving = object.__new__(OpenAIServingResponses)
    serving.msg_store = {"resp_previous": [
        {"role": "user", "content": "ORIGINAL_GOAL"},
        {"role": "assistant", "content": "CURRENT_ACTION"},
    ]}
    previous = SimpleNamespace(id="resp_previous", output=[])
    request = ResponsesRequest(
        input=[{"type": "function_call_output", "call_id": "call_1", "output": "CURRENT_TOOL_FAILURE"}],
        reasoning=reasoning,
    )
    request = QwenExoRuntime._enable_default_response_thinking(request)
    selected = await serving._resolve_reasoning_effort(request, previous, runtime)
    assert selected.reasoning.effort == "high"
    assert selected.reasoning.summary == (reasoning or {}).get("summary")
    assert "effort" not in request.reasoning.model_fields_set
    rendered = TextTokenizer().decode(runner.prompts[0])
    assert "ORIGINAL_GOAL" in rendered
    assert "CURRENT_ACTION" not in rendered
    assert "CURRENT_TOOL_FAILURE" not in rendered
    params, applied = serving._single_phase_reasoning_sampling_params(
        {"max_new_tokens": 10000}, 99,
        runtime.reasoning_budget_for_effort(selected.reasoning.effort),
    )
    assert applied
    assert params["custom_params"][QWEN_EXO_REASONING_BUDGET_KEY]["max_tokens"] == 4090
    for effort in ("none", "min", "low", "medium", "high", "xhigh"):
        explicit = ResponsesRequest(input="task", reasoning={"effort": effort})
        assert await serving._resolve_reasoning_effort(explicit, None, runtime) is explicit
    runtime.config.reasoning_mode = "fixed"
    assert await serving._resolve_reasoning_effort(request, previous, runtime) is request
    assert len(runner.prompts) == 1


@pytest.mark.asyncio
async def test_dynamic_scoring_failure_returns_service_error_before_generation():
    from qwen_exo_booster.runtime import QwenExoRuntime
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses

    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        reasoning_mode="dynamic", reasoning_budget_for_effort=lambda effort: 512,
        reasoning_session_once=False,
    )
    runtime.reasoning_planner = DynamicReasoningPlanner(
        SystemOneEvaluator(ScoreRunner(fail=True), TextTokenizer(), max_prompt_tokens=10000),
        context_tokens=512,
    )
    serving = object.__new__(OpenAIServingResponses)
    serving.tokenizer_manager = SimpleNamespace()
    serving.use_harmony = False
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(qwen_exo_runtime=runtime)))
    response = await serving.create_responses(ResponsesRequest(input="task"), raw)
    assert response.status_code == 503
    assert json.loads(response.body)["error"]["type"] == "reasoning_planning_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fixed", "dynamic"])
@pytest.mark.parametrize("effort", [None, "none", "min", "minimal", "low", "medium", "high", "xhigh", "max"])
async def test_forced_auto_overrides_every_effort_and_fixed_mode(mode, effort):
    from qwen_exo_booster.runtime import QwenExoRuntime
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses

    runner = ScoreRunner(winner=1)
    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        reasoning_mode=mode, force_auto_reasoning=True,
        reasoning_session_once=False,
        reasoning_budget_for_effort=lambda effort: {"min":512,"low":1024,"medium":2048,"high":4090,"xhigh":8192}[effort],
    )
    runtime.reasoning_planner = DynamicReasoningPlanner(
        SystemOneEvaluator(runner, TextTokenizer(), max_prompt_tokens=10000), context_tokens=512
    )
    runtime.telemetry = SimpleNamespace(emit=lambda *args: None)
    serving = object.__new__(OpenAIServingResponses)
    request = ResponsesRequest(
        input="Decide the next step", reasoning=None if effort is None else {"effort":effort}
    )
    selected = await serving._resolve_reasoning_effort(request, None, runtime)
    assert selected.reasoning.effort == "low"
    assert runtime.reasoning_budget_for_effort(selected.reasoning.effort) == 1024
    assert len(runner.prompts) == 1
    assert request.reasoning is None if effort is None else request.reasoning.effort == effort
    runtime.config.force_auto_reasoning = False
    if effort is not None or mode == "fixed":
        assert await serving._resolve_reasoning_effort(request, None, runtime) is request
        assert len(runner.prompts) == 1


def task_runtime(runner, *, once=True, capacity=512):
    from qwen_exo_booster.runtime import QwenExoRuntime
    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        reasoning_mode="dynamic", force_auto_reasoning=True,
        reasoning_session_once=once,
        reasoning_budget_for_effort=lambda effort: {"min":512,"low":1024,"medium":2048,"high":4090,"xhigh":8192}[effort],
    )
    runtime.reasoning_planner = DynamicReasoningPlanner(
        SystemOneEvaluator(runner, TextTokenizer(), max_prompt_tokens=10000), context_tokens=512
    )
    runtime._request_conversation_keys = {}
    runtime._max_conversation_keys = capacity
    runtime.telemetry = SimpleNamespace(emit=lambda *args: None)
    return runtime


async def choose_task(runtime, rid, messages, *, scope="conversation", session="session", previous=None, raw=None):
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    runtime._request_conversation_keys[rid] = scope
    request = ResponsesRequest(
        request_id=rid, input=messages if raw is None else raw,
        session_id=session, previous_response_id=previous,
    )
    return await runtime.plan_reasoning_effort(rid, messages, request=request, previous_response_id=previous)


@pytest.mark.asyncio
async def test_task_budget_reuses_tool_rounds_and_reselects_new_user_or_session():
    runner = ScoreRunner(winner=3)
    runtime = task_runtime(runner)
    first = [{"role":"user","content":"TASK_START"}]
    assert await choose_task(runtime,"resp_first",first) == "high"
    runner.winner = 0
    continuation = [*first,{"role":"assistant","content":"MODEL_TRACE"},{"role":"tool","content":"TOOL_EVIDENCE"}]
    assert await choose_task(runtime,"resp_second",continuation) == "high"
    assert len(runner.prompts) == 1
    rendered = TextTokenizer().decode(runner.prompts[0])
    assert "TASK_START" in rendered and "MODEL_TRACE" not in rendered and "TOOL_EVIDENCE" not in rendered
    assert await choose_task(runtime,"resp_tool_only",[],previous="resp_second",raw=[{"type":"function_call_output","call_id":"call_1","output":"result"}]) == "high"
    assert len(runner.prompts) == 1
    # The same user text at a later genuine turn is a NEW task.
    next_task = [*continuation,{"role":"user","content":"TASK_START"}]
    assert await choose_task(runtime,"resp_new",next_task) == "min"
    assert len(runner.prompts) == 2
    assert await choose_task(runtime,"resp_other",continuation,session="other-session") == "min"
    assert len(runner.prompts) == 3
    await runtime.clear_reasoning_sessions(["conversation"])
    runner.winner = 2
    assert await choose_task(runtime,"resp_cleared",continuation) == "medium"
    assert len(runner.prompts) == 4


@pytest.mark.asyncio
async def test_latest_user_only_ignores_full_decision_context_and_old_task():
    runner = ScoreRunner(winner=2)
    runtime = task_runtime(runner)
    runtime._decision_contexts = {"resp_latest": object()}
    messages = [
        {"role": "user", "content": "OLD_TASK_IGNORED"},
        {"role": "assistant", "content": "WORKFLOW_IGNORED " * 20000},
        {"role": "tool", "content": "TOOL_RESULT_IGNORED " * 20000},
        {"role": "user", "content": "CURRENT_USER_TASK"},
    ]
    assert await choose_task(runtime, "resp_latest", messages) == "medium"
    rendered = TextTokenizer().decode(runner.prompts[0])
    assert "CURRENT_USER_TASK" in rendered
    assert all(marker not in rendered for marker in (
        "OLD_TASK_IGNORED", "WORKFLOW_IGNORED", "TOOL_RESULT_IGNORED"
    ))
    continuation = [*messages, {"role": "assistant", "content": "action"},
                    {"role": "tool", "content": "new result"}]
    assert await choose_task(runtime, "resp_continue", continuation) == "medium"
    assert len(runner.prompts) == 1

@pytest.mark.asyncio
async def test_task_mode_off_and_fresh_identical_submissions_reselect():
    runner = ScoreRunner()
    runtime = task_runtime(runner, once=False)
    messages = [{"role":"user","content":"TASK"},{"role":"assistant","content":"checking"}]
    await choose_task(runtime,"resp_a",messages)
    await choose_task(runtime,"resp_b",messages)
    assert len(runner.prompts) == 2
    runtime.config.reasoning_session_once = True
    await choose_task(runtime,"resp_c",messages[:1])
    await choose_task(runtime,"resp_d",messages[:1])
    assert len(runner.prompts) == 4


@pytest.mark.asyncio
async def test_task_selection_failure_is_not_cached_and_lru_is_bounded():
    runner = ScoreRunner(fail=True)
    runtime = task_runtime(runner, capacity=2)
    messages = [{"role":"user","content":"TASK"},{"role":"assistant","content":"checking"}]
    with pytest.raises(RuntimeError,match="backend unavailable"):
        await choose_task(runtime,"resp_fail",messages)
    assert not runtime._reasoning_task_selections
    runner.fail = False
    for index in range(4):
        await choose_task(runtime,f"resp_{index}",messages,scope=f"scope-{index}")
    assert len(runtime._reasoning_task_selections) == 2
    assert len(runtime._reasoning_tasks_by_response) <= 2
    await runtime.clear_reasoning_sessions()
    assert not runtime._reasoning_task_selections


@pytest.mark.asyncio
async def test_concurrent_task_selection_survives_one_cancelled_waiter():
    class DelayedRunner(ScoreRunner):
        async def run_option_score_batch(self,jobs,prompts,labels):
            started.set()
            await release.wait()
            return await super().run_option_score_batch(jobs,prompts,labels)
    started,release = asyncio.Event(),asyncio.Event()
    runner = DelayedRunner(winner=4)
    runtime = task_runtime(runner)
    messages = [{"role":"user","content":"TASK"}]
    left = asyncio.create_task(choose_task(runtime,"resp_left",messages))
    await started.wait()
    right = asyncio.create_task(choose_task(runtime,"resp_right",messages))
    await asyncio.sleep(0)
    left.cancel()
    with pytest.raises(asyncio.CancelledError):
        await left
    release.set()
    assert await right == "xhigh"
    assert len(runner.prompts) == 1
    assert len(runtime._reasoning_task_selections) == 1


@pytest.mark.asyncio
async def test_unknown_search_history_is_input_error_not_retryable_jev_failure():
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
    runtime = task_runtime(ScoreRunner())
    serving = object.__new__(OpenAIServingResponses)
    serving.tokenizer_manager = SimpleNamespace()
    serving.use_harmony = False
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(qwen_exo_runtime=runtime)))
    request = ResponsesRequest(input=[{"role":"user","content":"task"},{"type":"web_search_call","id":"ws_external","status":"completed"}])
    response = await serving.create_responses(request,raw)
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["type"] == "invalid_request_error"
    assert not runtime.reasoning_planner.evaluator.runner.prompts
