import importlib.util
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import BaseModel
from scripts.qwen_exo.run_deep_swe_compressed import _parser, build_command
import scripts.qwen_exo.run_deep_swe_compressed as runner


class _InteractiveAgentConfig(BaseModel):
    system_template: str = ""
    instance_template: str = ""
    step_limit: int = 0
    cost_limit: float = 0
    mode: str = "yolo"
    confirm_exit: bool = False
    max_consecutive_format_errors: int = 3


class _InteractiveAgent:
    def __init__(self, model=None, env=None, *, config_class, **kwargs):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env
        self.messages = []
        self.n_calls = 0

    def add_messages(self, *messages):
        self.messages.extend(messages)
        return list(messages)

    def run(self, task="", **kwargs):
        return {"task": task, **kwargs}

    def query(self):
        return {}

    def serialize(self, *extra_dicts):
        return {"info": {}, "messages": self.messages}


def _load_compacting_agent():
    fake_interactive = ModuleType("minisweagent.agents.interactive")
    fake_interactive.InteractiveAgent = _InteractiveAgent
    fake_interactive.InteractiveAgentConfig = _InteractiveAgentConfig
    fake_agents = ModuleType("minisweagent.agents")
    fake_package = ModuleType("minisweagent")

    module_names = {
        "minisweagent": fake_package,
        "minisweagent.agents": fake_agents,
        "minisweagent.agents.interactive": fake_interactive,
    }
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(module_names)
    try:
        source = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "qwen_exo"
            / "agents"
            / "qwen_exo_compacting_agent.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen_exo_compacting_agent_under_test", source
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.CompactingInteractiveAgent
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _CancelledMiniSweAgent:
    def __init__(self, *args, **kwargs):
        self.commands = []

    async def exec_as_agent(self, _environment, *, command, env):
        self.commands.append((command, env))

    def build_process_env(self):
        return {"AGENT": "qwen-exo"}

    async def run(self, _instruction, _environment, _context):
        raise asyncio.CancelledError


def _load_pier_agent():
    fake_agent_module = ModuleType("pier.agents.installed.mini_swe_agent")
    fake_agent_module.MiniSweAgent = _CancelledMiniSweAgent
    fake_environment_module = ModuleType("pier.environments.base")
    fake_environment_module.BaseEnvironment = type("BaseEnvironment", (), {})
    fake_context_module = ModuleType("pier.models.agent.context")
    fake_context_module.AgentContext = type("AgentContext", (), {})

    module_names = {
        "pier": ModuleType("pier"),
        "pier.agents": ModuleType("pier.agents"),
        "pier.agents.installed": ModuleType("pier.agents.installed"),
        "pier.agents.installed.mini_swe_agent": fake_agent_module,
        "pier.environments": ModuleType("pier.environments"),
        "pier.environments.base": fake_environment_module,
        "pier.models": ModuleType("pier.models"),
        "pier.models.agent": ModuleType("pier.models.agent"),
        "pier.models.agent.context": fake_context_module,
    }
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(module_names)
    try:
        source = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "qwen_exo"
            / "agents"
            / "qwen_exo_pier_agent.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen_exo_pier_agent_under_test", source
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.QwenExoMiniSweAgent
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.mark.asyncio
async def test_pier_agent_commits_worktree_when_cancelled():
    agent_class = _load_pier_agent()
    agent = agent_class()

    with pytest.raises(asyncio.CancelledError):
        await agent.run("task", object(), object())

    assert len(agent.commands) == 2
    preserve_command, preserve_env = agent.commands[-1]
    assert 'git -C "$root" add -A' in preserve_command
    assert 'git -C "$root" diff --cached --quiet' in preserve_command
    assert "commit --no-verify" in preserve_command
    assert preserve_env == {"AGENT": "qwen-exo"}


def _history(last_prompt_tokens=800):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for index in range(8):
        prompt_tokens = last_prompt_tokens if index == 7 else 100 + index * 50
        messages.extend(
            (
                {
                    "object": "response",
                    "id": f"response-{index}",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 10,
                    },
                    "output": [
                        {
                            "type": "function_call",
                            "name": "bash",
                            "arguments": json.dumps(
                                {"command": f"inspect-target-{index}"}
                            ),
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call-{index}",
                    "output": json.dumps(
                        {"returncode": 0, "output": f"result-{index}"}
                    ),
                },
            )
        )
    return messages


def _agent():
    agent_class = _load_compacting_agent()
    return agent_class(
        system_template="system",
        instance_template="task",
        context_window_tokens=1000,
        compaction_threshold=0.70,
        compaction_keep_model_turns=3,
        compaction_summary_chars=4000,
        compaction_min_messages=6,
        compaction_cooldown_model_turns=4,
    )


def test_compacts_at_seventy_percent_and_preserves_full_trajectory():
    agent = _agent()
    agent.messages = _history()
    agent._trajectory_messages = list(agent.messages)
    assert agent._compact_if_needed()

    assert len(agent.messages) == 10
    assert "<context_compaction>" in agent.messages[2]["content"]
    assert "inspect-target-4" in agent.messages[2]["content"]
    assert [
        message.get("id")
        for message in agent.messages
        if message.get("object") == "response"
    ] == ["response-5", "response-6", "response-7"]
    assert agent.messages[-1]["role"] == "user"
    assert agent.messages[-1]["content"].startswith("继续（Continue）")
    assert agent.messages[-1]["extra"]["context_compaction_continue"] is True

    event = agent._compaction_events[0]
    assert event["threshold_tokens"] == 700
    assert event["before_message_count"] == 18
    assert event["after_message_count"] == 10
    assert len(agent.serialize()["messages"]) == 20


def test_compaction_preserves_causal_evidence_across_old_history():
    agent = _agent()
    messages = []
    for index in range(30):
        reasoning = f"Routine inspection {index}"
        output = f"routine result {index}"
        if index == 2:
            reasoning = (
                "Architectural decision: materialize derived routes during registration "
                "because the implementation currently owns that lifecycle."
            )
        if index == 9:
            output = "Traceback: AssertionError: derived route was duplicated"
        messages.extend(
            (
                {
                    "object": "response",
                    "id": f"response-{index}",
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [{"text": reasoning}],
                        },
                        {
                            "type": "function_call",
                            "name": "bash",
                            "arguments": json.dumps(
                                {"command": f"inspect-target-{index}"}
                            ),
                        },
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call-{index}",
                    "output": json.dumps(
                        {"returncode": 1 if index == 9 else 0, "output": output}
                    ),
                },
            )
        )

    summary = agent._build_execution_digest(messages)

    assert "Architectural decision" in summary
    assert "derived route was duplicated" in summary
    assert "inspect-target-29" in summary
    assert "by recency and causal salience" in summary
    assert "routine entries" in summary


def test_compaction_honors_threshold_and_cooldown():
    below_threshold = _agent()
    below_threshold.messages = _history(last_prompt_tokens=600)
    below_threshold._trajectory_messages = list(below_threshold.messages)
    assert not below_threshold._compact_if_needed()

    agent = _agent()
    agent.messages = _history()
    agent._trajectory_messages = list(agent.messages)
    agent.n_calls = 3
    assert agent._compact_if_needed()

    agent.messages.extend(
        (
            {
                "object": "response",
                "id": "response-new",
                "usage": {"prompt_tokens": 900, "completion_tokens": 10},
                "output": [],
            },
            {
                "type": "function_call_output",
                "call_id": "call-new",
                "output": json.dumps({"returncode": 0, "output": "new"}),
            },
        )
    )
    agent.n_calls = 4
    assert not agent._compact_if_needed()
    agent.n_calls = 7
    assert agent._compact_if_needed()


def test_compaction_can_be_disabled_for_ab_baseline():
    agent_class = _load_compacting_agent()
    agent = agent_class(
        system_template="system",
        instance_template="task",
        compaction_enabled=False,
        context_window_tokens=1000,
        compaction_threshold=0.70,
        compaction_keep_model_turns=3,
        compaction_summary_chars=4000,
        compaction_min_messages=6,
        compaction_cooldown_model_turns=4,
    )
    agent.messages = _history()
    agent._trajectory_messages = list(agent.messages)

    assert not agent._compact_if_needed()
    assert agent.serialize()["info"]["compaction"]["enabled"] is False


def test_runner_injects_agent_without_replacing_pier_log_mounts():
    args = _parser().parse_args(
        [
            "--task",
            "/task",
            "--job-name",
            "comparison",
            "--jobs-dir",
            "/jobs",
            "--disable-compaction",
            "--skip-verification",
            "--n-tasks",
            "3",
            "--sample-seed",
            "17",
        ]
    )

    command = build_command(args)

    assert "--mounts-json" not in command
    import_index = command.index("--agent-import-path")
    assert command[import_index + 1] == ("qwen_exo_pier_agent:QwenExoMiniSweAgent")
    assert "PYTHONPATH=/tmp/qwen-exo-agent" in command
    assert "QWEN_EXO_COMPACTION_ENABLED=0" in command
    assert "--disable-verification" in command
    timeout_index = command.index("--agent-timeout-multiplier")
    assert command[timeout_index + 1] == "1.5"
    concurrency_index = command.index("--n-concurrent")
    assert command[concurrency_index + 1] == "1"
    attempts_index = command.index("--n-attempts")
    assert command[attempts_index + 1] == "1"
    task_count_index = command.index("--n-tasks")
    assert command[task_count_index + 1] == "3"
    sample_seed_index = command.index("--sample-seed")
    assert command[sample_seed_index + 1] == "17"


def test_runtime_reset_waits_for_base_health(monkeypatch, capsys):
    calls = []
    health_attempts = 0

    def fake_request(url, method, *, timeout=30):
        nonlocal health_attempts
        calls.append((url, method, timeout))
        if url.endswith("/health"):
            health_attempts += 1
            if health_attempts == 1:
                raise TimeoutError("warmup still running")
            return ""
        if url.endswith("/flush_cache"):
            return "Cache flushed."
        return '{"status":"cleared"}'

    monkeypatch.setattr(runner, "_request", fake_request)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner._reset_runtime("http://runtime/")

    assert calls == [
        ("http://runtime/health", "GET", 10),
        ("http://runtime/health", "GET", 10),
        ("http://runtime/flush_cache", "POST", 120),
        ("http://runtime/qwen-exo/recall-trace", "DELETE", 30),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "Cache flushed.",
        '{"status":"cleared"}',
    ]
