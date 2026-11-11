"""Behavioral coverage for the startup-only master memory switch."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from qwen_exo_booster.config import QwenExoConfig, memory_enabled_from_environment
from qwen_exo_booster.hybrid_state import HybridRuntimePolicy
from qwen_exo_booster.initial_gdn_request import (
    INITIAL_GDN_DISABLED_CACHE_NAMESPACE,
    INITIAL_GDN_PARAM,
    bind_initial_gdn_request,
)
from qwen_exo_booster.memory_tools import (
    MEMORY_READ_TOOL,
    MEMORY_SEARCH_TOOL,
    MEMORY_WRITE_TOOL,
)
from qwen_exo_booster.runtime import QwenExoRuntime


def _server_args(tmp_path):
    return SimpleNamespace(
        qwen_exo_state_dir=str(tmp_path / "state"),
        qwen_exo_knowledge_dir=str(tmp_path / "knowledge"),
        qwen_exo_max_internal_fanout=8,
        qwen_exo_internal_mamba_reserve=6,
        qwen_exo_max_internal_tokens=12288,
        qwen_exo_observer_mode="active",
        qwen_exo_enable_hybrid_prefix=True,
        qwen_exo_enable_capsule=True,
        qwen_exo_enable_policy_data=True,
        qwen_exo_experimental_activation_training=True,
        qwen_exo_score_bias_mode="trajectory_active",
        qwen_exo_reflection_memory_mode="active",
        qwen_exo_response_compaction_mode="active",
        qwen_exo_latent_transplant_enabled=True,
        qwen_exo_activation_editor_enabled=True,
        model_path="model",
        tp_size=2,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3",
        context_length=32768,
    )


@pytest.mark.parametrize("value,enabled", [(None, True), ("0", False), ("1", True)])
def test_memory_environment_defaults_on_and_accepts_only_binary_values(
    tmp_path, monkeypatch, value, enabled
):
    if value is None:
        monkeypatch.delenv("QWEN_EXO_MEMORY_ENABLED", raising=False)
    else:
        monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", value)
    assert memory_enabled_from_environment() is enabled
    config = QwenExoConfig.from_server_args(_server_args(tmp_path))
    assert config.memory_enabled is enabled
    assert config.public_dict()["memory_enabled"] is enabled


@pytest.mark.parametrize("value", ["", "true", "false", "True", "yes", "2", "-1", " 0", "1 "])
def test_memory_environment_rejects_ambiguous_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", value)
    with pytest.raises(ValueError, match="QWEN_EXO_MEMORY_ENABLED must be 0 or 1"):
        QwenExoConfig.from_server_args(_server_args(tmp_path))


@pytest.mark.parametrize("value", [0, 1, None, "0", "1"])
def test_memory_config_requires_boolean(tmp_path, monkeypatch, value):
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "1")
    config = QwenExoConfig.from_server_args(_server_args(tmp_path))
    with pytest.raises(ValueError, match="memory_enabled must be a boolean"):
        replace(config, memory_enabled=value)


def test_disabled_config_normalizes_memory_without_disabling_caching(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "1")
    enabled = QwenExoConfig.from_server_args(_server_args(tmp_path))
    disabled = replace(enabled, memory_enabled=False)
    for name in ("capsule", "observer", "policy_data", "score_bias", "activation_training"):
        assert getattr(enabled.feature_flags, name) is True
        assert getattr(disabled.feature_flags, name) is False
    for name in ("observer_mode", "reflection_memory_mode", "response_compaction_mode", "score_bias_mode"):
        assert getattr(enabled, name) != "off"
        assert getattr(disabled, name) == "off"
    for name in ("latent_transplant_enabled", "activation_editor_enabled"):
        assert getattr(enabled, name) is True
        assert getattr(disabled, name) is False
    assert disabled.feature_flags.hybrid_prefix is enabled.feature_flags.hybrid_prefix is True
    for name in ("kv_cache_dtype", "context_length", "max_output_tokens", "max_reasoning_tokens",
                 "reasoning_mode", "force_auto_reasoning", "reasoning_session_once"):
        assert getattr(disabled, name) == getattr(enabled, name)
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "0")
    assert QwenExoConfig.from_server_args(_server_args(tmp_path)) == disabled


@pytest.fixture
def disabled_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "0")
    config = QwenExoConfig.from_server_args(_server_args(tmp_path))
    return QwenExoRuntime(
        config,
        SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))),
        HybridRuntimePolicy(
            tp_size=2, dtype="bfloat16", page_size=64,
            mamba_strategy="extra_buffer_lazy", mamba_state_dtype="bfloat16",
        ),
    )


def test_disabled_runtime_ignores_existing_personality_gdn_and_hints(disabled_runtime):
    runtime = disabled_runtime
    runtime.cognition.upsert("identity.md", "---\nsource_kind: gpt_cognition_identity_card\n---\nPRIVATE_COGNITION_SENTINEL")
    runtime.policy_data.upsert("policy.md", "PRIVATE_POLICY_SENTINEL")
    runtime.file_knowledge.upsert("fact.md", b"Unique transport retry memory evidence")
    runtime.file_knowledge.upsert("background.md", b"Unrelated gardening flowers and soil")
    runtime.memory_hints.refresh(runtime.file_knowledge.metadata())
    assert runtime.memory_hints.lookup("transport retry memory evidence")
    assert runtime.cognition.snapshot.documents
    assert runtime.policy_data.snapshot.documents
    # Even a stale loaded snapshot or an admin persona toggle cannot bypass the master gate.
    runtime.policy_data_enabled = True
    runtime._session_initial_gdn_value = {"source_digest": "old", "state_identity": "old"}
    request = SimpleNamespace(instructions="Keep my instructions", metadata={})
    assert runtime._cognition_text() is None
    assert runtime.personality_instructions() is None
    assert runtime._attach_cognition_text(request) is request
    assert runtime._attach_policy_data(request) is request
    assert runtime._build_session_initial_gdn_prompt() is None
    assert runtime.initial_gdn_selection(response_id="disabled-turn") is None
    assert runtime.latent_transplant_payload(request) is None
    assert not runtime._session_initial_gdn_pins
    runtime._refresh_memory_hints_if_stale()
    assert runtime._memory_hints_refresh is None
    assert asyncio.run(runtime.memory_block_for_turn(
        "disabled-turn", blocked_reason=None, query_text="transport retry memory evidence"
    )) is None
    assert not runtime._memory_tool_stats
    assert runtime._schedule_session_initial_gdn_refresh(reason="new_memory") is None
    assert asyncio.run(runtime._refresh_session_initial_gdn(reason="startup")) is None
    assert runtime._session_initial_gdn_refresh_task is None
    assert runtime.memory_summary_status() == {"available": False, "reason": "memory_disabled"}
    assert runtime.memory_dream_status() == {"available": False, "reason": "memory_disabled"}


def test_disabled_binding_preserves_normal_cache_key_and_nonmemory_params(disabled_runtime):
    original = {
        INITIAL_GDN_PARAM: {"state_identity": "client-supplied"},
        "qwen_exo_latent_transplant": {"artifact": "old-memory"},
        "qwen_exo_activation_editor": {"editor": "old-memory"},
        "qwen_exo_disable_activation_editor": False,
        "unrelated": "preserved",
    }
    params, namespace = bind_initial_gdn_request(
        disabled_runtime, custom_params=original, extra_key="client-prefix", response_id="r1"
    )
    assert params == {"unrelated": "preserved", "qwen_exo_disable_activation_editor": True}
    assert namespace == INITIAL_GDN_DISABLED_CACHE_NAMESPACE + "|client-prefix"
    assert INITIAL_GDN_PARAM in original
    assert bind_initial_gdn_request(
        disabled_runtime, custom_params=original, extra_key="client-prefix",
        previous_response_id="r1", response_id="r2",
    ) == (params, namespace)


def test_disabled_memory_keeps_client_functions_and_blocks_existing_tool_server(tmp_path, monkeypatch):
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
    from sglang.srt.entrypoints.openai.tool_server import FileMemoryToolServer

    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "1")
    server = FileMemoryToolServer(tmp_path / "knowledge", tmp_path / "state")
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.tool_server = server
    serving.use_harmony = False
    serving.tool_call_parser = "qwen"
    request = ResponsesRequest(model="model", input="Use my tool", tools=[{
        "type": "function", "name": "client_lookup", "description": "Client lookup",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }])
    assert len(serving._chat_tools_for_request(request)) > 1
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "0")
    assert serving._memory_tools_blocked_reason(request) == "memory_disabled"
    tools = serving._chat_tools_for_request(request)
    assert len(tools) == 1
    assert tools[0].function.name == "client_lookup"
    assert tools[0].function.parameters == request.tools[0].parameters
    assert server.chat_tool_schemas_for_memory_scope(None, "shared") == ()
    assert not server.has_tool(MEMORY_READ_TOOL)
    assert server.get_tool_description(MEMORY_READ_TOOL) is None
    with pytest.raises(RuntimeError, match="memory_disabled"):
        server.executor_for_memory_scope(None, "shared")
    with pytest.raises(RuntimeError, match="memory_disabled"):
        FileMemoryToolServer(tmp_path / "other", tmp_path / "other-state")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_existing_executor_fails_closed_for_reads_searches_and_writes(tmp_path, monkeypatch, asynchronous):
    from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
    from qwen_exo_booster.memory_tools import MemoryToolExecutor

    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "1")
    index = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    executor = MemoryToolExecutor(index)
    ok, result = executor.run_sync(MEMORY_WRITE_TOOL, {
        "path": "note.md", "title": "Existing note", "content": "PRIVATE_STORED_EVIDENCE",
        "mode": "create",
    })
    assert ok, result
    path = index.path_of("work-memory/note.md")
    before = path.read_bytes()
    assert executor.run_sync(MEMORY_READ_TOOL, {"path": "work-memory/note.md"})[0]
    monkeypatch.setenv("QWEN_EXO_MEMORY_ENABLED", "0")
    calls = [
        (MEMORY_READ_TOOL, {"path": "work-memory/note.md"}),
        (MEMORY_SEARCH_TOOL, {"pattern": "PRIVATE_STORED_EVIDENCE"}),
        (MEMORY_WRITE_TOOL, {"path": "new.md", "title": "New", "content": "blocked", "mode": "create"}),
    ]
    calls.extend((MEMORY_WRITE_TOOL, {"path": "note.md", "content": "blocked", "mode": mode})
                 for mode in ("append", "edit", "replace", "delete"))
    for name, arguments in calls:
        result = asyncio.run(executor.run(name, arguments)) if asynchronous else executor.run_sync(name, arguments)
        assert result == (False, "Error: memory_disabled")
    assert path.read_bytes() == before
    assert not (index.root / "work-memory" / "new.md").exists()
