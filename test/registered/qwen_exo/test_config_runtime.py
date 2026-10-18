import argparse
import asyncio
import json
import sys
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qwen_exo_booster.config import PROJECT_NAME, QwenExoConfig
from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState


def server_args(tmp_path, **overrides):
    values = {
        "qwen_exo_state_dir": str(tmp_path / "state"),
        "qwen_exo_knowledge_dir": str(tmp_path / "knowledge"),
        "qwen_exo_policy_data_dir": str(tmp_path / "policydata"),
        "qwen_exo_max_internal_fanout": 32,
        "qwen_exo_internal_mamba_reserve": 6,
        "qwen_exo_max_internal_tokens": 4096,
        "qwen_exo_max_reasoning_tokens": 3072,
        "qwen_exo_max_output_tokens": 8192,
        "qwen_exo_observer_mode": "shadow",
        "qwen_exo_enable_hybrid_prefix": True,
        "qwen_exo_enable_policy_data": True,
        "qwen_exo_enable_capsule": True,
        "model_path": "model",
        "tp_size": 2,
        "dtype": "bfloat16",
        "page_size": 64,
        "mamba_radix_cache_strategy": "extra_buffer",
        "disable_radix_cache": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_has_no_external_learning_surface(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))
    public = config.public_dict()

    assert public["project"] == PROJECT_NAME
    assert public["tp_size"] == 2
    assert all("learning" not in key.lower() for key in public)


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("min", 512), ("minimal", 512), ("low", 1024), ("medium", 2048),
        ("high", 4090), ("xhigh", 8192), ("max", 8192), ("none", 0),
        (None, 3072), (0, 3072), (0.5, 3072),
    ],
)
def test_reasoning_budget_resolves_requested_tier(tmp_path, effort, expected):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))
    assert config.reasoning_budget_for_effort(effort) == expected


@pytest.mark.parametrize("tier", ["min", "low", "medium", "high", "xhigh"])
def test_reasoning_budget_override_is_independent_of_fallback(tmp_path, tier):
    config = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_max_reasoning_tokens=128,
            **{f"qwen_exo_reasoning_budget_{tier}": 9011},
        )
    )
    assert config.reasoning_budget_for_effort(tier) == 9011
    assert config.reasoning_budget_for_effort(None) == 128
    assert config.public_dict()[f"reasoning_budget_{tier}"] == 9011


@pytest.mark.parametrize("tier", ["min", "low", "medium", "high", "xhigh"])
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "512"])
def test_reasoning_budget_rejects_nonpositive_or_noninteger(tmp_path, tier, invalid):
    with pytest.raises(ValueError, match=f"reasoning_budget_{tier}"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, **{f"qwen_exo_reasoning_budget_{tier}": invalid})
        )


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("qwen_exo_max_reasoning_tokens", 0),
        ("qwen_exo_reasoning_mode", "auto"),
        ("qwen_exo_force_auto_reasoning", 1),
        ("qwen_exo_force_auto_reasoning", "false"),
        ("qwen_exo_reasoning_session_once", 0),
        ("qwen_exo_reasoning_session_once", 1),
        ("qwen_exo_reasoning_session_once", "false"),
        ("qwen_exo_reasoning_session_once", None),
        ("qwen_exo_reasoning_dynamic_context_tokens", 0),
        ("qwen_exo_reasoning_dynamic_context_tokens", 255),
        ("qwen_exo_reasoning_dynamic_context_tokens", True),
        ("qwen_exo_reasoning_dynamic_context_tokens", 1.5),
    ],
)
def test_reasoning_configuration_rejects_invalid_controls(tmp_path, key, invalid):
    with pytest.raises(ValueError, match=key):
        QwenExoConfig.from_server_args(server_args(tmp_path, **{key: invalid}))


@pytest.mark.parametrize("session_once", [True, False])
def test_managed_reasoning_configuration_preserves_tier_resolution(
    tmp_path, session_once
):
    from qwen_exo_booster.service_config import (
        apply_values_to_args,
        default_values,
        values_from_args,
    )

    values = default_values()
    values.update(
        qwen_exo_max_reasoning_tokens=128,
        qwen_exo_reasoning_budget_min=111,
        qwen_exo_reasoning_budget_low=222,
        qwen_exo_reasoning_budget_medium=333,
        qwen_exo_reasoning_budget_high=444,
        qwen_exo_reasoning_budget_xhigh=9999,
        qwen_exo_reasoning_mode="dynamic",
        qwen_exo_force_auto_reasoning=False,
        qwen_exo_reasoning_session_once=session_once,
        qwen_exo_reasoning_dynamic_context_tokens=1024,
    )
    argv = apply_values_to_args(
        [
            "--qwen-exo-reasoning-budget-xhigh", "512",
            "--qwen-exo-reasoning-session-once",
            "--no-qwen-exo-reasoning-session-once",
        ],
        values,
    )
    config = QwenExoConfig.from_server_args(
        server_args(tmp_path, **values_from_args(argv))
    )
    if sys.platform != "win32":
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--model", "dummy", *argv])
        config = QwenExoConfig.from_server_args(parsed)
    assert [config.reasoning_budget_for_effort(tier) for tier in (
        "min", "low", "medium", "high", "xhigh", "none", None
    )] == [111, 222, 333, 444, 9999, 0, 128]
    assert config.reasoning_mode == "dynamic"
    assert config.force_auto_reasoning is False
    assert config.reasoning_session_once is session_once
    assert config.public_dict()["reasoning_session_once"] is session_once
    assert config.reasoning_dynamic_context_tokens == 1024


def test_experimental_activation_training_defaults_off_and_is_public(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))

    assert config.feature_flags.activation_training is False
    assert config.public_dict()["features"]["activation_training"] is False

    enabled = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_experimental_activation_training=True)
    )
    assert enabled.feature_flags.activation_training is True
    assert enabled.public_dict()["features"]["activation_training"] is True


def test_response_output_budget_is_public_and_positive(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_max_output_tokens=12288)
    )

    assert config.max_output_tokens == 12288
    assert config.public_dict()["max_output_tokens"] == 12288
    with pytest.raises(ValueError, match="max_output_tokens"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_max_output_tokens=0)
        )


def test_trajectory_score_bias_defaults_are_bounded_and_public(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_score_bias_mode="trajectory_shadow")
    )

    assert config.feature_flags.score_bias is True
    assert config.score_bias_mode == "trajectory_shadow"
    assert config.score_bias_max == 0.05
    assert config.score_bias_max_blocks == 8
    assert config.score_bias_min_age_steps == 2
    assert config.score_bias_tail_tokens == 4096
    assert config.score_bias_selected_blocks == 2
    assert config.public_dict()["score_bias_tail_ratio"] == 0.15
    assert config.score_bias_min_relevance == 0.01
    assert config.score_bias_anchor_bias == 0.01

    explicit_zero = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_score_bias_mode="trajectory_shadow",
            qwen_exo_score_bias_min_relevance=0.0,
            qwen_exo_score_bias_anchor_bias=0.0,
        )
    )
    assert explicit_zero.score_bias_min_relevance == 0.0
    assert explicit_zero.score_bias_anchor_bias == 0.0

    with pytest.raises(ValueError, match="score_bias_mode"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_score_bias_mode="global")
        )


def test_score_bias_requires_observer_and_valid_selection_limits(tmp_path):
    with pytest.raises(ValueError, match="requires an active or shadow observer"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_score_bias_mode="trajectory_active",
                qwen_exo_observer_mode="off",
            )
        )

    with pytest.raises(ValueError, match="trajectory selection limits"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_score_bias_mode="trajectory_shadow",
                qwen_exo_score_bias_selected_blocks=9,
                qwen_exo_score_bias_max_blocks=8,
            )
        )


def test_policy_data_cli_arguments_are_registered():
    if sys.platform == "win32":
        pytest.skip("SGLang ServerArgs imports the POSIX resource module")
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    parsed = parser.parse_args(
        [
            "--model",
            "dummy",
            "--qwen-exo-policy-data-dir",
            "/data/policydata",
            "--qwen-exo-enable-policy-data",
            "--qwen-exo-max-reasoning-tokens",
            "6144",
            "--qwen-exo-max-output-tokens",
            "12288",
        ]
    )

    assert parsed.qwen_exo_policy_data_dir == "/data/policydata"
    assert parsed.qwen_exo_enable_policy_data is True
    assert not hasattr(parsed, "qwen_exo_max_policy_tokens")  # persona is injected in full
    assert parsed.qwen_exo_max_reasoning_tokens == 6144
    assert parsed.qwen_exo_max_output_tokens == 12288
    assert not hasattr(parsed, "qwen_exo_tensor_bank_max_document_tokens")


def test_immediate_uncertainty_retrieval_cli_argument_is_registered():
    if sys.platform == "win32":
        pytest.skip("SGLang ServerArgs imports the POSIX resource module")
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model", "dummy", "--qwen-exo-immediate-uncertainty-retrieval"]
    )

    assert parsed.qwen_exo_immediate_uncertainty_retrieval is True


def test_response_compaction_config_is_bounded_and_public(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_response_compaction_mode="active",
            qwen_exo_response_compaction_max_history_tokens=4096,
            qwen_exo_response_compaction_max_dropped_items=4,
            qwen_exo_response_compaction_max_output_tokens=1024,
        )
    )

    assert config.response_compaction_mode == "active"
    assert config.public_dict()["response_compaction_max_dropped_items"] == 4
    with pytest.raises(ValueError, match="output budget cannot exceed"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_response_compaction_mode="active",
                qwen_exo_max_internal_tokens=1024,
                qwen_exo_response_compaction_max_output_tokens=2048,
            )
        )
    with pytest.raises(ValueError, match="response_compaction_mode"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_response_compaction_mode="shadow")
        )
    with pytest.raises(ValueError, match="history budget"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_response_compaction_max_history_tokens=512,
            )
        )


def test_reflection_memory_rejects_shadow(tmp_path):
    with pytest.raises(ValueError, match="reflection_memory_mode"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_reflection_memory_mode="shadow")
        )


def test_immediate_uncertainty_retrieval_requires_active_observer(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_observer_mode="active",
            qwen_exo_immediate_uncertainty_retrieval=True,
        )
    )
    assert config.immediate_uncertainty_retrieval is True
    assert config.public_dict()["immediate_uncertainty_retrieval"] is True

    with pytest.raises(ValueError, match="requires qwen_exo_observer_mode=active"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_immediate_uncertainty_retrieval=True,
            )
        )


def test_policy_data_flag_follows_server_args(tmp_path):
    value = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_enable_policy_data=False)
    )

    assert value.feature_flags.policy_data is False


def test_observer_rejects_unimplemented_multiple_refresh_attempts(tmp_path):
    with pytest.raises(ValueError, match="trigger limits"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_observer_max_triggers=2)
        )


def test_runtime_creates_authoritative_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    model_path = tmp_path / "model"
    model_path.mkdir()
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "attn_output_gate": True,
                    "num_hidden_layers": 64,
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "head_dim": 256,
                    "full_attention_interval": 4,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "partial_rotary_factor": 0.25,
                    "vocab_size": 248320,
                    "rope_parameters": {"rope_theta": 10000000},
                    "layer_types": layer_types,
                    "max_position_embeddings": 262144,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}),
        encoding="utf-8",
    )
    runtime = QwenExoRuntime.from_server_args(
        server_args(tmp_path, model_path=str(model_path)),
        SimpleNamespace(
            tokenizer=SimpleNamespace(
                encode=lambda text, **kwargs: [ord(char) for char in text]
            )
        ),
    )

    asyncio.run(runtime.start())
    assert runtime.state is QwenExoRuntimeState.READY
    assert runtime.config.state_directory.is_dir()
    assert runtime.config.knowledge_directory.is_dir()
    assert runtime.config.policy_data_directory.is_dir()
    assert runtime.config.cognition_directory.is_dir()
    assert runtime.status()["external_learning"] is False
    assert runtime.status()["hybrid_state"]["atomic_full_gdn_lifecycle"] is True
    assert runtime.status()["knowledge"]["document_count"] == 0
    assert runtime.status()["cognition"] == {
        "source_digest": runtime.cognition.snapshot.source_digest,
        "document_count": 0,
        "always_on": False,
        "route": "text_instructions",
    }
    assert runtime.status()["policy_data"] == {
        "source_digest": runtime.policy_data.snapshot.source_digest,
        "document_count": 0,
        "enabled": True,
        "always_on": False,
        "route": "text_instructions",
    }
    assert runtime.status()["internal_services"] == {
        "capsule": True,
        "memory_tools": True,
        "policy_data": True,
        "cognition": False,
    }

    asyncio.run(runtime.close())
    assert runtime.state is QwenExoRuntimeState.STOPPED


def test_session_initial_gdn_keeps_target_only_under_replayssm_spec():
    runtime = object.__new__(QwenExoRuntime)
    runtime.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(enable_gdn_replayssm_spec=True)
    )
    assert runtime._session_initial_gdn_dflash_mode() == "target_only"
    runtime.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace())
    assert runtime._session_initial_gdn_dflash_mode() == "eligible"


@pytest.mark.parametrize("grade", ["verified", "supported", "unresolved"])
def test_global_gdn_accepts_active_memory_independent_of_causal_grade(grade):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(x) for x in text]

        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, ensure_ascii=False)

    entry = {
        "entry_id": "admitted", "version": 1, "admission_status": "active",
        "causal_status": grade, "title": "条件性缓存经验", "mechanism": "机制待查",
        "rule": "先核对缓存键", "missing_evidence": ["缺少隔离变量比较"],
    }
    record = {"created_at": 1, "causal_schema": 1, "causal_entries": [entry]}
    runtime = object.__new__(QwenExoRuntime)
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())
    runtime.model_identity = SimpleNamespace(fingerprint="model")
    runtime.config = SimpleNamespace(context_length=10000)
    runtime.reflection_memory_store = SimpleNamespace(list=lambda: [record])
    prompt = runtime._build_session_initial_gdn_prompt()
    assert prompt["memory_count"] == 1
    assert "缺少隔离变量比较" in prompt["prompt"]
    assert grade in prompt["prompt"]
    original_identity = prompt["state_identity"]
    entry["missing_evidence"] = ["缺少回退复现"]
    assert runtime._build_session_initial_gdn_prompt()["state_identity"] != original_identity
    for status in ("candidate", "retired"):
        entry["admission_status"] = status
        assert runtime._build_session_initial_gdn_prompt() is None
    entry["admission_status"] = "active"
    entry["rule"] = "oversized" * 10000
    # No partial entry is substituted when the admitted entry exceeds the budget.
    assert runtime._build_session_initial_gdn_prompt() is None


def test_startup_without_memories_serves_without_initial_gdn():
    """A fresh deployment has no reflection memories and must still come up.

    With an empty store the consolidation has nothing to prefill: the runtime
    records ``no_memory``, keeps the neutral generation warmup, and hands out
    no selection so requests run from a zero recurrent state instead of
    failing closed.
    """
    jobs_run = []
    events = []

    class Runner:
        async def run_batch(self, jobs, prompts, sampling_params, **kwargs):
            jobs_run.append(tuple(jobs))
            return (
                SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    latency_seconds=0.1,
                    finish_reason={"type": "stop"},
                ),
            )

    runtime = object.__new__(QwenExoRuntime)
    runtime.internal_jobs = Runner()
    runtime.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(speculative_algorithm="DFLASH")
    )
    runtime._session_initial_gdn_refresh_lock = asyncio.Lock()
    runtime._session_initial_gdn_value_lock = threading.RLock()
    runtime._session_initial_gdn_value = None
    runtime._session_initial_gdn_status = {"status": "not_started"}
    runtime._build_session_initial_gdn_prompt = lambda: None
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (event_type, payload)
        )
    )

    asyncio.run(runtime._run_startup_generation_warmup())

    assert runtime._session_initial_gdn_status["status"] == "no_memory"
    assert runtime.initial_gdn_selection() is None
    assert [job.job_id for batch in jobs_run for job in batch] == [
        "qwen-exo-startup-generation-warmup"
    ]
    assert events[-1][0] == "runtime.startup_generation_warmup"
    assert events[-1][1]["session_initial_gdn"] is False


def test_runtime_startup_generation_builds_global_session_gdn(tmp_path):
    calls = []
    finished = []
    events = []

    class Runner:
        async def run_batch(self, jobs, prompts, sampling_params, **kwargs):
            calls.append((tuple(jobs), tuple(prompts), sampling_params, kwargs))
            return (
                SimpleNamespace(
                    text="consolidated reflection",
                    prompt_tokens=60,
                    completion_tokens=16,
                    latency_seconds=0.25,
                    finish_reason={"type": "stop"},
                    metadata={"qwen_exo_session_initial_gdn_status": ["exported"]},
                ),
            )

        async def finish_parent(self, parent_id):
            finished.append(parent_id)

    runtime = object.__new__(QwenExoRuntime)
    runtime.internal_jobs = Runner()
    runtime.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(enable_gdn_replayssm_spec=False)
    )
    runtime.config = SimpleNamespace(
        context_length=128, max_internal_tokens=32, state_directory=tmp_path
    )
    runtime._session_initial_gdn_refresh_lock = asyncio.Lock()
    runtime._session_initial_gdn_value_lock = threading.RLock()
    runtime._session_initial_gdn_value = None
    runtime._session_initial_gdn_status = {"status": "not_started"}
    runtime._build_session_initial_gdn_prompt = lambda: {
        "prompt": "rendered recent memories",
        "prompt_tokens": 60,
        "source_tokens": 40,
        "input_budget": 64,
        "source_digest": "a" * 64,
        "state_identity": "b" * 64,
        "memory_count": 2,
        "available_memory_count": 3,
    }
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )

    asyncio.run(runtime._run_startup_generation_warmup())

    assert len(calls) == 1
    jobs, prompts, sampling_params, kwargs = calls[0]
    assert jobs[0].job_type.value == "reflection_memory"
    assert jobs[0].deadline_monotonic is None
    assert prompts == ("rendered recent memories",)
    # DFLASH-eligible so user requests co-schedule with the consolidation
    # (target-only requests exclude eligible ones from the batch), with strict
    # acceptance so the exported state follows the target distribution.
    assert sampling_params["custom_params"] == {
        "qwen_exo_dflash": "eligible",
        "qwen_exo_dflash_think_phase": False,
    }
    assert kwargs["custom_params_per_job"] == (
        {
            "qwen_exo_session_initial_gdn_export": {
                "source_digest": "a" * 64,
                "state_identity": "b" * 64,
            }
        },
    )
    assert finished == [jobs[0].parent_request_id]
    assert runtime._session_initial_gdn_value["state_identity"] == "b" * 64
    assert runtime._session_initial_gdn_value["truncated"] is False
    assert runtime._session_initial_gdn_status["status"] == "ready"
    assert [event[1] for event in events] == [
        "runtime.session_initial_gdn_updated",
        "runtime.session_initial_gdn_generation",
        "runtime.startup_generation_warmup",
    ]
    sidecar = json.loads(
        (tmp_path / "session-initial-gdn" / f"{'b' * 64}.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["reflection"] == "consolidated reflection"
    assert sidecar["finish_reason"] == "stop"


class _CharacterTokenizer:
    def __init__(self):
        self.template_kwargs = []

    @staticmethod
    def encode(value, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in str(value)]

    @staticmethod
    def decode(values, **_kwargs):
        return "".join(chr(int(value)) for value in values)

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append(kwargs)
        return "\n".join(str(message["content"]) for message in messages)


def _session_gdn_prompt_runtime(records, *, context_length):
    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(context_length=context_length)
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=_CharacterTokenizer())
    runtime.model_identity = SimpleNamespace(fingerprint="model-fingerprint")
    runtime.reflection_memory_store = SimpleNamespace(list=lambda: list(records))
    return runtime


def test_session_initial_gdn_prompt_is_newest_first_and_half_context_bounded():
    runtime = _session_gdn_prompt_runtime(
        [
            {
                "source_digest": "old",
                "created_at": 1.0,
                "title": "OLD_MEMORY_MARKER",
                "reflection": "old evidence",
            },
            {
                "source_digest": "new",
                "created_at": 2.0,
                "title": "NEW_MEMORY_MARKER",
                "reflection": "new evidence",
            },
        ],
        context_length=8000,
    )

    prompt = runtime._build_session_initial_gdn_prompt()

    assert prompt is not None
    assert prompt["prompt_tokens"] <= runtime.config.context_length // 2
    assert prompt["memory_count"] == 2
    assert prompt["memory_digests"] == ("new", "old")
    assert prompt["prompt"].index("NEW_MEMORY_MARKER") < prompt["prompt"].index(
        "OLD_MEMORY_MARKER"
    )
    # The consolidation reasons over the memories before writing the reflection.
    tokenizer = runtime.tokenizer_manager.tokenizer
    assert all(kw["enable_thinking"] is True for kw in tokenizer.template_kwargs)


def test_session_initial_gdn_prompt_never_cuts_a_memory_record():
    """Budget overflow drops whole records instead of truncating token runs.

    The previous builder sliced the concatenated token stream at the budget,
    which left the last memory as a half record. A record that does not fit
    is skipped so older, smaller records still make it in intact.
    """
    runtime = _session_gdn_prompt_runtime(
        [
            {
                "source_digest": "old",
                "created_at": 1.0,
                "title": "OLD_MEMORY_MARKER",
                "reflection": "short",
            },
            {
                "source_digest": "new",
                "created_at": 2.0,
                "title": "NEW_MEMORY_MARKER",
                "reflection": "x" * 5000,
            },
        ],
        context_length=8000,
    )

    prompt = runtime._build_session_initial_gdn_prompt()

    assert prompt is not None
    assert prompt["prompt_tokens"] <= 4000
    assert prompt["memory_count"] == 1
    assert prompt["memory_digests"] == ("old",)
    assert "NEW_MEMORY_MARKER" not in prompt["prompt"]
    assert prompt["prompt"].count("<reflection_memory ") == prompt["prompt"].count(
        "</reflection_memory>"
    )


def test_new_memory_refresh_runs_in_background_and_coalesces():
    """Storing memories must not block on consolidation, and bursts coalesce.

    Consolidation prefills every memory and decodes a full reflection. The
    organization job stores several memories back to back; awaiting a refresh
    per memory serialized minutes of generation into that job. A burst before
    the pass starts folds into it; a burst during it yields one follow-up pass.
    """
    reasons = []
    release = asyncio.Event()

    async def refresh(*, reason):
        reasons.append(reason)
        await release.wait()

    runtime = _quiet_runtime()
    runtime._refresh_session_initial_gdn = refresh

    async def scenario():
        for _ in range(3):
            await runtime._on_reflection_memory_stored(SimpleNamespace())
        await asyncio.sleep(0)
        assert reasons == ["new_memory"]
        for _ in range(3):
            await runtime._on_reflection_memory_stored(SimpleNamespace())
        release.set()
        await runtime._session_initial_gdn_refresh_task
        assert reasons == ["new_memory", "new_memory"]
        assert runtime._session_initial_gdn_refresh_requested is None

    asyncio.run(scenario())


def _quiet_runtime():
    runtime = object.__new__(QwenExoRuntime)
    runtime._session_initial_gdn_refresh_task = None
    runtime._session_initial_gdn_refresh_requested = None
    runtime._request_questions = {}
    runtime._finalize_tasks = {}
    runtime._foreground_last_activity = time.monotonic() - 3600
    return runtime


def test_new_memory_refresh_waits_for_user_turns_and_reflections():
    """Consolidation decodes thousands of tokens; it must not start while a
    user request is in flight or a reflection is still publishing memories."""
    reasons = []

    async def refresh(*, reason):
        reasons.append(reason)

    runtime = _quiet_runtime()
    runtime._refresh_session_initial_gdn = refresh
    runtime._request_questions = {"resp-user": "question"}

    async def scenario():
        reflection = asyncio.create_task(asyncio.sleep(3600))
        runtime._server_session_reflections = {reflection}
        await runtime._on_reflection_memory_stored(SimpleNamespace())
        await asyncio.sleep(0.05)
        assert reasons == []
        runtime._request_questions.clear()
        await asyncio.sleep(0.05)
        assert reasons == []
        reflection.cancel()
        await asyncio.gather(reflection, return_exceptions=True)
        await asyncio.wait_for(runtime._session_initial_gdn_refresh_task, 10)
        assert reasons == ["new_memory"]

    with patch("qwen_exo_booster.runtime._FOREGROUND_POLL_SECONDS", 0.01):
        asyncio.run(scenario())


def _selection_runtime(identity):
    runtime = object.__new__(QwenExoRuntime)
    runtime._session_initial_gdn_value_lock = threading.RLock()
    runtime._session_initial_gdn_pins = OrderedDict()
    runtime._session_initial_gdn_value = {
        "source_digest": "a" * 64,
        "state_identity": identity,
    }
    return runtime


def _swap_global_identity(runtime, identity):
    with runtime._session_initial_gdn_value_lock:
        runtime._session_initial_gdn_value = {
            "source_digest": "b" * 64,
            "state_identity": identity,
        }


def test_initial_gdn_selection_uses_global_snapshot():
    runtime = _selection_runtime("1" * 64)

    first = runtime.initial_gdn_selection()
    same_snapshot = runtime.initial_gdn_selection()
    _swap_global_identity(runtime, "2" * 64)
    refreshed = runtime.initial_gdn_selection()

    assert first["state_identity"] == "1" * 64
    assert same_snapshot == first
    assert refreshed["state_identity"] == "2" * 64
    assert refreshed["cache_namespace"] != first["cache_namespace"]
    assert first["scope"] == "global"
    assert refreshed["scope"] == "global"


def test_continued_conversation_keeps_its_initial_gdn_across_refresh():
    """A refresh must not change the identity of an in-flight conversation.

    Switching a 97K-token conversation to the new namespace threw away its
    whole radix-cached prefix and forced a full re-prefill on the next turn,
    which is what produced the multi-minute first-token stalls after every
    refresh. The turn that continues a response keeps that response's pin;
    only a conversation without a known parent picks up the refreshed state.
    """
    runtime = _selection_runtime("1" * 64)

    opened = runtime.initial_gdn_selection(response_id="resp_1")
    _swap_global_identity(runtime, "2" * 64)
    continued = runtime.initial_gdn_selection(
        previous_response_id="resp_1", response_id="resp_2"
    )
    chained = runtime.initial_gdn_selection(
        previous_response_id="resp_2", response_id="resp_3"
    )
    fresh = runtime.initial_gdn_selection(response_id="resp_new")
    unknown_parent = runtime.initial_gdn_selection(
        previous_response_id="resp_from_before_restart", response_id="resp_x"
    )

    assert opened["state_identity"] == "1" * 64
    assert continued == opened
    assert chained == opened
    assert fresh["state_identity"] == "2" * 64
    assert unknown_parent["state_identity"] == "2" * 64
    # Read-only lookups (no response_id) never create pins.
    runtime.initial_gdn_selection(previous_response_id="resp_1")
    assert set(runtime._session_initial_gdn_pins) == {
        "resp_1",
        "resp_2",
        "resp_3",
        "resp_new",
        "resp_x",
    }


def test_initial_gdn_pins_are_bounded_lru():
    runtime = _selection_runtime("1" * 64)
    from qwen_exo_booster import runtime as runtime_module

    limit = runtime_module._SESSION_INITIAL_GDN_MAX_PINS
    for index in range(limit + 5):
        runtime.initial_gdn_selection(response_id=f"resp_{index}")
    assert len(runtime._session_initial_gdn_pins) == limit
    assert "resp_0" not in runtime._session_initial_gdn_pins
    assert f"resp_{limit + 4}" in runtime._session_initial_gdn_pins


def test_responses_default_to_high_thinking_without_overriding_explicit_choice():
    class Request:
        def __init__(self, reasoning):
            self.reasoning = reasoning

        def model_copy(self, update):
            return Request(update.get("reasoning", self.reasoning))

    enabled = QwenExoRuntime._enable_default_response_thinking(Request(None))
    explicit = SimpleNamespace(effort="none")
    preserved = QwenExoRuntime._enable_default_response_thinking(Request(explicit))

    assert enabled.reasoning.effort == "high"
    assert preserved.reasoning is explicit


def test_initial_gdn_uses_active_entries_not_raw_summaries_or_old_versions():
    active = {
        "entry_id": "kept",
        "version": 2,
        "title": "VERIFIED_RULE",
        "causal_status": "verified",
        "admission_status": "active",
        "rule": "Verified scoped rule",
        "scope": "exact environment",
        "versions": [{"rule": "OLD_VERSION_NOT_REPLAYED"}],
    }
    record = {
        "source_digest": "causal",
        "created_at": 1,
        "causal_schema": 1,
        "reflection": "RAW_SUMMARY_NOT_REPLAYED",
        "causal_entries": [
            active,
            {
                **active,
                "entry_id": "candidate",
                "title": "CANDIDATE_NOT_REPLAYED",
                "causal_status": "supported",
                "admission_status": "candidate",
            },
            {
                **active,
                "entry_id": "recallable",
                "title": "RECALLABLE_SUPPORTED",
                "causal_status": "supported",
            },
            {
                **active,
                "entry_id": "retired",
                "title": "RETIRED_NOT_REPLAYED",
                "admission_status": "retired",
            },
        ],
    }
    value = _session_gdn_prompt_runtime([record], context_length=16000)
    prompt = value._build_session_initial_gdn_prompt()
    assert "VERIFIED_RULE" in prompt["prompt"]
    assert "RECALLABLE_SUPPORTED" in prompt["prompt"]
    assert prompt["memory_count"] == 2
    for excluded in (
        "RAW_SUMMARY_NOT_REPLAYED",
        "CANDIDATE_NOT_REPLAYED",
        "RETIRED_NOT_REPLAYED",
        "OLD_VERSION_NOT_REPLAYED",
    ):
        assert excluded not in prompt["prompt"]
    record["causal_entries"] = [
        entry for entry in record["causal_entries"]
        if entry["admission_status"] != "active"
    ]
    assert value._build_session_initial_gdn_prompt() is None


def test_retiring_last_memory_clears_global_initial_gdn():
    value = _session_gdn_prompt_runtime([], context_length=16000)
    value._session_initial_gdn_refresh_lock = asyncio.Lock()
    value._session_initial_gdn_value_lock = threading.RLock()
    value._session_initial_gdn_value = {"state_identity": "retired-memory-state"}
    value._session_initial_gdn_status = {"status": "ready"}
    assert (
        asyncio.run(value._refresh_session_initial_gdn(reason="memory_retired")) is None
    )
    assert value.initial_gdn_selection() is None
