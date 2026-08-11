import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
from qwen_exo_booster.config import PROJECT_NAME, QwenExoConfig
from qwen_exo_booster.refresh import RefreshRecord
from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState


def server_args(tmp_path, **overrides):
    values = {
        "qwen_exo_state_dir": str(tmp_path / "state"),
        "qwen_exo_knowledge_dir": str(tmp_path / "knowledge"),
        "qwen_exo_policy_data_dir": str(tmp_path / "policydata"),
        "qwen_exo_max_internal_fanout": 32,
        "qwen_exo_max_internal_tokens": 4096,
        "qwen_exo_max_candidates": 8,
        "qwen_exo_max_memory_tokens": 8192,
        "qwen_exo_max_reasoning_tokens": 3072,
        "qwen_exo_max_output_tokens": 8192,
        "qwen_exo_observer_mode": "shadow",
        "qwen_exo_enable_hybrid_prefix": True,
        "qwen_exo_enable_external_memory": True,
        "qwen_exo_enable_policy_data": True,
        "qwen_exo_enable_reference_judge": True,
        "qwen_exo_enable_capsule": True,
        "qwen_exo_enable_adaptive_refresh": False,
        "qwen_exo_context_evidence_mode": "off",
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


def test_reasoning_budget_is_public_and_positive(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_max_reasoning_tokens=6144)
    )

    assert config.max_reasoning_tokens == 6144
    assert config.public_dict()["max_reasoning_tokens"] == 6144
    with pytest.raises(ValueError, match="max_reasoning_tokens"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_max_reasoning_tokens=0)
        )


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


def test_document_tensor_bank_contract_is_bounded_and_public(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_tensor_bank_max_document_tokens=16384,
            qwen_exo_tensor_bank_salient_token_budget=1024,
            qwen_exo_tensor_bank_surprisal_threshold=5.5,
            qwen_exo_tensor_bank_span_tokens=24,
        )
    )

    assert config.tensor_bank_max_document_tokens == 16384
    assert config.tensor_bank_salient_token_budget == 1024
    assert config.tensor_bank_surprisal_threshold == 5.5
    assert config.tensor_bank_span_tokens == 24
    assert config.public_dict()["tensor_bank_salient_token_budget"] == 1024

    with pytest.raises(ValueError, match="64-token aligned"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_tensor_bank_salient_token_budget=1000)
        )

    with pytest.raises(ValueError, match="cannot exceed context_length - 2048"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                context_length=16384,
                qwen_exo_tensor_bank_max_document_tokens=14400,
                qwen_exo_tensor_bank_salient_token_budget=64,
            )
        )

    with pytest.raises(ValueError, match="leave 2048 tokens"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                context_length=2048,
                qwen_exo_tensor_bank_max_document_tokens=64,
                qwen_exo_tensor_bank_salient_token_budget=64,
            )
        )


def test_qk_expansion_margin_is_bounded_and_public(tmp_path):
    config = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_qk_expansion_margin=0.02)
    )

    assert config.qk_expansion_margin == 0.02
    assert config.public_dict()["qk_expansion_margin"] == 0.02

    default_config = QwenExoConfig.from_server_args(server_args(tmp_path))
    assert default_config.qk_expansion_margin == 0.02
    assert default_config.qk_admission_margin == 0.02
    assert default_config.qk_admission_gates == (0.0, 0.02)

    broad = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_qk_recall_preset="broad",
            qwen_exo_qk_expansion_margin=0.0,
        )
    )
    assert broad.qk_admission_margin == 0.0
    assert (
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_qk_recall_preset="broad",
                qwen_exo_qk_expansion_margin=0.0,
                qwen_exo_qk_only_knowledge=True,
            )
        ).qk_admission_margin
        == 0.005
    )
    assert (
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_qk_recall_preset="strict",
                qwen_exo_qk_expansion_margin=0.0,
            )
        ).qk_admission_margin
        == 0.02
    )
    assert QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_qk_recall_preset="strict",
            qwen_exo_qk_expansion_margin=0.0,
        )
    ).qk_admission_gates == (8.0, 0.02)
    with pytest.raises(ValueError, match="expansion margin"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_qk_expansion_margin=-0.01)
        )


def test_request_recall_has_no_direct_text_admission_controls(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))

    assert config.feature_flags.reference_judge is True
    assert config.public_dict()["features"]["external_memory"] is True


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
            "--qwen-exo-max-policy-tokens",
            "2048",
            "--qwen-exo-max-reasoning-tokens",
            "6144",
            "--qwen-exo-max-output-tokens",
            "12288",
            "--qwen-exo-tensor-bank-max-document-tokens",
            "16384",
            "--qwen-exo-tensor-bank-salient-token-budget",
            "1024",
            "--qwen-exo-tensor-bank-surprisal-threshold",
            "5.5",
            "--qwen-exo-tensor-bank-span-tokens",
            "24",
        ]
    )

    assert parsed.qwen_exo_policy_data_dir == "/data/policydata"
    assert parsed.qwen_exo_enable_policy_data is True
    assert parsed.qwen_exo_max_policy_tokens == 2048
    assert parsed.qwen_exo_max_reasoning_tokens == 6144
    assert parsed.qwen_exo_max_output_tokens == 12288
    assert parsed.qwen_exo_tensor_bank_max_document_tokens == 16384
    assert parsed.qwen_exo_tensor_bank_salient_token_budget == 1024
    assert parsed.qwen_exo_tensor_bank_surprisal_threshold == 5.5
    assert parsed.qwen_exo_tensor_bank_span_tokens == 24


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


def test_context_evidence_cli_argument_is_registered():
    if sys.platform == "win32":
        pytest.skip("SGLang ServerArgs imports the POSIX resource module")
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model", "dummy", "--qwen-exo-context-evidence-mode", "active"]
    )

    assert parsed.qwen_exo_context_evidence_mode == "active"


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
    with pytest.raises(ValueError, match="requires external memory"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_response_compaction_mode="active",
                qwen_exo_enable_external_memory=False,
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


def test_context_features_require_adaptive_refresh_and_reject_shadow(tmp_path):
    with pytest.raises(ValueError, match="requires adaptive refresh"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_context_evidence_mode="active")
        )

    config = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_observer_mode="active",
            qwen_exo_enable_adaptive_refresh=True,
            qwen_exo_context_evidence_mode="active",
            qwen_exo_context_integrity_mode="active",
            context_length=204000,
            qwen_exo_context_integrity_context_divisor=3,
        )
    )
    assert config.context_evidence_mode == "active"
    assert config.context_integrity_mode == "active"
    assert config.context_integrity_max_tokens == 68000
    assert config.public_dict()["context_integrity_context_divisor"] == 3
    with pytest.raises(ValueError, match="context_evidence_mode"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_observer_mode="active",
                qwen_exo_enable_adaptive_refresh=True,
                qwen_exo_context_evidence_mode="shadow",
            )
        )
    with pytest.raises(ValueError, match="context_integrity_mode"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_observer_mode="active",
                qwen_exo_enable_adaptive_refresh=True,
                qwen_exo_context_integrity_mode="shadow",
            )
        )


def test_qk_prefilter_and_reflection_memory_reject_shadow(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))
    assert config.qk_prefilter_mode == "active"
    with pytest.raises(ValueError, match="qk_prefilter_mode"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_qk_prefilter_mode="shadow")
        )
    with pytest.raises(ValueError, match="reflection_memory_mode"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_reflection_memory_mode="shadow")
        )


def test_active_refresh_requires_active_observer(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_observer_mode="shadow",
                qwen_exo_enable_adaptive_refresh=True,
            )
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


@pytest.mark.parametrize(
    "disabled_flag",
    ["qwen_exo_enable_reference_judge", "qwen_exo_enable_external_memory"],
)
def test_active_refresh_requires_semantic_memory_dependencies(tmp_path, disabled_flag):
    with pytest.raises(ValueError, match="requires reference judge"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_observer_mode="active",
                qwen_exo_enable_adaptive_refresh=True,
                **{disabled_flag: False},
            )
        )


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
        SimpleNamespace(tokenizer=object()),
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
        "route": "native_tensor_bank_conditioning",
        "qk_ranked": False,
    }
    assert runtime.status()["policy_data"] == {
        "source_digest": runtime.policy_data.snapshot.source_digest,
        "document_count": 0,
        "enabled": True,
        "always_on": False,
        "semantic_eligibility_required": False,
        "qk_relevance_required": False,
        "reference_judge_required": False,
        "route": "attention_q_native_tensor_bank",
        "max_tokens": runtime.config.max_policy_tokens,
    }
    assert runtime.status()["internal_services"] == {
        "reference_judge": True,
        "capsule": True,
        "memory_pipeline": True,
        "self_ask_refresh": False,
        "policy_data": True,
        "tensor_bank": True,
        "cognition": False,
        "query_probe": False,
        "causal_replay": False,
        "adaptive_retrieval": False,
    }

    asyncio.run(runtime.close())
    assert runtime.state is QwenExoRuntimeState.STOPPED


def test_runtime_builds_think_context_from_real_refresh_record():
    record = RefreshRecord(
        parent_request_id="request-real",
        turn_id="request-real:post_tool:0",
        status="ready_for_safe_replay",
        question="Which wrapper is stale?",
        answer="The generated async wrapper still has the old signature.",
        selected_document_ids=("document-1",),
        selected_reference_digests=("digest-1",),
        decision_ids=("decision-1",),
        created_monotonic=1.0,
        purpose="post_tool",
    )

    injection = QwenExoRuntime._think_context_from_record(record)

    assert injection is not None
    assert injection.question == "Which wrapper is stale?"
    assert "generated async wrapper" in injection.answer
    assert injection.text == (
        "\n\nSelf-question: Which wrapper is stale?\n"
        "Self-answer: The generated async wrapper still has the old signature.\n"
    )


def test_runtime_builds_context_evidence_think_context_without_document():
    record = RefreshRecord(
        parent_request_id="request-context",
        turn_id="request-context:post_tool:0",
        status="context_evidence_ready",
        question="Which wrapper is stale?",
        answer="The direct tool result identified the generated async wrapper.",
        selected_document_ids=(),
        selected_reference_digests=(),
        decision_ids=("decision-context",),
        created_monotonic=1.0,
        purpose="post_tool",
        selected_lanes=("context",),
        context_status="eligible",
        context_source_digests=("context-digest",),
        context_decision_ids=("decision-context",),
    )

    injection = QwenExoRuntime._think_context_from_record(record)

    assert injection is not None
    assert injection.text == (
        "\n\nSelf-question: Which wrapper is stale?\n"
        "Self-answer: The direct tool result identified the generated async wrapper.\n"
    )


def test_runtime_post_tool_recall_queues_admitted_think_context(tmp_path):
    calls = []

    class RefreshService:
        async def refresh(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="ready_for_safe_replay",
                question="Q?",
                answer="A",
                selected_document_ids=("document-1",),
                decision_ids=("decision-1",),
                maybe_decision="admit_post_tool",
                reflection_kind="none",
            )

    class Telemetry:
        def __init__(self):
            self.events = []

        def emit(self, request_id, event_type, payload):
            self.events.append((request_id, event_type, payload))

    runtime = object.__new__(QwenExoRuntime)
    runtime._request_tool_calls = {}
    runtime._request_tool_observations = {}
    runtime.adaptive_retrieval = None
    runtime.observer = SimpleNamespace(mode="active")
    runtime.refresh_service = RefreshService()
    runtime._request_questions = {"request-1": "Original task"}
    runtime._request_outputs = {"request-1": "Reasoning so far"}
    runtime._pending_think_contexts = {}
    runtime._consumed_think_contexts = set()
    runtime.telemetry = Telemetry()

    injection = asyncio.run(
        runtime.recall_after_tool("request-1", "direct tool result", generation_index=2)
    )

    assert calls[0]["purpose"] == "post_tool"
    assert calls[0]["turn_id"] == "request-1:post_tool:2"
    assert calls[0]["partial_output"] == "Reasoning so far"
    assert calls[0]["latest_tool_observation"] == "direct tool result"
    assert injection is not None
    assert injection.question == "Q?"
    assert injection.answer == "A"
    assert runtime._pending_think_contexts["request-1"] == injection
    assert runtime.telemetry.events[0][1] == "post_tool_recall.completed"
    assert runtime.telemetry.events[0][2]["think_context_ready"] is True
    assert runtime.telemetry.events[0][2]["text_injected"] is False
