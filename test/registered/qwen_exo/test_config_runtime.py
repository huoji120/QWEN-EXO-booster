import argparse
import asyncio
import json
import sys
import threading
from collections import OrderedDict
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
        "qwen_exo_internal_mamba_reserve": 6,
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
        "qwen_exo_experimental_context_integrity": False,
        "qwen_exo_context_integrity_mode": "active",
        "qwen_exo_context_integrity_context_divisor": 3,
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


def test_experimental_activation_training_defaults_off_and_is_public(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))

    assert config.feature_flags.activation_training is False
    assert config.public_dict()["features"]["activation_training"] is False

    enabled = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_experimental_activation_training=True)
    )
    assert enabled.feature_flags.activation_training is True
    assert enabled.public_dict()["features"]["activation_training"] is True


def test_qk_layer_and_head_selection_are_parsed_validated_and_public(tmp_path):
    """Layer/head selection defaults to legacy behaviour and rejects MLX.

    The MLX backend captures probe Q on its final layer only, so a configured
    recall layer would pair Q and K from different layers; it fails at config
    time instead of silently ranking against mismatched geometry.
    """
    default = QwenExoConfig.from_server_args(server_args(tmp_path))
    assert default.qk_layer_id is None
    assert default.qk_query_heads == ()
    assert default.qk_query_pooling == "windows"

    tuned = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_qk_layer=35,
            qwen_exo_qk_query_heads="7, 3,7,11",
            qwen_exo_qk_query_pooling="sentence",
        )
    )
    assert tuned.qk_layer_id == 35
    assert tuned.qk_query_heads == (3, 7, 11)
    assert tuned.public_dict()["qk_query_heads"] == [3, 7, 11]
    assert tuned.public_dict()["qk_layer_id"] == 35

    with pytest.raises(ValueError, match="qk_query_pooling"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_qk_query_pooling="tokens")
        )
    with pytest.raises(ValueError, match="MLX"):
        QwenExoConfig.from_server_args(
            server_args(tmp_path, qwen_exo_backend="mlx", qwen_exo_qk_layer=35)
        )


def test_experimental_context_integrity_defaults_off_and_can_be_enabled(tmp_path):
    disabled = QwenExoConfig.from_server_args(
        server_args(tmp_path, qwen_exo_context_integrity_mode="active")
    )
    assert disabled.context_integrity_mode == "off"
    assert disabled.feature_flags.context_integrity is False

    enabled = QwenExoConfig.from_server_args(
        server_args(
            tmp_path,
            qwen_exo_observer_mode="active",
            qwen_exo_enable_adaptive_refresh=True,
            qwen_exo_context_integrity_mode="active",
            qwen_exo_experimental_context_integrity=True,
        )
    )
    assert enabled.context_integrity_mode == "active"
    assert enabled.feature_flags.context_integrity is True
    assert enabled.public_dict()["features"]["context_integrity"] is True


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
            qwen_exo_experimental_context_integrity=True,
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
                qwen_exo_experimental_context_integrity=True,
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
        "route": "text_instructions",
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
        "route": "text_instructions",
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


def test_runtime_startup_warmup_records_native_probe_result():
    calls = []
    events = []

    class Probe:
        async def warmup(self):
            calls.append("warmup")
            return SimpleNamespace(
                status="ready",
                prompt_tokens=264,
                latency_seconds=0.25,
                cache_hit=False,
            )

    runtime = object.__new__(QwenExoRuntime)
    runtime.query_probe = Probe()
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )

    async def generation_warmup():
        calls.append("generation_warmup")

    runtime._run_startup_generation_warmup = generation_warmup

    asyncio.run(runtime._run_startup_warmup())

    assert calls == ["warmup", "generation_warmup"]
    assert events[0][0:2] == ("runtime", "runtime.startup_warmup")
    assert events[0][2]["query_probe_status"] == "ready"
    assert events[0][2]["query_probe_prompt_tokens"] == 264
    assert events[0][2]["cache_hit"] is False


def test_session_initial_gdn_keeps_target_only_under_replayssm_spec():
    runtime = object.__new__(QwenExoRuntime)
    runtime.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(enable_gdn_replayssm_spec=True)
    )
    assert runtime._session_initial_gdn_dflash_mode() == "target_only"
    runtime.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace())
    assert runtime._session_initial_gdn_dflash_mode() == "eligible"


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
    per memory serialized minutes of generation into that job. A burst now
    yields the running pass plus exactly one follow-up pass.
    """
    reasons = []
    release = asyncio.Event()

    async def refresh(*, reason):
        reasons.append(reason)
        await release.wait()

    runtime = object.__new__(QwenExoRuntime)
    runtime._session_initial_gdn_refresh_task = None
    runtime._session_initial_gdn_refresh_requested = None
    runtime._refresh_session_initial_gdn = refresh

    async def scenario():
        for _ in range(3):
            await runtime._on_reflection_memory_stored(SimpleNamespace())
        await asyncio.sleep(0)
        assert reasons == ["new_memory"]
        release.set()
        await runtime._session_initial_gdn_refresh_task
        assert reasons == ["new_memory", "new_memory"]
        assert runtime._session_initial_gdn_refresh_requested is None

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


def test_initial_gdn_uses_only_active_verified_causal_entries():
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
                "title": "RECALLABLE_NOT_REPLAYED",
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
    for excluded in (
        "RAW_SUMMARY_NOT_REPLAYED",
        "CANDIDATE_NOT_REPLAYED",
        "RECALLABLE_NOT_REPLAYED",
        "RETIRED_NOT_REPLAYED",
        "OLD_VERSION_NOT_REPLAYED",
    ):
        assert excluded not in prompt["prompt"]
    record["causal_entries"] = record["causal_entries"][1:]
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
