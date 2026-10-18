from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_exo_booster.service_config import (
    SERVICE_SETTINGS,
    ServiceConfigError,
    ServiceConfigStore,
    apply_values_to_args,
    default_values,
    validate_values,
    values_from_args,
)


@pytest.mark.parametrize(
    "key",
    [
        "qwen_exo_reasoning_budget_min",
        "qwen_exo_reasoning_budget_low",
        "qwen_exo_reasoning_budget_medium",
        "qwen_exo_reasoning_budget_high",
        "qwen_exo_reasoning_budget_xhigh",
    ],
)
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "2048", 32769])
def test_managed_reasoning_budgets_reject_invalid_values(key, invalid):
    values = default_values()
    values[key] = invalid
    with pytest.raises(ServiceConfigError) as error:
        validate_values(values)
    assert error.value.field == key


@pytest.mark.parametrize("invalid", [0, 255, True, 1.5, "2048"])
def test_managed_dynamic_context_rejects_invalid_values(invalid):
    values = default_values()
    values["qwen_exo_reasoning_dynamic_context_tokens"] = invalid
    with pytest.raises(ServiceConfigError) as error:
        validate_values(values)
    assert error.value.field == "qwen_exo_reasoning_dynamic_context_tokens"


def test_managed_reasoning_mode_rejects_unknown_mode():
    values = default_values()
    values["qwen_exo_reasoning_mode"] = "auto"
    with pytest.raises(ServiceConfigError) as error:
        validate_values(values)
    assert error.value.field == "qwen_exo_reasoning_mode"


@pytest.mark.parametrize(
    "key", ["qwen_exo_force_auto_reasoning", "qwen_exo_reasoning_session_once"]
)
@pytest.mark.parametrize("invalid", [0, 1, "false", "true", None, []])
def test_managed_reasoning_switches_reject_nonbooleans(key, invalid):
    values = default_values()
    values[key] = invalid
    with pytest.raises(ServiceConfigError) as error:
        validate_values(values)
    assert error.value.field == key
    assert error.value.code == "invalid_type"


def test_score_bias_defaults_enable_anchor_and_positive_relevance_floor():
    values = default_values()

    assert values["qwen_exo_score_bias_anchor_bias"] == 0.01
    assert values["qwen_exo_score_bias_min_relevance"] == 0.01

    validated = validate_values(values)
    assert validated["qwen_exo_score_bias_anchor_bias"] == 0.01
    assert validated["qwen_exo_score_bias_min_relevance"] == 0.01


def test_default_thinking_settings_map_to_chat_template_kwargs():
    values = default_values()
    assert values["default_enable_thinking"] is False
    assert values["default_preserve_thinking"] is False

    values["default_enable_thinking"] = True
    effective_args = apply_values_to_args(
        [
            "--model-path",
            "/models/qwen-exo",
            "--default-chat-template-kwargs",
            '{"enable_thinking":false,"preserve_thinking":true,"extra":"drop"}',
        ],
        values,
    )

    flag_index = effective_args.index("--default-chat-template-kwargs")
    template_kwargs = json.loads(effective_args[flag_index + 1])
    assert template_kwargs == {
        "enable_thinking": True,
        "preserve_thinking": False,
    }
    assert values_from_args(effective_args)["default_enable_thinking"] is True
    assert values_from_args(effective_args)["default_preserve_thinking"] is False
    assert "--default-enable-thinking" not in effective_args
    assert "--no-default-preserve-thinking" not in effective_args


def test_invalid_default_chat_template_kwargs_fail_closed():
    with pytest.raises(ServiceConfigError, match="必须是 JSON 对象"):
        values_from_args(["--default-chat-template-kwargs", "[]"])


def test_values_round_trip_through_managed_server_args():
    values = default_values()
    values.update(
        context_length=131072,
        max_prefill_tokens=32768,
        qwen_exo_enable_policy_data=False,
        qwen_exo_telemetry_text_mode="all",
        qwen_exo_reasoning_mode="dynamic",
        qwen_exo_console_trace_default_scope="all",
    )

    effective_args = apply_values_to_args(
        [
            "--model-path",
            "/models/qwen-exo",
            "--context-length",
            "8192",
            "--qwen-exo-enable-policy-data",
            "--qwen-exo-telemetry-text-mode",
            "off",
            "--qwen-exo-reasoning-mode",
            "fixed",
        ],
        values,
    )

    assert effective_args[:2] == ["--model-path", "/models/qwen-exo"]
    assert values_from_args(effective_args) == validate_values(values)
    assert effective_args.count("--context-length") == 1
    assert "--no-qwen-exo-enable-policy-data" in effective_args
    assert "--qwen-exo-telemetry-text-mode" in effective_args
    assert effective_args.count("--qwen-exo-reasoning-mode") == 1
    assert values_from_args(effective_args)["qwen_exo_reasoning_mode"] == "dynamic"
    assert values_from_args(effective_args)["qwen_exo_max_output_tokens"] == 8192
    assert effective_args.count("--qwen-exo-console-trace-default-scope") == 1
    assert (
        values_from_args(effective_args)["qwen_exo_console_trace_default_scope"]
        == "all"
    )


def test_store_persists_revision_and_marks_exact_config_applied(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial = store.ensure([])

    updated = store.update(
        {"qwen_exo_max_internal_fanout": 12},
        expected_revision=initial["revision"],
    )

    assert updated["pending_restart"] is True
    assert updated["revision"] != initial["revision"]
    assert updated["values"]["qwen_exo_max_internal_fanout"] == 12

    applied, effective_args = store.mark_applied([])

    assert applied["applied_revision"] == updated["revision"]
    assert values_from_args(effective_args)["qwen_exo_max_internal_fanout"] == 12
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["applied_revision"] == persisted["revision"]


@pytest.mark.parametrize("enabled", [True, False])
def test_store_applies_task_chain_reuse_toggle(tmp_path: Path, enabled):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    positive = "--qwen-exo-reasoning-session-once"
    negative = "--no-qwen-exo-reasoning-session-once"
    old_args = [negative if enabled else positive]
    initial = store.ensure(old_args)
    updated = store.update(
        {"qwen_exo_reasoning_session_once": enabled},
        expected_revision=initial["revision"],
    )
    assert updated["pending_restart"] is True
    assert updated["revision"] != initial["revision"]

    applied, argv = store.mark_applied(old_args)
    assert argv.count(positive if enabled else negative) == 1
    assert (negative if enabled else positive) not in argv
    assert values_from_args(argv)["qwen_exo_reasoning_session_once"] is enabled
    assert applied["applied_revision"] == updated["revision"]
    persisted = ServiceConfigStore(store.path).public_document()
    assert persisted["values"]["qwen_exo_reasoning_session_once"] is enabled


def test_store_rejects_stale_revision_without_overwriting(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial = store.ensure([])
    store.update(
        {"qwen_exo_max_internal_fanout": 10},
        expected_revision=initial["revision"],
    )

    with pytest.raises(ServiceConfigError, match="其他会话") as exc_info:
        store.update(
            {"qwen_exo_max_internal_fanout": 11},
            expected_revision=initial["revision"],
        )

    assert exc_info.value.code == "revision_conflict"
    assert store.public_document()["values"]["qwen_exo_max_internal_fanout"] == 10


def test_second_unhealthy_boot_rolls_back_to_last_healthy_revision(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial, _ = store.mark_applied([])
    assert store.mark_healthy() is True
    store.update(
        {"qwen_exo_max_internal_fanout": 12},
        expected_revision=initial["revision"],
    )

    failed_attempt, _ = store.mark_applied([])
    rolled_back, effective_args = store.mark_applied([])

    assert failed_attempt["revision"] != initial["revision"]
    assert rolled_back["revision"] == initial["revision"]
    assert rolled_back["last_failed_revision"] == failed_attempt["revision"]
    assert values_from_args(effective_args)["qwen_exo_max_internal_fanout"] == 8


def test_validation_rejects_incompatible_runtime_contract():
    values = default_values()
    values["qwen_exo_observer_mode"] = "shadow"

    with pytest.raises(ServiceConfigError, match="Immediate uncertainty retrieval") as exc_info:
        validate_values(values)

    assert exc_info.value.code == "invalid_contract"


def test_internal_budget_covers_reflection_retries_and_compaction():
    values = default_values()

    assert values["qwen_exo_max_internal_tokens"] == 12288
    assert values["qwen_exo_reflection_memory_max_output_tokens"] == 4096
    assert values["qwen_exo_reflection_memory_max_history_tokens"] == 92160
    values["qwen_exo_max_internal_tokens"] = 8192
    with pytest.raises(ServiceConfigError, match="retry output budget cannot exceed"):
        validate_values(values)

    values = default_values()
    values.update(
        qwen_exo_reflection_memory_mode="off",
        qwen_exo_response_compaction_mode="active",
        qwen_exo_max_internal_tokens=1024,
    )
    with pytest.raises(ServiceConfigError, match="output budget cannot exceed"):
        validate_values(values)


def test_console_trace_default_scope_rejects_unknown_filter():
    values = default_values()
    values["qwen_exo_console_trace_default_scope"] = "admitted"

    with pytest.raises(ServiceConfigError, match="必须是以下值之一"):
        validate_values(values)


def test_ensure_backfills_new_settings_for_legacy_documents(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    store.ensure([])
    document = json.loads((tmp_path / "service-config.json").read_text("utf-8"))
    document["values"]["qwen_exo_activation_editor_enabled"] = True
    document["values"].pop("qwen_exo_reasoning_mode")
    document["values"].pop("qwen_exo_console_trace_default_scope")
    document["values"]["qwen_exo_latent_transplant_default"] = "legacy-name"
    (tmp_path / "service-config.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )

    migrated = store.ensure([])

    assert "qwen_exo_activation_editor_enabled" not in migrated["values"]
    assert migrated["values"]["qwen_exo_reasoning_mode"] == "fixed"
    assert migrated["values"]["qwen_exo_console_trace_default_scope"] == "activity"
    assert "qwen_exo_latent_transplant_default" not in migrated["values"]


def test_ensure_drops_retired_qk_recall_settings_and_keeps_the_rest(tmp_path: Path):
    path = tmp_path / "service-config.json"
    store = ServiceConfigStore(path)
    store.ensure([])
    document = json.loads(path.read_text("utf-8"))
    kept = dict(document["values"], qwen_exo_max_internal_fanout=12)
    document["values"] = {
        **kept,
        "qwen_exo_enable_external_memory": False,
        "qwen_exo_enable_reference_judge": True,
        "qwen_exo_qk_recall_preset": "strict",
        "qwen_exo_tensor_bank_span_tokens": 16,
        "qwen_exo_replay_minimum_gain": 0.02,
        "qwen_exo_context_evidence_mode": "active",
    }
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    migrated = store.ensure([])

    assert migrated["values"] == validate_values(kept)
    assert migrated["previous_values"]["qwen_exo_qk_recall_preset"] == "strict"
    assert "--qwen-exo-qk-recall-preset" not in store.mark_applied([])[1]
