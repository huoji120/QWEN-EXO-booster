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


def test_tensor_bank_defaults_reserve_context_and_full_attention_capacity():
    values = default_values()

    assert values["context_length"] == 102400
    assert values["qwen_exo_tensor_bank_max_document_tokens"] == 100352
    assert values["qwen_exo_tensor_bank_salient_token_budget"] == 4096

    values["qwen_exo_tensor_bank_max_document_tokens"] = 100416
    with pytest.raises(ServiceConfigError, match="上下文长度减 2048"):
        validate_values(values)


def test_values_round_trip_through_managed_server_args():
    values = default_values()
    values.update(
        context_length=131072,
        max_prefill_tokens=32768,
        qwen_exo_enable_policy_data=False,
        qwen_exo_telemetry_text_mode="all",
        qwen_exo_qk_recall_preset="strict",
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
            "--qwen-exo-qk-recall-preset",
            "broad",
        ],
        values,
    )

    assert effective_args[:2] == ["--model-path", "/models/qwen-exo"]
    assert values_from_args(effective_args) == validate_values(values)
    assert effective_args.count("--context-length") == 1
    assert "--no-qwen-exo-enable-policy-data" in effective_args
    assert "--qwen-exo-telemetry-text-mode" in effective_args
    assert effective_args.count("--qwen-exo-qk-recall-preset") == 1
    assert values_from_args(effective_args)["qwen_exo_qk_recall_preset"] == "strict"
    assert values_from_args(effective_args)["qwen_exo_max_output_tokens"] == 8192
    assert (
        values_from_args(effective_args)["qwen_exo_context_integrity_context_divisor"]
        == 3
    )
    assert effective_args.count("--qwen-exo-console-trace-default-scope") == 1
    assert (
        values_from_args(effective_args)["qwen_exo_console_trace_default_scope"]
        == "all"
    )


def test_store_persists_revision_and_marks_exact_config_applied(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial = store.ensure([])

    updated = store.update(
        {"qwen_exo_max_candidates": 12},
        expected_revision=initial["revision"],
    )

    assert updated["pending_restart"] is True
    assert updated["revision"] != initial["revision"]
    assert updated["values"]["qwen_exo_max_candidates"] == 12

    applied, effective_args = store.mark_applied([])

    assert applied["applied_revision"] == updated["revision"]
    assert values_from_args(effective_args)["qwen_exo_max_candidates"] == 12
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["applied_revision"] == persisted["revision"]


def test_store_rejects_stale_revision_without_overwriting(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial = store.ensure([])
    store.update(
        {"qwen_exo_max_candidates": 10},
        expected_revision=initial["revision"],
    )

    with pytest.raises(ServiceConfigError, match="其他会话") as exc_info:
        store.update(
            {"qwen_exo_max_candidates": 11},
            expected_revision=initial["revision"],
        )

    assert exc_info.value.code == "revision_conflict"
    assert store.public_document()["values"]["qwen_exo_max_candidates"] == 10


def test_second_unhealthy_boot_rolls_back_to_last_healthy_revision(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial, _ = store.mark_applied([])
    assert store.mark_healthy() is True
    store.update(
        {"qwen_exo_max_candidates": 12},
        expected_revision=initial["revision"],
    )

    failed_attempt, _ = store.mark_applied([])
    rolled_back, effective_args = store.mark_applied([])

    assert failed_attempt["revision"] != initial["revision"]
    assert rolled_back["revision"] == initial["revision"]
    assert rolled_back["last_failed_revision"] == failed_attempt["revision"]
    assert values_from_args(effective_args)["qwen_exo_max_candidates"] == 8


def test_validation_rejects_incompatible_runtime_contract():
    values = default_values()
    values["qwen_exo_observer_mode"] = "shadow"

    with pytest.raises(ServiceConfigError, match="Adaptive refresh") as exc_info:
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


def test_activation_editor_strength_is_the_only_managed_editor_setting():
    values = default_values()
    assert "qwen_exo_activation_editor_enabled" not in values
    assert values["qwen_exo_activation_editor_strength"] == "2.0"

    values["qwen_exo_activation_editor_strength"] = "8.0"
    with pytest.raises(ServiceConfigError, match="必须是以下值之一"):
        validate_values(values)


def test_new_runtime_features_default_active_with_distinct_groups():
    values = default_values()

    assert values["qwen_exo_qk_prefilter_mode"] == "active"
    assert values["qwen_exo_context_evidence_mode"] == "active"
    assert values["qwen_exo_context_integrity_mode"] == "active"
    assert values["qwen_exo_reflection_memory_mode"] == "active"
    assert values["qwen_exo_response_compaction_mode"] == "active"
    assert values["qwen_exo_context_integrity_context_divisor"] == 3

    settings = {setting.key: setting for setting in SERVICE_SETTINGS}
    assert settings["qwen_exo_context_evidence_mode"].group == "post_tool_evidence"
    assert settings["qwen_exo_context_integrity_mode"].group == "context_integrity"
    assert (
        settings["qwen_exo_context_integrity_context_divisor"].group
        == "context_integrity"
    )
    assert settings["qwen_exo_reflection_memory_mode"].group == "reflection_memory"
    assert settings["qwen_exo_response_compaction_mode"].group == "compaction"
    assert settings["qwen_exo_qk_prefilter_mode"].choices == ("off", "active")
    assert settings["qwen_exo_context_integrity_mode"].choices == ("off", "active")


def test_managed_qk_expansion_margin_defaults_to_production_rank_gate():
    values = default_values()
    settings = {setting.key: setting for setting in SERVICE_SETTINGS}
    margin = settings["qwen_exo_qk_expansion_margin"]

    assert values["qwen_exo_qk_expansion_margin"] == 0.02
    assert margin.default == 0.02
    assert "推荐值：0.02" in margin.description


def test_ensure_migrates_legacy_modes_to_active_reflection_memory(tmp_path: Path):
    store = ServiceConfigStore(tmp_path / "service-config.json")
    initial = store.ensure([])
    document = json.loads(store.path.read_text(encoding="utf-8"))
    for key in (
        "qwen_exo_reflection_memory_mode",
        "qwen_exo_reflection_memory_idle_seconds",
        "qwen_exo_reflection_memory_min_events",
        "qwen_exo_reflection_memory_min_tokens",
        "qwen_exo_reflection_memory_max_attempts",
        "qwen_exo_reflection_memory_max_output_tokens",
        "qwen_exo_reflection_memory_max_history_tokens",
    ):
        document["values"].pop(key)
    document["values"].update(
        qwen_exo_qk_prefilter_mode="shadow",
        qwen_exo_context_evidence_mode="off",
        qwen_exo_context_integrity_mode="shadow",
        qwen_exo_dream_reflection_mode="off",
        qwen_exo_dream_reflection_min_events=5,
        qwen_exo_response_compaction_mode="off",
    )
    store.path.write_text(json.dumps(document), encoding="utf-8")

    migrated = store.ensure([])

    assert migrated["revision"] != initial["revision"]
    assert migrated["values"]["qwen_exo_qk_prefilter_mode"] == "active"
    assert migrated["values"]["qwen_exo_context_evidence_mode"] == "active"
    assert migrated["values"]["qwen_exo_context_integrity_mode"] == "active"
    assert migrated["values"]["qwen_exo_reflection_memory_mode"] == "active"
    assert migrated["values"]["qwen_exo_reflection_memory_min_events"] == 5
    assert migrated["values"]["qwen_exo_response_compaction_mode"] == "active"
    assert "qwen_exo_dream_reflection_mode" not in migrated["values"]


def test_qk_recall_preset_rejects_free_form_thresholds():
    values = default_values()
    values["qwen_exo_qk_recall_preset"] = "0.42"

    with pytest.raises(ServiceConfigError, match="必须是以下值之一"):
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
    document["values"].pop("qwen_exo_qk_recall_preset")
    document["values"].pop("qwen_exo_console_trace_default_scope")
    document["values"]["qwen_exo_latent_transplant_default"] = "legacy-name"
    (tmp_path / "service-config.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )

    migrated = store.ensure([])

    assert "qwen_exo_activation_editor_enabled" not in migrated["values"]
    assert migrated["values"]["qwen_exo_qk_recall_preset"] == "balanced"
    assert migrated["values"]["qwen_exo_console_trace_default_scope"] == "activity"
    assert "qwen_exo_latent_transplant_default" not in migrated["values"]
