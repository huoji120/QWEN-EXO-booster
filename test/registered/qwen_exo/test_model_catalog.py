from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_exo_booster.model_catalog import ModelCatalogError, ModelCatalogStore
from qwen_exo_booster import activation_training, service_launcher
from qwen_exo_booster.service_config import ServiceConfigStore


def write_model(
    root: Path,
    architecture: str,
    *,
    large_moe: bool = False,
    quantization_config: dict | None = None,
) -> None:
    root.mkdir()
    moe = architecture == "Qwen3_5MoeForConditionalGeneration"
    layer_count = 48 if large_moe else (40 if moe else 64)
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "head_dim": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
        "full_attention_interval": 4,
        "num_hidden_layers": layer_count,
        "intermediate_size": None if moe else 17408,
        "hidden_size": 3072 if large_moe else (2048 if moe else 5120),
        "num_attention_heads": 32 if large_moe else (16 if moe else 24),
        "num_key_value_heads": 2 if moe else 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64 if large_moe else (32 if moe else 48),
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(layer_count)
        ],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 10_000_000},
    }
    if moe:
        text.update(
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=1024 if large_moe else 512,
            shared_expert_intermediate_size=1024 if large_moe else 512,
        )
    config = {
        "architectures": [architecture],
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": text,
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / name).write_text(name, encoding="utf-8")


def test_model_catalog_clones_sources_and_keeps_profiles_isolated(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    write_model(moe, "Qwen3_5MoeForConditionalGeneration")

    data = tmp_path / "data"
    (data / "knowledge").mkdir(parents=True)
    (data / "policydata").mkdir()
    (data / "cognition").mkdir()
    (data / "trajectories").mkdir()
    (data / "knowledge" / "dense.md").write_text("# Dense", encoding="utf-8")
    (data / "policydata" / "policy.md").write_text("# Policy", encoding="utf-8")

    store = ModelCatalogStore([models], data)
    initial = store.ensure(dense)
    dense_fingerprint = initial["active_model_fingerprint"]
    catalog = store.public_document()
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in catalog["models"]
        if model["model_path"] == str(moe.resolve())
    )

    selected = store.select(
        moe_fingerprint,
        expected_revision=initial["revision"],
        clone_sources=True,
    )
    assert selected["active_model_fingerprint"] == moe_fingerprint
    moe_knowledge = data / "model-profiles" / moe_fingerprint / "knowledge"
    (moe_knowledge / "moe.md").write_text("# MoE", encoding="utf-8")
    assert not (
        data / "model-profiles" / dense_fingerprint / "knowledge" / "moe.md"
    ).exists()

    _, args, selected_model = store.mark_applied(
        [
            "--model-path",
            str(dense),
            "--qwen-exo-state-dir",
            "/data/qwen-exo/state-cuda",
            "--qwen-exo-knowledge-dir",
            "/data/qwen-exo/knowledge",
            "--qwen-exo-policy-data-dir",
            "/data/qwen-exo/policydata",
            "--qwen-exo-cognition-dir",
            "/data/qwen-exo/cognition",
        ]
    )
    assert selected_model["model_fingerprint"] == moe_fingerprint
    assert str(moe.resolve()) in args
    assert str(moe_knowledge) in args


def test_gptq_122b_catalog_uses_required_moe_wna16_runtime(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    gptq = models / "gptq-122b"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    write_model(
        gptq,
        "Qwen3_5MoeForConditionalGeneration",
        large_moe=True,
        quantization_config={
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "dynamic": {
                "-:.*attn.*": {},
                "-:.*shared_expert.*": {},
            },
        },
    )
    store = ModelCatalogStore([models], tmp_path / "data")
    initial = store.ensure(dense)
    target = next(
        model
        for model in store.discover_models()
        if model["model_path"] == str(gptq.resolve())
    )

    assert target["variant"] == "moe-122b-a10b"
    assert target["checkpoint_quantization"] == "gptq"
    assert target["runtime_quantization"] == "moe_wna16"
    assert target["checkpoint_quantization_bits"] == 4
    assert target["checkpoint_quantization_group_size"] == 128
    assert target["checkpoint_quantization_exclusions"] == [
        ".*attn.*",
        ".*shared_expert.*",
    ]

    store.select(
        target["model_fingerprint"],
        expected_revision=initial["revision"],
    )
    _, args, selected = store.mark_applied(
        [
            "--model-path",
            str(dense),
            "--quantization",
            "fp8",
            "--qwen-exo-state-dir",
            "/data/qwen-exo/state-cuda",
        ]
    )

    quantization_index = args.index("--quantization")
    assert args[quantization_index + 1] == "moe_wna16"
    assert args.count("--quantization") == 1
    assert selected["model_path"] == str(gptq.resolve())


def test_gptq_122b_requires_desc_act_false(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    gptq = models / "gptq-122b"
    write_model(
        gptq,
        "Qwen3_5MoeForConditionalGeneration",
        large_moe=True,
        quantization_config={
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": True,
            "sym": True,
        },
    )

    assert ModelCatalogStore([models], tmp_path / "data").discover_models() == []


def test_first_catalog_boot_keeps_legacy_runtime_paths(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    data = tmp_path / "data"
    store = ModelCatalogStore([models], data)
    _, args, selected_model = store.mark_applied(
        [
            "--model-path",
            str(dense),
            "--qwen-exo-state-dir",
            str(data / "state-cuda"),
            "--qwen-exo-knowledge-dir",
            str(data / "knowledge"),
            "--qwen-exo-policy-data-dir",
            str(data / "policydata"),
            "--qwen-exo-cognition-dir",
            str(data / "cognition"),
        ]
    )

    assert selected_model["model_path"] == str(dense.resolve())
    assert str(data / "state-cuda") in args
    assert str(data / "knowledge") in args
    assert str(data / "policydata") in args
    assert str(data / "cognition") in args

    document = store.public_document()
    assert document["legacy_model_fingerprint"] == selected_model["model_fingerprint"]


def test_model_catalog_rolls_back_second_unhealthy_boot(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    write_model(moe, "Qwen3_5MoeForConditionalGeneration")
    store = ModelCatalogStore([models], tmp_path / "data")
    initial = store.ensure(dense)
    dense_fingerprint = initial["active_model_fingerprint"]
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in store.public_document()["models"]
        if model["model_path"] == str(moe.resolve())
    )
    store.mark_healthy(dense_fingerprint)
    selected = store.select(moe_fingerprint, expected_revision=initial["revision"])
    store.mark_applied(["--model-path", str(dense)])
    _, _, restored_model = store.mark_applied(["--model-path", str(dense)])
    assert selected["active_model_fingerprint"] == moe_fingerprint
    assert restored_model["model_fingerprint"] == dense_fingerprint
    assert store.public_document()["last_failed_model_fingerprint"] == moe_fingerprint


def test_model_catalog_success_clears_matching_failed_marker(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    store = ModelCatalogStore([models], tmp_path / "data")
    document, _, model = store.mark_applied(["--model-path", str(dense)])
    document["last_failed_model_fingerprint"] = model["model_fingerprint"]
    document["last_rollback_at"] = "earlier"
    store._write_document(document)

    assert store.mark_healthy(model["model_fingerprint"]) is True
    public = store.public_document()
    assert public["last_failed_model_fingerprint"] is None
    assert public["last_rollback_at"] is None


def test_catalog_file_without_legacy_marker_uses_model_profile(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    data = tmp_path / "data"
    store = ModelCatalogStore([models], data)
    document = store.ensure(dense)
    document.pop("legacy_model_fingerprint")
    document["applied_model_fingerprint"] = document["active_model_fingerprint"]
    document["boot_attempts"] = 3
    store._write_document(document)

    _, args, _ = store.mark_applied(
        [
            "--model-path",
            str(dense),
            "--qwen-exo-state-dir",
            str(data / "state-cuda"),
            "--qwen-exo-knowledge-dir",
            str(data / "knowledge"),
            "--qwen-exo-policy-data-dir",
            str(data / "policydata"),
            "--qwen-exo-cognition-dir",
            str(data / "cognition"),
        ]
    )

    profile = data / "model-profiles" / document["active_model_fingerprint"]
    assert str(profile / "knowledge") in args
    assert str(profile / "policydata") in args


def test_model_catalog_rejects_stale_revision_without_initializing_target(
    tmp_path: Path,
):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    write_model(moe, "Qwen3_5MoeForConditionalGeneration")
    store = ModelCatalogStore([models], tmp_path / "data")
    initial = store.ensure(dense)
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in store.public_document()["models"]
        if model["model_path"] == str(moe.resolve())
    )

    with pytest.raises(ModelCatalogError, match="刷新后重试") as captured:
        store.select(moe_fingerprint, expected_revision="stale")

    assert captured.value.code == "revision_conflict"
    assert not (tmp_path / "data" / "model-profiles" / moe_fingerprint).exists()
    assert store.public_document()["revision"] == initial["revision"]


def test_model_catalog_reports_running_and_native_bank_state(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    store = ModelCatalogStore([models], tmp_path / "data")
    initial = store.ensure(dense)
    fingerprint = initial["active_model_fingerprint"]
    profile = tmp_path / "data" / "model-profiles" / fingerprint
    (profile / "state-cuda" / "model-native").mkdir(parents=True)

    public = store.public_document(running_model_fingerprint=fingerprint)

    assert public["models"][0]["active"] is True
    assert public["models"][0]["running"] is True
    assert public["models"][0]["native_bank_ready"] is True


def test_service_launcher_uses_selected_model_profile(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    write_model(dense, "Qwen3_5ForConditionalGeneration")
    write_model(moe, "Qwen3_5MoeForConditionalGeneration")
    data = tmp_path / "data"
    store = ModelCatalogStore([models], data)
    initial = store.ensure(dense)
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in store.public_document()["models"]
        if model["model_path"] == str(moe.resolve())
    )
    store.select(moe_fingerprint, expected_revision=initial["revision"])
    service_config_path = data / "service-config.json"
    ServiceConfigStore(service_config_path).ensure([])
    monkeypatch.setenv("QWEN_EXO_MODEL_CATALOG_ROOTS", str(models))
    monkeypatch.setenv(
        "QWEN_EXO_MODEL_CATALOG_CONFIG", str(data / "model-catalog.json")
    )
    monkeypatch.setenv("QWEN_EXO_MODEL_DATA_ROOT", str(data))
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(service_config_path))
    monkeypatch.setattr(
        activation_training,
        "run_pending_activation_training",
        lambda **kwargs: None,
    )
    executed = {}
    monkeypatch.setattr(
        service_launcher.os,
        "execvp",
        lambda executable, args: executed.update(executable=executable, args=args),
    )
    monkeypatch.setattr(
        service_launcher.sys,
        "argv",
        [
            "service_launcher",
            "--",
            "--enable-qwen-exo",
            "--model-path",
            str(dense),
            "--qwen-exo-state-dir",
            str(data / "state-cuda"),
            "--qwen-exo-knowledge-dir",
            str(data / "knowledge"),
            "--qwen-exo-policy-data-dir",
            str(data / "policydata"),
            "--qwen-exo-cognition-dir",
            str(data / "cognition"),
        ],
    )

    service_launcher.main()

    profile = data / "model-profiles" / moe_fingerprint
    assert str(moe.resolve()) in executed["args"]
    assert str(profile / "knowledge") in executed["args"]
    assert str(profile / "policydata") in executed["args"]
    assert str(profile / "cognition") in executed["args"]
    assert service_launcher.os.environ["QWEN_EXO_ACTIVE_MODEL_PROFILE"] == str(profile)
