import json

import pytest
from qwen_exo_booster.fingerprint import (
    ModelIdentity,
    main as fingerprint_main,
    validate_qwen_exo_config,
    validate_qwen_exo_model_path,
)
from qwen_exo_booster.service_launcher import _validate_qwen_exo_model_arguments


def write_model(root, *, architecture="Qwen3_5ForConditionalGeneration"):
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    config = {
        "architectures": [architecture],
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "model_type": "qwen3_5_text",
            "attn_output_gate": True,
            "layer_types": layer_types,
            "max_position_embeddings": 262144,
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
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 55562855904}, "weight_map": {}}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("tokenizer-v1", encoding="utf-8")


def write_moe_model(root):
    write_model(root, architecture="Qwen3_5MoeForConditionalGeneration")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    text_config.update(
        {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "intermediate_size": None,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "linear_num_value_heads": 32,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(40)
            ],
        }
    )
    config["model_type"] = "qwen3_5_moe"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def write_122b_moe_model(root):
    write_moe_model(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    text_config.update(
        {
            "num_hidden_layers": 48,
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "linear_num_value_heads": 64,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(48)
            ],
        }
    )
    text_config.pop("intermediate_size")
    text_config.pop("partial_rotary_factor")
    text_config["rope_parameters"]["partial_rotary_factor"] = 0.25
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 127152313312}, "weight_map": {}}),
        encoding="utf-8",
    )


def test_122b_moe_model_accepts_exact_published_hybrid_layout(tmp_path):
    write_122b_moe_model(tmp_path)

    assert validate_qwen_exo_model_path(tmp_path) == "moe-122b-a10b"
    identity = ModelIdentity.from_path(tmp_path)
    assert identity.layer_count == 48
    assert identity.full_attention_layers == 12
    assert identity.linear_attention_layers == 36
    assert identity.weight_bytes == 127152313312
    assert identity.fingerprint
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert validate_qwen_exo_config(config) == "moe-122b-a10b"


def test_122b_moe_layout_accepts_q4_checkpoint_metadata(tmp_path):
    write_122b_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert validate_qwen_exo_model_path(tmp_path) == "moe-122b-a10b"


def test_model_path_rejects_git_lfs_pointer_weight(tmp_path):
    write_122b_moe_model(tmp_path)
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1024},
                "weight_map": {"model.weight": shard},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / shard).write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 1024\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="still a Git LFS pointer"):
        validate_qwen_exo_model_path(tmp_path)


def test_model_path_rejects_missing_weight_shard(tmp_path):
    write_122b_moe_model(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1024},
                "weight_map": {"model.weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weight shard is missing"):
        validate_qwen_exo_model_path(tmp_path)


def test_122b_moe_model_rejects_nearby_unverified_shape(tmp_path):
    write_122b_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["moe_intermediate_size"] = 768
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="moe-122b-a10b.*moe_intermediate_size"):
        validate_qwen_exo_model_path(tmp_path)


def test_moe_model_identity_accepts_hybrid_layout(tmp_path):
    write_moe_model(tmp_path)

    identity = ModelIdentity.from_path(tmp_path)
    identity.validate_qwen_exo_model()

    assert identity.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert identity.layer_count == 40
    assert identity.full_attention_layers == 10
    assert identity.linear_attention_layers == 30


def test_moe_runtime_config_accepts_sglang_normalized_intermediate_size():
    from types import SimpleNamespace

    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    text = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=40,
        hidden_size=2048,
        intermediate_size=5632,
        head_dim=256,
        full_attention_interval=4,
        num_attention_heads=16,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        max_position_embeddings=262144,
        vocab_size=248320,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        layer_types=layer_types,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_parameters={"rope_theta": 10_000_000},
    )
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        model_type="qwen3_5_moe",
        text_config=text,
    )

    assert validate_qwen_exo_config(config) == "moe-35b-a3b"


def test_moe_122b_model_identity_accepts_hybrid_layout(tmp_path):
    write_122b_moe_model(tmp_path)

    identity = ModelIdentity.from_path(tmp_path)
    identity.validate_qwen_exo_model()

    assert identity.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert identity.layer_count == 48
    assert identity.full_attention_layers == 12
    assert identity.linear_attention_layers == 36
    assert validate_qwen_exo_model_path(tmp_path) == "moe-122b-a10b"


def test_moe_122b_model_rejects_wrong_expert_dimension(tmp_path):
    write_122b_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["moe_intermediate_size"] = 512
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="moe_intermediate_size"):
        validate_qwen_exo_model_path(tmp_path)


def test_moe_model_path_accepts_omitted_unused_intermediate_size(tmp_path):
    write_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"].pop("intermediate_size")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert validate_qwen_exo_model_path(tmp_path) == "moe-35b-a3b"


def test_moe_model_path_accepts_sglang_materialized_unused_intermediate_size(
    tmp_path,
):
    write_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["intermediate_size"] = 5632
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert validate_qwen_exo_model_path(tmp_path) == "moe-35b-a3b"


def test_moe_model_path_accepts_null_unused_intermediate_size(tmp_path):
    write_moe_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["intermediate_size"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert validate_qwen_exo_model_path(tmp_path) == "moe-35b-a3b"


def test_model_identity_freezes_hybrid_layout(tmp_path):
    write_model(tmp_path)

    identity = ModelIdentity.from_path(tmp_path)
    identity.validate_qwen_exo_model()

    assert identity.layer_count == 64
    assert identity.full_attention_layers == 16
    assert identity.linear_attention_layers == 48
    assert identity.weight_bytes == 55562855904


def test_model_fingerprint_changes_with_tokenizer(tmp_path):
    write_model(tmp_path)
    first = ModelIdentity.from_path(tmp_path)
    (tmp_path / "tokenizer.json").write_text("tokenizer-v2", encoding="utf-8")
    second = ModelIdentity.from_path(tmp_path)

    assert first.fingerprint != second.fingerprint


def test_wrong_architecture_is_rejected(tmp_path):
    write_model(tmp_path, architecture="OtherModel")
    identity = ModelIdentity.from_path(tmp_path)

    with pytest.raises(ValueError, match="exact verified Dense 27B"):
        identity.validate_qwen_exo_model()


def test_model_path_preflight_uses_config_structure_not_directory_name(tmp_path):
    model_path = tmp_path / "Qwen3.8-27B-marketing-name"
    model_path.mkdir()
    write_model(model_path)

    assert validate_qwen_exo_model_path(model_path) == "dense-27b"


def test_model_path_accepts_current_nested_partial_rotary_factor(tmp_path):
    write_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    text_config.pop("partial_rotary_factor")
    text_config["rope_parameters"]["partial_rotary_factor"] = 0.25
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert validate_qwen_exo_model_path(tmp_path) == "dense-27b"


def test_model_path_rejects_conflicting_top_level_partial_rotary_factor(tmp_path):
    write_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    text_config["partial_rotary_factor"] = 0.5
    text_config["rope_parameters"]["partial_rotary_factor"] = 0.25
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="partial_rotary_factor"):
        validate_qwen_exo_model_path(tmp_path)


def test_model_path_preflight_rejects_false_compatible_directory_name(tmp_path, capsys):
    model_path = tmp_path / "Qwen3.8-27B-marketing-name"
    model_path.mkdir()
    write_model(model_path, architecture="OtherModel")

    with pytest.raises(SystemExit) as raised:
        fingerprint_main([str(model_path)])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "Directory names and marketing labels are never trusted" in error
    assert "Set QWEN_EXO_MODEL_PATH to a Qwen-series checkpoint" in error
    assert "Dense 27B, MoE 35B-A3B, or MoE 122B-A10B" in error


def test_model_path_preflight_missing_config_has_actionable_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        fingerprint_main([str(tmp_path)])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "model config was not found" in error
    assert "Directory names and marketing labels are never trusted" in error
    assert "Set QWEN_EXO_MODEL_PATH to a Qwen-series checkpoint" in error
    assert "Dense 27B, MoE 35B-A3B, or MoE 122B-A10B" in error


def test_service_launcher_missing_model_path_has_actionable_error():
    with pytest.raises(SystemExit) as raised:
        _validate_qwen_exo_model_arguments(["--enable-qwen-exo"])

    error = str(raised.value)
    assert "--model-path is required" in error
    assert "Directory names and marketing labels are never trusted" in error
    assert "Set QWEN_EXO_MODEL_PATH to a Qwen-series checkpoint" in error


def test_service_launcher_blocks_non_qwen_runtime_structure_before_exec(tmp_path):
    write_model(tmp_path, architecture="OtherModel")

    with pytest.raises(SystemExit, match="exact verified Dense 27B"):
        _validate_qwen_exo_model_arguments(
            ["--enable-qwen-exo", "--model-path", str(tmp_path)]
        )


def test_service_launcher_preserves_upstream_models_when_qwen_exo_is_disabled(
    tmp_path,
):
    write_model(tmp_path, architecture="OtherModel")

    assert _validate_qwen_exo_model_arguments(["--model-path", str(tmp_path)]) is None


def test_unverified_dense_shape_is_rejected(tmp_path):
    write_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["hidden_size"] = 4096
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unverified.*hidden_size"):
        ModelIdentity.from_path(tmp_path).validate_qwen_exo_model()


def test_hf_config_inherited_moe_defaults_do_not_change_dense_variant():
    from types import SimpleNamespace

    from qwen_exo_booster.fingerprint import validate_qwen_exo_config

    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=64,
        layer_types=layer_types,
        max_position_embeddings=262144,
        hidden_size=5120,
        intermediate_size=17408,
        head_dim=256,
        full_attention_interval=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        vocab_size=248320,
        rope_parameters={"rope_theta": 10000000},
        attn_output_gate=True,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
    config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=text_config,
    )
    assert validate_qwen_exo_config(config) == "dense-27b"

    text_config.layer_types = None
    text_config.layers_block_type = [
        "attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    assert validate_qwen_exo_config(config) == "dense-27b"
