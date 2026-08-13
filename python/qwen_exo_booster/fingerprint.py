from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FINGERPRINT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


_QWEN_EXO_DENSE_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_QWEN_EXO_MOE_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"

_COMMON_TEXT_STRUCTURE = {
    "head_dim": 256,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 262144,
    "vocab_size": 248320,
    "full_attention_interval": 4,
}
_QWEN_EXO_LAYOUTS = {
    _QWEN_EXO_DENSE_ARCHITECTURE: (
        {
            "variant": "dense-27b",
            "model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "text_structure": {
                **_COMMON_TEXT_STRUCTURE,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
            },
        },
    ),
    _QWEN_EXO_MOE_ARCHITECTURE: (
        {
            "variant": "moe-35b-a3b",
            "model_type": "qwen3_5_moe",
            "text_model_type": "qwen3_5_moe_text",
            "text_structure": {
                **_COMMON_TEXT_STRUCTURE,
                "num_hidden_layers": 40,
                "hidden_size": 2048,
                "intermediate_size": None,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
            },
        },
        {
            "variant": "moe-122b-a10b",
            "model_type": "qwen3_5_moe",
            "text_model_type": "qwen3_5_moe_text",
            "text_structure": {
                **_COMMON_TEXT_STRUCTURE,
                "num_hidden_layers": 48,
                "hidden_size": 3072,
                "intermediate_size": None,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 64,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
            },
        },
    ),
}
_COMPATIBILITY_GUIDANCE = (
    "Directory names and marketing labels are never trusted. Set "
    "QWEN_EXO_MODEL_PATH to a Qwen-series checkpoint with one of the exact "
    "verified Dense 27B, MoE 35B-A3B, or MoE 122B-A10B Qwen3_5* runtime "
    "structures"
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _normalize_layer_type(value: Any) -> str:
    raw = getattr(value, "value", value)
    name = str(raw)
    if name.startswith("HybridLayerType."):
        name = name.rsplit(".", 1)[-1]
    return "full_attention" if name == "attention" else name


def _layout_mismatches(
    config: Any, text: Any, architecture: str, expected: dict[str, Any]
) -> list[str]:
    layer_types = tuple(
        _normalize_layer_type(value)
        for value in (
            _config_value(text, "layer_types", None)
            or _config_value(text, "layers_block_type", ())
            or ()
        )
    )
    mismatches: list[str] = []
    model_type = str(_config_value(config, "model_type", "") or "")
    if model_type != expected["model_type"]:
        mismatches.append(
            f"model_type={model_type!r} (expected {expected['model_type']!r})"
        )
    text_model_type = str(_config_value(text, "model_type", "") or "")
    if text_model_type != expected["text_model_type"]:
        mismatches.append(
            f"text_config.model_type={text_model_type!r} "
            f"(expected {expected['text_model_type']!r})"
        )
    text_structure = expected["text_structure"]
    for name, wanted in text_structure.items():
        observed = _config_value(text, name, None)
        if (
            architecture == _QWEN_EXO_MOE_ARCHITECTURE
            and name == "intermediate_size"
            and observed in {None, 5632}
        ):
            continue
        if observed != wanted:
            mismatches.append(f"{name}={observed!r} (expected {wanted!r})")

    expected_pattern = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(int(text_structure["num_hidden_layers"]))
    )
    if layer_types != expected_pattern:
        mismatches.append("layer_types does not match the verified 3:1 GDN/Full layout")
    if _config_value(text, "attn_output_gate", None) is not True:
        mismatches.append("attn_output_gate is not enabled")
    rope = _config_value(text, "rope_parameters", None) or _config_value(
        text, "rope_scaling", None
    )
    partial_rotary_factor = _config_value(text, "partial_rotary_factor", None)
    if partial_rotary_factor is None:
        partial_rotary_factor = _config_value(rope or {}, "partial_rotary_factor", 0.0)
    if float(partial_rotary_factor or 0.0) != 0.25:
        mismatches.append("partial_rotary_factor is not 0.25")
    if int(_config_value(rope or {}, "rope_theta", 0) or 0) != 10_000_000:
        mismatches.append("rope_theta is not 10000000")
    return mismatches


def validate_qwen_exo_config(config: Any) -> str:
    """Accept only exact Qwen hybrid tensor layouts verified for QWEN-EXO."""

    architectures = tuple(_config_value(config, "architectures", ()) or ())
    architecture = str(architectures[0]) if len(architectures) == 1 else ""
    layouts = _QWEN_EXO_LAYOUTS.get(architecture)
    if layouts is None:
        observed = repr(architectures) if architectures else "missing"
        raise ValueError(
            "QWEN-EXO rejected the checkpoint architecture "
            f"{observed}. {_COMPATIBILITY_GUIDANCE}"
        )

    text = _config_value(config, "text_config", None) or config
    failures: list[str] = []
    for expected in layouts:
        mismatches = _layout_mismatches(config, text, architecture, expected)
        if not mismatches:
            return str(expected["variant"])
        failures.append(f"{expected['variant']}: " + "; ".join(mismatches))

    raise ValueError(
        f"QWEN-EXO rejected unverified {architecture} structure: "
        + " | ".join(failures)
        + f". {_COMPATIBILITY_GUIDANCE}"
    )


def _load_weight_index(root: Path) -> dict[str, Any]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"model weight index was not found: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model weight index is not valid JSON: {index_path}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"model weight index has no weight_map object: {index_path}")
    return index


def _validate_weight_artifacts(root: Path, index: dict[str, Any]) -> None:
    filenames = sorted({str(value) for value in index["weight_map"].values()})
    if not filenames:
        return
    total_size = 0
    for filename in filenames:
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"model weight path escapes checkpoint root: {filename}")
        if not path.is_file():
            raise ValueError(f"model weight shard is missing: {path}")
        size = path.stat().st_size
        total_size += size
        if size <= 1024 and path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            raise ValueError(f"model weight shard is still a Git LFS pointer: {path}")
    expected_size = int(index.get("metadata", {}).get("total_size") or 0)
    if expected_size and total_size < expected_size:
        raise ValueError(
            "model weight shards are incomplete: "
            f"found {total_size} bytes, expected at least {expected_size}"
        )


def validate_qwen_exo_model_path(model_path: Path | str) -> str:
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"model config was not found: {config_path}. {_COMPATIBILITY_GUIDANCE}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"model config is not valid JSON: {config_path}. "
            f"{_COMPATIBILITY_GUIDANCE}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"model config must be a JSON object: {config_path}. "
            f"{_COMPATIBILITY_GUIDANCE}"
        )
    variant = validate_qwen_exo_config(config)
    index = _load_weight_index(root)
    _validate_weight_artifacts(root, index)
    return variant


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    fingerprint: str
    model_path: str
    architecture: str
    model_type: str
    layer_count: int
    full_attention_layers: int
    linear_attention_layers: int
    max_position_embeddings: int
    weight_bytes: int
    file_hashes: dict[str, str | None]

    @classmethod
    def from_path(cls, model_path: Path | str) -> ModelIdentity:
        root = Path(model_path).expanduser().resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config was not found: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = _load_weight_index(root)
        _validate_weight_artifacts(root, index)
        text_config = config.get("text_config") or config
        architectures = config.get("architectures") or []
        architecture = str(architectures[0]) if architectures else ""
        raw_layer_types = (
            text_config.get("layer_types") or text_config.get("layers_block_type") or ()
        )
        layer_types = tuple(_normalize_layer_type(value) for value in raw_layer_types)
        layer_count = int(text_config.get("num_hidden_layers") or len(layer_types))
        full_attention_layers = layer_types.count("full_attention")
        linear_attention_layers = layer_types.count("linear_attention")
        file_hashes = {name: _file_sha256(root / name) for name in _FINGERPRINT_FILES}
        fingerprint_payload = {
            "architecture": architecture,
            "model_type": config.get("model_type"),
            "text_config": {
                "hidden_size": text_config.get("hidden_size"),
                "head_dim": text_config.get("head_dim"),
                "num_attention_heads": text_config.get("num_attention_heads"),
                "num_key_value_heads": text_config.get("num_key_value_heads"),
                "linear_num_key_heads": text_config.get("linear_num_key_heads"),
                "linear_num_value_heads": text_config.get("linear_num_value_heads"),
                "linear_key_head_dim": text_config.get("linear_key_head_dim"),
                "linear_value_head_dim": text_config.get("linear_value_head_dim"),
                "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
                "layer_types": layer_types,
                "max_position_embeddings": text_config.get("max_position_embeddings"),
                "rope_parameters": text_config.get("rope_parameters"),
            },
            "weight_bytes": index.get("metadata", {}).get("total_size"),
            "file_hashes": file_hashes,
        }
        encoded = json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        return cls(
            fingerprint=fingerprint,
            model_path=str(root),
            architecture=architecture,
            model_type=str(config.get("model_type") or ""),
            layer_count=layer_count,
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            max_position_embeddings=int(
                text_config.get("max_position_embeddings") or 0
            ),
            weight_bytes=int(index.get("metadata", {}).get("total_size") or 0),
            file_hashes=file_hashes,
        )

    def validate_qwen_exo_model(self) -> None:
        validate_qwen_exo_model_path(self.model_path)

    def public_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "model_path": self.model_path,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "layer_count": self.layer_count,
            "full_attention_layers": self.full_attention_layers,
            "linear_attention_layers": self.linear_attention_layers,
            "max_position_embeddings": self.max_position_embeddings,
            "weight_bytes": self.weight_bytes,
            "file_hashes": dict(self.file_hashes),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a checkpoint against QWEN-EXO's verified tensor layouts."
    )
    parser.add_argument("model_path", help="Local Hugging Face checkpoint directory")
    args = parser.parse_args(argv)
    try:
        variant = validate_qwen_exo_model_path(args.model_path)
    except ValueError as exc:
        parser.exit(2, f"QWEN-EXO startup blocked: {exc}\n")
    print(f"QWEN-EXO model accepted: {variant} ({Path(args.model_path).resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
