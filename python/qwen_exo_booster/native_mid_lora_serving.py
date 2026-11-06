"""Startup-only native shared-expert LoRA; never imports the training model.

The schema-1 ``qwen_exo_native_mid_lora`` config field contains exactly:
``schema``, ``name``, ``adapter_path``, ``adapter_sha256``, ``base_profile``,
``base_config_sha256``, ``base_index_sha256``, and ``base_identity`` (the original
NativePLEIdentity dictionary). Paths are absolute. The candidate's config must
be a separate file equal to the original config after removing only this field;
all indexed checkpoint shards must reference the original files via hardlinks.

Call install after the MAIN Qwen4Exp weight loader has finished, before warmup,
quantization postprocessing and graph capture. Its forward hooks preserve the
original projection modules/parameter names and run before the shared gate.
The caller must reject subsequent model weight updates before they write any
weights, and reject dynamic LoRA APIs. A process restart is the only cutover.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from qwen_exo_booster.fingerprint import _external_ple_manifest, _file_sha256
from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity

_FIELD = "qwen_exo_native_mid_lora"
_MARKER = "_qwen_exo_native_mid_lora"
_BUFFER_A = "_qwen_exo_native_mid_lora_A"
_BUFFER_B = "_qwen_exo_native_mid_lora_B"
_SCHEMA = "native-mid-lora-shared-down-v2-fp32"
_REFERENCE = "frozen NVFP4 decoded W4A16, not serving W4A4"
_MANIFEST_KEYS = frozenset({"schema", "name", "adapter_path", "adapter_sha256",
    "base_profile", "base_config_sha256", "base_index_sha256", "base_identity"})
_CONFIG_KEYS = frozenset({"first_layer", "last_layer", "rank", "alpha",
    "gradient_checkpointing", "checkpoint_group_size", "offload_checkpoints"})
_TARGETS = tuple(f"model.layers.{i}.mlp.shared_expert.down_proj" for i in range(16, 32))
_MAX_ADAPTER_BYTES = 64 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"native_mid_lora_duplicate_json_key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"native_mid_lora_nonfinite_json: {value}")


def _json(value):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_bad_constant)


def _same_json(left, right):
    # JSON true is not the number 1, despite Python container equality.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"native_mid_lora_invalid_sha256: {label}")
    return value


def _absolute(value, label):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"native_mid_lora_absolute_path_required: {label}")
    return Path(value).resolve(strict=True)


def _inside(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("native_mid_lora_invalid_checkpoint_shard")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("native_mid_lora_checkpoint_shard_escapes_profile")
    return path


@dataclass(frozen=True, slots=True)
class NativeMidLoraManifest:
    name: str
    adapter_path: Path
    adapter_sha256: str
    profile: Path
    profile_config_sha256: str
    base_profile: Path
    base_config_sha256: str
    base_index_sha256: str
    base_identity_fingerprint: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    _base_config_json: str = field(repr=False)
    _tensors: Mapping[str, torch.Tensor] = field(repr=False, compare=False)

    def report(self) -> dict[str, Any]:
        """Identity evidence only; installation does not claim a forward occurred."""
        return {"name": self.name, "adapter_sha256": self.adapter_sha256,
            "target_count": len(self.targets), "targets": list(self.targets),
            "base_identity_fingerprint": self.base_identity_fingerprint,
            "profile_config_sha256": self.profile_config_sha256,
            "startup_only": True}


def _adapter(path, expected_hash, identity, text):
    # Hash and parse the SAME bounded byte snapshot, not two opens of a mutable
    # pathname. No safetensors mapping survives startup validation.
    with path.open("rb") as source:
        payload = source.read(_MAX_ADAPTER_BYTES + 1)
    if len(payload) > _MAX_ADAPTER_BYTES or len(payload) < 8:
        raise ValueError("native_mid_lora_adapter_file_size_invalid")
    if _digest(payload) != expected_hash:
        raise ValueError("native_mid_lora_adapter_sha256_mismatch")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if not 0 < header_size <= min(len(payload) - 8, 1024 * 1024):
        raise ValueError("native_mid_lora_adapter_header_invalid")
    header = _json(payload[8:8 + header_size])
    if not isinstance(header, dict):
        raise ValueError("native_mid_lora_adapter_header_invalid")
    strings = header.get("__metadata__")
    if (not isinstance(strings, dict) or set(strings) != {"native_mid_lora"}
            or not isinstance(strings["native_mid_lora"], str)):
        raise ValueError("native_mid_lora_adapter_metadata_missing")
    metadata = _json(strings["native_mid_lora"])
    if (not isinstance(metadata, dict) or set(metadata) != {"schema", "identity", "config",
            "targets", "shapes", "adapter_dtype", "weight_reference"}
            or metadata["schema"] != _SCHEMA or not _same_json(metadata["identity"], identity)
            or metadata["targets"] != list(_TARGETS)
            or metadata["adapter_dtype"] != "torch.float32"
            or metadata["weight_reference"] != _REFERENCE):
        raise ValueError("native_mid_lora_adapter_metadata_mismatch")
    config = metadata["config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("native_mid_lora_adapter_config_invalid")
    for key, expected in (("first_layer", 16), ("last_layer", 31), ("rank", 8)):
        if type(config[key]) is not int or config[key] != expected:
            raise ValueError(f"native_mid_lora_unsupported_adapter_config: {key}")
    if (type(config["alpha"]) not in (int, float) or not math.isfinite(config["alpha"])
            or config["alpha"] != 16.0
            or type(config["gradient_checkpointing"]) is not bool
            or type(config["offload_checkpoints"]) is not bool
            or type(config["checkpoint_group_size"]) is not int
            or not 1 <= config["checkpoint_group_size"] <= text["num_hidden_layers"]):
        raise ValueError("native_mid_lora_adapter_config_invalid")
    shapes = {name + suffix: shape for name in _TARGETS for suffix, shape in
        ((".lora_A", [8, text["shared_expert_intermediate_size"]]),
         (".lora_B", [text["hidden_size"], 8]))}
    if metadata["shapes"] != shapes or set(header) != set(shapes) | {"__metadata__"}:
        raise ValueError("native_mid_lora_adapter_tensor_set_mismatch")
    for name, shape in shapes.items():
        entry = header[name]
        if (not isinstance(entry, dict) or entry.get("dtype") != "F32"
                or entry.get("shape") != shape):
            raise ValueError(f"native_mid_lora_adapter_tensor_header_invalid: {name}")
    from safetensors.torch import load
    tensors = load(payload)
    if set(tensors) != set(shapes):
        raise ValueError("native_mid_lora_adapter_tensor_set_mismatch")
    for name, tensor in tensors.items():
        if (list(tensor.shape) != shapes[name] or tensor.dtype != torch.float32
                or tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"native_mid_lora_adapter_tensor_invalid: {name}")
    return MappingProxyType(tensors)


def load_manifest(profile: str | Path) -> NativeMidLoraManifest | None:
    """Validate the original profile and exact adapter entirely on CPU.

    Absence is the sole disabled state; a present null/false/malformed field is
    rejected. There is deliberately no metadata-only bypass of tensor checks.
    """
    root = Path(profile).resolve(strict=True)
    config_bytes = (root / "config.json").read_bytes()
    config = _json(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("native_mid_lora_model_config_invalid")
    if _FIELD not in config:
        return None
    declared = config[_FIELD]
    if (not isinstance(declared, dict) or set(declared) != _MANIFEST_KEYS
            or type(declared.get("schema")) is not int or declared["schema"] != 1):
        raise ValueError("native_mid_lora_manifest_schema_invalid")
    name = declared["name"]
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
        raise ValueError("native_mid_lora_manifest_name_invalid")
    base = _absolute(declared["base_profile"], "base_profile")
    adapter_path = _absolute(declared["adapter_path"], "adapter_path")
    adapter_hash = _hash(declared["adapter_sha256"], "adapter_sha256")
    config_hash = _hash(declared["base_config_sha256"], "base_config_sha256")
    index_hash = _hash(declared["base_index_sha256"], "base_index_sha256")
    if root == base or (root / "config.json").samefile(base / "config.json"):
        raise ValueError("native_mid_lora_candidate_requires_separate_config")
    original_bytes = (base / "config.json").read_bytes()
    if _digest(original_bytes) != config_hash:
        raise ValueError("native_mid_lora_base_config_sha256_mismatch")
    original = _json(original_bytes)
    if (not isinstance(original, dict) or _FIELD in original
            or not _same_json({key: value for key, value in config.items() if key != _FIELD}, original)):
        raise ValueError("native_mid_lora_candidate_changed_base_config")
    if (original.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
            or original.get("model_type") != "qwen4_exp"
            or "qwen_exo_runtime_lora" in original):
        raise ValueError("native_mid_lora_base_model_unsupported")
    text = original.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("native_mid_lora_base_text_config_invalid")
    for key, expected in (("hidden_size", 2560), ("shared_expert_intermediate_size", 640),
            ("num_hidden_layers", 48), ("hc_count", 4)):
        if type(text.get(key)) is not int or text[key] != expected:
            raise ValueError(f"native_mid_lora_base_geometry_unsupported: {key}")
    index_bytes = (base / "model.safetensors.index.json").read_bytes()
    if (_digest(index_bytes) != index_hash
            or _file_sha256(root / "model.safetensors.index.json") != index_hash):
        raise ValueError("native_mid_lora_base_index_sha256_mismatch")
    index = _json(index_bytes)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("native_mid_lora_base_weight_index_invalid")
    for shard in set(weight_map.values()):
        if not _inside(root, shard).samefile(_inside(base, shard)):
            raise ValueError("native_mid_lora_candidate_requires_original_weight_shards")
    for target_name in _TARGETS:
        if "model.language_model." + target_name[len("model."):] + ".weight" not in weight_map:
            raise ValueError(f"native_mid_lora_base_target_missing: {target_name}")
    # The external PLE validator checks its hash-bound manifest, absolute source
    # root, shard provenance/size and geometry without hashing the giant table.
    _external_ple_manifest(base, original)
    _external_ple_manifest(root, config)
    base_identity = NativePLEIdentity.from_profile(base)
    expected_identity = base_identity.to_dict()
    if (not _same_json(declared["base_identity"], expected_identity)
            or expected_identity["model_config_sha256"] != config_hash
            or expected_identity["index_sha256"] != index_hash):
        raise ValueError("native_mid_lora_original_identity_mismatch")
    candidate_identity = NativePLEIdentity.from_profile(root).to_dict()
    if candidate_identity["model_config_sha256"] != _digest(config_bytes):
        raise ValueError("native_mid_lora_candidate_config_changed_during_load")
    candidate_identity["model_config_sha256"] = config_hash
    if not _same_json(candidate_identity, expected_identity):
        raise ValueError("native_mid_lora_candidate_native_ple_identity_mismatch")
    tensors = _adapter(adapter_path, adapter_hash, expected_identity, text)
    return NativeMidLoraManifest(name, adapter_path, adapter_hash, root,
        _digest(config_bytes), base, config_hash, index_hash, base_identity.fingerprint(),
        _TARGETS, 8, 16.0, json.dumps(original, sort_keys=True, allow_nan=False), tensors)


def _residual_hook(scale, target, adapter_name, adapter_sha256):
    observed = False
    def apply(module, args, kwargs, output):
        nonlocal observed
        hidden = args[0] if args else kwargs["input_"]
        # Captured execution uses tensor metadata and fixed resident buffers.
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("native_mid_lora_projection_output_contract_changed")
        result, bias = output
        a, b = getattr(module, _BUFFER_A), getattr(module, _BUFFER_B)
        if (hidden.dtype != torch.bfloat16 or result.dtype != torch.bfloat16
                or a.dtype != torch.float32 or b.dtype != torch.float32
                or hidden.device != a.device or result.device != a.device
                or hidden.shape[-1] != a.shape[1] or result.shape[-1] != b.shape[0]):
            raise RuntimeError("native_mid_lora_projection_activation_contract_changed")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = F.linear(F.linear(hidden.to(torch.float32), a), b)
        if not observed and delta.numel() and (
                delta.device.type != "cuda" or not torch.cuda.is_current_stream_capturing()):
            # Exactly one eager observation per target; never item()/logging
            # during capture. A legitimate zero-input application may report 0.
            delta_l1 = float(delta.detach().abs().sum().item())
            _LOGGER.info("Native mid LoRA first eager application name=%s sha256=%s "
                "target=%s fp32_delta_l1=%s", adapter_name, adapter_sha256, target, delta_l1)
            observed = True
        # Match the training reference's cast-before-scale boundary exactly.
        return result + delta.to(result.dtype) * scale, bias
    return apply


def install(model, profile: str | Path, tp_size: int, pp_size: int) -> dict[str, Any] | None:
    """Attach once, transactionally; return identity, never execution evidence.

    All CPU validation and every device copy precede registration. No module or
    parameter is replaced, so quantization postprocessing retains ownership of
    its original weights. Any attachment failure removes all hooks/buffers.
    """
    if hasattr(model, _MARKER):
        raise RuntimeError("native_mid_lora_already_installed_restart_required")
    manifest = load_manifest(profile)
    if manifest is None:
        return None
    if type(tp_size) is not int or type(pp_size) is not int or (tp_size, pp_size) != (1, 1):
        raise ValueError("native_mid_lora_requires_tp1_pp1")
    if (type(model).__name__ != "Qwen4ExpForConditionalGeneration"
            or type(model).__module__ != "sglang.srt.models.qwen4_exp"
            or getattr(model, "is_nextn", False) or getattr(model, "is_draft_worker", False)):
        raise ValueError("native_mid_lora_requires_main_qwen4_model_not_draft")
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", ())
    if (len(layers) != 48 or getattr(core, "start_layer", None) != 0
            or getattr(core, "end_layer", None) != 48):
        raise ValueError("native_mid_lora_requires_complete_main_layer_range")
    original = _json(manifest._base_config_json)["text_config"]
    model_config = getattr(model, "config", None)
    for key in ("hidden_size", "shared_expert_intermediate_size", "num_hidden_layers", "hc_count"):
        if getattr(model_config, key, None) != original[key]:
            raise ValueError(f"native_mid_lora_runtime_geometry_mismatch: {key}")
    staged = []
    seen = set()
    for name in manifest.targets:
        layer_id = int(name.split(".")[2])
        layer = layers[layer_id]
        mlp = getattr(layer, "mlp", None)
        if (getattr(layer, "is_nextn", False) or getattr(mlp, "is_nextn", False)
                or getattr(mlp, "num_fused_shared_experts", 0) != 0
                or getattr(mlp, "enable_shared_expert_fusion", False)
                or getattr(mlp, "tp_size", None) != 1):
            raise ValueError(f"native_mid_lora_fused_sharded_or_draft_target: {name}")
        shared = getattr(mlp, "shared_expert", None)
        projection = getattr(shared, "down_proj", None)
        if (projection is None or type(projection).__name__ != "RowParallelLinear"
                or type(projection).__module__ != "sglang.srt.layers.linear"
                or getattr(projection, "tp_size", None) != 1
                or getattr(projection, "tp_rank", None) != 0
                or getattr(projection, "input_size", None) != 640
                or getattr(projection, "input_size_per_partition", None) != 640
                or getattr(projection, "output_size", None) != 2560
                or id(projection) in seen):
            raise ValueError(f"native_mid_lora_projection_geometry_invalid: {name}")
        if any(hasattr(projection, key) for key in (_BUFFER_A, _BUFFER_B, _MARKER)):
            raise RuntimeError(f"native_mid_lora_projection_already_adapted: {name}")
        weight = getattr(projection, "weight", None)
        if (not isinstance(weight, torch.Tensor) or weight.device.type not in {"cpu", "cuda"}
                or tuple(weight.shape) != (2560, 640) or weight.dtype != torch.bfloat16):
            raise ValueError(f"native_mid_lora_projection_weight_invalid: {name}")
        seen.add(id(projection))
        a = manifest._tensors[name + ".lora_A"].to(device=weight.device, copy=True)
        b = manifest._tensors[name + ".lora_B"].to(device=weight.device, copy=True)
        staged.append((name, projection, a, b))
    handles, attached = [], []
    identity = MappingProxyType(manifest.report())
    try:
        for name, projection, a, b in staged:
            attached.append(projection)
            projection.register_buffer(_BUFFER_A, a, persistent=False)
            projection.register_buffer(_BUFFER_B, b, persistent=False)
            handles.append(projection.register_forward_hook(
                _residual_hook(manifest.alpha / manifest.rank, name,
                    manifest.name, manifest.adapter_sha256), with_kwargs=True))
            setattr(projection, _MARKER, identity)
        setattr(model, _MARKER, identity)
    except BaseException:
        for handle in handles:
            handle.remove()
        for projection in attached:
            for key in (_BUFFER_A, _BUFFER_B, _MARKER):
                if hasattr(projection, key):
                    delattr(projection, key)
        if hasattr(model, _MARKER):
            delattr(model, _MARKER)
        raise
    return manifest.report()
