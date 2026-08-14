from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qwen_exo_booster.fingerprint import ModelIdentity, validate_qwen_exo_model_path

_MODEL_CATALOG_SCHEMA = 1
_DEFAULT_CATALOG_ROOTS = (Path("/models/catalog"),)
_DEFAULT_DATA_ROOT = Path("/data/qwen-exo")


class ModelCatalogError(ValueError):
    def __init__(
        self, code: str, message: str, *, model_fingerprint: str | None = None
    ):
        super().__init__(message)
        self.code = code
        self.model_fingerprint = model_fingerprint

    def public_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": str(self)}
        if self.model_fingerprint:
            payload["model_fingerprint"] = self.model_fingerprint
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _revision(active_model_fingerprint: str, updated_at: str) -> str:
    encoded = json.dumps(
        {
            "active_model_fingerprint": active_model_fingerprint,
            "updated_at": updated_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _argument_value(arguments: Iterable[str], option: str) -> str | None:
    argv = list(arguments)
    value = None
    prefix = option + "="
    for index, token in enumerate(argv):
        if token.startswith(prefix):
            value = token[len(prefix) :]
        elif token == option and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def _replace_argument(arguments: Iterable[str], option: str, value: str) -> list[str]:
    argv = list(arguments)
    cleaned: list[str] = []
    index = 0
    prefix = option + "="
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 2
            continue
        if token.startswith(prefix):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    cleaned.extend((option, value))
    return cleaned


def _state_directory_name(arguments: Iterable[str], model: dict[str, Any]) -> str:
    existing = Path(_argument_value(arguments, "--qwen-exo-state-dir") or "state").name
    runtime_quantization = model.get("runtime_quantization")
    has_topology = any(
        _argument_value(arguments, option) is not None
        for option in ("--tp-size", "--kv-cache-dtype")
    )
    if runtime_quantization is None and not has_topology:
        return existing
    tp_size = _argument_value(arguments, "--tp-size") or "1"
    quantization = str(
        runtime_quantization or _argument_value(arguments, "--quantization") or "auto"
    )
    kv_cache_dtype = _argument_value(arguments, "--kv-cache-dtype") or "auto"
    return f"state-cuda-tp{tp_size}-{quantization}-{kv_cache_dtype}"


def _count_markdown(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )


def _runtime_quantization(config: dict[str, Any], variant: str) -> str | None:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return None
    method = str(quantization.get("quant_method") or "").lower()
    if method == "fp8":
        return "fp8"
    if (
        method == "gptq"
        and quantization.get("bits") == 4
        and quantization.get("desc_act") is False
        and variant in {"dense-27b", "moe-122b-a10b"}
    ):
        return "gptq" if variant == "dense-27b" else "moe_wna16"
    raise ValueError(
        f"unsupported QWEN-EXO checkpoint quantization for {variant}: {quantization!r}"
    )


def _checkpoint_quantization(config: dict[str, Any]) -> dict[str, Any]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return {
            "checkpoint_quantization": None,
            "checkpoint_quantization_bits": None,
            "checkpoint_quantization_group_size": None,
            "checkpoint_quantization_exclusions": [],
        }
    dynamic = quantization.get("dynamic")
    exclusions = (
        sorted(
            pattern.removeprefix("-:")
            for pattern in dynamic
            if isinstance(pattern, str) and pattern.startswith("-:")
        )
        if isinstance(dynamic, dict)
        else []
    )
    return {
        "checkpoint_quantization": quantization.get("quant_method"),
        "checkpoint_quantization_bits": quantization.get("bits"),
        "checkpoint_quantization_group_size": quantization.get("group_size"),
        "checkpoint_quantization_exclusions": exclusions,
    }


class ModelCatalogStore:
    def __init__(
        self,
        catalog_roots: Iterable[Path | str],
        data_root: Path | str,
        path: Path | str | None = None,
    ):
        roots = tuple(Path(root).expanduser().resolve() for root in catalog_roots)
        if not roots:
            raise ModelCatalogError("catalog_roots_missing", "模型目录根路径不能为空")
        self.catalog_roots = roots
        self.data_root = Path(data_root).expanduser().resolve()
        self.profiles_root = self.data_root / "model-profiles"
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.data_root / "model-catalog.json"
        )

    @classmethod
    def from_environment(cls) -> ModelCatalogStore:
        raw_roots = os.getenv("QWEN_EXO_MODEL_CATALOG_ROOTS", "")
        roots = (
            tuple(Path(value) for value in raw_roots.split(os.pathsep) if value.strip())
            or _DEFAULT_CATALOG_ROOTS
        )
        catalog_path = Path(
            os.getenv(
                "QWEN_EXO_MODEL_CATALOG_CONFIG",
                str(_DEFAULT_DATA_ROOT / "model-catalog.json"),
            )
        )
        data_root = Path(
            os.getenv("QWEN_EXO_MODEL_DATA_ROOT", str(catalog_path.parent))
        )
        return cls(
            roots,
            data_root,
            path=catalog_path,
        )

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    document, stream, ensure_ascii=False, sort_keys=True, indent=2
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_document(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelCatalogError(
                "catalog_unreadable", f"无法读取模型目录配置：{exc}"
            ) from exc
        if payload.get("schema") != _MODEL_CATALOG_SCHEMA:
            raise ModelCatalogError(
                "catalog_schema_mismatch", "模型目录 schema 不受支持"
            )
        return payload

    def discover_models(self) -> list[dict[str, Any]]:
        discovered: dict[str, dict[str, Any]] = {}
        for root_index, root in enumerate(self.catalog_roots):
            if not root.is_dir():
                continue
            candidates = [root] if (root / "config.json").is_file() else []
            candidates.extend(
                path
                for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
                if path.is_dir() and (path / "config.json").is_file()
            )
            for candidate in candidates:
                try:
                    variant = validate_qwen_exo_model_path(candidate)
                    identity = ModelIdentity.from_path(candidate)
                    config = json.loads(
                        (candidate / "config.json").read_text(encoding="utf-8")
                    )
                    runtime_quantization = _runtime_quantization(config, variant)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                discovered.setdefault(
                    identity.fingerprint,
                    {
                        "model_fingerprint": identity.fingerprint,
                        "name": candidate.name,
                        "model_path": str(candidate),
                        "architecture": identity.architecture,
                        "model_type": identity.model_type,
                        "variant": variant,
                        "layer_count": identity.layer_count,
                        "full_attention_layers": identity.full_attention_layers,
                        "linear_attention_layers": identity.linear_attention_layers,
                        "max_position_embeddings": identity.max_position_embeddings,
                        "weight_bytes": identity.weight_bytes,
                        **_checkpoint_quantization(config),
                        "runtime_quantization": runtime_quantization,
                        "catalog_root_index": root_index,
                    },
                )
        return sorted(
            discovered.values(),
            key=lambda item: (str(item["variant"]), str(item["name"]).lower()),
        )

    def _find_model(
        self, model_fingerprint: str, models: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        for model in models or self.discover_models():
            if model["model_fingerprint"] == model_fingerprint:
                return model
        raise ModelCatalogError(
            "model_not_found",
            "模型目录中找不到指定模型，或模型结构不受支持",
            model_fingerprint=model_fingerprint,
        )

    def _profile_root(self, model_fingerprint: str) -> Path:
        return self.profiles_root / model_fingerprint

    def _profile_initialized(self, model_fingerprint: str) -> bool:
        return self._profile_root(model_fingerprint).is_dir()

    def _initialize_profile(self, model_fingerprint: str) -> Path:
        profile_root = self._profile_root(model_fingerprint)
        profile_root.mkdir(parents=True, exist_ok=True)
        return profile_root

    def ensure(self, default_model_path: Path | str | None = None) -> dict[str, Any]:
        models = self.discover_models()
        if not models:
            raise ModelCatalogError(
                "catalog_empty", "模型目录中没有结构兼容的 QWEN-EXO 模型"
            )
        document = self._read_document()
        if document is None:
            selected = None
            if default_model_path is not None:
                wanted = Path(default_model_path).expanduser().resolve()
                selected = next(
                    (
                        model
                        for model in models
                        if Path(str(model["model_path"])).resolve() == wanted
                    ),
                    None,
                )
            selected = selected or models[0]
            updated_at = _utc_now()
            fingerprint = str(selected["model_fingerprint"])
            document = {
                "schema": _MODEL_CATALOG_SCHEMA,
                "revision": _revision(fingerprint, updated_at),
                "active_model_fingerprint": fingerprint,
                "applied_model_fingerprint": None,
                "healthy_model_fingerprint": None,
                "previous_model_fingerprint": None,
                "legacy_model_fingerprint": fingerprint,
                "updated_at": updated_at,
                "applied_at": None,
                "healthy_at": None,
                "boot_attempts": 0,
                "last_failed_model_fingerprint": None,
                "last_rollback_at": None,
            }
            self._initialize_profile(fingerprint)
            self._write_document(document)
            return document

        active = str(document.get("active_model_fingerprint") or "")
        self._find_model(active, models)
        self._initialize_profile(active)
        return document

    def select(
        self,
        model_fingerprint: str,
        *,
        expected_revision: str,
    ) -> dict[str, Any]:
        models = self.discover_models()
        document = self.ensure()
        if expected_revision != document.get("revision"):
            raise ModelCatalogError(
                "revision_conflict", "模型目录已被其他操作更新，请刷新后重试"
            )
        target = self._find_model(model_fingerprint, models)
        target_fingerprint = str(target["model_fingerprint"])
        current_fingerprint = str(document["active_model_fingerprint"])
        if target_fingerprint == current_fingerprint:
            return document
        self._initialize_profile(target_fingerprint)
        updated_at = _utc_now()
        document.update(
            previous_model_fingerprint=current_fingerprint,
            active_model_fingerprint=target_fingerprint,
            revision=_revision(target_fingerprint, updated_at),
            updated_at=updated_at,
            boot_attempts=0,
        )
        self._write_document(document)
        return document

    def mark_applied(
        self, base_args: Iterable[str]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        argv = list(base_args)
        default_model_path = _argument_value(argv, "--model-path")
        document = self.ensure(default_model_path)
        models = self.discover_models()
        active = str(document["active_model_fingerprint"])
        if (
            document.get("healthy_model_fingerprint") != active
            and int(document.get("boot_attempts", 0)) >= 1
            and document.get("previous_model_fingerprint")
        ):
            document["last_failed_model_fingerprint"] = active
            document["last_rollback_at"] = _utc_now()
            active = str(document.pop("previous_model_fingerprint"))
            document["active_model_fingerprint"] = active
            updated_at = _utc_now()
            document["updated_at"] = updated_at
            document["revision"] = _revision(active, updated_at)

        model = self._find_model(active, models)
        profile_root = self._initialize_profile(active)
        if document.get("healthy_model_fingerprint") == active:
            document["boot_attempts"] = 0
        else:
            document["boot_attempts"] = int(document.get("boot_attempts", 0)) + 1
        document["applied_model_fingerprint"] = active
        document["applied_at"] = _utc_now()
        self._write_document(document)

        rewritten = _replace_argument(argv, "--model-path", str(model["model_path"]))
        if model.get("runtime_quantization"):
            rewritten = _replace_argument(
                rewritten,
                "--quantization",
                str(model["runtime_quantization"]),
            )
        state_name = _state_directory_name(argv, model)
        rewritten = _replace_argument(
            rewritten, "--qwen-exo-state-dir", str(profile_root / state_name)
        )
        return document, rewritten, model

    def mark_healthy(self, model_fingerprint: str) -> bool:
        document = self._read_document()
        if document is None:
            return False
        if (
            document.get("active_model_fingerprint") != model_fingerprint
            or document.get("applied_model_fingerprint") != model_fingerprint
        ):
            return False
        document["healthy_model_fingerprint"] = model_fingerprint
        document["healthy_at"] = _utc_now()
        document["boot_attempts"] = 0
        document["previous_model_fingerprint"] = None
        if document.get("last_failed_model_fingerprint") == model_fingerprint:
            document["last_failed_model_fingerprint"] = None
            document["last_rollback_at"] = None
        self._write_document(document)
        return True

    def public_document(
        self, *, running_model_fingerprint: str | None = None
    ) -> dict[str, Any]:
        document = self.ensure()
        models = self.discover_models()
        public_models = []
        for model in models:
            fingerprint = str(model["model_fingerprint"])
            profile_root = self._profile_root(fingerprint)
            state_directories = (
                [
                    path
                    for path in profile_root.iterdir()
                    if path.is_dir() and path.name.startswith("state")
                ]
                if profile_root.is_dir()
                else []
            )
            public_models.append(
                {
                    **model,
                    "active": fingerprint == document.get("active_model_fingerprint"),
                    "running": fingerprint == running_model_fingerprint,
                    "profile_initialized": self._profile_initialized(fingerprint),
                    "profile_root": str(profile_root),
                    "knowledge_document_count": _count_markdown(
                        self.data_root / "knowledge"
                    ),
                    "policy_document_count": _count_markdown(
                        self.data_root / "policydata"
                    ),
                    "cognition_document_count": _count_markdown(
                        self.data_root / "cognition"
                    ),
                    "native_bank_ready": any(
                        (state / "model-native").is_dir() for state in state_directories
                    ),
                }
            )
        return {
            **document,
            "models": public_models,
            "catalog_roots": [str(root) for root in self.catalog_roots],
            "profiles_root": str(self.profiles_root),
            "source_root": str(self.data_root),
            "sources_shared": True,
            "running_model_fingerprint": running_model_fingerprint,
            "managed_restart": os.getenv("QWEN_EXO_MANAGED_RESTART", "0") == "1",
        }
