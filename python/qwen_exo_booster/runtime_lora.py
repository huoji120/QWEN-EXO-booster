from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qwen_exo_booster.fingerprint import _file_sha256


@dataclass(frozen=True)
class RuntimeLoRA:
    """One immutable adapter bound to a catalog model's config fingerprint."""

    name: str
    path: Path

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> RuntimeLoRA | None:
        root = Path(model_path).expanduser().resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        manifest = config.get("qwen_exo_runtime_lora")
        if manifest is None:
            return None
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            raise ValueError("Invalid QWEN-EXO runtime LoRA manifest")
        name, relative = manifest.get("name"), manifest.get("path")
        if not isinstance(name, str) or not name or any(x in name for x in (":", "=", "/")):
            raise ValueError("Invalid runtime LoRA name")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Runtime LoRA path is required")
        path = (root / relative).resolve()
        hashes = manifest.get("file_hashes")
        if not isinstance(hashes, dict) or set(hashes) not in (
            {"adapter_config.json", "adapter_model.bin"},
            {"adapter_config.json", "adapter_model.safetensors"},
        ):
            raise ValueError("Runtime LoRA requires exact config and weight hashes")
        for filename, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64 or _file_sha256(path / filename) != expected:
                raise ValueError(f"Runtime LoRA file changed: {filename}")
        return cls(name, path)

    def bind_request(self, requested: str | list[str | None] | None):
        """Bind before batch normalization; all internal and external jobs agree."""
        if isinstance(requested, list):
            return [self.bind_request(value) for value in requested]
        if requested is None or requested == self.name:
            return self.name
        raise ValueError("This model profile serves only its fixed runtime LoRA")

    def validate_loaded(self, enabled: bool, adapters) -> None:
        refs = tuple(adapters or ())
        if not enabled or len(refs) != 1:
            raise ValueError("Runtime LoRA profile requires exactly one enabled adapter")
        ref = refs[0]
        if ref.lora_name != self.name or Path(ref.lora_path).resolve() != self.path:
            raise ValueError("Loaded LoRA does not match the model profile")
