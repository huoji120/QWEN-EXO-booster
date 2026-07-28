from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_NAME = "QWEN-EXO-booster"
PROJECT_API_VERSION = "1"
_OBSERVER_MODES = frozenset({"off", "shadow", "active"})


@dataclass(frozen=True, slots=True)
class QwenExoFeatureFlags:
    hybrid_prefix: bool
    external_memory: bool
    reference_judge: bool
    capsule: bool
    observer: bool
    adaptive_refresh: bool


@dataclass(frozen=True, slots=True)
class QwenExoConfig:
    state_directory: Path
    knowledge_directory: Path
    max_internal_fanout: int
    max_internal_tokens: int
    observer_mode: str
    feature_flags: QwenExoFeatureFlags
    model_path: str
    tp_size: int

    def __post_init__(self) -> None:
        if self.max_internal_fanout < 1:
            raise ValueError("qwen_exo_max_internal_fanout must be positive")
        if self.max_internal_tokens < 1:
            raise ValueError("qwen_exo_max_internal_tokens must be positive")
        if self.observer_mode not in _OBSERVER_MODES:
            raise ValueError(
                f"qwen_exo_observer_mode must be one of {sorted(_OBSERVER_MODES)}"
            )
        if self.tp_size < 1:
            raise ValueError("tp_size must be positive")
        if self.feature_flags.adaptive_refresh and self.observer_mode != "active":
            raise ValueError(
                "Adaptive refresh requires qwen_exo_observer_mode=active"
            )
        if self.feature_flags.observer != (self.observer_mode != "off"):
            raise ValueError("Observer feature flag and observer mode disagree")

    @classmethod
    def from_server_args(cls, server_args: Any) -> QwenExoConfig:
        observer_mode = str(server_args.qwen_exo_observer_mode)
        return cls(
            state_directory=Path(server_args.qwen_exo_state_dir).expanduser(),
            knowledge_directory=Path(
                server_args.qwen_exo_knowledge_dir
            ).expanduser(),
            max_internal_fanout=int(server_args.qwen_exo_max_internal_fanout),
            max_internal_tokens=int(server_args.qwen_exo_max_internal_tokens),
            observer_mode=observer_mode,
            feature_flags=QwenExoFeatureFlags(
                hybrid_prefix=bool(server_args.qwen_exo_enable_hybrid_prefix),
                external_memory=bool(server_args.qwen_exo_enable_external_memory),
                reference_judge=bool(server_args.qwen_exo_enable_reference_judge),
                capsule=bool(server_args.qwen_exo_enable_capsule),
                observer=observer_mode != "off",
                adaptive_refresh=bool(server_args.qwen_exo_enable_adaptive_refresh),
            ),
            model_path=str(server_args.model_path),
            tp_size=int(server_args.tp_size),
        )

    def public_dict(self) -> dict[str, Any]:
        flags = asdict(self.feature_flags)
        return {
            "project": PROJECT_NAME,
            "api_version": PROJECT_API_VERSION,
            "state_directory": str(self.state_directory),
            "knowledge_directory": str(self.knowledge_directory),
            "max_internal_fanout": self.max_internal_fanout,
            "max_internal_tokens": self.max_internal_tokens,
            "observer_mode": self.observer_mode,
            "features": flags,
            "model_path": self.model_path,
            "tp_size": self.tp_size,
        }
