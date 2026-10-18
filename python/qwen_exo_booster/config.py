from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from qwen_exo_booster.score_bias import SCORE_BIAS_KERNEL_MAX_BLOCKS
from qwen_exo_booster.hybrid_state import (
    qwen_exo_topology_key,
    resolve_qwen_exo_backend,
)

PROJECT_NAME = "QWEN-EXO-booster"
PROJECT_API_VERSION = "1"
_OBSERVER_MODES = frozenset({"off", "shadow", "active"})
_REFLECTION_MEMORY_MODES = frozenset({"off", "active"})
_RESPONSE_COMPACTION_MODES = frozenset({"off", "active"})


_SCORE_BIAS_MODES = frozenset({"off", "trajectory_shadow", "trajectory_active"})
LATENT_TRANSPLANT_MAX_STRENGTH = 0.5



@dataclass(frozen=True, slots=True)
class QwenExoFeatureFlags:
    hybrid_prefix: bool
    capsule: bool
    observer: bool
    policy_data: bool = True
    score_bias: bool = False
    activation_training: bool = False


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
    backend: str = "cuda"
    dtype: str = "auto"
    quantization: str = "none"
    kv_cache_dtype: str = "auto"
    max_running_requests: int = 10
    context_length: int = 131072
    internal_mamba_reserve: int = 6
    policy_data_directory: Path | None = None
    cognition_directory: Path | None = None
    telemetry_include_text: bool = False
    telemetry_text_mode: str = "off"
    reflection_memory_mode: str = "off"
    reflection_memory_idle_seconds: float = 600.0
    reflection_memory_min_events: int = 3
    reflection_memory_min_tokens: int = 256
    reflection_memory_max_attempts: int = 3
    reflection_memory_max_output_tokens: int = 4096
    reflection_memory_max_history_tokens: int = 92160
    response_compaction_mode: str = "off"
    response_compaction_max_history_tokens: int = 8192
    response_compaction_max_dropped_items: int = 16
    response_compaction_max_output_tokens: int = 2048
    score_bias_mode: str = "off"
    score_bias_min_relevance: float = 0.01

    score_bias_min_surprisal: float = 0.8
    score_bias_max: float = 0.05
    score_bias_half_life_steps: float = 4.0
    score_bias_max_blocks: int = 8
    score_bias_min_age_steps: int = 2
    score_bias_max_age_steps: int = 16
    score_bias_tail_tokens: int = 4096
    score_bias_tail_ratio: float = 0.15
    score_bias_selected_blocks: int = 2
    score_bias_query_window: int = 8
    score_bias_relevance_margin: float = 0.005
    score_bias_anchor_bias: float = 0.01
    score_bias_anchor_max_blocks: int = 2
    latent_transplant_enabled: bool = False
    latent_transplant_strength: float = 0.05
    activation_editor_enabled: bool = False
    activation_editor_strength: float = 2.0
    max_output_tokens: int = 8192
    # Fallback for omitted/numeric effort and internal generation, not a tier ceiling.
    max_reasoning_tokens: int = 3072
    reasoning_budget_min: int = 512
    reasoning_budget_low: int = 1024
    reasoning_budget_medium: int = 2048
    reasoning_budget_high: int = 4090
    reasoning_budget_xhigh: int = 8192
    reasoning_mode: str = "fixed"
    force_auto_reasoning: bool = True
    reasoning_session_once: bool = True
    reasoning_dynamic_context_tokens: int = 2048
    observer_surprisal_threshold: float = 0.8
    observer_surprisal_window: int = 8
    observer_surprisal_margin: float = 0.2
    observer_surprisal_sustain: int = 1
    observer_q_drift_threshold: float = 0.35
    observer_cooldown_tokens: int = 64
    observer_max_triggers: int = 1
    observer_q_pre_tokens: int = 8
    observer_q_post_tokens: int = 4
    observer_recovery_tokens: int = 8
    immediate_uncertainty_retrieval: bool = False
    # When think generation hits its resolved reasoning budget and is truncated, inject
    # this many short, concrete self-check questions before the stop-and-go line
    # so the model reconsiders before committing. ``0`` disables the questions and
    # keeps the plain reasoning-budget cutoff.
    reasoning_cutoff_question_count: int = 3
    knowledge_retrieval_mode: Literal["file"] = "file"

    def __post_init__(self) -> None:
        if self.policy_data_directory is None:
            object.__setattr__(
                self,
                "policy_data_directory",
                self.knowledge_directory.parent / "policydata",
            )
        if self.cognition_directory is None:
            object.__setattr__(
                self,
                "cognition_directory",
                self.knowledge_directory.parent / "cognition",
            )
        if self.max_internal_fanout < 1:
            raise ValueError("qwen_exo_max_internal_fanout must be positive")
        if self.internal_mamba_reserve < 0:
            raise ValueError("qwen_exo_internal_mamba_reserve must be non-negative")
        if self.max_internal_tokens < 1:
            raise ValueError("qwen_exo_max_internal_tokens must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("qwen_exo_max_output_tokens must be positive")
        if self.max_reasoning_tokens < 1:
            raise ValueError("qwen_exo_max_reasoning_tokens must be positive")
        if self.reasoning_mode not in ("fixed", "dynamic"):
            raise ValueError("qwen_exo_reasoning_mode must be fixed or dynamic")
        if type(self.force_auto_reasoning) is not bool:
            raise ValueError("qwen_exo_force_auto_reasoning must be a boolean")
        if type(self.reasoning_session_once) is not bool:
            raise ValueError("qwen_exo_reasoning_session_once must be a boolean")
        if (
            type(self.reasoning_dynamic_context_tokens) is not int
            or self.reasoning_dynamic_context_tokens < 256
        ):
            raise ValueError(
                "qwen_exo_reasoning_dynamic_context_tokens must be an integer >= 256"
            )
        for effort in ("min", "low", "medium", "high", "xhigh"):
            budget = getattr(self, f"reasoning_budget_{effort}")
            if type(budget) is not int or budget < 1:
                raise ValueError(
                    f"qwen_exo_reasoning_budget_{effort} must be a positive integer"
                )
        if self.observer_mode not in _OBSERVER_MODES:
            raise ValueError(
                f"qwen_exo_observer_mode must be one of {sorted(_OBSERVER_MODES)}"
            )
        if self.tp_size < 1:
            raise ValueError("tp_size must be positive")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be positive")
        if self.reflection_memory_mode not in _REFLECTION_MEMORY_MODES:
            raise ValueError(
                "qwen_exo_reflection_memory_mode must be one of "
                f"{sorted(_REFLECTION_MEMORY_MODES)}"
            )
        if self.reflection_memory_idle_seconds < 60:
            raise ValueError("Reflection memory idle time must be at least 60 seconds")
        if self.reflection_memory_min_events < 2:
            raise ValueError("Reflection memory requires at least two tool events")
        if self.reflection_memory_min_tokens < 0:
            raise ValueError("Reflection memory minimum tokens cannot be negative")
        if not 1 <= self.reflection_memory_max_attempts <= 3:
            raise ValueError("Reflection memory attempts must be between 1 and 3")
        if not 512 <= self.reflection_memory_max_output_tokens <= 8192:
            raise ValueError(
                "Reflection memory output budget must be between 512 and 8192"
            )
        if not 1024 <= self.reflection_memory_max_history_tokens <= 96256:
            raise ValueError(
                "Reflection memory history budget must be between 1024 and 96256"
            )
        if (
            self.reflection_memory_mode == "active"
            and self.reflection_memory_max_output_tokens
            * self.reflection_memory_max_attempts
            > self.max_internal_tokens
        ):
            raise ValueError(
                "Reflection memory retry output budget cannot exceed internal job token budget"
            )

        if self.response_compaction_mode not in _RESPONSE_COMPACTION_MODES:
            raise ValueError(
                "qwen_exo_response_compaction_mode must be one of "
                f"{sorted(_RESPONSE_COMPACTION_MODES)}"
            )
        if (
            self.response_compaction_mode == "active"
            and self.response_compaction_max_output_tokens > self.max_internal_tokens
        ):
            raise ValueError(
                "Response compaction output budget cannot exceed internal job token budget"
            )
        if self.response_compaction_max_history_tokens < 1024:
            raise ValueError("Response compaction history budget must be at least 1024")
        if self.response_compaction_max_dropped_items < 0:
            raise ValueError(
                "Response compaction dropped-item limit cannot be negative"
            )
        if self.response_compaction_max_output_tokens < 256:
            raise ValueError("Response compaction output budget must be at least 256")

        if self.reasoning_cutoff_question_count < 0:
            raise ValueError(
                "qwen_exo_reasoning_cutoff_question_count must be non-negative"
            )

        if self.score_bias_mode not in _SCORE_BIAS_MODES:
            raise ValueError(
                "qwen_exo_score_bias_mode must be one of "
                f"{sorted(_SCORE_BIAS_MODES)}"
            )
        if self.feature_flags.score_bias != (self.score_bias_mode != "off"):
            raise ValueError("Score Bias feature flag and mode disagree")
        if self.feature_flags.observer != (self.observer_mode != "off"):
            raise ValueError("Observer feature flag and observer mode disagree")
        if self.immediate_uncertainty_retrieval and self.observer_mode != "active":
            raise ValueError(
                "Immediate uncertainty retrieval requires qwen_exo_observer_mode=active"
            )
        if (
            self.observer_surprisal_threshold < 0
            or self.observer_surprisal_window < 2
            or self.observer_surprisal_margin < 0
            or self.observer_surprisal_sustain < 1
            or self.observer_q_drift_threshold < 0
        ):
            raise ValueError("Observer thresholds and windows are invalid")
        if (
            self.observer_cooldown_tokens < 1
            or self.observer_max_triggers not in {0, 1}
            or self.observer_q_pre_tokens < 1
            or self.observer_q_post_tokens < 1
            or self.observer_recovery_tokens < 1
        ):
            raise ValueError("Observer trigger limits are invalid")

        if self.score_bias_mode != "off" and self.observer_mode == "off":
            raise ValueError("Score Bias requires an active or shadow observer")
        if self.score_bias_min_surprisal < 0 or self.score_bias_max < 0:
            raise ValueError("Score Bias thresholds and cap must be non-negative")
        if (
            self.score_bias_half_life_steps <= 0
            or self.score_bias_max_blocks < 1
            or self.score_bias_max_blocks > SCORE_BIAS_KERNEL_MAX_BLOCKS
            or self.score_bias_min_age_steps < 1
            or self.score_bias_max_age_steps < self.score_bias_min_age_steps
            or self.score_bias_tail_tokens < 0
            or not 0 <= self.score_bias_tail_ratio < 1
            or self.score_bias_selected_blocks < 1
            or self.score_bias_selected_blocks > self.score_bias_max_blocks
            or self.score_bias_query_window < 1
            or not -1 <= self.score_bias_min_relevance <= 1
            or self.score_bias_relevance_margin < 0
            or not 0 <= self.score_bias_anchor_bias <= self.score_bias_max
            or self.score_bias_anchor_max_blocks < 1
            or self.score_bias_anchor_max_blocks > self.score_bias_max_blocks
        ):
            raise ValueError("Score Bias trajectory selection limits are invalid")
        if self.latent_transplant_enabled and not (
            0 < self.latent_transplant_strength <= LATENT_TRANSPLANT_MAX_STRENGTH
        ):
            raise ValueError(
                "Latent transplant strength must be in "
                f"(0, {LATENT_TRANSPLANT_MAX_STRENGTH}]"
            )
        object.__setattr__(
            self,
            "activation_editor_strength",
            float(self.activation_editor_strength),
        )
        if not 0 < self.activation_editor_strength <= 4.0:
            raise ValueError("Activation editor strength must be in (0, 4]")
        if self.telemetry_text_mode not in {"off", "edited", "all"}:
            raise ValueError("telemetry_text_mode must be off/edited/all")

    @classmethod
    def from_server_args(cls, server_args: Any) -> QwenExoConfig:
        observer_mode = str(server_args.qwen_exo_observer_mode)
        score_bias_mode = str(getattr(server_args, "qwen_exo_score_bias_mode", "off"))
        score_bias_enabled = score_bias_mode != "off"
        backend = resolve_qwen_exo_backend(server_args)
        if score_bias_enabled and backend != "mlx":
            attention_backends = (
                getattr(server_args, "prefill_attention_backend", None)
                or getattr(server_args, "attention_backend", None),
                getattr(server_args, "decode_attention_backend", None)
                or getattr(server_args, "attention_backend", None),
            )
            if any(
                attention_backend is not None and attention_backend != "triton"
                for attention_backend in attention_backends
            ):
                raise ValueError(
                    "QWEN-EXO CUDA Score Bias requires the Triton attention backend"
                )
        return cls(
            state_directory=Path(server_args.qwen_exo_state_dir).expanduser(),
            knowledge_directory=Path(server_args.qwen_exo_knowledge_dir).expanduser(),
            policy_data_directory=Path(
                getattr(
                    server_args,
                    "qwen_exo_policy_data_dir",
                    Path(server_args.qwen_exo_knowledge_dir).parent / "policydata",
                )
            ).expanduser(),
            cognition_directory=Path(
                getattr(
                    server_args,
                    "qwen_exo_cognition_dir",
                    Path(server_args.qwen_exo_knowledge_dir).parent / "cognition",
                )
            ).expanduser(),
            max_internal_fanout=int(server_args.qwen_exo_max_internal_fanout),
            internal_mamba_reserve=int(server_args.qwen_exo_internal_mamba_reserve),
            max_internal_tokens=int(server_args.qwen_exo_max_internal_tokens),
            max_output_tokens=int(
                getattr(server_args, "qwen_exo_max_output_tokens", 8192)
            ),
            max_reasoning_tokens=int(
                getattr(server_args, "qwen_exo_max_reasoning_tokens", 3072)
            ),
            reasoning_budget_min=getattr(server_args, "qwen_exo_reasoning_budget_min", 512),
            reasoning_budget_low=getattr(server_args, "qwen_exo_reasoning_budget_low", 1024),
            reasoning_budget_medium=getattr(
                server_args, "qwen_exo_reasoning_budget_medium", 2048
            ),
            reasoning_budget_high=getattr(server_args, "qwen_exo_reasoning_budget_high", 4090),
            reasoning_budget_xhigh=getattr(
                server_args, "qwen_exo_reasoning_budget_xhigh", 8192
            ),
            reasoning_mode=getattr(server_args, "qwen_exo_reasoning_mode", "fixed"),
            force_auto_reasoning=getattr(server_args, "qwen_exo_force_auto_reasoning", True),
            reasoning_session_once=getattr(
                server_args, "qwen_exo_reasoning_session_once", True
            ),
            reasoning_dynamic_context_tokens=getattr(
                server_args, "qwen_exo_reasoning_dynamic_context_tokens", 2048
            ),
            observer_mode=observer_mode,
            feature_flags=QwenExoFeatureFlags(
                hybrid_prefix=bool(server_args.qwen_exo_enable_hybrid_prefix),
                capsule=bool(server_args.qwen_exo_enable_capsule),
                observer=observer_mode != "off",
                policy_data=bool(
                    getattr(server_args, "qwen_exo_enable_policy_data", False)
                ),
                score_bias=score_bias_enabled,
                activation_training=bool(
                    getattr(
                        server_args, "qwen_exo_experimental_activation_training", False
                    )
                ),
            ),
            max_running_requests=int(
                getattr(server_args, "max_running_requests", 10) or 10
            ),
            context_length=int(getattr(server_args, "context_length", 131072)),
            model_path=str(server_args.model_path),
            tp_size=int(server_args.tp_size),
            backend=backend,
            dtype=str(getattr(server_args, "dtype", None) or "auto"),
            quantization=str(getattr(server_args, "quantization", None) or "none"),
            kv_cache_dtype=str(getattr(server_args, "kv_cache_dtype", None) or "auto"),
            knowledge_retrieval_mode="file",
            telemetry_include_text=bool(
                getattr(server_args, "qwen_exo_telemetry_include_text", False)
            ),
            telemetry_text_mode=(
                str(getattr(server_args, "qwen_exo_telemetry_text_mode", "") or "")
                or (
                    "all"
                    if bool(
                        getattr(server_args, "qwen_exo_telemetry_include_text", False)
                    )
                    else "off"
                )
            ),
            reflection_memory_mode=str(
                getattr(server_args, "qwen_exo_reflection_memory_mode", "off")
            ),
            reflection_memory_idle_seconds=float(
                getattr(server_args, "qwen_exo_reflection_memory_idle_seconds", 600.0)
            ),
            reflection_memory_min_events=int(
                getattr(server_args, "qwen_exo_reflection_memory_min_events", 3)
            ),
            reflection_memory_min_tokens=int(
                getattr(server_args, "qwen_exo_reflection_memory_min_tokens", 256)
            ),
            reflection_memory_max_attempts=int(
                getattr(server_args, "qwen_exo_reflection_memory_max_attempts", 3)
            ),
            reflection_memory_max_output_tokens=int(
                getattr(
                    server_args, "qwen_exo_reflection_memory_max_output_tokens", 4096
                )
            ),
            reflection_memory_max_history_tokens=int(
                getattr(
                    server_args, "qwen_exo_reflection_memory_max_history_tokens", 92160
                )
            ),
            response_compaction_mode=str(
                getattr(server_args, "qwen_exo_response_compaction_mode", "off")
            ),
            response_compaction_max_history_tokens=int(
                getattr(
                    server_args, "qwen_exo_response_compaction_max_history_tokens", 8192
                )
            ),
            response_compaction_max_dropped_items=int(
                getattr(
                    server_args, "qwen_exo_response_compaction_max_dropped_items", 16
                )
            ),
            response_compaction_max_output_tokens=int(
                getattr(
                    server_args, "qwen_exo_response_compaction_max_output_tokens", 2048
                )
            ),
            score_bias_mode=score_bias_mode,
            score_bias_min_surprisal=float(
                getattr(server_args, "qwen_exo_score_bias_min_surprisal", 0.8)
            ),
            score_bias_max=float(getattr(server_args, "qwen_exo_score_bias_max", 0.05)),
            score_bias_half_life_steps=float(
                getattr(server_args, "qwen_exo_score_bias_half_life_steps", 4.0)
            ),
            score_bias_max_blocks=int(
                getattr(server_args, "qwen_exo_score_bias_max_blocks", 8)
            ),
            score_bias_min_age_steps=int(
                getattr(server_args, "qwen_exo_score_bias_min_age_steps", 2)
            ),
            score_bias_max_age_steps=int(
                getattr(server_args, "qwen_exo_score_bias_max_age_steps", 16)
            ),
            score_bias_tail_tokens=int(
                getattr(server_args, "qwen_exo_score_bias_tail_tokens", 4096)
            ),
            score_bias_tail_ratio=float(
                getattr(server_args, "qwen_exo_score_bias_tail_ratio", 0.15)
            ),
            score_bias_selected_blocks=int(
                getattr(server_args, "qwen_exo_score_bias_selected_blocks", 2)
            ),
            score_bias_query_window=int(
                getattr(server_args, "qwen_exo_score_bias_query_window", 8)
            ),
            score_bias_min_relevance=float(
                getattr(server_args, "qwen_exo_score_bias_min_relevance", 0.01)
            ),
            score_bias_relevance_margin=float(
                getattr(server_args, "qwen_exo_score_bias_relevance_margin", 0.005)
            ),
            score_bias_anchor_bias=float(
                getattr(server_args, "qwen_exo_score_bias_anchor_bias", 0.01)
            ),
            score_bias_anchor_max_blocks=int(
                getattr(server_args, "qwen_exo_score_bias_anchor_max_blocks", 2)
            ),
            latent_transplant_enabled=bool(
                getattr(server_args, "qwen_exo_latent_transplant_enabled", False)
            ),
            latent_transplant_strength=float(
                getattr(server_args, "qwen_exo_latent_transplant_strength", 0.05)
            ),
            activation_editor_enabled=bool(
                getattr(server_args, "qwen_exo_activation_editor_enabled", False)
            ),
            activation_editor_strength=float(
                getattr(server_args, "qwen_exo_activation_editor_strength", 2.0)
            ),
            observer_surprisal_threshold=float(
                getattr(server_args, "qwen_exo_observer_surprisal_threshold", 0.8)
            ),
            observer_surprisal_window=int(
                getattr(server_args, "qwen_exo_observer_surprisal_window", 8)
            ),
            observer_surprisal_margin=float(
                getattr(server_args, "qwen_exo_observer_surprisal_margin", 0.2)
            ),
            observer_surprisal_sustain=int(
                getattr(server_args, "qwen_exo_observer_surprisal_sustain", 1)
            ),
            observer_q_drift_threshold=float(
                getattr(server_args, "qwen_exo_observer_q_drift_threshold", 0.35)
            ),
            observer_cooldown_tokens=int(
                getattr(server_args, "qwen_exo_observer_cooldown_tokens", 64)
            ),
            observer_max_triggers=int(
                getattr(server_args, "qwen_exo_observer_max_triggers", 1)
            ),
            observer_q_pre_tokens=int(
                getattr(server_args, "qwen_exo_observer_q_pre_tokens", 8)
            ),
            observer_q_post_tokens=int(
                getattr(server_args, "qwen_exo_observer_q_post_tokens", 4)
            ),
            observer_recovery_tokens=int(
                getattr(server_args, "qwen_exo_observer_recovery_tokens", 8)
            ),
            immediate_uncertainty_retrieval=bool(
                getattr(server_args, "qwen_exo_immediate_uncertainty_retrieval", False)
            ),
            reasoning_cutoff_question_count=int(
                getattr(
                    server_args, "qwen_exo_reasoning_cutoff_question_count", 3
                )
            ),
        )

    @property
    def topology_key(self) -> str:
        return qwen_exo_topology_key(
            backend=self.backend,
            tp_size=self.tp_size,
            dtype=self.dtype,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    @property
    def model_state_directory(self) -> Path:
        return self.state_directory / "model-native" / self.topology_key

    def reasoning_budget_for_effort(self, effort: str | int | float | None) -> int:
        """Resolve the API tier before any model-template effort normalization."""
        if effort is None or isinstance(effort, (int, float)):
            return self.max_reasoning_tokens
        effort = {"minimal": "min", "max": "xhigh"}.get(effort, effort)
        if effort == "none":
            return 0
        if effort not in ("min", "low", "medium", "high", "xhigh"):
            raise ValueError(f"Unsupported reasoning effort: {effort}")
        return getattr(self, f"reasoning_budget_{effort}")

    def public_dict(self) -> dict[str, Any]:
        flags = asdict(self.feature_flags)
        return {
            "project": PROJECT_NAME,
            "api_version": PROJECT_API_VERSION,
            "state_directory": str(self.state_directory),
            "model_state_directory": str(self.model_state_directory),
            "knowledge_directory": str(self.knowledge_directory),
            "policy_data_directory": str(self.policy_data_directory),
            "cognition_directory": str(self.cognition_directory),
            "max_internal_fanout": self.max_internal_fanout,
            "internal_mamba_reserve": self.internal_mamba_reserve,
            "max_internal_tokens": self.max_internal_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "reasoning_budget_min": self.reasoning_budget_min,
            "reasoning_budget_low": self.reasoning_budget_low,
            "reasoning_budget_medium": self.reasoning_budget_medium,
            "reasoning_budget_high": self.reasoning_budget_high,
            "reasoning_budget_xhigh": self.reasoning_budget_xhigh,
            "reasoning_mode": self.reasoning_mode,
            "force_auto_reasoning": self.force_auto_reasoning,
            "reasoning_session_once": self.reasoning_session_once,
            "reasoning_dynamic_context_tokens": self.reasoning_dynamic_context_tokens,
            "max_running_requests": self.max_running_requests,
            "context_length": self.context_length,
            "knowledge_retrieval_mode": self.knowledge_retrieval_mode,
            "observer_mode": self.observer_mode,
            "telemetry_include_text": self.telemetry_include_text,
            "reflection_memory_mode": self.reflection_memory_mode,
            "reflection_memory_max_attempts": self.reflection_memory_max_attempts,
            "reflection_memory_idle_seconds": self.reflection_memory_idle_seconds,
            "reflection_memory_min_events": self.reflection_memory_min_events,
            "reflection_memory_min_tokens": self.reflection_memory_min_tokens,
            "reflection_memory_max_output_tokens": self.reflection_memory_max_output_tokens,
            "reflection_memory_max_history_tokens": self.reflection_memory_max_history_tokens,
            "response_compaction_mode": self.response_compaction_mode,
            "response_compaction_max_history_tokens": self.response_compaction_max_history_tokens,
            "response_compaction_max_dropped_items": self.response_compaction_max_dropped_items,
            "response_compaction_max_output_tokens": self.response_compaction_max_output_tokens,
            "score_bias_mode": self.score_bias_mode,
            "score_bias_min_relevance": self.score_bias_min_relevance,
            "score_bias_relevance_margin": self.score_bias_relevance_margin,
            "score_bias_tail_tokens": self.score_bias_tail_tokens,
            "score_bias_tail_ratio": self.score_bias_tail_ratio,
            "score_bias_selected_blocks": self.score_bias_selected_blocks,
            "score_bias_query_window": self.score_bias_query_window,
            "latent_transplant_enabled": self.latent_transplant_enabled,
            "latent_transplant_strength": self.latent_transplant_strength,
            "observer_surprisal_threshold": self.observer_surprisal_threshold,
            "observer_surprisal_window": self.observer_surprisal_window,
            "observer_surprisal_margin": self.observer_surprisal_margin,
            "observer_surprisal_sustain": self.observer_surprisal_sustain,
            "observer_q_drift_threshold": self.observer_q_drift_threshold,
            "observer_cooldown_tokens": self.observer_cooldown_tokens,
            "observer_max_triggers": self.observer_max_triggers,
            "observer_q_pre_tokens": self.observer_q_pre_tokens,
            "observer_q_post_tokens": self.observer_q_post_tokens,
            "observer_recovery_tokens": self.observer_recovery_tokens,
            "immediate_uncertainty_retrieval": self.immediate_uncertainty_retrieval,
            "reasoning_cutoff_question_count": self.reasoning_cutoff_question_count,
            "features": flags,
            "model_path": self.model_path,
            "tp_size": self.tp_size,
            "backend": self.backend,
            "topology_key": self.topology_key,
        }
