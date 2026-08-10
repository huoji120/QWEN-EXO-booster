from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_exo_booster.score_bias import SCORE_BIAS_KERNEL_MAX_BLOCKS
from qwen_exo_booster.hybrid_state import (
    qwen_exo_topology_key,
    resolve_qwen_exo_backend,
)

PROJECT_NAME = "QWEN-EXO-booster"
PROJECT_API_VERSION = "1"
_OBSERVER_MODES = frozenset({"off", "shadow", "active"})
_CONTEXT_EVIDENCE_MODES = frozenset({"off", "active"})
_CONTEXT_INTEGRITY_MODES = frozenset({"off", "active"})
_REFLECTION_MEMORY_MODES = frozenset({"off", "active"})
_RESPONSE_COMPACTION_MODES = frozenset({"off", "active"})
_QK_PREFILTER_MODES = frozenset({"off", "active"})
TENSOR_BANK_CONTEXT_RESERVE_TOKENS = 2048
DEFAULT_TENSOR_BANK_MAX_DOCUMENT_TOKENS = 100352
DEFAULT_TENSOR_BANK_SALIENT_TOKEN_BUDGET = 4096


_SCORE_BIAS_MODES = frozenset({"off", "trajectory_shadow", "trajectory_active"})
LATENT_TRANSPLANT_MAX_STRENGTH = 0.5

QK_RECALL_PRESETS: dict[str, tuple[float, float]] = {
    "broad": (-0.05, 0.0),
    "balanced": (0.0, 0.0),
    "strict": (8.0, 0.02),
}


def qk_recall_gates(preset: str) -> tuple[float, float]:
    try:
        return QK_RECALL_PRESETS[str(preset)]
    except KeyError as exc:
        raise ValueError(
            f"qwen_exo_qk_recall_preset must be one of {sorted(QK_RECALL_PRESETS)}"
        ) from exc


@dataclass(frozen=True, slots=True)
class QwenExoFeatureFlags:
    hybrid_prefix: bool
    external_memory: bool
    reference_judge: bool
    capsule: bool
    observer: bool
    adaptive_refresh: bool
    policy_data: bool = True
    score_bias: bool = False


@dataclass(frozen=True, slots=True)
class QwenExoConfig:
    state_directory: Path
    knowledge_directory: Path
    max_internal_fanout: int
    max_internal_tokens: int
    max_candidates: int
    max_memory_tokens: int
    observer_mode: str
    feature_flags: QwenExoFeatureFlags
    model_path: str
    tp_size: int
    backend: str = "cuda"
    dtype: str = "auto"
    quantization: str = "none"
    kv_cache_dtype: str = "auto"
    max_running_requests: int = 4
    context_length: int = 102400
    policy_data_directory: Path | None = None
    cognition_directory: Path | None = None
    max_policy_tokens: int = 4096
    tensor_bank_max_document_tokens: int = DEFAULT_TENSOR_BANK_MAX_DOCUMENT_TOKENS
    tensor_bank_salient_token_budget: int = DEFAULT_TENSOR_BANK_SALIENT_TOKEN_BUDGET
    tensor_bank_surprisal_threshold: float = 6.0
    tensor_bank_span_tokens: int = 16
    telemetry_include_text: bool = False
    telemetry_text_mode: str = "off"
    context_evidence_mode: str = "off"
    context_integrity_mode: str = "off"
    context_integrity_context_divisor: int = 3
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
    score_bias_min_relevance: float = 0.0

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
    score_bias_anchor_bias: float = 0.0
    score_bias_anchor_max_blocks: int = 2
    latent_transplant_enabled: bool = False
    latent_transplant_strength: float = 0.05
    activation_editor_enabled: bool = False
    activation_editor_strength: float = 2.0
    max_output_tokens: int = 8192
    max_reasoning_tokens: int = 3072
    observer_surprisal_threshold: float = 0.8
    observer_surprisal_window: int = 8
    observer_surprisal_margin: float = 0.2
    observer_q_drift_threshold: float = 0.35
    observer_cooldown_tokens: int = 64
    observer_max_triggers: int = 1
    observer_q_pre_tokens: int = 8
    observer_q_post_tokens: int = 4
    observer_recovery_tokens: int = 8
    immediate_uncertainty_retrieval: bool = False
    replay_observation_tokens: int = 8
    replay_prefix_tokens: int = 1024
    replay_max_candidates: int = 2
    replay_reference_tokens: int = 128
    replay_minimum_gain: float = 0.02
    replay_switch_margin: float = 0.05
    replay_maybe_kl_cap: float = 4.0
    qk_expansion_margin: float = 0.01
    qk_only_knowledge: bool = False
    qk_recall_preset: str = "balanced"
    qk_prefilter_mode: str = "active"
    qk_prefilter_min_score: float | None = None
    qk_prefilter_min_margin: float | None = None
    qk_max_candidates_per_document: int = 1

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
        if self.max_internal_tokens < 1:
            raise ValueError("qwen_exo_max_internal_tokens must be positive")
        if self.max_candidates < 1:
            raise ValueError("qwen_exo_max_candidates must be positive")
        if self.max_policy_tokens < 1:
            raise ValueError("qwen_exo_max_policy_tokens must be positive")
        max_compilable_document_tokens = (
            self.context_length - TENSOR_BANK_CONTEXT_RESERVE_TOKENS
        )
        if max_compilable_document_tokens < 64:
            raise ValueError(
                "context_length must leave 2048 tokens plus one 64-token radix "
                "page for Tensor Bank compilation"
            )
        if self.tensor_bank_max_document_tokens < 64:
            raise ValueError("tensor_bank_max_document_tokens must be at least 64")
        if self.tensor_bank_max_document_tokens > max_compilable_document_tokens:
            raise ValueError(
                "tensor_bank_max_document_tokens cannot exceed context_length - "
                "2048 tokens"
            )
        if (
            self.tensor_bank_salient_token_budget < 64
            or self.tensor_bank_salient_token_budget % 64
            or self.tensor_bank_salient_token_budget
            > self.tensor_bank_max_document_tokens
        ):
            raise ValueError(
                "tensor_bank_salient_token_budget must be 64-token aligned and "
                "no larger than tensor_bank_max_document_tokens"
            )
        if (
            self.tensor_bank_surprisal_threshold < 0
            or self.tensor_bank_span_tokens < 1
            or self.tensor_bank_span_tokens > self.tensor_bank_salient_token_budget
        ):
            raise ValueError("Tensor Bank surprisal and span settings are invalid")
        if self.max_output_tokens < 1:
            raise ValueError("qwen_exo_max_output_tokens must be positive")
        if self.max_reasoning_tokens < 1:
            raise ValueError("qwen_exo_max_reasoning_tokens must be positive")
        if self.observer_mode not in _OBSERVER_MODES:
            raise ValueError(
                f"qwen_exo_observer_mode must be one of {sorted(_OBSERVER_MODES)}"
            )
        if self.tp_size < 1:
            raise ValueError("tp_size must be positive")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be positive")
        if self.feature_flags.adaptive_refresh and self.observer_mode != "active":
            raise ValueError("Adaptive refresh requires qwen_exo_observer_mode=active")
        if self.feature_flags.adaptive_refresh and not (
            self.feature_flags.reference_judge and self.feature_flags.external_memory
        ):
            raise ValueError(
                "Adaptive refresh requires reference judge and external memory"
            )
        if self.context_evidence_mode not in _CONTEXT_EVIDENCE_MODES:
            raise ValueError(
                "qwen_exo_context_evidence_mode must be one of "
                f"{sorted(_CONTEXT_EVIDENCE_MODES)}"
            )
        if (
            self.context_evidence_mode != "off"
            and not self.feature_flags.adaptive_refresh
        ):
            raise ValueError("Context Evidence Check requires adaptive refresh")
        if self.context_integrity_mode not in _CONTEXT_INTEGRITY_MODES:
            raise ValueError(
                "qwen_exo_context_integrity_mode must be one of "
                f"{sorted(_CONTEXT_INTEGRITY_MODES)}"
            )
        if self.context_integrity_context_divisor < 2:
            raise ValueError(
                "qwen_exo_context_integrity_context_divisor must be at least 2"
            )
        if (
            self.context_integrity_mode != "off"
            and not self.feature_flags.adaptive_refresh
        ):
            raise ValueError("Context Integrity Check requires adaptive refresh")
        if self.reflection_memory_mode not in _REFLECTION_MEMORY_MODES:
            raise ValueError(
                "qwen_exo_reflection_memory_mode must be one of "
                f"{sorted(_REFLECTION_MEMORY_MODES)}"
            )
        if (
            self.reflection_memory_mode != "off"
            and not self.feature_flags.external_memory
        ):
            raise ValueError("Reflection memory requires external memory")
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
        if self.response_compaction_mode != "off" and not (
            self.feature_flags.external_memory
        ):
            raise ValueError("Response compaction requires external memory")
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
        if (
            self.replay_observation_tokens < 2
            or self.replay_prefix_tokens < 1
            or self.replay_max_candidates < 1
            or self.replay_reference_tokens < 1
            or self.replay_minimum_gain < 0
            or self.replay_switch_margin < 0
            or self.replay_maybe_kl_cap < 0
        ):
            raise ValueError("Causal replay limits are invalid")
        if self.qk_expansion_margin < 0:
            raise ValueError("Q/K expansion margin must be non-negative")
        qk_recall_gates(self.qk_recall_preset)
        if self.qk_prefilter_mode not in _QK_PREFILTER_MODES:
            raise ValueError(
                "qwen_exo_qk_prefilter_mode must be one of "
                f"{sorted(_QK_PREFILTER_MODES)}"
            )
        if self.qk_prefilter_min_score is not None and not math.isfinite(
            self.qk_prefilter_min_score
        ):
            raise ValueError("Q/K prefilter minimum score must be finite")
        if (
            self.qk_prefilter_min_margin is not None
            and self.qk_prefilter_min_margin < 0
        ):
            raise ValueError("Q/K prefilter minimum margin must be non-negative")
        if self.qk_max_candidates_per_document < 1:
            raise ValueError("Q/K per-document candidate limit must be positive")

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
            max_internal_tokens=int(server_args.qwen_exo_max_internal_tokens),
            max_candidates=int(server_args.qwen_exo_max_candidates),
            max_memory_tokens=int(server_args.qwen_exo_max_memory_tokens),
            max_policy_tokens=int(
                getattr(server_args, "qwen_exo_max_policy_tokens", 4096)
            ),
            tensor_bank_max_document_tokens=int(
                getattr(
                    server_args,
                    "qwen_exo_tensor_bank_max_document_tokens",
                    DEFAULT_TENSOR_BANK_MAX_DOCUMENT_TOKENS,
                )
            ),
            tensor_bank_salient_token_budget=int(
                getattr(
                    server_args,
                    "qwen_exo_tensor_bank_salient_token_budget",
                    DEFAULT_TENSOR_BANK_SALIENT_TOKEN_BUDGET,
                )
            ),
            tensor_bank_surprisal_threshold=float(
                getattr(server_args, "qwen_exo_tensor_bank_surprisal_threshold", 6.0)
            ),
            tensor_bank_span_tokens=int(
                getattr(server_args, "qwen_exo_tensor_bank_span_tokens", 16)
            ),
            max_output_tokens=int(
                getattr(server_args, "qwen_exo_max_output_tokens", 8192)
            ),
            max_reasoning_tokens=int(
                getattr(server_args, "qwen_exo_max_reasoning_tokens", 3072)
            ),
            observer_mode=observer_mode,
            feature_flags=QwenExoFeatureFlags(
                hybrid_prefix=bool(server_args.qwen_exo_enable_hybrid_prefix),
                external_memory=bool(server_args.qwen_exo_enable_external_memory),
                reference_judge=bool(server_args.qwen_exo_enable_reference_judge),
                capsule=bool(server_args.qwen_exo_enable_capsule),
                observer=observer_mode != "off",
                adaptive_refresh=bool(server_args.qwen_exo_enable_adaptive_refresh),
                policy_data=bool(
                    getattr(server_args, "qwen_exo_enable_policy_data", False)
                ),
                score_bias=score_bias_enabled,
            ),
            max_running_requests=int(
                getattr(server_args, "max_running_requests", 4) or 4
            ),
            context_length=int(getattr(server_args, "context_length", 102400)),
            model_path=str(server_args.model_path),
            tp_size=int(server_args.tp_size),
            backend=backend,
            dtype=str(getattr(server_args, "dtype", None) or "auto"),
            quantization=str(getattr(server_args, "quantization", None) or "none"),
            kv_cache_dtype=str(getattr(server_args, "kv_cache_dtype", None) or "auto"),
            qk_expansion_margin=float(
                getattr(server_args, "qwen_exo_qk_expansion_margin", 0.01)
            ),
            qk_only_knowledge=bool(
                getattr(server_args, "qwen_exo_qk_only_knowledge", False)
            ),
            qk_recall_preset=str(
                getattr(server_args, "qwen_exo_qk_recall_preset", "balanced")
            ),
            qk_prefilter_mode=str(
                getattr(server_args, "qwen_exo_qk_prefilter_mode", "active")
            ),
            qk_prefilter_min_score=(
                None
                if getattr(server_args, "qwen_exo_qk_prefilter_min_score", None) is None
                else float(server_args.qwen_exo_qk_prefilter_min_score)
            ),
            qk_prefilter_min_margin=(
                None
                if getattr(server_args, "qwen_exo_qk_prefilter_min_margin", None)
                is None
                else float(server_args.qwen_exo_qk_prefilter_min_margin)
            ),
            qk_max_candidates_per_document=int(
                getattr(server_args, "qwen_exo_qk_max_candidates_per_document", 1)
            ),
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
            context_evidence_mode=str(
                getattr(server_args, "qwen_exo_context_evidence_mode", "off")
            ),
            context_integrity_mode=str(
                getattr(server_args, "qwen_exo_context_integrity_mode", "off")
            ),
            context_integrity_context_divisor=int(
                getattr(server_args, "qwen_exo_context_integrity_context_divisor", 3)
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
                getattr(server_args, "qwen_exo_score_bias_min_relevance", 0.0)
            ),
            score_bias_relevance_margin=float(
                getattr(server_args, "qwen_exo_score_bias_relevance_margin", 0.005)
            ),
            score_bias_anchor_bias=float(
                getattr(server_args, "qwen_exo_score_bias_anchor_bias", 0.0)
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
            replay_observation_tokens=int(
                getattr(server_args, "qwen_exo_replay_observation_tokens", 8)
            ),
            replay_prefix_tokens=int(
                getattr(server_args, "qwen_exo_replay_prefix_tokens", 1024)
            ),
            replay_max_candidates=int(
                getattr(server_args, "qwen_exo_replay_max_candidates", 2)
            ),
            replay_reference_tokens=int(
                getattr(server_args, "qwen_exo_replay_reference_tokens", 128)
            ),
            replay_minimum_gain=float(
                getattr(server_args, "qwen_exo_replay_minimum_gain", 0.02)
            ),
            replay_switch_margin=float(
                getattr(server_args, "qwen_exo_replay_switch_margin", 0.05)
            ),
            replay_maybe_kl_cap=float(
                getattr(server_args, "qwen_exo_replay_maybe_kl_cap", 4.0)
            ),
        )

    @property
    def context_integrity_max_tokens(self) -> int:
        return self.context_length // self.context_integrity_context_divisor

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
            "max_internal_tokens": self.max_internal_tokens,
            "max_candidates": self.max_candidates,
            "max_memory_tokens": self.max_memory_tokens,
            "max_policy_tokens": self.max_policy_tokens,
            "tensor_bank_max_document_tokens": self.tensor_bank_max_document_tokens,
            "tensor_bank_salient_token_budget": self.tensor_bank_salient_token_budget,
            "tensor_bank_surprisal_threshold": self.tensor_bank_surprisal_threshold,
            "tensor_bank_span_tokens": self.tensor_bank_span_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_running_requests": self.max_running_requests,
            "context_length": self.context_length,
            "qk_expansion_margin": self.qk_expansion_margin,
            "qk_only_knowledge": self.qk_only_knowledge,
            "qk_recall_preset": self.qk_recall_preset,
            "qk_prefilter_mode": self.qk_prefilter_mode,
            "qk_prefilter_min_score": self.qk_prefilter_min_score,
            "qk_prefilter_min_margin": self.qk_prefilter_min_margin,
            "qk_max_candidates_per_document": self.qk_max_candidates_per_document,
            "observer_mode": self.observer_mode,
            "telemetry_include_text": self.telemetry_include_text,
            "context_evidence_mode": self.context_evidence_mode,
            "context_integrity_mode": self.context_integrity_mode,
            "context_integrity_max_tokens": self.context_integrity_max_tokens,
            "context_integrity_context_divisor": (
                self.context_integrity_context_divisor
            ),
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
            "observer_q_drift_threshold": self.observer_q_drift_threshold,
            "observer_cooldown_tokens": self.observer_cooldown_tokens,
            "observer_max_triggers": self.observer_max_triggers,
            "observer_q_pre_tokens": self.observer_q_pre_tokens,
            "observer_q_post_tokens": self.observer_q_post_tokens,
            "observer_recovery_tokens": self.observer_recovery_tokens,
            "immediate_uncertainty_retrieval": self.immediate_uncertainty_retrieval,
            "replay_observation_tokens": self.replay_observation_tokens,
            "replay_prefix_tokens": self.replay_prefix_tokens,
            "replay_max_candidates": self.replay_max_candidates,
            "replay_reference_tokens": self.replay_reference_tokens,
            "replay_minimum_gain": self.replay_minimum_gain,
            "replay_switch_margin": self.replay_switch_margin,
            "replay_maybe_kl_cap": self.replay_maybe_kl_cap,
            "features": flags,
            "model_path": self.model_path,
            "tp_size": self.tp_size,
            "backend": self.backend,
            "topology_key": self.topology_key,
        }
