from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils import is_cuda, is_musa

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
QWEN_EXO_DFLASH_MODE_KEY = "qwen_exo_dflash"
QWEN_EXO_DFLASH_ELIGIBLE = "eligible"
QWEN_EXO_DFLASH_THINK_ACCEPT_MODE_KEY = "qwen_exo_dflash_think_accept_mode"
QWEN_EXO_DFLASH_THINK_ACCEPT_PROBABILITY_KEY = (
    "qwen_exo_dflash_think_accept_probability"
)
QWEN_EXO_DFLASH_THINK_PHASE_KEY = "qwen_exo_dflash_think_phase"
DFLASH_THINK_ACCEPT_MODES = frozenset({"off", "shadow", "active"})


logger = logging.getLogger(__name__)

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_TARGET_SAMPLING_KERNEL_ARGUMENT_COUNT = 0
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TritonAttnBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
        _DFLASH_TARGET_SAMPLING_KERNEL_ARGUMENT_COUNT = len(
            torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default._schema.arguments
        )
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def _run_dflash_target_only_sampling(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Invoke the installed target-only sampling op across sgl-kernel ABIs."""
    if _DFLASH_TARGET_SAMPLING_KERNEL_ARGUMENT_COUNT >= 15:
        stream = (
            torch.musa.current_stream().musa_stream
            if is_musa()
            else torch.cuda.current_stream().cuda_stream
        )
        torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            threshold_single,
            threshold_acc,
            deterministic,
            stream,
        )
        return
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def apply_dflash_verify_logits_adjustments(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
) -> None:
    """Apply sampling-time logit adjustments for DFlash verify in place.

    This keeps v1 and v2 verify semantics aligned while letting overlap scheduling
    use the cheaper precomputed `acc_linear_penalties` path instead of allocating a
    repeated `[bs * draft_token_num, vocab]` penalty tensor every step.
    """
    if sampling_info is None:
        return
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")

    bs = len(sampling_info)
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch for DFlash verify adjustments. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )

    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            next_token_logits,
            sampling_info,
            num_tokens_in_batch=draft_token_num,
        )

    acc_linear_penalties = getattr(sampling_info, "acc_linear_penalties", None)
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    vocab_mask = getattr(sampling_info, "vocab_mask", None)
    logit_bias = getattr(sampling_info, "logit_bias", None)

    logits_3d: Optional[torch.Tensor] = None

    def get_logits_3d() -> torch.Tensor:
        nonlocal logits_3d
        if logits_3d is None:
            logits_3d = next_token_logits.reshape(bs, draft_token_num, -1)
        return logits_3d

    # Dense fallback only when we need live penalizer application or a vocab mask.
    # In overlap scheduling the common path is `acc_linear_penalties`, which can be
    # broadcast over the verify block without materializing a repeated buffer.
    if (
        penalizer is not None and penalizer.is_required and acc_linear_penalties is None
    ) or vocab_mask is not None:
        linear_penalty = torch.zeros(
            (bs, next_token_logits.shape[1]),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        sampling_info.apply_logits_bias(linear_penalty)
        get_logits_3d().add_(
            linear_penalty[:, None, :].to(dtype=next_token_logits.dtype)
        )
        return

    if acc_linear_penalties is not None:
        if (
            acc_linear_penalties.device != next_token_logits.device
            or acc_linear_penalties.dtype != next_token_logits.dtype
        ):
            acc_linear_penalties = acc_linear_penalties.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(acc_linear_penalties[:, None, :])

    if logit_bias is not None:
        if (
            logit_bias.device != next_token_logits.device
            or logit_bias.dtype != next_token_logits.dtype
        ):
            logit_bias = logit_bias.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(logit_bias[:, None, :])


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_layer_types(config: Any) -> Optional[Sequence[str]]:
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> Optional[int]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    # HF sliding windows include the current token; SGLang stores window_left.
    return int(sliding_window) - 1


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: Optional[float]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    conv_kernel_size = _parse_optional_int(
        dflash_cfg.get("conv_kernel_size", 0),
        field_name="DFLASH conv_kernel_size",
        min_value=0,
    )
    conv_group_size = _parse_optional_int(
        dflash_cfg.get("conv_group_size", 0),
        field_name="DFLASH conv_group_size",
        min_value=0,
    )
    if bool(conv_kernel_size) != bool(conv_group_size):
        raise ValueError(
            "DFLASH grouped convolution needs conv_kernel_size and conv_group_size "
            f"together. Got conv_kernel_size={conv_kernel_size}, "
            f"conv_group_size={conv_group_size}."
        )
    selector_rank = _parse_optional_int(
        dflash_cfg.get("selector_rank", 0),
        field_name="DFLASH selector rank",
        min_value=0,
    )
    selector_top_k = _parse_optional_int(
        dflash_cfg.get("selector_top_k", 0),
        field_name="DFLASH selector top_k",
        min_value=0,
    )
    if bool(selector_rank) != bool(selector_top_k):
        raise ValueError(
            "DFLASH selector needs rank and top_k together. "
            f"Got rank={selector_rank}, top_k={selector_top_k}."
        )

    output_multiplier = float(dflash_cfg.get("output_multiplier", 1.0))
    if output_multiplier <= 0:
        raise ValueError("DFLASH output_multiplier must be positive.")
    softcap = float(dflash_cfg.get("final_logit_softcapping") or 0.0)
    final_logit_softcapping = softcap if softcap > 0 else None

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        conv_kernel_size=conv_kernel_size,
        conv_group_size=conv_group_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        output_multiplier=output_multiplier,
        final_logit_softcapping=final_logit_softcapping,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


# is_floating_point() is True for fp8; list dtypes explicitly.
_DENSE_HEAD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def is_dense_head_weight(weight: Any) -> bool:
    """Whether an lm_head weight can be read as a plain matrix. A quantized head
    stores packed values, which a dense matmul would read as if they were
    activations."""
    return weight is not None and weight.dtype in _DENSE_HEAD_DTYPES


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


def dflash_think_acceptance_mask(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    think_mask: torch.Tensor,
    probability_threshold: float,
    temperatures: Optional[torch.Tensor] = None,
    target_max_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Think-only force-accept mask from target logit confidence.

    The confidence is the candidate's probability relative to the highest
    probability at the same target position:
    ``exp((candidate_logit - max_logit) / temperature)``.  A threshold of
    ``0.60`` therefore admits a near-top candidate without changing body
    decoding.  This is intentionally non-lossless and is only used by the
    explicit DFLASH experiment mode.

    Returns ``(force_accept_mask, relative_probability)`` for the ``gamma``
    draft positions.  The mask is ``[bs, block_size - 1]`` and already
    includes the caller's Think/answer phase mask.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_logits.ndim != 2:
        raise ValueError(
            f"target_logits must be 2D, got shape={tuple(target_logits.shape)}"
        )
    bs, block_size = candidates.shape
    if bs < 1 or block_size < 1:
        raise ValueError("candidates must have positive batch and block dimensions")
    if target_logits.shape[0] != bs * block_size:
        raise ValueError(
            "target_logits row count must equal batch_size * block_size, "
            f"got {target_logits.shape[0]} for {bs * block_size}"
        )
    threshold = float(probability_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"probability_threshold must be finite and within [0, 1], got {threshold}"
        )
    gamma = block_size - 1
    if gamma == 0:
        empty_mask = torch.empty((bs, 0), dtype=torch.bool, device=candidates.device)
        empty_ratio = torch.empty(
            (bs, 0), dtype=torch.float32, device=candidates.device
        )
        return empty_mask, empty_ratio
    if think_mask.shape != (bs, gamma):
        raise ValueError(
            "think_mask must have shape [batch_size, block_size - 1], "
            f"got {tuple(think_mask.shape)} for {(bs, gamma)}"
        )
    if think_mask.device != candidates.device:
        raise ValueError("think_mask must be on the candidates device")
    logits = target_logits.reshape(bs, block_size, -1)[:, :-1]
    candidate_ids = candidates[:, 1:].to(dtype=torch.long)
    candidate_logits = logits.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
    if target_max_logits is None:
        max_logits = logits.amax(dim=-1)
    else:
        if target_max_logits.shape != (bs, gamma):
            raise ValueError(
                "target_max_logits must have shape [batch_size, block_size - 1], "
                f"got {tuple(target_max_logits.shape)} for {(bs, gamma)}"
            )
        if target_max_logits.device != candidates.device:
            raise ValueError("target_max_logits must be on the candidates device")
        max_logits = target_max_logits
    candidate_logits = candidate_logits.float()
    max_logits = max_logits.float()
    if temperatures is None:
        temperature = torch.ones((bs, 1), device=logits.device, dtype=torch.float32)
    else:
        if temperatures.numel() != bs:
            raise ValueError(
                f"temperatures must contain one value per request, got {temperatures.numel()} for {bs}"
            )
        temperature = temperatures.reshape(bs, 1).to(
            device=logits.device, dtype=torch.float32
        )
        temperature = temperature.clamp_min(1e-5)
    relative_probability = torch.exp(
        ((candidate_logits - max_logits) / temperature).clamp(min=-80.0, max=0.0)
    )
    relative_probability = torch.nan_to_num(
        relative_probability, nan=0.0, posinf=1.0, neginf=0.0
    )
    force_accept_mask = think_mask.to(dtype=torch.bool) & (
        relative_probability >= threshold
    )
    return force_accept_mask, relative_probability


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
    force_accept_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if force_accept_mask is not None:
        expected_force_shape = (bs, max(draft_token_num - 1, 0))
        if tuple(force_accept_mask.shape) != expected_force_shape:
            raise ValueError(
                "force_accept_mask must have shape [batch_size, draft_token_num - 1], "
                f"got {tuple(force_accept_mask.shape)} for {expected_force_shape}"
            )
        if force_accept_mask.device != candidates.device:
            raise ValueError("force_accept_mask must be on the candidates device")
    if threshold_single is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_single = get_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_acc = get_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    target_probs = build_dflash_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
        bs=bs,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
        use_sparse_topk=use_sparse_topk,
    )
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    if force_accept_mask is None:
        _run_dflash_target_only_sampling(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates_i64,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
        )
    else:
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates_i64,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
            force_accept_mask=force_accept_mask,
        )

    correct_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


def build_dflash_verify_target_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
    bs: int,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    use_sparse_topk: bool = True,
) -> torch.Tensor:
    device = next_token_logits.device
    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        if max_top_k is None:
            max_top_k = int(repeated_top_ks.max().item())
        else:
            max_top_k = int(max_top_k)
        if max_top_k < 1:
            max_top_k = 1
        elif max_top_k > vocab_size:
            max_top_k = vocab_size

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    return target_probs.view(bs, draft_token_num, -1).contiguous()


def build_dflash_grammar_vocab_mask(
    *,
    reqs: List[Req],
    draft_tokens: torch.Tensor,
    vocab_size: int,
    verify_input: DFlashVerifyInput,
) -> Optional[torch.Tensor]:
    """Per-draft-position grammar bitmask for a DFLASH verify block.

    The block is a linear chain whose position 0 is the bonus token each
    grammar has already accepted. Every valid draft token is accepted to fill
    the next row's mask, then rolled back; rows of non-grammar requests and
    rows past the first grammar-invalid draft stay all-allowed (acceptance
    stops there). Sets ``verify_input.grammar`` for ``apply_vocab_mask``.
    """
    from sglang.srt.speculative.spec_utils import generate_token_bitmask

    bs, block_size = draft_tokens.shape
    next_token = torch.arange(1, block_size + 1, dtype=torch.int64)
    next_token[-1] = -1
    vocab_mask = generate_token_bitmask(
        reqs,
        verify_input,
        next_token.expand(bs, block_size),
        torch.full((bs, block_size), -1, dtype=torch.int64),
        draft_tokens.cpu(),
        vocab_size,
    )
    if vocab_mask is None:
        return None
    return vocab_mask.to(draft_tokens.device, non_blocking=True)


def is_dflash_target_only_request(req: Req) -> bool:
    """Return whether DFLASH must route this request through the target only."""
    custom_params = req.sampling_params.custom_params or {}
    if bool(req.return_hidden_states):
        return True
    if custom_params.get("qwen_exo_kind") == "internal":
        return custom_params.get(QWEN_EXO_DFLASH_MODE_KEY) != QWEN_EXO_DFLASH_ELIGIBLE
    return False


def dflash_request_needs_target_logprobs(req: Req) -> bool:
    """Return whether this request has a consumer for target token logprobs."""
    custom_params = req.sampling_params.custom_params or {}
    return (
        bool(req.return_logprob)
        or custom_params.get("qwen_exo_kind") == "user"
        or (custom_params.get("qwen_exo_job_type") == "query_probe")
    )


def validate_dflash_request(req: Req, enable_overlap: bool) -> Optional[str]:
    if is_dflash_target_only_request(req):
        return None
    if enable_overlap and req.return_hidden_states:
        return "DFLASH speculative decoding does not support return_hidden_states yet."
    return None
