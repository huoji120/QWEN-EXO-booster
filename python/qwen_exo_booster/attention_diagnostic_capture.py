from __future__ import annotations

import math
from typing import Any

import torch

ATTENTION_DIAGNOSTIC_MAX_TOKENS = 32768
ATTENTION_DIAGNOSTIC_MAX_LAYERS = 2
ATTENTION_DIAGNOSTIC_ERROR_KEY = "qwen_exo_attention_error"


def attention_diagnostic_specs(requests) -> list[dict[str, Any] | None] | None:
    """Keep diagnostics request-local; malformed requests become error metadata."""
    if not any(
        "qwen_exo_attention_diagnostic"
        in (getattr(req.sampling_params, "custom_params", None) or {})
        for req in requests
    ):
        return None
    specs = []
    for req in requests:
        custom = getattr(req.sampling_params, "custom_params", None) or {}
        if "qwen_exo_attention_diagnostic" not in custom:
            specs.append(None)
            continue
        raw = custom["qwen_exo_attention_diagnostic"]
        count = len(req.origin_input_ids)
        spec = {"token_count": count, "layer_ids": (), "error": 2}
        if (
            isinstance(raw, dict)
            and set(raw) == {"token_count", "layer_ids"}
            and type(raw.get("token_count")) is int
            and raw["token_count"] == count
            and 1 <= count <= ATTENTION_DIAGNOSTIC_MAX_TOKENS
            and isinstance(raw.get("layer_ids"), list)
            and 1 <= len(raw["layer_ids"]) <= ATTENTION_DIAGNOSTIC_MAX_LAYERS
            and all(type(value) is int and value >= 0 for value in raw["layer_ids"])
            and len(set(raw["layer_ids"])) == len(raw["layer_ids"])
            and custom.get("qwen_exo_kind") == "internal"
            and custom.get("qwen_exo_job_type") == "attention_diagnostic"
            and custom.get("qwen_exo_dflash") == "target_only"
            and req.sampling_params.max_new_tokens == 1
            and bool(getattr(req, "extra_key", None))
            and not getattr(req, "multimodal_inputs", None)
            and getattr(req, "input_embeds", None) is None
            and getattr(req, "positional_embed_overrides", None) is None
            and not getattr(req, "session_id", None)
            and not getattr(req, "lora_id", None)
            and not any(
                key.startswith(
                    (
                        "qwen_exo_memory",
                        "qwen_exo_native",
                        "qwen_exo_bank",
                        "qwen_exo_session_initial",
                        "qwen_exo_score_bias",
                        "qwen_exo_trajectory",
                        "qwen_exo_latent",
                        "qwen_exo_activation_editor",
                    )
                )
                for key in custom
            )
        ):
            spec.update(layer_ids=tuple(raw["layer_ids"]), error=0)
        specs.append(spec)
    return specs


def _diagnostic_backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    # These wrappers preserve the ordinary full-attention request mapping.
    for _ in range(3):
        name = type(backend).__name__
        if name == "HybridLinearAttnBackend":
            backend = backend.full_attn_backend
        elif name == "HybridAttnBackend":
            backend = backend.prefill_backend
        else:
            break
    if (
        type(backend).__name__ not in ("TritonAttnBackend", "FlashInferAttnBackend")
        or getattr(backend, "dcp_size", 1) != 1
        or getattr(backend, "use_mla", False)
        or getattr(backend, "prefill_uses_dequant_workspace", False)
        or getattr(backend, "enable_mis", False)
    ):
        return None
    pool = backend.token_to_kv_pool
    if type(pool).__name__ == "HybridLinearKVPool":
        if pool.use_mla:
            return None
        pool = pool.full_kv_pool
    if (
        type(pool).__name__ != "MHATokenToKVPool"
        or getattr(pool, "use_hnd", False)
        or getattr(pool, "is_quantized_kv_cache", False)
        or getattr(pool, "kv_cache_layout", None) == "vectorized_5d"
    ):
        return None
    return backend


def _key_descale(layer, dtype: torch.dtype) -> float | None:
    ordinary = (torch.float16, torch.bfloat16, torch.float32)
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if dtype not in ordinary + fp8:
        return None
    # The backend applies the checkpoint's scalar descale to cached K. No dynamic
    # block/head quantization is supported, and ordinary caches must be unscaled.
    if layer.k_scale is None and layer.v_scale is None:
        return 1.0
    if layer.k_scale is None or layer.v_scale is None:
        return None
    value = layer.k_scale_float
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    if dtype in ordinary and value != 1.0:
        return None
    return float(value)


def mean_cached_attention(query, keys, mapping, *, scaling: float, key_descale: float):
    """FP32 softmax per query head, then head mean; one KV head resident at a time.

    Q and cached K already include RoPE. The result reconstructs cached-key
    attention, not fused-kernel probabilities (fresh FP8-prefill K may differ).
    """
    num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[1] if keys.ndim == 3 else 0
    if (
        keys.ndim != 3
        or keys.shape[2] != head_dim
        or num_kv_heads < 1
        or num_heads < 1
        or num_heads % num_kv_heads != 0
        or not 1 <= mapping.numel() <= ATTENTION_DIAGNOSTIC_MAX_TOKENS
    ):
        raise ValueError("Invalid diagnostic attention geometry")
    group_size = num_heads // num_kv_heads
    result = torch.zeros(mapping.numel(), dtype=torch.float32, device=query.device)
    for kv_head in range(num_kv_heads):
        source = keys[:, kv_head, :]
        if keys.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Byte gather also works on torch versions without FP8 index_select.
            selected = (
                source.view(torch.uint8)
                .index_select(0, mapping)
                .view(keys.dtype)
                .float()
            )
        else:
            selected = source.index_select(0, mapping).float()
        # Scaling Q instead of making a second dequantized K allocation is
        # algebraically identical for the scalar-scaled cache contract.
        for head in range(kv_head * group_size, (kv_head + 1) * group_size):
            scores = torch.mv(selected, query[head].float() * (scaling * key_descale))
            result.add_(torch.softmax(scores, dim=0))
    return result.div_(num_heads)


def diagnostic_position_matches(positions, end: int, count: int) -> bool:
    """Validate a text prefix for scalar RoPE or three identical MRoPE axes."""
    if end < 1 or count < 1:
        return False
    if positions.ndim == 1:
        return end <= positions.shape[0] and bool(
            (positions[end - 1] == count - 1).item()
        )
    if positions.ndim == 2 and positions.shape[0] == 3:
        return end <= positions.shape[1] and bool(
            (positions[:, end - 1] == count - 1).all().item()
        )
    return False


def capture_attention_diagnostic(decoder, query, positions, forward_batch) -> None:
    """Called strictly after the actual attention backend has written K/V.

    Error codes: 1 unsupported backend/cache/topology; 2 invalid specification;
    3 incomplete/mismatched request mapping or nonfinite attention. Errors are
    transported rather than raised through the shared inference worker.
    """
    specs = forward_batch.qwen_exo_attention_diagnostics
    if specs is None or not forward_batch.forward_mode.is_extend():
        return
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.runtime_context import get_parallel

    lengths = forward_batch.extend_seq_lens_cpu
    prefixes = forward_batch.extend_prefix_lens_cpu
    if lengths is None or prefixes is None:
        return  # Missing captures are rejected by the HTTP coordinator.
    final_rows = [
        row
        for row, spec in enumerate(specs)
        if spec is not None
        and int(prefixes[row]) + int(lengths[row]) == spec["token_count"]
    ]
    if not final_rows:
        return
    errors = torch.zeros(len(specs), dtype=torch.float32, device=query.device)
    parallel = get_parallel()
    backend = (
        None
        if (
            get_is_capture_mode()
            or query.device.type != "cuda"
            or parallel.pp_size != 1
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.attn_dcp_size != 1
            or forward_batch.tbo_split_seq_index is not None
        )
        else _diagnostic_backend()
    )
    layer = decoder.attn
    full_layers = {
        index
        for index, kind in enumerate(decoder.config.layers_block_type)
        if kind == "attention"
    }
    valid_rows = []
    has_error = False
    for row in final_rows:
        spec = specs[row]
        if spec["error"] or not set(spec["layer_ids"]).issubset(full_layers):
            errors[row] = 2
            has_error = True
        elif backend is None:
            errors[row] = 1
            has_error = True
        elif layer.layer_id in spec["layer_ids"]:
            valid_rows.append(row)
    if not valid_rows and not has_error:
        return
    weights = None
    if valid_rows:
        metadata = backend.forward_metadata
        unsupported = (
            layer.is_cross_attention
            or layer.logit_cap != 0
            or layer.sliding_window_size not in (None, -1)
            or layer.xai_temperature_len not in (None, -1)
            or layer.attn_type.value != "decoder"
            or layer.pos_encoding_mode != "NONE"
            or getattr(metadata, "custom_mask", None) is not None
            or getattr(forward_batch, "cross_attention_custom_mask", None) is not None
            or getattr(metadata, "multi_item_params", None) is not None
        )
        if unsupported:
            errors[valid_rows] = 1
        else:
            keys = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
            descale = _key_descale(layer, keys.dtype)
            if (
                descale is None
                or keys.ndim != 3
                or keys.shape[1:] != (decoder.num_kv_heads, decoder.head_dim)
            ):
                errors[valid_rows] = 1
            else:
                width = max(
                    spec["token_count"]
                    for spec in specs
                    if spec is not None and not spec["error"]
                )
                weights = torch.zeros(
                    (len(specs), width), dtype=torch.float32, device=query.device
                )
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + int(length))
                q = query.view(-1, decoder.num_heads, decoder.head_dim)
                mapping_table = backend.req_to_token_pool.req_to_token
                for row in valid_rows:
                    count = specs[row]["token_count"]
                    slot = int(forward_batch.req_pool_indices[row].item())
                    end = offsets[row + 1]
                    position_ok = diagnostic_position_matches(positions, end, count)
                    if (
                        int(lengths[row]) < 1
                        or end > q.shape[0]
                        or not 0 <= slot < mapping_table.shape[0]
                        or count > mapping_table.shape[1]
                        or not position_ok
                    ):
                        errors[row] = 3
                        continue
                    mapping = mapping_table[slot, :count].long()
                    if bool(((mapping <= 0) | (mapping >= keys.shape[0])).any().item()):
                        errors[row] = 3
                        continue
                    # Verify the final chunk is really the cache just written.
                    if not torch.equal(
                        mapping[int(prefixes[row]) :],
                        forward_batch.out_cache_loc[offsets[row] : end].long(),
                    ):
                        errors[row] = 3
                        continue
                    weights[row, :count] = mean_cached_attention(
                        q[end - 1],
                        keys,
                        mapping,
                        scaling=layer.scaling,
                        key_descale=descale,
                    )
                if parallel.attn_tp_size > 1:
                    weights = parallel.attn_tp_group.all_reduce(weights)
                    weights.div_(parallel.attn_tp_size)
                finite = torch.isfinite(weights).all(dim=1)
                errors = torch.where(finite, errors, torch.full_like(errors, 3))
    if parallel.attn_tp_size > 1:
        # Any rank-local mapping error invalidates that row on every rank.
        errors = (
            parallel.attn_tp_group.all_gather(errors, dim=0)
            .view(parallel.attn_tp_size, len(specs))
            .amax(dim=0)
        )
    info = getattr(forward_batch, "qwen_exo_customized_info", None)
    if info is None:
        info = {}
        forward_batch.qwen_exo_customized_info = info
    previous_errors = info.get(ATTENTION_DIAGNOSTIC_ERROR_KEY)
    info[ATTENTION_DIAGNOSTIC_ERROR_KEY] = (
        errors if previous_errors is None else torch.maximum(previous_errors, errors)
    )
    if weights is not None:
        weights[errors > 0] = 0
        info[f"qwen_exo_attention_layer_{layer.layer_id}"] = weights
