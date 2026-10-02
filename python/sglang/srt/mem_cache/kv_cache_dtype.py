import logging
from typing import Optional

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import is_float4_e2m1fn_x2

logger = logging.getLogger(__name__)

_is_hip = is_hip()

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def configure_kv_cache_dtype(
    *,
    server_args_kv_cache_dtype: str,
    model: nn.Module,
    model_dtype: torch.dtype,
    is_draft_worker: bool,
    is_dflash: bool,
    speculative_draft_attention_backend: str,
) -> tuple[Optional[str], torch.dtype]:
    resolved_kv_cache_dtype: Optional[str] = None
    if server_args_kv_cache_dtype == "auto":
        quant_config = getattr(model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            kv_cache_dtype = fp8_dtype if _is_hip else torch.float8_e4m3fn
            resolved_kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
        else:
            kv_cache_dtype = model_dtype
    elif server_args_kv_cache_dtype == "fp8_e5m2":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e5m2
    elif server_args_kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype in ("bf16", "bfloat16"):
        kv_cache_dtype = torch.bfloat16
    elif server_args_kv_cache_dtype == "fp4_e2m1":
        raise ValueError(
            "--kv-cache-dtype=fp4_e2m1 is deprecated. "
            "Use --kv-cache-dtype=fp4_mx_block16."
        )
    elif server_args_kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
        if hasattr(torch, "float4_e2m1fn_x2"):
            kv_cache_dtype = torch.float4_e2m1fn_x2
            logger.warning(
                "%s KV Cache might lead to an accuracy drop!",
                server_args_kv_cache_dtype.upper(),
            )
        else:
            raise ValueError(
                f"--kv-cache-dtype={server_args_kv_cache_dtype} requires "
                "torch.float4_e2m1fn_x2 support. Please use PyTorch 2.8.0+ "
                "with CUDA 12.8+."
            )
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {server_args_kv_cache_dtype}.")

    if is_draft_worker and is_dflash:
        draft_kv_cache_dtype = dflash_draft_kv_cache_dtype(
            target_kv_cache_dtype=kv_cache_dtype,
            model_dtype=model_dtype,
            speculative_draft_attention_backend=speculative_draft_attention_backend,
        )
        if draft_kv_cache_dtype != kv_cache_dtype:
            logger.info(
                "DFLASH draft: overriding KV cache dtype %s -> %s.",
                kv_cache_dtype,
                draft_kv_cache_dtype,
            )
            kv_cache_dtype = draft_kv_cache_dtype

    return resolved_kv_cache_dtype, kv_cache_dtype


def dflash_draft_kv_cache_dtype(
    *,
    target_kv_cache_dtype: torch.dtype,
    model_dtype: torch.dtype,
    speculative_draft_attention_backend: Optional[str],
) -> torch.dtype:
    """KV dtype of the DFLASH draft worker's own pool, given the target's.

    Shared by the draft worker and the target's pool budget, which must size
    the draft pool in the same dtype.
    """
    # fa4 draft attention can't read the target's quantized KV (needs
    # K.dtype == Q.dtype), so it gets compute-dtype KV.
    if speculative_draft_attention_backend == "fa4":
        return model_dtype
    # The draft pool has no FP4 dequant-workspace or native-FP4 attention path.
    if is_float4_e2m1fn_x2(target_kv_cache_dtype):
        return torch.float8_e4m3fn
    return target_kv_cache_dtype
