"""Model-bound SM120 fast preset. Importing this module never loads a model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SM120_PROFILE_KEY = "qwen_exo_sm120_fast"
SM120_PROFILE_MARKER = {
    "schema": 1,
    "precision": "w4a16-int8",
    "sampling": "exact-rs-truncated-v1",
}


def _value(config: Any, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def sm120_profile_enabled(config: Any) -> bool:
    root = _value(config, SM120_PROFILE_KEY)
    text = _value(config, "text_config")
    nested = _value(text, SM120_PROFILE_KEY) if text is not None else None
    if root is not None and nested is not None and root != nested:
        raise ValueError("SM120 fast profile root/text markers disagree")
    marker = root if root is not None else nested
    if marker is None:
        return False
    if not isinstance(marker, dict) or marker != SM120_PROFILE_MARKER or type(marker.get("schema")) is not int:
        raise ValueError("Unsupported qwen_exo_sm120_fast profile marker")
    return True


def _argument_value(arguments: list[str], option: str):
    value = None
    for index, token in enumerate(arguments):
        if token == option and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif token.startswith(option + "="):
            value = token.split("=", 1)[1]
    return value


def _replace_argument(arguments: list[str], option: str, value: str) -> list[str]:
    result = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    result.extend((option, value))
    return result


def configure_sm120_profile_arguments(arguments: list[str]) -> list[str]:
    """Apply a marked model's coupled numerical contract after managed settings."""
    model_path = _argument_value(arguments, "--model-path")
    if model_path is None:
        if "--qwen-exo-sm120-fast" in arguments:
            raise ValueError("SM120 fast profile requires a marked --model-path")
        return arguments
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if not sm120_profile_enabled(config):
        if "--qwen-exo-sm120-fast" in arguments:
            raise ValueError("--qwen-exo-sm120-fast requires an independent marked model profile")
        return arguments
    from qwen_exo_booster.fingerprint import validate_qwen_exo_config

    validate_qwen_exo_config(config)
    if "--enable-qwen-exo" not in arguments:
        raise ValueError("SM120 fast profile requires --enable-qwen-exo for state isolation")
    if config.get("qwen_exo_runtime_lora") is not None:
        raise ValueError("SM120 fast profile does not support a runtime LoRA adapter")
    text = config.get("text_config", config)
    if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        raise ValueError("SM120 fast INT8 head requires untied embeddings")
    result = list(arguments)
    values = {
        "--quantization": "modelopt_mixed",
        "--dtype": "bfloat16",
        "--qwen4-ple-disk-path": model_path,
        "--moe-runner-backend": "marlin",
        "--speculative-algorithm": "EAGLE",
        "--speculative-num-steps": "3",
        "--speculative-eagle-topk": "1",
        "--speculative-num-draft-tokens": "4",
        "--speculative-draft-model-path": model_path,
        "--speculative-draft-model-quantization": "modelopt_mixed",
        "--speculative-moe-runner-backend": "triton",
        "--speculative-accept-threshold-single": "1.0",
        "--speculative-accept-threshold-acc": "1.0",
    }
    for option, value in values.items():
        result = _replace_argument(result, option, value)
    for flag in ("--qwen-exo-sm120-fast", "--speculative-use-rejection-sampling", "--disable-shared-experts-fusion"):
        if flag not in result:
            result.append(flag)
    return result


def enforce_sm120_profile_runtime(config: Any) -> bool:
    """Fail before allocations when the preset and numerical runtime disagree."""
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    enabled = sm120_profile_enabled(config)
    if enabled != bool(getattr(args, "qwen_exo_sm120_fast", False)):
        raise ValueError("SM120 fast model marker and launch flag disagree")
    if not enabled:
        return False
    validate_sm120_profile_args(args)
    text = _value(config, "text_config", config)
    if _value(text, "tie_word_embeddings", _value(config, "tie_word_embeddings", False)):
        raise ValueError("SM120 fast INT8 head requires untied embeddings")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("SM120 fast profile requires a CUDA SM120 GPU")
    if torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count < 188:
        raise ValueError("SM120 fast HC requires the PRO6000 resident-grid geometry (at least 188 SMs)")
    return True


def validate_sm120_profile_args(args: Any) -> None:
    if not bool(getattr(args, "qwen_exo_sm120_fast", False)):
        return
    if not args.enable_qwen_exo:
        raise ValueError("SM120 fast profile requires QWEN-EXO state isolation")
    for name in ("tp_size", "pp_size", "dp_size", "ep_size", "moe_dp_size", "attn_cp_size", "dcp_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"SM120 fast profile requires {name}=1")
    if args.quantization != "modelopt_mixed" or args.moe_runner_backend != "marlin":
        raise ValueError("SM120 fast profile requires ModelOpt mixed weights and Marlin experts")
    if str(args.speculative_algorithm).upper() != "EAGLE":
        raise ValueError("SM120 fast profile requires native EAGLE MTP")
    if (args.speculative_num_steps, args.speculative_eagle_topk, args.speculative_num_draft_tokens) != (3, 1, 4):
        raise ValueError("SM120 fast profile requires EAGLE steps=3, topk=1, verify width=4")
    if not args.speculative_use_rejection_sampling:
        raise ValueError("SM120 fast profile requires exact rejection sampling")
    if (args.speculative_accept_threshold_single, args.speculative_accept_threshold_acc) != (1.0, 1.0):
        raise ValueError("Exact rejection sampling requires both accept thresholds to be 1")
    if bool(getattr(args, "enable_lora", False)):
        raise ValueError("SM120 fast profile does not support runtime LoRA")
    if bool(getattr(args, "qwen_exo_moe_cpu_offload", False)):
        raise ValueError("SM120 fast profile requires resident GPU experts")
    if bool(getattr(args, "enable_fp32_lm_head", False)):
        raise ValueError("SM120 fast profile requires BF16 LM-head computation")
    if getattr(args, "dtype", "bfloat16") not in {"bfloat16", "auto"}:
        raise ValueError("SM120 fast profile requires BF16 target activations")
