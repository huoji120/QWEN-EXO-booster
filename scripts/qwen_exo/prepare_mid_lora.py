"""Prepare a private Flash-Next mid-layer LoRA plan; never start training.

The default inspects CPU metadata and the native checkpoint inventory. Optional
--verify-code runs only the synthetic tiny native CPU backward/adapter gate.
Neither mode loads the real backbone, uses a corpus, or performs optimizer steps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

# Set before importing torch or the native backend. This is preparation, not a
# device-selection option for an otherwise executable GPU training entrypoint.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
REPO = Path(__file__).resolve().parents[2]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(profile, output, *, sequence_length=32768, batch_size=1,
            first_layer=16, last_layer=31, rank=8, alpha=16.0,
            verify_code=False):
    """Inspect a real profile and optionally prove the tiny CPU code path."""
    output = Path(output).resolve()
    if output.is_relative_to(REPO):
        raise ValueError("Private preparation plans must remain outside the public source tree")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing preparation plan: {output}")
    if batch_size != 1:
        raise ValueError("This preparation contract supports batch size 1 only")
    if sequence_length < 2 or sequence_length > 32768:
        raise ValueError("Preparation sequence length must be between 2 and 32768")
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and finite alpha must be positive")

    profile = Path(profile).resolve()
    config_path = profile / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("model_type") != "qwen4_exp":
        raise ValueError("Expected an exact Qwen4Exp Flash-Next profile")
    text = raw["text_config"]
    layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    hc_count = int(text["hc_count"])
    if hidden < 1 or hc_count != 4:
        raise ValueError("Expected the native four-branch HC layout")
    if not 0 <= first_layer <= last_layer < layers:
        raise ValueError("LoRA layer range must be inclusive and inside the native model")
    if sequence_length > int(text.get("max_position_embeddings", sequence_length)):
        raise ValueError("Requested sequence exceeds the profile context limit")

    import torch
    from qwen_exo_booster.native_mid_lora import (
        MidLoraConfig,
        inspect_mid_lora_profile,
        verify_tiny_mid_lora_backward,
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation requires a fresh CPU-only process, without initialized CUDA")
    torch.set_num_threads(2)
    lora_config = MidLoraConfig(first_layer=first_layer, last_layer=last_layer,
                               rank=rank, alpha=alpha)
    inventory = inspect_mid_lora_profile(profile, config=lora_config)
    tiny = verify_tiny_mid_lora_backward() if verify_code else None
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")

    code_paths = (
        "scripts/qwen_exo/prepare_mid_lora.py",
        "python/qwen_exo_booster/native_mid_lora.py",
        "python/qwen_exo_booster/native_ple_checkpoint.py",
        "python/qwen_exo_booster/native_ple_training_backend.py",
        "python/qwen_exo_booster/native_ple_knowledge.py",
    )
    report = {
        "schema": 1,
        "status": ("cpu_code_verified_gpu_capacity_pending" if verify_code
                   else "metadata_prepared_cpu_code_gate_pending"),
        "training_ready": False,
        "training_started": False,
        "automatic_start": False,
        "optimizer_steps": 0,
        "GPU_used": False,
        "gpu_model_loaded": False,
        "live_service_changes": False,
        "corpus_loaded": False,
        "model_path": str(profile),
        "config_sha256": digest_file(config_path),
        "environment": {
            "python_executable": sys.executable,
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_initialized": False,
        },
        "request": {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "first_layer": first_layer,
            "last_layer_inclusive": last_layer,
            "rank": rank,
            "alpha": alpha,
        },
        "architecture": {
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "hc_count": hc_count,
            "prefix_layers": list(range(first_layer)),
            "lora_layers": list(range(first_layer, last_layer + 1)),
            "frozen_upper_layers_with_input_backward": list(range(last_layer + 1, layers)),
            "adapter_targets": [
                f"model.layers.{layer}.mlp.shared_expert.down_proj"
                for layer in range(first_layer, last_layer + 1)
            ],
            "router_and_routed_experts_frozen": True,
        },
        "checkpoint_inventory": inventory,
        "reference_precision": (
            "frozen original NVFP4 packed expert weights with BF16/F32 activations (W4A16); "
            "not bit-exact serving W4A4"
        ),
        "boundary_contract": {
            "state": "complete native HC branches plus causal positions/masks and PLE history",
            "mixed_hidden_alone_sufficient": False,
            "reusable_only_for_identical_input_and_initial_state": True,
            "fresh_sequence_only": True,
            "recurrent_or_kv_continuation_supported": False,
            "bf16_hc_payload_bytes": batch_size * sequence_length * hidden * hc_count * 2,
            "payload_estimate_excludes": [
                "position/mask/history storage", "backward activations and recomputation workspace",
                "model and adapter weights", "optimizer states", "allocator overhead",
            ],
            "payload_estimate_is_gpu_capacity_proof": False,
        },
        "tiny_native_cpu_gate": tiny,
        "full_checkpoint_gpu_backward_verified": False,
        "short_sequence_gpu_capacity_verified": False,
        "requested_sequence_gpu_capacity_verified": False,
        "serving_activation_quantization_parity": False,
        "quality_evaluated": False,
        "pending_runtime_gates": [
            "Explicitly authorized isolated GPU window; no capacity shared with live inference is assumed",
            "Real-profile short synthetic sequence forward/backward with finite LoRA gradients and frozen weights unchanged",
            "Requested sequence/batch real-profile forward/backward, measured peak memory and loss, without treating a short gate as 32K proof",
            "Separate authorization and implementation for optimizer-driven training and corpus selection",
            "Serving compatibility and matched behavioral evaluation; CPU adapter serialization is not deployment readiness",
        ],
        "code_sha256": {name: digest_file(REPO / name) for name in code_paths},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New private JSON plan outside the source tree")
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--first-layer", type=int, default=16)
    parser.add_argument("--last-layer", type=int, default=31,
                        help="Inclusive, zero-based final LoRA layer")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--verify-code", action="store_true",
                        help="Also run synthetic tiny native CPU backward and adapter round-trip; no optimizer")
    parser.add_argument("--start-training", action="store_true",
                        help="Rejected: this entrypoint is preparation-only")
    args = parser.parse_args()
    if args.start_training:
        parser.error("This CPU-only preparation command cannot start training. "
                     "Real-profile GPU backward and 32K capacity remain separate authorized gates.")
    report = prepare(args.profile, args.output, sequence_length=args.sequence_length,
                     batch_size=args.batch_size, first_layer=args.first_layer,
                     last_layer=args.last_layer, rank=args.rank, alpha=args.alpha,
                     verify_code=args.verify_code)
    print(json.dumps({key: report[key] for key in (
        "status", "training_ready", "training_started", "automatic_start",
        "optimizer_steps", "GPU_used", "request", "pending_runtime_gates",
    )}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
