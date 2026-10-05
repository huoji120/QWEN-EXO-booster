"""CPU-only inspection and backward code gate for the native frozen PLE backend.

This command never starts training, performs optimizer steps, loads the complete
backbone or touches a serving process. Use the installed isolated HF environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def inspect(profile):
    import torch
    from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity, SparseNativePLEDelta
    from qwen_exo_booster.native_ple_training_backend import load_native_ple_training_model
    identity = NativePLEIdentity.from_profile(profile)
    delta = SparseNativePLEDelta(torch.tensor([identity.hash.head_offsets[0]], dtype=torch.int64), identity)
    report = load_native_ple_training_model(profile, delta, metadata_only=True)
    return {"identity_sha256": identity.fingerprint(), "checkpoint_inventory": report,
            "training_started": False, "optimizer_steps": 0, "GPU_used": False}


def verify_real_slice(profile):
    import torch
    from qwen_exo_booster.native_ple_checkpoint import IndexedCheckpoint, frozen_linear
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    with IndexedCheckpoint(profile) as source:
        weight = source.decode_linear(prefix, device="cpu", dtype=torch.float32)
        packed = source.read(prefix + ".weight")
        scale = source.read(prefix + ".weight_scale").float()
        global_scale = source.read(prefix + ".weight_scale_2").float()
        # Independent scalar expansion for a bounded actual weight slice.
        expected = torch.empty((3, 48), dtype=torch.float32)
        values = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
        for row in range(3):
            for col in range(48):
                code = (int(packed[row, col // 2]) >> (4 * (col % 2))) & 15
                expected[row, col] = values[code] * float(scale[row, col // 16]) * float(global_scale)
        torch.testing.assert_close(weight[:3, :48], expected, rtol=1e-6, atol=1e-8)
        inputs = torch.linspace(-.1, .1, 2 * weight.shape[1]).reshape(2, -1).requires_grad_()
        output = frozen_linear(inputs, source, prefix)
        direction = torch.linspace(-.2, .2, output.numel()).reshape_as(output)
        output.backward(direction)
        torch.testing.assert_close(inputs.grad, direction @ weight, rtol=1e-5, atol=1e-6)
        if not bool(torch.isfinite(inputs.grad).all()) or not bool((inputs.grad != 0).any()):
            raise AssertionError("Real checkpoint frozen-linear activation gradient is invalid")
        return {"prefix": prefix, "shape": list(weight.shape), "independent_decode_parity": True,
                "activation_gradient_parity": True, "activation_gradient_abs_max": float(inputs.grad.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--verify-code", action="store_true")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError("Backend preparation checks must not see a CUDA device")
    report = inspect(args.profile)
    if args.verify_code:
        from qwen_exo_booster.native_ple_training_backend import verify_tiny_native_ple_backward
        report["actual_weight_slice"] = verify_real_slice(args.profile)
        report["tiny_native_backward"] = verify_tiny_native_ple_backward()
        report["status"] = "code_prepared_cpu_verified"
    else:
        report["status"] = "checkpoint_inventory_verified"
    report["full_checkpoint_gpu_backward_verified"] = False
    report["32k_gpu_backward_capacity_verified"] = False
    report["serving_activation_quantization_parity"] = False
    report["training_reference"] = "original frozen packed weights, W4A16 activations; not serving W4A4"
    from qwen_exo_booster.native_ple_knowledge import sha256_file
    root = Path(__file__).resolve().parents[2]
    paths = ("python/qwen_exo_booster/native_ple_checkpoint.py",
             "python/qwen_exo_booster/native_ple_training_backend.py",
             "python/qwen_exo_booster/native_ple_knowledge.py",
             "scripts/qwen_exo/check_native_ple_backend.py")
    report["backend_code_sha256"] = {name: sha256_file(root / name) for name in paths}
    if args.output is not None:
        output = args.output.resolve()
        if output.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Backend receipts must remain outside the public source tree")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2)
            target.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
