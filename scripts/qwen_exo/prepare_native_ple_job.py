"""Prepare/check a native PLE knowledge job; never start training or inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def backend_status(backend_python=None):
    if backend_python is not None:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [str(backend_python), "-B", __file__, "--backend-probe"],
            env=environment, capture_output=True, text=True, check=True, timeout=120,
        )
        return json.loads(result.stdout)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )
    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    model = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("qwen4_exp")
    conditional = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get("qwen4_exp")
    config = CONFIG_MAPPING_NAMES.get("qwen4_exp")
    return {"transformers_version": importlib.metadata.version("transformers"),
            "python_executable": sys.executable,
            "modelopt_mixed_loader_registered": "modelopt" in AUTO_QUANTIZER_MAPPING,
            "config_class": config, "causal_model_class": model,
            "conditional_model_class": conditional,
            "native_architecture_registered": bool(config and (model or conditional)),
            "native_backbone_gradient_gate": "not_run",
            "quantized_backbone_autograd_verified": False}


def prepare(data, output, backend_python=None):
    from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_not_started" or manifest.get("training_started"):
        raise ValueError("Native data must be prepared and training must not have started")
    identity = NativePLEIdentity.from_profile(manifest["model_path"])
    if identity.fingerprint() != manifest["native_identity_sha256"]:
        raise ValueError("Native model/reader/table identity changed after data preparation")
    backend = backend_status(backend_python)
    reasons = []
    if not backend["native_architecture_registered"]:
        reasons.append("Installed Transformers has no exact Qwen4Exp differentiable model/config registration")
    if not backend["modelopt_mixed_loader_registered"]:
        reasons.append("Exact checkpoint modelopt MIXED_PRECISION loader is absent; generic NVFP4 is not interchangeable")
    reasons.append("Real native NVFP4 frozen-backbone-to-PLE gradient gate has not run; serving no_grad is not a training backend")
    code_paths = (
        "python/qwen_exo_booster/native_ple_knowledge.py",
        "scripts/qwen_exo/prepare_native_ple_data.py",
        "scripts/qwen_exo/prepare_native_ple_job.py",
    )
    output = output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError("Private job plans must remain outside the public source tree")
    report = {
        "schema": 1, "status": "blocked_native_training_backend",
        "training_started": False, "automatic_start": False,
        "requires_explicit_training_start": True,
        "live_service_changes": False, "gpu_model_loaded": False,
        "data_directory": str(data.resolve()),
        "data_manifest_sha256": digest_file(manifest_path),
        "model_identity": identity.to_dict(), "model_identity_sha256": identity.fingerprint(),
        "tokenizer": manifest["tokenizer"], "source_records": manifest["source_records"],
        "splits": manifest["splits"], "split_provenance": manifest["split_provenance"],
        "backend": backend, "blocking_prerequisites": reasons,
        "adaptation": {
            "trainable": "separate sparse native PLE delta rows only",
            "frozen": ["native backbone", "original disk PLE", "native PLE reader"],
            "native_layer_id": identity.layer_id, "row_head_dim": identity.head_dim,
            "epochs": 1, "seed": 20261005,
            "optimizer": "sparse Adam; use only after the real native gradient gate",
            "learning_rate": 1e-4, "sequence_tokens": manifest["max_sequence"],
            "causal_loss": "logits[:-1] versus labels[1:]; -100 ignored",
            "no_old_27b_reader": True,
            "32k_gpu_backward_capacity_verified": False,
        },
        "evaluation": {
            "modes": ["off", "real", "shuffled"],
            "heldout": manifest["evaluation"]["matched_inputs_labels_files"],
            "shuffle_seed": manifest["evaluation"]["shuffle_seed"],
            "shuffle_contract": "delta values permuted within native hash heads; frozen base PLE never shuffled",
            "teacher_forced_nll_is_auxiliary": True,
            "behavior_required": ["heldout task success", "knowledge-dependent decision accuracy",
                                  "tool-call validity", "regression versus knowledge off"],
            "quality_evaluated": False,
        },
        "code_sha256": {name: digest_file(REPO / name) for name in code_paths},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    return report


def check(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != 1 or report.get("training_started") or report.get("automatic_start"):
        raise ValueError("Native job has an invalid execution state")
    data = Path(report["data_directory"])
    if digest_file(data / "manifest.json") != report["data_manifest_sha256"]:
        raise ValueError("Native prepared data changed")
    for name, sha in report["code_sha256"].items():
        if digest_file(REPO / name) != sha:
            raise ValueError("Native preparation code changed")
    from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if NativePLEIdentity.from_profile(manifest["model_path"]).fingerprint() != report["model_identity_sha256"]:
        raise ValueError("Native model/reader/table changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--backend-probe", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--backend-python", type=Path)
    args = parser.parse_args()
    if args.backend_probe:
        print(json.dumps(backend_status(), sort_keys=True))
        return
    if args.plan is None:
        parser.error("--prepare/--check requires --plan")
    if args.prepare and args.data is None:
        parser.error("--prepare requires --data")
    report = prepare(args.data.resolve(), args.plan, args.backend_python) if args.prepare else check(args.plan)
    print(json.dumps({key: report[key] for key in (
        "status", "training_started", "automatic_start", "gpu_model_loaded",
        "source_records", "splits", "backend", "blocking_prerequisites")}, sort_keys=True))


if __name__ == "__main__":
    main()
