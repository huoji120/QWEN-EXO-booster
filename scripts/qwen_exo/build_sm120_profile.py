#!/usr/bin/env python3
"""Create an independent PRO6000 fast view without changing checkpoint weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from qwen_exo_booster.fingerprint import ModelIdentity, validate_qwen_exo_config
from qwen_exo_booster.sm120_profile import SM120_PROFILE_KEY, SM120_PROFILE_MARKER, sm120_profile_enabled

_METADATA = (
    "model.safetensors.index.json", "generation_config.json", "hf_quant_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    "native-ple.json", "download-manifest.json", "recovered-mtp-receipt.json",
    "native-ple-profile-receipt.json",
)


def build_profile(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"Refusing existing profile destination: {output}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_qwen_exo_config(config)
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("PRO6000 fast profile requires native Qwen4Exp Flash-Next")
    if sm120_profile_enabled(config):
        raise ValueError("Source is already an SM120 fast profile")
    if config.get("qwen_exo_runtime_lora") is not None:
        raise ValueError("PRO6000 fast profile cannot inherit a runtime LoRA")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("PRO6000 fast profile requires the native ModelOpt mixed checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    filenames = set(index["weight_map"].values())
    for name in filenames:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (source / relative).is_file():
            raise ValueError(f"Invalid or incomplete indexed checkpoint shard: {name}")
    config[SM120_PROFILE_KEY] = dict(SM120_PROFILE_MARKER)
    if isinstance(config.get("text_config"), dict):
        config["text_config"][SM120_PROFILE_KEY] = dict(SM120_PROFILE_MARKER)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        for name in sorted(filenames | {p for p in _METADATA if (source / p).is_file()}):
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name in filenames:
                os.link(source / name, destination)
            else:
                # Editable metadata must never alias the original checkpoint.
                shutil.copyfile(source / name, destination)
        (temporary / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_identity = ModelIdentity.from_path(source)
        identity = ModelIdentity.from_path(temporary)
        if identity.fingerprint == source_identity.fingerprint:
            raise ValueError("Fast profile did not obtain an independent model fingerprint")
        receipt = {"schema": 1, "source_model_fingerprint": source_identity.fingerprint,
            "model_fingerprint": identity.fingerprint, "profile": dict(SM120_PROFILE_MARKER),
            "linked_weight_files": len(filenames), "checkpoint_weights_modified": False}
        (temporary / "sm120-profile-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = build_profile(args.source, args.output)
    print(json.dumps({"profile": str(args.output.resolve()), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
