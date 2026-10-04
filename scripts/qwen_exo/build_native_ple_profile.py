#!/usr/bin/env python3
"""Package unchanged Flash-Next main/MTP tensors with a frozen disk PLE source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


_METADATA_FILES = (
    "config.json", "generation_config.json", "hf_quant_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "merges.txt", "vocab.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header(path: Path) -> dict:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 128 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path}")
        encoded = source.read(length)
        if len(encoded) != length:
            raise ValueError(f"truncated safetensors header: {path}")
    records = json.loads(encoded)
    file_size = path.stat().st_size
    for name, record in records.items():
        if name == "__metadata__":
            continue
        start, end = record["data_offsets"]
        if not 0 <= start <= end or end + length + 8 > file_size:
            raise ValueError(f"incomplete tensor payload: {path}:{name}")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--ple-root", required=True)
    parser.add_argument("--engram-manifest", required=True)
    parser.add_argument("--recovered-mtp", required=True)
    parser.add_argument("--combined-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    ple_root = Path(args.ple_root).resolve()
    mtp = Path(args.recovered_mtp).resolve()
    combined_name = Path(args.combined_source).name
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"refusing existing output: {output}")
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("native PLE profile requires the Qwen4Exp architecture")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    authoritative = json.loads((source / "download-manifest.json").read_text())
    expected_files = {item["rfilename"]: item for item in authoritative["files"]}
    recovery = json.loads((source / "recovered-mtp-receipt.json").read_text())
    if not mtp.is_file():
        raise ValueError("recovered MTP is missing")
    mtp_hash = sha256(mtp)
    recorded_mtp_hash = recovery["file_sha256"]
    if recorded_mtp_hash != mtp_hash:
        raise ValueError("recovered MTP file does not match its recovery receipt")
    mtp_records = header(mtp)
    external_keys = {
        name for name in weight_map
        if ".ngram_embedding.shard_" in name or name.endswith(".ngram_embedding.weight_scale")
    }
    if not external_keys:
        raise ValueError("checkpoint index contains no native PLE table")
    expected_mtp = {name for name, filename in weight_map.items()
                    if filename == combined_name and name not in external_keys}
    if expected_mtp != set(mtp_records) - {"__metadata__"}:
        raise ValueError("recovered MTP tensor names do not cover the original combined index")
    new_map = {
        name: "recovered-mtp.safetensors" if filename == combined_name else filename
        for name, filename in weight_map.items() if name not in external_keys
    }
    inputs = {"recovered-mtp.safetensors": mtp}
    digests = {}
    for filename in sorted(set(new_map.values()) - {"recovered-mtp.safetensors"}):
        path = (source / filename).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"main checkpoint shard is incomplete: {filename}")
        expected = expected_files[filename]
        digest = sha256(path)
        if path.stat().st_size != expected["size"] or digest != expected["lfs"]["sha256"]:
            raise ValueError(f"main checkpoint shard fails authoritative SHA: {filename}")
        inputs[filename] = path
        digests[filename] = digest
    tensor_bytes = 0
    for filename, path in inputs.items():
        records = mtp_records if filename == "recovered-mtp.safetensors" else header(path)
        for name in (name for name, mapped in new_map.items() if mapped == filename):
            if name not in records:
                raise ValueError(f"checkpoint index points to a missing tensor: {filename}:{name}")
            begin, end = records[name]["data_offsets"]
            tensor_bytes += end - begin
    old_manifest_path = Path(args.engram_manifest).resolve()
    old_manifest = json.loads(old_manifest_path.read_text())
    table = old_manifest["table"]
    if table.get("dtype") != "fp8_e4m3_rowscale":
        raise ValueError("this builder expects a frozen per-row FP8 PLE source")
    shards = []
    for number in range(int(table["num_shards"])):
        path = ple_root / f"shard_{number}.safetensors"
        records = header(path)
        data, scale = records["data"], records["scale"]
        rows, width = data["shape"]
        if (data["dtype"] != "F8_E4M3" or width != table["head_dim"]
                or scale["dtype"] != "F32" or scale["shape"] != [rows]):
            raise ValueError(f"invalid per-row PLE shard: {path}")
        shards.append({"file": path.name, "rows": rows, "size": path.stat().st_size,
                       "sha256": sha256(path), "data_tensor": "data", "scale_tensor": "scale"})
    native_manifest = {
        "schema": 1, "storage": "fp8_e4m3_rowscale", "root": str(ple_root),
        "num_shards": len(shards), "rows_per_shard": int(table["rows_per_shard"]),
        "num_embeddings": sum(item["rows"] for item in shards),
        "embedding_dim": int(table["head_dim"]), "global_scale": 1.0,
        "hash": old_manifest["hash"], "source_manifest_sha256": sha256(old_manifest_path),
        "shards": shards,
    }
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        for filename in _METADATA_FILES:
            if (source / filename).is_file():
                shutil.copyfile(source / filename, temporary / filename)
        for filename, path in inputs.items():
            os.link(path, temporary / filename)
        (temporary / "native-ple.json").write_text(json.dumps(native_manifest, indent=2, sort_keys=True) + "\n")
        manifest_hash = sha256(temporary / "native-ple.json")
        config["qwen_exo_native_ple"] = {
            "schema": 1, "manifest": "native-ple.json", "manifest_sha256": manifest_hash,
        }
        (temporary / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        new_index = {**index, "weight_map": new_map,
                     "metadata": {**index.get("metadata", {}), "total_size": tensor_bytes}}
        (temporary / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2, sort_keys=True) + "\n")
        receipt = {
            "source": str(source), "original_repository": authoritative["repository"],
            "original_revision": authoritative["revision"], "source_main_files": digests,
            "recovered_mtp": {"file": mtp.name, "sha256": mtp_hash, "recovery_provenance": recovery},
            "ple_manifest_sha256": manifest_hash, "ple_source": str(ple_root),
            "external_ple_tensors": sorted(external_keys), "tensor_bytes": tensor_bytes,
            "ple_quantization": "reused per-row FP8, not byte-identical NVIDIA global-FP8",
        }
        (temporary / "native-ple-profile-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(output)


if __name__ == "__main__":
    main()
