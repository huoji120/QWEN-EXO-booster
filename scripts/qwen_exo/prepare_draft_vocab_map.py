#!/usr/bin/env python3
"""Prepare a local, identity-bound Qwen4 native MTP vocabulary map on CPU.

--token-ids-json is a plain JSON list of integer target-vocabulary IDs. Input
order and duplicates are normalized; all tokenizer special IDs and model
BOS/EOS IDs are retained. No downloads, tensor loading, or model edits occur.
The optional source URL/revision are provenance labels, never fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "qwen4-draft-vocab-v1"
MAX_MAP_BYTES = 8 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, max_bytes: int | None = None):
    with path.open("rb") as handle:
        raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("Token ID input exceeds the 8 MiB limit")
    return json.loads(raw, object_pairs_hook=_unique_object), hashlib.sha256(raw).hexdigest()


def _local_file(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def _token_id(value, vocab_size: int) -> int:
    if type(value) is not int or not 0 <= value < vocab_size:
        raise ValueError("Token IDs must be integers in [0, vocab_size), not booleans")
    return value


def _required_ids(config: dict, text_config: dict, tokenizer: dict, vocab_size: int):
    required = set()
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError("tokenizer added_tokens must be a list")
    for token in added:
        if not isinstance(token, dict) or type(token.get("special")) is not bool:
            raise ValueError("Each added token must be an object with a boolean special flag")
        if token["special"]:
            required.add(_token_id(token.get("id"), vocab_size))
    for section in (config, text_config):
        for key in ("bos_token_id", "eos_token_id"):
            value = section.get(key)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            required.update(_token_id(item, vocab_size) for item in values)
    return required


def prepare(args: argparse.Namespace) -> dict:
    config_path = _local_file(args.model_path, "config.json")
    tokenizer_root = args.tokenizer_path
    if tokenizer_root is None:
        tokenizer_root = args.model_path if args.model_path.is_dir() else args.model_path.parent
    tokenizer_path = _local_file(tokenizer_root, "tokenizer.json")
    sources = (config_path, tokenizer_path, args.token_ids_json)
    for source in sources:
        if args.output.resolve() == source.resolve() or (
            args.output.exists() and source.exists() and args.output.samefile(source)
        ):
            raise ValueError("Output must not replace a model, tokenizer, or token ID source file")
    if args.output.exists() and not args.overwrite:
        raise ValueError("Output already exists; pass --overwrite to replace it")

    config, config_sha = _read_json(config_path)
    tokenizer, tokenizer_sha = _read_json(tokenizer_path)
    source_ids, source_sha = _read_json(args.token_ids_json, max_bytes=MAX_MAP_BYTES)
    if not isinstance(config, dict) or not isinstance(tokenizer, dict):
        raise ValueError("Model config and tokenizer must be JSON objects")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen4_exp_text":
        raise ValueError("Model text_config.model_type must be qwen4_exp_text")
    dimensions = {}
    for key in ("vocab_size", "hidden_size"):
        value = text_config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Model {key} must be a positive integer")
        dimensions[key] = value
    vocab_size = dimensions["vocab_size"]
    if not isinstance(source_ids, list):
        raise ValueError("--token-ids-json must contain a plain JSON list of integer IDs")
    selected = {_token_id(value, vocab_size) for value in source_ids}
    selected.update(_required_ids(config, text_config, tokenizer, vocab_size))
    if not 0 < len(selected) < vocab_size:
        raise ValueError("Selected vocabulary must be nonempty and smaller than the full vocabulary")
    provenance = {"source_sha256": source_sha}
    for key in ("source_url", "source_revision"):
        value = getattr(args, key)
        if value is not None:
            if not value.strip():
                raise ValueError(f"{key} must be nonempty when supplied")
            provenance[key] = value
    return {
        "schema": SCHEMA,
        "model_config_sha256": config_sha,
        "tokenizer_sha256": tokenizer_sha,
        **dimensions,
        "token_ids": sorted(selected),
        "provenance": provenance,
    }


def _write_map(path: Path, document: dict, *, overwrite: bool) -> None:
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(raw) > MAX_MAP_BYTES:
        raise ValueError("Prepared map exceeds the runtime 8 MiB limit")
    if not overwrite:
        # Exclusive creation also protects against an output appearing after validation.
        with path.open("xb") as handle:
            handle.write(raw)
        return
    # Validation/serialization finish before an existing map is replaced.
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            staged = Path(handle.name)
            handle.write(raw)
        os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Local model directory or config.json")
    parser.add_argument("--tokenizer-path", type=Path, help="Local tokenizer directory or tokenizer.json; defaults to model directory")
    parser.add_argument("--token-ids-json", type=Path, required=True, help="Local plain JSON list of integer target-vocabulary IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", help="Optional provenance URL (not downloaded)")
    parser.add_argument("--source-revision", help="Optional provenance revision")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output map, never an input file")
    args = parser.parse_args()
    try:
        document = prepare(args)
        _write_map(args.output, document, overwrite=args.overwrite)
    except (OSError, ValueError, RecursionError) as exc:
        parser.error(str(exc))
    print(f"Prepared {len(document['token_ids'])}/{document['vocab_size']} target-vocabulary IDs")


if __name__ == "__main__":
    main()
