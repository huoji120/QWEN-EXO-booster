"""CPU-only guards for serial Think pre-generation artifacts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[3] / "scripts" / "qwen_exo" / "pregenerate_mid_lora_think.py"
_SPEC = importlib.util.spec_from_file_location("pregenerate_mid_lora_think", _SCRIPT)
_PREGEN = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_PREGEN)


def test_rejects_output_path_alias_before_creating_or_replacing(tmp_path):
    source = tmp_path / "prepared.jsonl"
    source.write_bytes(b"immutable source")

    with pytest.raises(ValueError, match="source_collides_with_output_artifact"):
        _PREGEN._reject_source_artifact_collisions(source.resolve(), source.resolve())

    assert source.read_bytes() == b"immutable source"


def test_rejects_existing_hardlinked_temporary_artifact(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"immutable source")
    output = tmp_path / "run"
    output.mkdir()
    partial = output / "prepared.jsonl.partial"
    try:
        partial.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")

    with pytest.raises(ValueError, match="source_collides_with_output_artifact"):
        _PREGEN._reject_source_artifact_collisions(source.resolve(), output.resolve())

    assert partial.read_bytes() == b"immutable source"


def test_rejects_resolved_symlink_artifact(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"immutable source")
    output = tmp_path / "run"
    output.mkdir()
    artifact = output / "prepared.jsonl"
    try:
        artifact.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable: {error}")

    with pytest.raises(ValueError, match="source_collides_with_output_artifact"):
        _PREGEN._reject_source_artifact_collisions(source.resolve(), output.resolve())

    assert artifact.resolve() == source.resolve()
