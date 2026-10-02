import hashlib
import json
from types import SimpleNamespace

import pytest

from qwen_exo_booster.runtime_lora import RuntimeLoRA


def profile(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":4}')
    (adapter / "adapter_model.bin").write_bytes(b"synthetic-adapter")
    manifest = {
        "schema": 1, "name": "test-adapter", "path": "adapter",
        "file_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()},
    }
    (tmp_path / "config.json").write_text(json.dumps({"qwen_exo_runtime_lora": manifest}))
    return RuntimeLoRA.from_model_path(tmp_path)


def test_profile_routes_missing_adapter_for_all_batch_rows(tmp_path):
    adapter = profile(tmp_path)
    assert adapter.bind_request(None) == "test-adapter"
    assert adapter.bind_request([None, "test-adapter", None]) == ["test-adapter"] * 3
    with pytest.raises(ValueError):
        adapter.bind_request(["test-adapter", "another"])
    with pytest.raises(ValueError):
        adapter.bind_request("")


def test_profile_requires_exact_preloaded_adapter(tmp_path):
    adapter = profile(tmp_path)
    ref = SimpleNamespace(lora_name=adapter.name, lora_path=str(adapter.path))
    adapter.validate_loaded(True, [ref])
    for enabled, refs in [(False, [ref]), (True, []), (True, [ref, ref])]:
        with pytest.raises(ValueError):
            adapter.validate_loaded(enabled, refs)
    with pytest.raises(ValueError):
        adapter.validate_loaded(True, [SimpleNamespace(lora_name="other", lora_path=str(adapter.path))])


def test_profile_rejects_modified_adapter_weights(tmp_path):
    adapter = profile(tmp_path)
    (adapter.path / "adapter_model.bin").write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="file changed"):
        RuntimeLoRA.from_model_path(tmp_path)


def test_plain_base_profile_has_no_implicit_adapter(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert RuntimeLoRA.from_model_path(tmp_path) is None
