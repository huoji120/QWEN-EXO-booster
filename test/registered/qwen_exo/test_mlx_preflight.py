from __future__ import annotations

from types import SimpleNamespace

import pytest
from qwen_exo_booster.mlx_preflight import check_mlx_environment


def _modules(*, mps_available: bool = True):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )
    )
    values = {
        "mlx.core": object(),
        "mlx_lm": object(),
        "mlx_lm.models.qwen3_5": object(),
        "mlx_lm.models.qwen3_5_moe": object(),
        "torch": torch,
        "sglang.srt.hardware_backend.mlx.model_runner": object(),
        "sglang.srt.hardware_backend.mlx.qwen_exo": object(),
        "sglang.srt.hardware_backend.mlx.native_state_bank": object(),
    }

    def load(name: str):
        return values[name]

    return load


def _versions(**overrides: str):
    values = {"mlx": "0.32.0", "mlx-lm": "0.31.3", "torch": "2.11.0"}
    values.update(overrides)
    return values.__getitem__


def test_mlx_preflight_accepts_supported_apple_silicon_runtime():
    report = check_mlx_environment(
        system="Darwin",
        machine="arm64",
        module_loader=_modules(),
        distribution_version=_versions(),
        smoke_runner=lambda _mx: {"checksum": 120.0, "device": {"arch": "gpu"}},
    )

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["metal"]["torch_mps_available"] is True
    assert report["modules"]["mlx_lm.models.qwen3_5"] == "ok"
    assert report["modules"]["mlx_lm.models.qwen3_5_moe"] == "ok"
    assert report["modules"]["sglang.srt.hardware_backend.mlx.qwen_exo"] == "ok"


def test_mlx_preflight_rejects_intel_mac_and_missing_mps():
    report = check_mlx_environment(
        system="Darwin",
        machine="x86_64",
        module_loader=_modules(mps_available=False),
        distribution_version=_versions(),
        smoke_runner=lambda _mx: {"checksum": 120.0, "device": {}},
    )

    assert report["ok"] is False
    assert any("Apple Silicon arm64" in error for error in report["errors"])
    assert "PyTorch MPS is unavailable" in report["errors"]


def test_mlx_preflight_requires_qwen35_cache_fix_versions():
    report = check_mlx_environment(
        system="Darwin",
        machine="arm64",
        module_loader=_modules(),
        distribution_version=_versions(mlx="0.31.1", **{"mlx-lm": "0.31.0"}),
        smoke_runner=lambda _mx: {"checksum": 120.0, "device": {}},
    )

    assert report["ok"] is False
    assert "mlx>=0.31.2 is required, found 0.31.1" in report["errors"]
    assert "mlx-lm>=0.31.2 is required, found 0.31.0" in report["errors"]


def test_mlx_preflight_reports_missing_qwen35_model_module():
    loader = _modules()

    def missing_dense(name: str):
        if name == "mlx_lm.models.qwen3_5":
            raise ModuleNotFoundError(name)
        return loader(name)

    report = check_mlx_environment(
        system="Darwin",
        machine="arm64",
        module_loader=missing_dense,
        distribution_version=_versions(),
        smoke_runner=lambda _mx: {"checksum": 120.0, "device": {}},
    )

    assert report["ok"] is False
    assert any(
        "cannot import mlx_lm.models.qwen3_5" in error for error in report["errors"]
    )


def test_qwen_exo_server_args_fail_when_requested_mlx_is_unavailable(
    monkeypatch,
):
    from sglang.srt import server_args as server_args_module

    monkeypatch.setenv("SGLANG_USE_MLX", "1")
    monkeypatch.setattr(server_args_module, "use_mlx", lambda: False)
    values = SimpleNamespace(enable_qwen_exo=True, device="mps")

    with pytest.raises(ValueError, match="mlx runtime is unavailable"):
        server_args_module.ServerArgs._handle_qwen_exo_runtime(values)


def test_qwen_exo_server_args_reject_mlx_with_non_mps_device(monkeypatch):
    from sglang.srt import server_args as server_args_module

    monkeypatch.setenv("SGLANG_USE_MLX", "1")
    monkeypatch.setattr(server_args_module, "use_mlx", lambda: True)
    values = SimpleNamespace(enable_qwen_exo=True, device="cuda")

    with pytest.raises(ValueError, match="requires --device mps"):
        server_args_module.ServerArgs._handle_qwen_exo_runtime(values)
