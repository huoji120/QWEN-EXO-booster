from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
from collections.abc import Callable
from typing import Any

_MIN_MLX_VERSION = (0, 31, 2)
_MIN_MLX_LM_VERSION = (0, 31, 2)
_REQUIRED_MODULES = (
    "mlx.core",
    "mlx_lm",
    "mlx_lm.models.qwen3_5",
    "mlx_lm.models.qwen3_5_moe",
    "torch",
    "sglang.srt.hardware_backend.mlx.model_runner",
    "sglang.srt.hardware_backend.mlx.qwen_exo",
    "sglang.srt.hardware_backend.mlx.native_state_bank",
)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    observed = _version_tuple(value)
    width = max(len(observed), len(minimum))
    return observed + (0,) * (width - len(observed)) >= minimum + (0,) * (
        width - len(minimum)
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _mlx_smoke(mx: Any) -> dict[str, Any]:
    source = mx.arange(16, dtype=mx.float32).reshape((4, 4))
    result = source @ mx.eye(4, dtype=mx.float32)
    checksum = mx.sum(result)
    mx.eval(result, checksum)
    observed = float(checksum.item())
    if abs(observed - 120.0) > 1e-4:
        raise RuntimeError(f"MLX matrix smoke returned checksum={observed!r}")
    device_info = mx.device_info()
    return {
        "checksum": observed,
        "device": {str(key): str(value) for key, value in device_info.items()},
    }


def check_mlx_environment(
    *,
    system: str | None = None,
    machine: str | None = None,
    module_loader: Callable[[str], Any] = importlib.import_module,
    distribution_version: Callable[[str], str] = _distribution_version,
    smoke_runner: Callable[[Any], dict[str, Any]] = _mlx_smoke,
) -> dict[str, Any]:
    """Validate the Apple Silicon runtime required by QWEN-EXO's MLX path."""

    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    report: dict[str, Any] = {
        "platform": {"system": system, "machine": machine},
        "modules": {},
        "versions": {},
        "metal": {},
    }
    errors: list[str] = []
    warnings: list[str] = []

    if system != "Darwin":
        errors.append(f"MLX requires macOS (Darwin), found {system!r}")
    if machine not in {"arm64", "aarch64"}:
        errors.append(f"MLX requires Apple Silicon arm64, found {machine!r}")

    imported: dict[str, Any] = {}
    for name in _REQUIRED_MODULES:
        try:
            imported[name] = module_loader(name)
            report["modules"][name] = "ok"
        except Exception as exc:  # noqa: BLE001 - report every broken native import
            report["modules"][name] = f"{type(exc).__name__}: {exc}"
            errors.append(f"cannot import {name}: {exc}")

    versions = {
        "mlx": distribution_version("mlx"),
        "mlx-lm": distribution_version("mlx-lm"),
        "torch": distribution_version("torch"),
    }
    report["versions"] = versions
    for name, minimum in (
        ("mlx", _MIN_MLX_VERSION),
        ("mlx-lm", _MIN_MLX_LM_VERSION),
    ):
        value = versions[name]
        if value == "unknown":
            errors.append(f"cannot determine installed {name} version")
        elif not _version_at_least(value, minimum):
            wanted = ".".join(str(part) for part in minimum)
            errors.append(f"{name}>={wanted} is required, found {value}")

    torch = imported.get("torch")
    if torch is not None:
        try:
            mps_available = bool(torch.backends.mps.is_available())
        except Exception as exc:  # noqa: BLE001 - backend probes are diagnostic
            mps_available = False
            errors.append(f"cannot query PyTorch MPS availability: {exc}")
        report["metal"]["torch_mps_available"] = mps_available
        if not mps_available:
            errors.append("PyTorch MPS is unavailable")

    mx = imported.get("mlx.core")
    if mx is not None:
        try:
            report["metal"].update(smoke_runner(mx))
        except Exception as exc:  # noqa: BLE001 - backend probes are diagnostic
            errors.append(f"MLX Metal smoke failed: {exc}")

    try:
        import psutil

        total_gib = psutil.virtual_memory().total / (1024**3)
        report["memory_gib"] = round(total_gib, 2)
        if total_gib < 32:
            warnings.append(
                "less than 32 GiB unified memory: the verified 27B/35B models "
                "may not fit even with 4-bit weights and a reduced context"
            )
    except Exception:  # noqa: BLE001 - memory reporting is optional
        warnings.append("physical memory size could not be determined")

    report["ok"] = not errors
    report["errors"] = errors
    report["warnings"] = warnings
    return report


__all__ = ["check_mlx_environment"]
