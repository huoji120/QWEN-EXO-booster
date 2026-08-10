#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

import torch

EXPECTED_TORCH_CUDA = "12.6"
MIN_DEVICE_MEMORY_BYTES = 48_000_000_000


def main() -> int:
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    errors = []

    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
    if torch.cuda.device_count() != 2:
        errors.append(f"expected exactly 2 GPUs, found {torch.cuda.device_count()}")
    if str(torch.version.cuda or "") != EXPECTED_TORCH_CUDA:
        errors.append(
            f"expected CUDA {EXPECTED_TORCH_CUDA}, found {torch.version.cuda!r}"
        )

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        device_report = {
            "index": index,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": list(capability),
            "bf16_supported": torch.cuda.is_bf16_supported(index),
        }
        report["devices"].append(device_report)
        if capability != (8, 9):
            errors.append(
                f"GPU {index} expected SM89, found SM{capability[0]}{capability[1]}"
            )
        if not device_report["bf16_supported"]:
            errors.append(f"GPU {index} does not support BF16")
        if properties.total_memory < MIN_DEVICE_MEMORY_BYTES:
            errors.append(
                f"GPU {index} requires at least {MIN_DEVICE_MEMORY_BYTES} bytes, "
                f"found {properties.total_memory}"
            )

        if torch.cuda.is_available():
            device = torch.device("cuda", index)
            left = torch.arange(4096, dtype=torch.bfloat16, device=device).reshape(
                64, 64
            )
            result = left @ left.T
            if not torch.isfinite(result.float()).all().item():
                errors.append(f"GPU {index} BF16 matmul returned non-finite values")
            del left, result

    report["ok"] = not errors
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
