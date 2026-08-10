#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

BASIC_MODULES = (
    "qwen_exo_booster",
    "qwen_exo_booster.runtime",
    "sglang",
)
FULL_MODULES = BASIC_MODULES + (
    "sglang.srt.server_args",
    "sglang.srt.entrypoints.http_server",
    "sglang.srt.entrypoints.openai.serving_responses",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.scheduler_components.batch_result_processor",
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.layers.attention.vision",
    "sglang.srt.models.qwen3_5",
    "sglang.srt.models.qwen3_vl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the reviewed source overlay")
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Skip GPU-driver-dependent SGLang modules during image build",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = BASIC_MODULES if args.basic else FULL_MODULES
    imported = {name: importlib.import_module(name) for name in modules}
    source = Path(imported["sglang"].__file__).resolve()
    if Path("/sgl-workspace/sglang") not in source.parents:
        raise RuntimeError(f"SGLang did not load from the reviewed overlay: {source}")

    import torch

    report = {
        "ok": True,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "sglang_source": str(source),
        "mode": "basic" if args.basic else "full",
        "modules": list(imported),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
