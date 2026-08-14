#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "python"))

from qwen_exo_booster.mlx_preflight import check_mlx_environment


def main() -> int:
    report = check_mlx_environment()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
