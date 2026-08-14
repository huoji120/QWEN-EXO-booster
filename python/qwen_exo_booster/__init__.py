"""QWEN-EXO-booster integration for the SGLang runtime."""

from typing import Any

from qwen_exo_booster.config import PROJECT_NAME, QwenExoConfig

__all__ = [
    "PROJECT_NAME",
    "QwenExoConfig",
    "QwenExoRuntime",
    "QwenExoRuntimeState",
]


def __getattr__(name: str) -> Any:
    if name in {"QwenExoRuntime", "QwenExoRuntimeState"}:
        from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState

        return {
            "QwenExoRuntime": QwenExoRuntime,
            "QwenExoRuntimeState": QwenExoRuntimeState,
        }[name]
    raise AttributeError(name)
