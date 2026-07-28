from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from qwen_exo_booster.config import PROJECT_NAME, QwenExoConfig


class QwenExoRuntimeState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class QwenExoRuntime:
    """Owns QWEN-EXO services inside the SGLang HTTP process.

    Model work is submitted through ``tokenizer_manager`` so internal work enters
    SGLang's scheduler instead of recursively calling the HTTP server.
    """

    def __init__(self, config: QwenExoConfig, tokenizer_manager: Any):
        self.config = config
        self.tokenizer_manager = tokenizer_manager
        self.state = QwenExoRuntimeState.CREATED
        self._lifecycle_lock = asyncio.Lock()

    @classmethod
    def from_server_args(
        cls, server_args: Any, tokenizer_manager: Any
    ) -> QwenExoRuntime:
        return cls(QwenExoConfig.from_server_args(server_args), tokenizer_manager)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.state is QwenExoRuntimeState.READY:
                return
            if self.state not in {
                QwenExoRuntimeState.CREATED,
                QwenExoRuntimeState.STOPPED,
            }:
                raise RuntimeError(
                    f"Cannot start QWEN-EXO runtime from {self.state.value}"
                )
            self.state = QwenExoRuntimeState.STARTING
            try:
                self.config.state_directory.mkdir(parents=True, exist_ok=True)
                self.config.knowledge_directory.mkdir(parents=True, exist_ok=True)
                self.state = QwenExoRuntimeState.READY
            except Exception:
                self.state = QwenExoRuntimeState.FAILED
                raise

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self.state in {
                QwenExoRuntimeState.CREATED,
                QwenExoRuntimeState.STOPPED,
            }:
                self.state = QwenExoRuntimeState.STOPPED
                return
            if self.state is QwenExoRuntimeState.STOPPING:
                return
            self.state = QwenExoRuntimeState.STOPPING
            self.state = QwenExoRuntimeState.STOPPED

    def status(self) -> dict[str, Any]:
        return {
            **self.config.public_dict(),
            "runtime_state": self.state.value,
            "scheduler_native_internal_jobs": True,
            "external_learning": False,
        }

    def health(self) -> dict[str, Any]:
        return {
            "project": PROJECT_NAME,
            "status": "ok" if self.state is QwenExoRuntimeState.READY else "unavailable",
            "runtime_state": self.state.value,
        }
