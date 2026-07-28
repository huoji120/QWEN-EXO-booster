import asyncio
from types import SimpleNamespace

import pytest

from qwen_exo_booster.config import PROJECT_NAME, QwenExoConfig
from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState


def server_args(tmp_path, **overrides):
    values = {
        "qwen_exo_state_dir": str(tmp_path / "state"),
        "qwen_exo_knowledge_dir": str(tmp_path / "knowledge"),
        "qwen_exo_max_internal_fanout": 32,
        "qwen_exo_max_internal_tokens": 4096,
        "qwen_exo_observer_mode": "shadow",
        "qwen_exo_enable_hybrid_prefix": True,
        "qwen_exo_enable_external_memory": True,
        "qwen_exo_enable_reference_judge": True,
        "qwen_exo_enable_capsule": True,
        "qwen_exo_enable_adaptive_refresh": False,
        "model_path": "model",
        "tp_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_has_no_external_learning_surface(tmp_path):
    config = QwenExoConfig.from_server_args(server_args(tmp_path))
    public = config.public_dict()

    assert public["project"] == PROJECT_NAME
    assert public["tp_size"] == 2
    assert all("learning" not in key.lower() for key in public)


def test_active_refresh_requires_active_observer(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        QwenExoConfig.from_server_args(
            server_args(
                tmp_path,
                qwen_exo_observer_mode="shadow",
                qwen_exo_enable_adaptive_refresh=True,
            )
        )


def test_runtime_creates_authoritative_directories(tmp_path):
    runtime = QwenExoRuntime.from_server_args(server_args(tmp_path), object())

    asyncio.run(runtime.start())
    assert runtime.state is QwenExoRuntimeState.READY
    assert runtime.config.state_directory.is_dir()
    assert runtime.config.knowledge_directory.is_dir()
    assert runtime.status()["external_learning"] is False

    asyncio.run(runtime.close())
    assert runtime.state is QwenExoRuntimeState.STOPPED
