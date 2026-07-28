from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from qwen_exo_booster.config import PROJECT_NAME
from qwen_exo_booster.runtime import QwenExoRuntimeState

router = APIRouter(prefix="/qwen-exo", tags=[PROJECT_NAME])


def _runtime(request: Request):
    runtime = getattr(request.app.state, "qwen_exo_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="QWEN-EXO runtime is disabled")
    return runtime


@router.get("/status")
async def status(request: Request):
    runtime = getattr(request.app.state, "qwen_exo_runtime", None)
    if runtime is None:
        return {
            "project": PROJECT_NAME,
            "enabled": False,
            "runtime_state": "disabled",
            "external_learning": False,
        }
    return {"enabled": True, **runtime.status()}


@router.get("/health")
async def health(request: Request):
    runtime = _runtime(request)
    payload = runtime.health()
    if runtime.state is not QwenExoRuntimeState.READY:
        raise HTTPException(status_code=503, detail=payload)
    return payload
