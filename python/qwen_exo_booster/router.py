from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from typing import Literal

from qwen_exo_booster.activation_training import (
    COMBINED_EDITOR_NAME,
    ActivationTrainingError,
    ActivationTrainingStore,
)

from qwen_exo_booster.config import PROJECT_NAME
from qwen_exo_booster.tags import TagValidationError, normalize_tags
from qwen_exo_booster.document_ingest import (
    KnowledgeIngestError,
    preview_knowledge_upload,
    validate_upload_batch,
)
from qwen_exo_booster.trajectory_store import (
    TrajectoryStore,
    TrajectoryStoreError,
    parse_trajectory_upload,
)
from qwen_exo_booster.recall_visualization import render_recall_trace_html
from qwen_exo_booster.runtime import QwenExoRuntimeState
from qwen_exo_booster.tensor_bank import TensorBankCompileError
from qwen_exo_booster.service_config import (
    ServiceConfigError,
    ServiceConfigStore,
    request_managed_restart,
)

_STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"
_APP_DIRECTORY = _STATIC_DIRECTORY / "app"
router = APIRouter(prefix="/qwen-exo", tags=[PROJECT_NAME])
compat_router = APIRouter(tags=[PROJECT_NAME])


class KnowledgeWriteRequest(BaseModel):
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list, max_length=16)


class KnowledgeUploadItem(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=6_000_000)


class KnowledgeIngestRequest(BaseModel):
    files: list[KnowledgeUploadItem] = Field(min_length=1, max_length=20)


class ServiceConfigWriteRequest(BaseModel):
    values: dict[str, object]
    expected_revision: str = Field(min_length=1)


class SourceSelectionRequest(BaseModel):
    lane: Literal["knowledge", "policydata"]
    relative_paths: list[str] = Field(min_length=1, max_length=1000)


class ReflectionSelectionRequest(BaseModel):
    conversation_keys: list[str] = Field(min_length=1, max_length=1000)


def _runtime(request: Request):
    runtime = getattr(request.app.state, "qwen_exo_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="QWEN-EXO 运行时未启用")
    return runtime


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
@router.get("/admin", include_in_schema=False)
async def workspace_console():
    return FileResponse(
        _APP_DIRECTORY / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/assets/{asset_path:path}", include_in_schema=False)
async def workspace_asset(asset_path: str):
    root = (_APP_DIRECTORY / "assets").resolve()
    candidate = (root / asset_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="未找到控制台资源")
    return FileResponse(
        candidate, headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )


@compat_router.get("/recall-trace", include_in_schema=False)
async def recall_trace_console(request: Request):
    payload = await asyncio.to_thread(_runtime(request).recall_trace, max_turns=10)
    html = await asyncio.to_thread(render_recall_trace_html, payload)
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-store"},
    )


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


@router.get("/service-config")
async def get_service_config():
    try:
        return ServiceConfigStore.from_environment().public_document()
    except ServiceConfigError as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": exc.public_dict()},
        )


@router.put("/service-config", status_code=202)
async def put_service_config(payload: ServiceConfigWriteRequest, request: Request):
    store = ServiceConfigStore.from_environment()
    if not store.managed_restart:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "restart_unmanaged",
                    "message": "当前服务未由自动重启策略托管，配置未写入",
                }
            },
        )
    try:
        document = store.update(
            payload.values,
            expected_revision=payload.expected_revision,
        )
        request_managed_restart()
    except ServiceConfigError as exc:
        status_code = 409 if exc.code == "revision_conflict" else 422
        return JSONResponse(
            status_code=status_code,
            content={"detail": exc.public_dict()},
        )
    runtime = getattr(request.app.state, "qwen_exo_runtime", None)
    if runtime is not None:
        runtime.telemetry.emit(
            "admin",
            "service_config.restart_requested",
            {"revision": document["revision"]},
        )
    return {**document, "restart_requested": True}


@router.get("/health")
async def health(request: Request):
    runtime = _runtime(request)
    payload = runtime.health()
    if runtime.state is not QwenExoRuntimeState.READY:
        return JSONResponse(status_code=503, content={"detail": payload})
    return payload


@router.get("/telemetry")
async def telemetry(
    request: Request,
    request_id: str | None = None,
    limit: int = Query(default=256, ge=1, le=1000),
):
    runtime = _runtime(request)
    return {
        "events": runtime.telemetry_events(request_id, limit=limit),
        "redacted": runtime.config.telemetry_text_mode == "off",
    }


@router.post("/telemetry/clear")
async def clear_telemetry(request: Request):
    runtime = _runtime(request)
    try:
        runtime.telemetry.clear()
    except OSError as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"清理遥测失败：{exc}"},
        )
    return {"cleared": True, "persistence": runtime.telemetry.persistence_status()}


@router.get("/request-traces")
async def request_traces(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    q: str = Query(default="", max_length=256),
):
    runtime = _runtime(request)
    return await asyncio.to_thread(runtime.request_traces, limit=limit, query=q)


@router.get("/recall-trace")
async def recall_trace(request: Request, limit: int = Query(default=10, ge=1, le=100)):
    return await asyncio.to_thread(_runtime(request).recall_trace, max_turns=limit)


@compat_router.get("/v1/recall-trace", include_in_schema=False)
async def recall_trace_compat(
    request: Request, limit: int = Query(default=10, ge=1, le=100)
):
    return await asyncio.to_thread(_runtime(request).recall_trace, max_turns=limit)


@router.delete("/recall-trace")
@compat_router.delete("/v1/recall-trace", include_in_schema=False)
async def clear_recall_trace(request: Request):
    return await _runtime(request).clear_recall_trace()


@router.get("/telemetry/stream")
async def telemetry_stream(
    request: Request,
    after: int = Query(default=-1, ge=-1),
):
    runtime = _runtime(request)
    last_event_id = request.headers.get("last-event-id")
    cursor = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

    async def generate():
        nonlocal cursor
        while not await request.is_disconnected():
            events = runtime.telemetry.events_after(cursor, limit=256)
            if events:
                for event in events:
                    cursor = event.event_id
                    data = json.dumps(
                        event.to_dict(), ensure_ascii=False, separators=(",", ":")
                    )
                    yield f"id: {event.event_id}\nevent: trace\ndata: {data}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/knowledge")
async def list_knowledge(request: Request, q: str = Query(default="", max_length=256)):
    runtime = _runtime(request)
    return {
        "source_digest": runtime.knowledge.snapshot.source_digest,
        "documents": runtime.knowledge_documents(query=q),
    }


@router.get("/policydata")
async def list_policy_data(
    request: Request, q: str = Query(default="", max_length=256)
):
    runtime = _runtime(request)
    return {
        "source_digest": runtime.policy_data.snapshot.source_digest,
        "documents": runtime.policy_data_documents(query=q),
        "always_on": bool(
            runtime.config.feature_flags.policy_data
            and getattr(runtime.policy_data.snapshot, "documents", ())
        ),
        "semantic_eligibility_required": False,
        "qk_relevance_required": False,
        "reference_judge_required": False,
        "route": "attention_q_native_tensor_bank",
        "max_tokens": runtime.config.max_policy_tokens,
    }


class TrajectoryWriteRequest(BaseModel):
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list, max_length=16)
    name: str | None = Field(default=None, max_length=96)


class TrajectoryRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=96)


class TrajectoryCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=96)
    content: str = Field(min_length=1)
    tags: list[str] = Field(min_length=1, max_length=16)


class TrajectoryPreviewRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class TrainingSelectionRequest(BaseModel):
    names: list[str] = Field(default_factory=list, max_length=16)


def _trajectory_store(request: Request) -> TrajectoryStore:
    runtime = _runtime(request)
    return TrajectoryStore(runtime.config.state_directory.parent / "trajectories")


def _activation_training_store(request: Request) -> ActivationTrainingStore:
    runtime = _runtime(request)
    return ActivationTrainingStore(runtime.config.state_directory.parent)


def _editors_root(request: Request) -> Path:
    runtime = _runtime(request)
    root = runtime.config.state_directory / "activation-editors"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _editor_metadata(path: Path) -> dict[str, object]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        sources = []
        for source in payload.get("sources") or []:
            if isinstance(source, dict) and source.get("name") and source.get("sha256"):
                sources.append(
                    {
                        "name": str(source["name"]),
                        "sha256": str(source["sha256"]),
                    }
                )
        return {
            "name": path.name[: -len(".editor.pt")],
            "layer": int(payload.get("layer") or 0),
            "rank": int(payload.get("rank") or 0),
            "window": int(payload.get("window") or 0),
            "hidden_size": int(payload.get("hidden_size") or 0),
            "bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
            "valid": True,
            "sources": sources,
        }
    except Exception:
        return {
            "name": path.name[: -len(".editor.pt")],
            "layer": 0,
            "rank": 0,
            "window": 0,
            "hidden_size": 0,
            "bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
            "valid": False,
            "sources": [],
        }


def _active_editor(root: Path) -> dict[str, object] | None:
    try:
        payload = json.loads((root / "active.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("editor"):
        return None
    if isinstance(payload.get("editors"), list) and len(payload["editors"]) > 1:
        return None
    return {
        "editor": str(payload["editor"]),
        "applied_at": payload.get("applied_at"),
    }


def _invalidate_combined_editor(request: Request) -> None:
    (_editors_root(request) / "active.json").unlink(missing_ok=True)


def _training_selection_status(request: Request) -> dict[str, object]:
    store = _activation_training_store(request)
    selection = store.selection()
    names = list(selection["trajectories"])
    records = store.source_records(names)
    editor_path = _editors_root(request) / f"{COMBINED_EDITOR_NAME}.editor.pt"
    editor = _editor_metadata(editor_path) if editor_path.is_file() else None
    expected_sources = [
        {"name": record["name"], "sha256": record["sha256"]} for record in records
    ]
    up_to_date = bool(
        names
        and editor
        and editor.get("valid")
        and editor.get("sources") == expected_sources
    )
    active = _active_editor(_editors_root(request))
    applied = bool(
        up_to_date and active and active.get("editor") == COMBINED_EDITOR_NAME
    )
    return {
        "names": names,
        "updated_at": selection.get("updated_at"),
        "sources": records,
        "editor": editor,
        "up_to_date": up_to_date,
        "applied": applied,
    }


def _read_trajectory_tags(name: str, store: TrajectoryStore) -> list[str]:
    try:
        return list(normalize_tags(store.get(name).get("tags")))
    except (TrajectoryStoreError, TagValidationError, json.JSONDecodeError):
        return []


@router.get("/trajectories")
async def list_trajectories(request: Request):
    return {"trajectories": _trajectory_store(request).list()}


@router.post("/trajectories/preview")
async def preview_trajectory(payload: TrajectoryPreviewRequest):
    try:
        data = base64.b64decode(payload.content_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 解码失败")
    try:
        normalized = parse_trajectory_upload(payload.filename, data)
    except TrajectoryStoreError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )
    suggested_name = Path(payload.filename).stem
    for suffix in (".jsonl", ".json", ".gz", ".zip"):
        if suggested_name.lower().endswith(suffix):
            suggested_name = suggested_name[: -len(suffix)]
    content = json.dumps(normalized, ensure_ascii=False, indent=2)
    return {
        "draft": True,
        "suggested_name": suggested_name.lower(),
        "content": content,
        "tags": normalized.get("tags", []),
        "messages": len(normalized["session"]["messages"]),
        "bytes": len(content.encode("utf-8")),
    }


@router.post("/trajectories", status_code=201)
async def create_trajectory(payload: TrajectoryCreateRequest, request: Request):
    try:
        document = json.loads(payload.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="内容必须是合法 JSON")
    try:
        return _trajectory_store(request).create(
            payload.name, document, tags=payload.tags
        )
    except TrajectoryStoreError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )


@router.get("/trajectories/{name}")
async def get_trajectory(name: str, request: Request):
    try:
        return _trajectory_store(request).get(name)
    except TrajectoryStoreError as exc:
        raise HTTPException(status_code=404, detail=exc.message)


@router.put("/trajectories/{name}")
async def put_trajectory(name: str, payload: TrajectoryWriteRequest, request: Request):
    store = _trajectory_store(request)
    target_name = str(payload.name).strip() if payload.name is not None else name
    try:
        document = json.loads(payload.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="内容必须是合法 JSON")
    try:
        if target_name != name:
            store.rename(name, target_name)
            result = store.save(target_name, document, tags=payload.tags)
            selected = _activation_training_store(request).rename_selection(
                name, result["name"]
            )
        else:
            result = store.save(name, document, tags=payload.tags)
            selected = _activation_training_store(request).touch_selection(
                result["name"]
            )
        if selected:
            _invalidate_combined_editor(request)
        return result
    except TrajectoryStoreError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )


@router.patch("/trajectories/{name}")
async def rename_trajectory(
    name: str, payload: TrajectoryRenameRequest, request: Request
):
    try:
        result = _trajectory_store(request).rename(name, payload.name)
        if _activation_training_store(request).rename_selection(name, result["name"]):
            _invalidate_combined_editor(request)
        return result
    except TrajectoryStoreError as exc:
        status_code = 404 if exc.code == "not_found" else 422
        return JSONResponse(
            status_code=status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )


@router.delete("/trajectories/{name}")
async def delete_trajectory(name: str, request: Request):
    try:
        _trajectory_store(request).delete(name)
    except TrajectoryStoreError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    if _activation_training_store(request).remove_selection(name):
        _invalidate_combined_editor(request)
    return {"deleted": True, "name": name}


@router.get("/editors")
async def list_editors(request: Request):
    root = _editors_root(request)
    trajectory_store = _trajectory_store(request)
    editors = []
    for path in sorted(root.glob("*.editor.pt")):
        metadata = _editor_metadata(path)
        tags = set()
        for source in metadata.get("sources") or []:
            tags.update(_read_trajectory_tags(str(source["name"]), trajectory_store))
        metadata["tags"] = sorted(tags)
        editors.append(metadata)
    return {"editors": editors, "active": _active_editor(root)}


@router.get("/editors/training-selection")
async def get_editor_training_selection(request: Request):
    try:
        return _training_selection_status(request)
    except ActivationTrainingError as exc:
        return JSONResponse(status_code=503, content={"detail": exc.public_dict()})


@router.put("/editors/training-selection")
async def put_editor_training_selection(
    payload: TrainingSelectionRequest, request: Request
):
    store = _activation_training_store(request)
    try:
        before = list(store.selection()["trajectories"])
        selection = store.set_selection(payload.names)
        if before != list(selection["trajectories"]):
            _invalidate_combined_editor(request)
        return _training_selection_status(request)
    except ActivationTrainingError as exc:
        status_code = 404 if exc.code == "trajectory_not_found" else 422
        return JSONResponse(
            status_code=status_code, content={"detail": exc.public_dict()}
        )


@router.get("/editors/training")
async def get_editor_training(request: Request):
    try:
        return _activation_training_store(request).public_status()
    except ActivationTrainingError as exc:
        return JSONResponse(status_code=503, content={"detail": exc.public_dict()})


@router.post("/editors/train", status_code=202)
async def train_editor(request: Request):
    if not ServiceConfigStore.from_environment().managed_restart:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "restart_unmanaged",
                    "message": "当前服务未由 Docker 托管，不能安全释放 GPU 进行训练",
                }
            },
        )
    runtime = _runtime(request)
    store = _activation_training_store(request)
    try:
        names = list(store.selection()["trajectories"])
        store.enqueue(names, state_directory=runtime.config.state_directory)
    except ActivationTrainingError as exc:
        if exc.code == "trajectory_not_found":
            status_code = 404
        elif exc.code == "training_busy":
            status_code = 409
        else:
            status_code = 422
        return JSONResponse(
            status_code=status_code, content={"detail": exc.public_dict()}
        )
    _invalidate_combined_editor(request)
    runtime.telemetry.emit(
        "admin",
        "activation_training.queued",
        {"editor": COMBINED_EDITOR_NAME, "trajectories": names},
    )
    request_managed_restart(delay_seconds=2.0)
    return {**store.public_status(), "restart_requested": True}


@router.get("/reflection-memory")
async def list_reflection_memories(request: Request):
    return {"reflections": _runtime(request).reflection_memories()}


@router.get("/reflection-memory/organize")
async def reflection_memory_organization_status(request: Request):
    return _runtime(request).reflection_memory_organization_status()


@router.post("/reflection-memory/organize", status_code=202)
async def start_reflection_memory_organization(request: Request):
    try:
        return _runtime(request).start_reflection_memory_organization()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/reflection-memory/pending")
async def list_pending_reflection_memories(request: Request):
    return {"pending": _runtime(request).pending_reflection_memories()}


@router.post("/reflection-memory/pending/reflect", status_code=202)
async def start_pending_reflection_memories(
    payload: ReflectionSelectionRequest, request: Request
):
    try:
        return _runtime(request).start_pending_reflections(payload.conversation_keys)
    except KeyError:
        raise HTTPException(status_code=404, detail="待反思轨迹不存在或已经完成")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/reflection-memory/pending/cancel")
async def cancel_pending_reflection_memories(
    payload: ReflectionSelectionRequest, request: Request
):
    try:
        return _runtime(request).cancel_pending_reflections(payload.conversation_keys)
    except KeyError:
        raise HTTPException(status_code=404, detail="待反思轨迹不存在或已经完成")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sources/delete")
async def delete_source_documents(payload: SourceSelectionRequest, request: Request):
    try:
        return _runtime(request).delete_source_documents(
            payload.lane, payload.relative_paths
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="选中的文档不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sources/compile")
async def compile_source_documents(payload: SourceSelectionRequest, request: Request):
    try:
        return await _runtime(request).compile_source_documents(
            payload.lane, payload.relative_paths
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="选中的文档不存在")
    except TensorBankCompileError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.public_dict()})
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/policydata/reindex")
async def reindex_policy_data(request: Request):
    return _runtime(request).reindex_policy_data()


@router.post("/tensor-bank/reindex")
async def reindex_tensor_bank(request: Request):
    runtime = _runtime(request)
    if runtime.tensor_bank is None:
        raise HTTPException(status_code=409, detail="Native Tensor Bank is disabled")
    try:
        return await runtime.reindex_tensor_bank()
    except TensorBankCompileError as error:
        return JSONResponse(
            status_code=422,
            content={"detail": error.public_dict()},
        )


@router.get("/policydata/{relative_path:path}")
async def get_policy_data(relative_path: str, request: Request):
    runtime = _runtime(request)
    try:
        return runtime.policy_data_document(relative_path)
    except KeyError:
        raise HTTPException(status_code=404, detail="未找到 PolicyData 文档")


@router.post("/knowledge/preview")
async def preview_knowledge(payload: KnowledgeIngestRequest):
    files = [item.model_dump() for item in payload.files]
    try:
        validate_upload_batch(files)
        drafts = [
            preview_knowledge_upload(
                str(item["filename"]), str(item["content_base64"])
            ).public_dict()
            for item in files
        ]
    except KnowledgeIngestError as error:
        return JSONResponse(
            status_code=422,
            content={"detail": error.public_dict()},
        )
    return {"drafts": drafts, "persisted": False}


@router.put("/policydata/{relative_path:path}")
async def put_policy_data(
    relative_path: str, payload: KnowledgeWriteRequest, request: Request
):
    runtime = _runtime(request)
    try:
        return runtime.upsert_policy_data(relative_path, payload.content, payload.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.delete("/policydata/{relative_path:path}")
async def delete_policy_data(relative_path: str, request: Request):
    runtime = _runtime(request)
    try:
        runtime.delete_policy_data(relative_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="未找到 PolicyData 文档")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"deleted": True, "relative_path": relative_path}


@router.post("/knowledge/ingest")
async def ingest_knowledge(payload: KnowledgeIngestRequest, request: Request):
    runtime = _runtime(request)
    try:
        return await runtime.ingest_knowledge_files(
            [item.model_dump() for item in payload.files]
        )
    except KnowledgeIngestError as error:
        return JSONResponse(
            status_code=422,
            content={"detail": error.public_dict()},
        )
    except TensorBankCompileError as error:
        return JSONResponse(
            status_code=422,
            content={"detail": error.public_dict()},
        )


@router.post("/knowledge/reindex")
async def reindex_knowledge(request: Request):
    return _runtime(request).reindex_knowledge()


@router.get("/knowledge/{relative_path:path}")
async def get_knowledge(relative_path: str, request: Request):
    runtime = _runtime(request)
    try:
        return runtime.knowledge_document(relative_path)
    except KeyError:
        raise HTTPException(status_code=404, detail="未找到知识库文档")


@router.put("/knowledge/{relative_path:path}")
async def put_knowledge(
    relative_path: str, payload: KnowledgeWriteRequest, request: Request
):
    runtime = _runtime(request)
    try:
        return runtime.upsert_knowledge(relative_path, payload.content, payload.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/knowledge/{relative_path:path}")
async def delete_knowledge(relative_path: str, request: Request):
    runtime = _runtime(request)
    try:
        runtime.delete_knowledge(relative_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="未找到知识库文档")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"deleted": True, "relative_path": relative_path}
