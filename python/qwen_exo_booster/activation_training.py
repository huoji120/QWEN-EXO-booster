from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from qwen_exo_booster.trajectory_store import (
    TrajectoryStoreError,
    normalize_chatml,
    validate_trajectory_name,
)

_TRAINING_SCHEMA = 1
_SELECTION_SCHEMA = 1
_MAX_ATTEMPTS = 2
_MAX_SOURCES = 16
_DEFAULT_LAYER = 47
_DEFAULT_RANK = 8
_DEFAULT_WINDOW = 16
_DEFAULT_EPOCHS = 0.25
_DEFAULT_LEARNING_RATE = 5e-4
_DEFAULT_MAX_CONTEXT_TOKENS = 512
_DEFAULT_MAX_TARGET_TOKENS = 2048
_DEFAULT_MAX_SEQUENCE_TOKENS = 2560
_JOB_LOCK = threading.Lock()
COMBINED_EDITOR_NAME = "combined-trajectories"


class ActivationTrainingError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_name(value: object) -> str:
    try:
        return Path(validate_trajectory_name(value)).stem
    except TrajectoryStoreError as exc:
        raise ActivationTrainingError(exc.code, exc.message) from exc


def _usable_sample_count(messages: Sequence[dict[str, Any]]) -> int:
    return sum(
        message["role"] == "assistant"
        and index >= 2
        and len(message["content"].strip()) >= 20
        for index, message in enumerate(messages)
    )


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ActivationTrainingStore:
    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / "activation-training"
        self.path = self.root / "job.json"
        self.selection_path = self.root / "selection.json"

    @classmethod
    def from_environment(cls) -> ActivationTrainingStore:
        config_path = Path(
            os.getenv("QWEN_EXO_SERVICE_CONFIG", "/data/qwen-exo/service-config.json")
        )
        return cls(config_path.parent)

    def _read(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ActivationTrainingError(
                "training_state_unreadable", f"无法读取训练任务状态：{exc}"
            ) from exc
        if payload.get("schema") != _TRAINING_SCHEMA:
            raise ActivationTrainingError(
                "training_schema_mismatch", "训练任务状态 schema 不受支持"
            )
        return payload

    def _write(self, document: dict[str, Any]) -> None:
        _atomic_write_json(self.path, document)

    def _read_selection(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.selection_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema": _SELECTION_SCHEMA,
                "trajectories": [],
                "updated_at": None,
            }
        except (OSError, json.JSONDecodeError) as exc:
            raise ActivationTrainingError(
                "training_selection_unreadable", f"无法读取训练成员状态：{exc}"
            ) from exc
        if payload.get("schema") != _SELECTION_SCHEMA or not isinstance(
            payload.get("trajectories"), list
        ):
            raise ActivationTrainingError(
                "training_selection_schema_mismatch", "训练成员状态 schema 不受支持"
            )
        names = [_normalize_name(name) for name in payload["trajectories"]]
        if len(names) != len(set(names)) or len(names) > _MAX_SOURCES:
            raise ActivationTrainingError(
                "training_selection_invalid", "训练成员列表重复或超过上限"
            )
        return {
            "schema": _SELECTION_SCHEMA,
            "trajectories": names,
            "updated_at": payload.get("updated_at"),
        }

    def _write_selection(self, names: Sequence[str]) -> dict[str, Any]:
        document = {
            "schema": _SELECTION_SCHEMA,
            "trajectories": list(names),
            "updated_at": _utc_now(),
        }
        _atomic_write_json(self.selection_path, document)
        return document

    def current(self) -> dict[str, Any] | None:
        with _JOB_LOCK:
            return self._read()

    def public_status(self) -> dict[str, Any]:
        job = self.current()
        if job is None:
            return {"status": "idle", "job": None}
        public_job = {
            key: value for key, value in job.items() if key not in {"state_directory"}
        }
        return {"status": str(job.get("status") or "unknown"), "job": public_job}

    def selection(self) -> dict[str, Any]:
        with _JOB_LOCK:
            return self._read_selection()

    def source_records(self, names: Sequence[object]) -> list[dict[str, Any]]:
        normalized_names = [_normalize_name(name) for name in names]
        if len(normalized_names) > _MAX_SOURCES:
            raise ActivationTrainingError(
                "too_many_trajectories", f"一次最多选择 {_MAX_SOURCES} 条训练轨迹"
            )
        if len(normalized_names) != len(set(normalized_names)):
            raise ActivationTrainingError("duplicate_trajectory", "训练轨迹不能重复")
        records = []
        for name in normalized_names:
            path = self.data_root / "trajectories" / f"{name}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                normalized = normalize_chatml(payload)
            except FileNotFoundError as exc:
                raise ActivationTrainingError(
                    "trajectory_not_found", f"训练语料不存在：{name}"
                ) from exc
            except (OSError, json.JSONDecodeError, TrajectoryStoreError) as exc:
                message = getattr(exc, "message", str(exc))
                raise ActivationTrainingError(
                    "trajectory_invalid", f"训练语料 {name} 不可用：{message}"
                ) from exc
            messages = normalized["session"]["messages"]
            records.append(
                {
                    "name": name,
                    "sha256": _sha256(path),
                    "message_count": len(messages),
                    "sample_count": _usable_sample_count(messages),
                }
            )
        return records

    def set_selection(self, names: Sequence[object]) -> dict[str, Any]:
        records = self.source_records(names)
        normalized_names = [str(record["name"]) for record in records]
        with _JOB_LOCK:
            return self._write_selection(normalized_names)

    def rename_selection(self, source: object, target: object) -> bool:
        source_name = _normalize_name(source)
        target_name = _normalize_name(target)
        with _JOB_LOCK:
            document = self._read_selection()
            names = list(document["trajectories"])
            if source_name not in names:
                return False
            names[names.index(source_name)] = target_name
            if len(names) != len(set(names)):
                raise ActivationTrainingError(
                    "duplicate_trajectory", "重命名后训练成员会重复"
                )
            self._write_selection(names)
            return True

    def remove_selection(self, name: object) -> bool:
        normalized = _normalize_name(name)
        with _JOB_LOCK:
            document = self._read_selection()
            names = list(document["trajectories"])
            if normalized not in names:
                return False
            names.remove(normalized)
            self._write_selection(names)
            return True

    def touch_selection(self, name: object) -> bool:
        normalized = _normalize_name(name)
        with _JOB_LOCK:
            document = self._read_selection()
            names = list(document["trajectories"])
            if normalized not in names:
                return False
            self._write_selection(names)
            return True

    def enqueue(
        self, trajectories: Sequence[object], *, state_directory: Path | str
    ) -> dict[str, Any]:
        records = self.source_records(trajectories)
        if not records:
            raise ActivationTrainingError(
                "no_training_trajectories", "至少激活一条轨迹后才能训练"
            )
        insufficient = [
            str(record["name"]) for record in records if int(record["sample_count"]) < 2
        ]
        if insufficient:
            raise ActivationTrainingError(
                "insufficient_samples",
                "以下轨迹至少需要 2 条可训练的 assistant 消息："
                + "、".join(insufficient),
            )
        state_path = Path(state_directory).resolve()
        state_root = state_path.parent
        if (
            state_root != self.data_root
            and state_root.parent != self.data_root / "model-profiles"
        ):
            raise ActivationTrainingError(
                "invalid_state_directory", "编辑器状态目录不属于当前 QWEN EXO 数据目录"
            )
        training_config = {
            "layer": _DEFAULT_LAYER,
            "rank": _DEFAULT_RANK,
            "window": _DEFAULT_WINDOW,
            "epochs": _DEFAULT_EPOCHS,
            "max_context_tokens": _DEFAULT_MAX_CONTEXT_TOKENS,
            "max_target_tokens": _DEFAULT_MAX_TARGET_TOKENS,
            "learning_rate": _DEFAULT_LEARNING_RATE,
            "max_sequence_tokens": _DEFAULT_MAX_SEQUENCE_TOKENS,
        }

        with _JOB_LOCK:
            current = self._read()
            if current and current.get("status") in {"queued", "running"}:
                raise ActivationTrainingError(
                    "training_busy",
                    f"联合编辑器 {current.get('editor') or ''} 正在训练，请等待完成",
                )
            now = _utc_now()
            document: dict[str, Any] = {
                "schema": _TRAINING_SCHEMA,
                "job_id": f"editor-{uuid.uuid4().hex[:16]}",
                "status": "queued",
                "stage": "waiting_for_restart",
                "editor": COMBINED_EDITOR_NAME,
                "trajectories": [record["name"] for record in records],
                "sources": records,
                "state_directory": str(state_path),
                "message_count": sum(
                    int(record["message_count"]) for record in records
                ),
                "sample_count": sum(int(record["sample_count"]) for record in records),
                "config": training_config,
                "attempts": 0,
                "requested_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
                "metrics": None,
                "error": None,
            }
            self._write(document)
            return document

    def transition(self, job_id: str, status: str, **changes: object) -> dict[str, Any]:
        with _JOB_LOCK:
            document = self._read()
            if document is None or document.get("job_id") != job_id:
                raise ActivationTrainingError(
                    "training_job_changed", "训练任务已被其他任务替换"
                )
            document.update(changes)
            document["status"] = status
            document["updated_at"] = _utc_now()
            self._write(document)
            return document

    def paths(self, job: dict[str, Any]) -> tuple[list[Path], Path, Path, Path, Path]:
        state_value = str(job.get("state_directory") or "")
        state_path = Path(state_value)
        if state_path.is_absolute():
            state_path = state_path.resolve()
        else:
            state_path = (self.data_root / state_value).resolve()
        allowed_root = state_path.parent == self.data_root or (
            state_path.parent.parent == self.data_root / "model-profiles"
        )
        if not allowed_root:
            raise ActivationTrainingError(
                "invalid_state_directory", "训练任务中的状态目录非法"
            )
        job_id = str(job.get("job_id") or "")
        if not job_id.startswith("editor-") or not job_id[7:].isalnum():
            raise ActivationTrainingError("invalid_job_id", "训练任务 ID 非法")
        names = list(job.get("trajectories") or [])
        if not names and job.get("trajectory"):
            names = [job["trajectory"]]
        sources = [
            self.data_root / "trajectories" / f"{_normalize_name(name)}.json"
            for name in names
        ]
        workspace = self.root / "jobs" / job_id
        candidate = workspace / f"{COMBINED_EDITOR_NAME}.editor.pt"
        report = workspace / "report.json"
        log = workspace / "training.log"
        target = state_path / "activation-editors" / f"{COMBINED_EDITOR_NAME}.editor.pt"
        return sources, candidate, report, log, target


TrainingRunner = Callable[[list[str], Path, dict[str, str]], int]


def _run_command(
    command: list[str], log_path: Path, environment: dict[str, str]
) -> int:
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        completed = subprocess.run(
            command,
            check=False,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=3600,
        )
    return int(completed.returncode)


def _log_tail(path: Path, limit: int = 4000) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[-limit:] or None


def _validated_metrics(payload: dict[str, Any], *, label: str) -> dict[str, float]:
    try:
        metrics = {
            key: float(payload[key])
            for key in ("baseline_nll", "random_editor_nll", "trained_editor_nll")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ActivationTrainingError(
            "training_report_invalid", f"训练报告 {label} 指标不可用：{exc}"
        ) from exc
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ActivationTrainingError(
            "training_report_invalid", f"训练报告 {label} 包含非有限损失值"
        )
    return metrics


def _validate_report(path: Path, expected_names: Sequence[str]) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationTrainingError(
            "training_report_invalid", f"训练报告不可用：{exc}"
        ) from exc
    aggregate = _validated_metrics(report, label="联合数据集")
    raw_sources = report.get("sources")
    if not isinstance(raw_sources, list):
        raise ActivationTrainingError(
            "training_report_invalid", "训练报告缺少逐轨迹质量指标"
        )
    validated_sources = []
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ActivationTrainingError(
                "training_report_invalid", "逐轨迹质量指标格式无效"
            )
        name = _normalize_name(source.get("name"))
        validated_sources.append(
            {"name": name, **_validated_metrics(source, label=f"轨迹 {name}")}
        )
    if [source["name"] for source in validated_sources] != list(expected_names):
        raise ActivationTrainingError(
            "training_report_invalid", "训练报告的轨迹集合与任务不一致"
        )
    return {**aggregate, "sources": validated_sources}


def _validate_editor(
    path: Path,
    expected: dict[str, object],
    expected_sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        summary: dict[str, Any] = {
            "schema": int(payload["schema"]),
            "layer": int(payload["layer"]),
            "rank": int(payload["rank"]),
            "window": int(payload["window"]),
            "hidden_size": int(payload["hidden_size"]),
        }
        state = payload["state_dict"]
        for key in ("projection", "transform", "bias"):
            if key not in state:
                raise KeyError(key)
        raw_sources = payload["sources"]
        sources = [
            {
                "name": _normalize_name(source["name"]),
                "sha256": str(source["sha256"]),
            }
            for source in raw_sources
        ]
    except Exception as exc:
        raise ActivationTrainingError(
            "editor_artifact_invalid", f"编辑器产物不可用：{exc}"
        ) from exc
    if summary["schema"] != 1 or any(
        summary[key] != int(expected[key]) for key in ("layer", "rank", "window")
    ):
        raise ActivationTrainingError(
            "editor_artifact_mismatch", "编辑器产物参数与训练任务不一致"
        )
    expected_identity = [
        {"name": str(source["name"]), "sha256": str(source["sha256"])}
        for source in expected_sources
    ]
    if sources != expected_identity:
        raise ActivationTrainingError(
            "editor_artifact_mismatch", "编辑器产物的训练轨迹与任务不一致"
        )
    summary["sources"] = sources
    return summary


def run_pending_activation_training(
    store: ActivationTrainingStore | None = None,
    *,
    runner: TrainingRunner | None = None,
    training_script: Path | str | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any] | None:
    store = store or ActivationTrainingStore.from_environment()
    job = store.current()
    if job is None or job.get("status") in {"succeeded", "failed"}:
        return job
    job_id = str(job.get("job_id") or "")
    if job.get("status") not in {"queued", "running"}:
        return store.transition(
            job_id,
            "failed",
            stage="failed",
            completed_at=_utc_now(),
            error="训练任务状态非法",
        )
    attempts = int(job.get("attempts") or 0)
    if attempts >= _MAX_ATTEMPTS:
        return store.transition(
            job_id,
            "failed",
            stage="failed",
            completed_at=_utc_now(),
            error="训练进程被中断次数过多，已停止自动重试",
        )

    sources, candidate, report, log, target = store.paths(job)
    workspace = candidate.parent
    workspace.mkdir(parents=True, exist_ok=True)
    candidate.unlink(missing_ok=True)
    report.unlink(missing_ok=True)
    config = dict(job.get("config") or {})
    job = store.transition(
        job_id,
        "running",
        stage="training",
        attempts=attempts + 1,
        started_at=job.get("started_at") or _utc_now(),
        completed_at=None,
        metrics=None,
        error=None,
    )

    script = Path(
        training_script
        or os.getenv(
            "QWEN_EXO_ACTIVATION_TRAINING_SCRIPT",
            "/sgl-workspace/sglang/scripts/qwen_exo/train_activation_editor.py",
        )
    )
    model = Path(model_path or "/models/qwen-exo")
    command = [sys.executable, str(script)]
    for source in sources:
        command.extend(("--trajectory", str(source)))
    command.extend(
        (
            "--model",
            str(model),
            "--layer",
            str(config["layer"]),
            "--rank",
            str(config["rank"]),
            "--window",
            str(config["window"]),
            "--epochs",
            str(config["epochs"]),
            "--max-context-tokens",
            str(config.get("max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS)),
            "--max-target-tokens",
            str(config.get("max_target_tokens", _DEFAULT_MAX_TARGET_TOKENS)),
            "--lr",
            str(config.get("learning_rate", _DEFAULT_LEARNING_RATE)),
            "--max-sequence-tokens",
            str(config.get("max_sequence_tokens", _DEFAULT_MAX_SEQUENCE_TOKENS)),
            "--editor-out",
            str(candidate),
            "--output",
            str(report),
        )
    )
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    execute = runner or _run_command
    source_records = list(job.get("sources") or [])
    expected_names = [str(source["name"]) for source in source_records]
    try:
        if len(sources) != len(source_records):
            raise ActivationTrainingError(
                "training_job_invalid", "训练任务缺少逐轨迹来源记录"
            )
        for source_path, source_record in zip(sources, source_records):
            if not source_path.is_file():
                raise ActivationTrainingError(
                    "trajectory_not_found",
                    f"训练开始前语料已不存在：{source_record['name']}",
                )
            if _sha256(source_path) != source_record.get("sha256"):
                raise ActivationTrainingError(
                    "trajectory_changed",
                    f"训练语料 {source_record['name']} 在任务提交后发生变化，拒绝继续",
                )
        return_code = execute(command, log, environment)
        if return_code != 0:
            raise ActivationTrainingError(
                "training_process_failed", f"训练进程退出码为 {return_code}"
            )
        metrics = _validate_report(report, expected_names)
        artifact = _validate_editor(candidate, config, source_records)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, target)
        _atomic_write_json(
            target.parent / "active.json",
            {
                "editor": COMBINED_EDITOR_NAME,
                "sources": expected_names,
                "applied_at": _utc_now(),
                "validation_schema": 1,
            },
        )
        return store.transition(
            job_id,
            "succeeded",
            stage="completed",
            completed_at=_utc_now(),
            metrics=metrics,
            artifact={**artifact, "bytes": target.stat().st_size},
            error=None,
        )
    except Exception as exc:
        message = exc.message if isinstance(exc, ActivationTrainingError) else str(exc)
        return store.transition(
            job_id,
            "failed",
            stage="failed",
            completed_at=_utc_now(),
            error=message or "训练失败",
            log_tail=_log_tail(log),
        )
