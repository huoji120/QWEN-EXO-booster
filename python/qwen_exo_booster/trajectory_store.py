from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from qwen_exo_booster.tags import TagValidationError, normalize_tags

_TRAJECTORY_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_MAX_STORED_BYTES = 8 * 1024 * 1024


class TrajectoryStoreError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_trajectory_name(value: object) -> str:
    name = str(value or "").strip().lower()
    if not name.endswith(".json"):
        name = f"{name}.json"
    stem = name[: -len(".json")]
    if not _TRAJECTORY_NAME.fullmatch(stem):
        raise TrajectoryStoreError(
            "invalid_name", "轨迹名称只能包含小写字母、数字、点、横线和下划线"
        )
    return name


def _normalize_message(index: int, message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise TrajectoryStoreError("invalid_message", f"第 {index} 条消息不是对象")
    role = str(message.get("role") or "").strip()
    if role not in _ALLOWED_ROLES:
        raise TrajectoryStoreError(
            "invalid_role", f"第 {index} 条消息的角色 {role!r} 不受支持"
        )
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return {"role": role, "content": content}


def normalize_chatml(payload: Any, *, tags: Any = None) -> dict[str, Any]:
    if isinstance(payload, dict):
        session = payload.get("session")
        messages = (
            session.get("messages")
            if isinstance(session, dict)
            else payload.get("messages")
        )
        tag_source = payload.get("tags") if tags is None else tags
    elif isinstance(payload, list):
        messages = payload
        tag_source = tags
    else:
        messages = None
        tag_source = tags
    if not isinstance(messages, list) or len(messages) < 2:
        raise TrajectoryStoreError(
            "invalid_format", "需要 ChatML 消息列表（至少 2 条消息）"
        )
    normalized = [_normalize_message(index, m) for index, m in enumerate(messages)]
    if not any(message["role"] == "assistant" for message in normalized):
        raise TrajectoryStoreError("no_assistant", "轨迹里至少需要一条助手消息")
    try:
        normalized_tags = normalize_tags(tag_source)
    except TagValidationError as exc:
        raise TrajectoryStoreError("invalid_tags", str(exc)) from exc
    return {
        "name": "uploaded-trajectory",
        "format": "chatml-v1",
        "tags": list(normalized_tags),
        "session": {"messages": normalized},
    }


def parse_trajectory_upload(filename: str, data: bytes) -> dict[str, Any]:
    lowered = filename.lower()
    if lowered.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except OSError as exc:
            raise TrajectoryStoreError("bad_gzip", "无法解压 gzip 文件") from exc
        lowered = lowered[: -len(".gz")]
    elif lowered.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/") and not name.startswith("__MACOSX")
                ]
                if len(members) != 1:
                    raise TrajectoryStoreError(
                        "bad_zip", "压缩包里必须只有一个轨迹文件"
                    )
                data = archive.read(members[0])
                lowered = members[0].lower()
        except zipfile.BadZipFile as exc:
            raise TrajectoryStoreError("bad_zip", "无法解压 zip 文件") from exc
    if len(data) > _MAX_STORED_BYTES:
        raise TrajectoryStoreError("too_large", "轨迹文件超过 8MB 上限")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrajectoryStoreError("bad_encoding", "文件必须是 UTF-8 编码") from exc
    if lowered.endswith(".jsonl"):
        messages = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryStoreError(
                    "bad_jsonl", f"第 {line_number} 行不是合法 JSON"
                ) from exc
            if isinstance(record, dict) and isinstance(record.get("messages"), list):
                messages.extend(record["messages"])
            else:
                messages.append(record)
        return normalize_chatml(messages)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrajectoryStoreError("bad_json", "不是合法的 ChatML JSON 文件") from exc
    return normalize_chatml(payload)


class TrajectoryStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / validate_trajectory_name(name)

    def list(self) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                messages = (payload.get("session") or {}).get("messages") or []
                entries.append(
                    {
                        "name": path.name[: -len(".json")],
                        "messages": len(messages),
                        "bytes": path.stat().st_size,
                        "modified_ns": path.stat().st_mtime_ns,
                        "tags": list(normalize_tags(payload.get("tags"))),
                        "valid": True,
                    }
                )
            except Exception:
                entries.append(
                    {
                        "name": path.name[: -len(".json")],
                        "messages": 0,
                        "bytes": path.stat().st_size,
                        "modified_ns": path.stat().st_mtime_ns,
                        "tags": [],
                        "valid": False,
                    }
                )
        return entries

    def get(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            raise TrajectoryStoreError("not_found", "轨迹文件不存在")
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = (payload.get("session") or {}).get("messages") or []
        return {
            "name": path.name[: -len(".json")],
            "content": path.read_text(encoding="utf-8"),
            "messages": len(messages),
            "bytes": path.stat().st_size,
            "tags": list(normalize_tags(payload.get("tags"))),
        }

    def save(
        self, name: str, payload: dict[str, Any], *, tags: Any = None
    ) -> dict[str, Any]:
        normalized = normalize_chatml(payload, tags=tags)
        path = self._path(name)
        normalized["name"] = path.stem
        text = json.dumps(normalized, ensure_ascii=False, indent=2)
        if len(text.encode("utf-8")) > _MAX_STORED_BYTES:
            raise TrajectoryStoreError("too_large", "轨迹文件超过 8MB 上限")
        path.write_text(text + "\n", encoding="utf-8")
        return {
            "name": path.stem,
            "messages": len(normalized["session"]["messages"]),
            "bytes": path.stat().st_size,
            "tags": list(normalized["tags"]),
        }

    def create(
        self, name: str, payload: dict[str, Any], *, tags: Any = None
    ) -> dict[str, Any]:
        path = self._path(name)
        if path.exists():
            raise TrajectoryStoreError("name_conflict", "同名轨迹已经存在")
        try:
            required_tags = normalize_tags(tags, required=True)
        except TagValidationError as exc:
            raise TrajectoryStoreError("invalid_tags", str(exc)) from exc
        return self.save(name, payload, tags=required_tags)

    def rename(self, name: str, new_name: str) -> dict[str, Any]:
        source = self._path(name)
        target = self._path(new_name)
        if source == target:
            return self.get(name)
        if not source.exists():
            raise TrajectoryStoreError("not_found", "轨迹文件不存在")
        if target.exists():
            raise TrajectoryStoreError("name_conflict", "同名轨迹已经存在")
        payload = json.loads(source.read_text(encoding="utf-8"))
        source.rename(target)
        payload["name"] = target.stem
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.get(target.stem)

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise TrajectoryStoreError("not_found", "轨迹文件不存在")
        path.unlink()
