"""Daily work memory: notes the model writes itself, kept apart from imports.

Everything lives under ``WORK_ROOT`` inside the knowledge directory, so the
memory tools search it together with imported memory, but only this area can be
written by the model (``qwen_exo_memory_write``) or reorganised by the nightly
dream. Each note is Markdown with a small front matter (title, created,
updated). Text that looks like a secret (keys, tokens, private keys,
passwords) is refused before anything is written. Deleted notes go to a trash
folder in the state directory and can be restored.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any

from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex

WORK_ROOT = "work-memory"
MAX_NOTE_BYTES = 16 * 1024
MAX_NOTE_TOTAL_BYTES = 256 * 1024
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# (label, pattern): matches are refused, so a note never stores a usable secret.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("API key (sk-)", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("credentials in URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]{3,}@")),
    (
        "password / secret assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token|"
            r"client[_-]?secret|private[_-]?key)\b\s*[:=]\s*[\"']?([^\s\"'<>{}$*]{8,})"
        ),
    ),
)
_PLACEHOLDER = re.compile(r"(?i)^(?:x+|\*+|redacted|placeholder|example|changeme|your[_-].*|<.*>)$")


def find_secrets(text: str) -> list[str]:
    """Labels of secret-looking content in ``text`` (placeholders are allowed)."""
    found: list[str] = []
    for label, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(match.lastindex or 0)
            if match.lastindex and _PLACEHOLDER.match(value):
                continue
            found.append(label)
            break
    return found


def reject_secrets(text: str) -> None:
    labels = find_secrets(text)
    if labels:
        raise FileKnowledgeError(
            "refused: the note looks like it contains secrets ("
            + ", ".join(labels)
            + "). Memory must never store keys, tokens, passwords or private keys; "
            "describe where the secret lives or write <REDACTED> instead, then retry."
        )


def work_path(path: str) -> str:
    """Normalize a model-given note path to ``work-memory/...md``."""
    raw = str(path or "").strip().replace("\\", "/").strip("/")
    if raw.startswith(WORK_ROOT + "/"):
        raw = raw[len(WORK_ROOT) + 1 :]
    if not raw or raw == WORK_ROOT:
        raise FileKnowledgeError("path must name a note, e.g. ctf/web/jdwp-attach.md")
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
        raise FileKnowledgeError("path contains an unsafe component")
    if not raw.lower().endswith((".md", ".markdown", ".txt")):
        raw += ".md"
    return f"{WORK_ROOT}/{raw}"


def work_directory(path: str) -> str:
    raw = str(path or "").strip().replace("\\", "/").strip("/")
    if raw.startswith(WORK_ROOT + "/"):
        raw = raw[len(WORK_ROOT) + 1 :]
    if raw in {"", WORK_ROOT}:
        return WORK_ROOT
    if any(part in {"", ".", ".."} or part.startswith(".") for part in PurePosixPath(raw).parts):
        raise FileKnowledgeError("path contains an unsafe component")
    return f"{WORK_ROOT}/{raw}"


def is_work_path(path: str) -> bool:
    return path == WORK_ROOT or path.startswith(WORK_ROOT + "/")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def parse_note(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            fields[key.strip()] = value.strip()
    return fields, text[match.end() :]


def render_note(*, title: str, body: str, created: str, updated: str, source: str) -> str:
    title = " ".join(str(title).split())[:160]
    return (
        f"---\ntitle: {title}\nsource: {source}\ncreated: {created}\nupdated: {updated}\n---\n"
        f"# {title}\n\n{body.strip()}\n"
    )


def _read_existing(index: FileKnowledgeIndex, relative: str) -> str | None:
    target = index.path_of(relative)
    if not target.exists():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def write_note(
    index: FileKnowledgeIndex,
    path: str,
    *,
    title: str,
    content: str,
    mode: str = "create",
    source: str = "model",
) -> dict[str, Any]:
    """Create, append to, or (``mode="replace"``, dream only) rewrite a note."""
    if mode not in {"create", "append", "replace"}:
        raise FileKnowledgeError("mode must be create or append")
    content = str(content or "").strip()
    if content.startswith("# "):
        # render_note writes the title heading itself.
        content = _without_heading(content).strip()
    if not content:
        raise FileKnowledgeError("content is empty")
    if len(content.encode("utf-8")) > MAX_NOTE_BYTES:
        raise FileKnowledgeError(
            f"content exceeds {MAX_NOTE_BYTES // 1024} KB; keep notes focused or split them"
        )
    reject_secrets(f"{title}\n{content}")
    relative = work_path(path)
    existing = _read_existing(index, relative)
    now = _now()
    if mode == "create" and existing is not None:
        raise FileKnowledgeError(
            f"{relative} already exists; use mode=append to add to it or choose another name"
        )
    if mode == "append" and existing is None:
        mode = "create"
    if mode == "append":
        fields, body = parse_note(existing)
        body = f"{body.rstrip()}\n\n## 补充（{now}）\n\n{content}"
        if len(body.encode("utf-8")) > MAX_NOTE_TOTAL_BYTES:
            raise FileKnowledgeError("note is full; start a new note for further details")
        text = render_note(
            title=fields.get("title") or title or PurePosixPath(relative).stem,
            body=_without_heading(body),
            created=fields.get("created") or now,
            updated=now,
            source=fields.get("source") or source,
        )
    else:
        fields = parse_note(existing)[0] if existing is not None else {}
        text = render_note(
            title=title or PurePosixPath(relative).stem,
            body=content,
            created=fields.get("created") or now,
            updated=now,
            source=source,
        )
    row = index.upsert(relative, text.encode("utf-8"))
    return {"path": relative, "bytes": int(row["byte_count"]), "mode": mode}


def _without_heading(body: str) -> str:
    return body.split("\n", 1)[1] if body.startswith("# ") else body


def existing_note(index: FileKnowledgeIndex, path: str) -> str:
    """Normalized path of a work note that must already exist."""
    relative = work_path(path)
    if not index.path_of(relative).is_file():
        raise FileKnowledgeError(
            f"{relative} does not exist; find the note with qwen_exo_memory_search first"
        )
    return relative


def edit_note(
    index: FileKnowledgeIndex,
    path: str,
    *,
    old_text: str,
    new_text: str,
    title: str = "",
    source: str = "model",
) -> dict[str, Any]:
    """Replace one exact passage of an existing note (and optionally its title)."""
    relative = existing_note(index, path)
    fields, body = parse_note(_read_existing(index, relative))
    if not old_text:
        raise FileKnowledgeError("old_text is empty; copy the exact passage to change")
    count = body.count(old_text)
    if count == 0:
        raise FileKnowledgeError(
            "old_text was not found in the note; read it with qwen_exo_memory_read and "
            "copy the passage exactly (without the line-number prefix)"
        )
    if count > 1:
        raise FileKnowledgeError(
            f"old_text appears {count} times; include more surrounding text so it is unique"
        )
    reject_secrets(f"{title}\n{new_text}")
    body = body.replace(old_text, new_text, 1)
    if len(body.encode("utf-8")) > MAX_NOTE_TOTAL_BYTES:
        raise FileKnowledgeError("the edited note is too large; split it into several notes")
    text = render_note(
        title=title or fields.get("title") or PurePosixPath(relative).stem,
        body=_without_heading(body),
        created=fields.get("created") or _now(),
        updated=_now(),
        source=fields.get("source") or source,
    )
    row = index.upsert(relative, text.encode("utf-8"))
    return {"path": relative, "bytes": int(row["byte_count"]), "mode": "edit"}


def list_notes(index: FileKnowledgeIndex, *, since_ns: int | None = None, until_ns: int | None = None) -> list[dict[str, Any]]:
    """Work notes with title and dates, newest first, optionally by mtime window."""
    notes = []
    for row in index.metadata():
        relative = str(row["relative_path"])
        if not is_work_path(relative):
            continue
        mtime = int(row["mtime_ns"])
        if since_ns is not None and mtime < since_ns:
            continue
        if until_ns is not None and mtime >= until_ns:
            continue
        notes.append(
            {
                "path": relative,
                "title": str(row["title"]),
                "bytes": int(row["byte_count"]),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime / 1e9)),
                "mtime_ns": mtime,
            }
        )
    notes.sort(key=lambda note: -note["mtime_ns"])
    return notes


class WorkMemoryTrash:
    """Deleted work notes/folders, kept under the state directory for restore."""

    def __init__(self, index: FileKnowledgeIndex):
        self.index = index
        self.root = index.state_directory / "work-memory-trash"

    def discard(self, path: str, *, batch: str) -> str:
        relative = work_directory(path)
        if not self.index.path_of(relative).is_dir():
            relative = work_path(path)
        if relative == WORK_ROOT:
            raise FileKnowledgeError("the work memory root cannot be deleted")
        source = self.index.path_of(relative)
        if not source.exists():
            raise FileNotFoundError(relative)
        target = self.root / batch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self.index.refresh()
        return relative

    def keep_copy(self, relative: str, *, batch: str, data: bytes | None = None) -> None:
        """Save a version of a note (``data``, or its current bytes) to the trash."""
        target = self.root / batch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            shutil.copy2(self.index.path_of(relative), target)
        else:
            target.write_bytes(data)

    def entries(self) -> list[dict[str, Any]]:
        found = []
        if not self.root.exists():
            return found
        for batch in sorted(os.listdir(self.root), reverse=True):
            batch_root = self.root / batch
            for directory, _dirs, files in os.walk(batch_root):
                for name in files:
                    full = os.path.join(directory, name)
                    found.append(
                        {
                            "batch": batch,
                            "path": os.path.relpath(full, batch_root).replace(os.sep, "/"),
                            "bytes": os.path.getsize(full),
                        }
                    )
        return found

    def restore(self, batch: str, path: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch):
            raise FileKnowledgeError("invalid trash batch")
        relative = work_path(path)
        source = self.root / batch / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = self.index.path_of(relative)
        if target.is_dir():
            raise FileKnowledgeError(f"{relative} is a folder")
        if target.exists():
            # Restoring an older version: the current one goes to the trash too.
            self.keep_copy(relative, batch=f"replaced-{time.time_ns()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self._prune_empty(source.parent, stop=self.root)
        self.index.refresh()
        return relative

    @staticmethod
    def _prune_empty(directory: Path, *, stop: Path) -> None:
        while directory != stop and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent


__all__ = [
    "MAX_NOTE_BYTES",
    "WORK_ROOT",
    "WorkMemoryTrash",
    "edit_note",
    "existing_note",
    "find_secrets",
    "is_work_path",
    "list_notes",
    "parse_note",
    "reject_secrets",
    "work_directory",
    "work_path",
    "write_note",
]
