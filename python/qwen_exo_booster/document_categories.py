from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qwen_exo_booster.knowledge import KnowledgeDocument


class DocumentCategoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentCategory:
    category_id: str
    title: str
    parent_id: str | None
    origin: str
    document_count: int

    def public_dict(self) -> dict[str, object]:
        return {
            "category_id": self.category_id,
            "title": self.title,
            "parent_id": self.parent_id,
            "origin": self.origin,
            "document_count": self.document_count,
        }


_DEFAULT_CATEGORIES: dict[str, tuple[str, str | None]] = {
    "agent-trajectories": ("Agent 轨迹", None),
    "reflection-memory": ("反思记忆", None),
    "references": ("参考资料", None),
    "uploads": ("用户上传", None),
    "policies": ("执行策略", None),
    "boeing_fable5_agent_trajectory": ("Fable 5 轨迹", "agent-trajectories"),
    "trajectory_reflection": ("轨迹反思", "reflection-memory"),
    "curated_reflection_memory": ("整理后的反思", "reflection-memory"),
    "uploaded_markdown": ("上传 Markdown", "uploads"),
    "uploaded_structured_text": ("上传结构化文本", "uploads"),
    "uploaded_text": ("上传文本", "uploads"),
    "threejs_production_reference": ("Three.js 参考", "references"),
    "frontend_design_reference": ("前端设计参考", "references"),
    "local_sdk_verified": ("已验证 SDK 参考", "references"),
    "software_engineering_reference": ("软件工程参考", "references"),
    "swe_task_targeted_reference": ("SWE 定向参考", "references"),
    "coding_agent_execution_policy": ("编码执行策略", "policies"),
}


def _validate_identifier(value: object) -> str:
    identifier = str(value or "").strip()
    if (
        not identifier
        or len(identifier) > 128
        or any(ord(char) < 32 for char in identifier)
    ):
        raise DocumentCategoryError("分类标识必须是 1–128 个可见字符")
    return identifier


def _validate_title(value: object) -> str:
    title = str(value or "").strip()
    if not title or len(title) > 128 or any(ord(char) < 32 for char in title):
        raise DocumentCategoryError("分类名称必须是 1–128 个可见字符")
    return title


class DocumentCategoryStore:
    """SQLite registry for stable categories and their current source mapping."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_categories (
                    category_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    parent_id TEXT REFERENCES document_categories(category_id),
                    origin TEXT NOT NULL CHECK(origin IN ('system', 'user', 'observed'))
                );
                CREATE TABLE IF NOT EXISTS document_category_assignments (
                    lane TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    category_id TEXT NOT NULL REFERENCES document_categories(category_id),
                    PRIMARY KEY (lane, relative_path)
                );
                CREATE INDEX IF NOT EXISTS document_category_assignments_category
                    ON document_category_assignments(category_id);
                """
            )
            for category_id, (title, parent_id) in _DEFAULT_CATEGORIES.items():
                connection.execute(
                    """
                    INSERT INTO document_categories(category_id, title, parent_id, origin)
                    VALUES (?, ?, ?, 'system')
                    ON CONFLICT(category_id) DO NOTHING
                    """,
                    (category_id, title, parent_id),
                )

    def ensure(
        self,
        category_id: object,
        *,
        title: object | None = None,
        parent_id: object | None = None,
        origin: str = "observed",
    ) -> str:
        category_id = _validate_identifier(category_id)
        if origin not in {"system", "user", "observed"}:
            raise DocumentCategoryError("分类来源无效")
        parent = _validate_identifier(parent_id) if parent_id is not None else None
        if parent == category_id:
            raise DocumentCategoryError("分类不能成为自身父级")
        category_title = _validate_title(title if title is not None else category_id)
        with self._lock, self._connect() as connection:
            if parent is not None:
                exists = connection.execute(
                    "SELECT 1 FROM document_categories WHERE category_id = ?", (parent,)
                ).fetchone()
                if exists is None:
                    raise DocumentCategoryError("父分类不存在")
            connection.execute(
                """
                INSERT INTO document_categories(category_id, title, parent_id, origin)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category_id) DO NOTHING
                """,
                (category_id, category_title, parent, origin),
            )
        return category_id

    def create(
        self, category_id: object, title: object, parent_id: object | None
    ) -> DocumentCategory:
        category_id = self.ensure(
            category_id, title=title, parent_id=parent_id, origin="user"
        )
        return self.get(category_id)

    def update(
        self, category_id: object, *, title: object, parent_id: object | None
    ) -> DocumentCategory:
        category_id = _validate_identifier(category_id)
        title = _validate_title(title)
        parent = _validate_identifier(parent_id) if parent_id is not None else None
        if parent == category_id:
            raise DocumentCategoryError("分类不能成为自身父级")
        with self._lock, self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM document_categories WHERE category_id = ?",
                    (category_id,),
                ).fetchone()
                is None
            ):
                raise DocumentCategoryError("分类不存在")
            if (
                parent is not None
                and connection.execute(
                    "SELECT 1 FROM document_categories WHERE category_id = ?", (parent,)
                ).fetchone()
                is None
            ):
                raise DocumentCategoryError("父分类不存在")
            connection.execute(
                "UPDATE document_categories SET title = ?, parent_id = ? WHERE category_id = ?",
                (title, parent, category_id),
            )
        return self.get(category_id)

    def sync_documents(self, lane: str, documents: Iterable[KnowledgeDocument]) -> None:
        lane = _validate_identifier(lane)
        rows = [
            (
                document.relative_path,
                str(
                    document.retrieval_category
                    or document.source_kind
                    or "unclassified"
                ),
            )
            for document in documents
        ]
        with self._lock, self._connect() as connection:
            current_paths = {path for path, _category_id in rows}
            for _path, category_id in rows:
                default = _DEFAULT_CATEGORIES.get(category_id)
                title, parent_id = (
                    default if default is not None else (category_id, None)
                )
                connection.execute(
                    """
                    INSERT INTO document_categories(category_id, title, parent_id, origin)
                    VALUES (?, ?, ?, 'observed')
                    ON CONFLICT(category_id) DO NOTHING
                    """,
                    (category_id, title, parent_id),
                )
            if current_paths:
                placeholders = ",".join("?" for _ in current_paths)
                connection.execute(
                    f"DELETE FROM document_category_assignments WHERE lane = ? AND relative_path NOT IN ({placeholders})",
                    (lane, *sorted(current_paths)),
                )
            else:
                connection.execute(
                    "DELETE FROM document_category_assignments WHERE lane = ?", (lane,)
                )
            connection.executemany(
                """
                INSERT INTO document_category_assignments(lane, relative_path, category_id)
                VALUES (?, ?, ?)
                ON CONFLICT(lane, relative_path) DO UPDATE SET category_id = excluded.category_id
                """,
                ((lane, path, category_id) for path, category_id in rows),
            )

    def categories(self) -> list[dict[str, object]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.category_id, c.title, c.parent_id, c.origin,
                       COUNT(a.relative_path) AS document_count
                FROM document_categories AS c
                LEFT JOIN document_category_assignments AS a
                    ON a.category_id = c.category_id
                GROUP BY c.category_id
                ORDER BY c.parent_id IS NOT NULL, c.parent_id, c.title COLLATE NOCASE
                """
            ).fetchall()
        return [
            DocumentCategory(
                category_id=str(row["category_id"]),
                title=str(row["title"]),
                parent_id=(
                    str(row["parent_id"]) if row["parent_id"] is not None else None
                ),
                origin=str(row["origin"]),
                document_count=int(row["document_count"]),
            ).public_dict()
            for row in rows
        ]

    def get(self, category_id: object) -> DocumentCategory:
        category_id = _validate_identifier(category_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.category_id, c.title, c.parent_id, c.origin,
                       COUNT(a.relative_path) AS document_count
                FROM document_categories AS c
                LEFT JOIN document_category_assignments AS a
                    ON a.category_id = c.category_id
                WHERE c.category_id = ?
                GROUP BY c.category_id
                """,
                (category_id,),
            ).fetchone()
        if row is None:
            raise DocumentCategoryError("分类不存在")
        return DocumentCategory(
            category_id=str(row["category_id"]),
            title=str(row["title"]),
            parent_id=(str(row["parent_id"]) if row["parent_id"] is not None else None),
            origin=str(row["origin"]),
            document_count=int(row["document_count"]),
        )
