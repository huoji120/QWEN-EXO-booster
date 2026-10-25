"""Authenticated-key private storage and a read-only shared/private search view."""

from __future__ import annotations

import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from qwen_exo_booster import memory_grep
from qwen_exo_booster.api_keys import ApiKeyStoreError, is_valid_api_key_id, validate_memory_scope
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex

PRIVATE_PREFIX = "private/"
OTHER_PRIVATE_PREFIX = "private-keys/"


def validate_memory_key_id(key_id: str) -> str:
    """Accept only a stable server-issued id, never a label or bearer token."""
    if not is_valid_api_key_id(key_id):
        raise FileKnowledgeError("invalid private memory key id")
    return key_id


def ensure_no_links(path: Path) -> None:
    """Reject symlinks and Windows junctions, including existing ancestors."""
    for component in (*reversed(path.parents), path):
        try:
            status = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode) or (
            getattr(status, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise FileKnowledgeError("symlink and junction memory paths are not allowed")


class _PrivateMemoryIndex(FileKnowledgeIndex):
    def __init__(self, root: Path, state_directory: Path, *, existing_only: bool = False):
        # Retain lexical paths: resolve() alone would hide an aliased ancestor.
        self._private_root = root
        self._private_state = state_directory
        self._existing_only = existing_only
        self._check_layout()
        if existing_only:
            # Enumeration must never recreate a removed owner's directories or DB.
            self.root = root
            self.state_directory = state_directory
            self.database_path = state_directory / "knowledge-index.sqlite3"
            self.index_path = self.database_path
            self._boilerplate = None
        else:
            super().__init__(root, state_directory)

    def _check_layout(self) -> None:
        ensure_no_links(self._private_root)
        ensure_no_links(self._private_state)
        for suffix in ("", "-wal", "-shm", "-journal"):
            ensure_no_links(self._private_state / ("knowledge-index.sqlite3" + suffix))
        if self._existing_only and not (
            self._private_root.is_dir()
            and self._private_state.is_dir()
            and (self._private_state / "knowledge-index.sqlite3").is_file()
        ):
            raise FileKnowledgeError("private memory space no longer exists")

    def _connect(self):
        self._check_layout()
        return super()._connect()

    def _safe_path(self, relative_path):
        self._check_layout()
        relative, path = super()._safe_path(relative_path)
        ensure_no_links(path)
        return relative, path

    def _is_state_path(self, candidate: Path) -> bool:
        # FileKnowledgeIndex calls this before descending or indexing a file.
        ensure_no_links(candidate)
        return super()._is_state_path(candidate)

    def _iter_regular_files(self):
        self._check_layout()
        return super()._iter_regular_files()


def get_private_memory_index(
    root: str | os.PathLike[str],
    state_directory: str | os.PathLike[str],
    key_id: str,
) -> FileKnowledgeIndex:
    """Create a private-only index for an authenticated managed key.

    Callers must obtain ``key_id`` from authentication, not tool arguments.
    The shared index, summarizer and dream worker cannot walk this directory.
    Mode switches leave these files in place without migrating them.
    """
    key_id = validate_memory_key_id(key_id)
    shared_root = Path(root).expanduser().resolve()
    state = Path(state_directory).expanduser().absolute()
    private_base = state / "private-memory"
    ensure_no_links(private_base)
    resolved_base = private_base.resolve()
    if resolved_base.is_relative_to(shared_root) or shared_root.is_relative_to(resolved_base):
        raise FileKnowledgeError("private memory storage must be outside the shared knowledge root")
    owner = private_base / key_id
    return _PrivateMemoryIndex(owner / "knowledge", owner / "state")

def private_memory_key_ids(state_directory: str | os.PathLike[str]) -> tuple[str, ...]:
    """List existing safe owner spaces, independently of the current key registry.

    No index constructor or mkdir is used here. Revoked/deleted owners remain
    readable while their files exist; links and incomplete spaces are excluded.
    """
    base = Path(state_directory).expanduser().absolute() / "private-memory"
    try:
        ensure_no_links(base)
        owners = tuple(base.iterdir())
    except (OSError, FileKnowledgeError):
        return ()
    result = []
    for owner in owners:
        if not is_valid_api_key_id(owner.name):
            continue
        try:
            _PrivateMemoryIndex(owner / "knowledge", owner / "state", existing_only=True)
        except (OSError, FileKnowledgeError):
            continue
        result.append(owner.name)
    return tuple(sorted(result))


class _ScopedSearchSource(FileKnowledgeIndex):
    """Apply virtual path filters before the existing iterator reads file bytes."""

    def __init__(self, index: FileKnowledgeIndex, prefix: str):
        self.index = index
        self.prefix = prefix
        self.table_name = index.table_name

    def _connect(self):
        return self.index._connect()

    def _safe_path(self, relative_path):
        return self.index._safe_path(relative_path)

    def _path_filter_matches(self, relative: str, path_filter: str | None) -> bool:
        if not self.prefix and relative.split("/", 1)[0] in {"private", "private-keys"}:
            return False
        return FileKnowledgeIndex._path_filter_matches(self.prefix + relative, path_filter)


class ScopedMemoryReadView(FileKnowledgeIndex):
    """Reuse the ordinary grep engine over one virtual, globally bounded corpus.

    No database or writable root belongs to this view. Mutations must use the
    owner's private index directly. Every operation enumerates current sources.
    """

    def __init__(self, shared: FileKnowledgeIndex, private: FileKnowledgeIndex, *,
                 key_id: str, memory_scope: str):
        try:
            memory_scope = validate_memory_scope(memory_scope)
        except ApiKeyStoreError as exc:
            raise FileKnowledgeError(str(exc)) from exc
        if memory_scope == "shared":
            raise FileKnowledgeError("shared memory does not use a private read view")
        self.shared = shared
        self.private = private
        self.key_id = validate_memory_key_id(key_id)
        self.memory_scope = memory_scope

    def read_sources(self) -> Iterator[tuple[str, FileKnowledgeIndex]]:
        """Yield virtual prefixes and existing read sources; never create others."""
        if self.memory_scope != "private_only":
            yield "", self.shared
        yield PRIVATE_PREFIX, self.private
        if self.memory_scope == "super":
            base = self.shared.state_directory / "private-memory"
            for key_id in private_memory_key_ids(self.shared.state_directory):
                if key_id == self.key_id:
                    continue
                owner = base / key_id
                try:
                    index = _PrivateMemoryIndex(owner / "knowledge", owner / "state", existing_only=True)
                except (OSError, FileKnowledgeError):
                    continue
                yield f"{OTHER_PRIVATE_PREFIX}{key_id}/", index

    def refresh(self):
        for _prefix, index in self.read_sources():
            index.refresh()

    def boilerplate_hashes(self) -> frozenset[int]:
        counts: Counter[int] = Counter()
        for _prefix, index in self.read_sources():
            with index._connect() as connection:
                counts.update(dict(connection.execute(
                    "SELECT hash, count(*) FROM knowledge_segments GROUP BY hash"
                ).fetchall()))
        return frozenset(
            value for value, count in counts.items()
            if count >= memory_grep.BOILERPLATE_MIN_FILES
        )

    def _search_documents(self, path_filter: str | None) -> Iterator[tuple[dict[str, Any], list[bytes]]]:
        for prefix, index in self.read_sources():
            source = _ScopedSearchSource(index, prefix)
            for entry, lines in source._search_documents(path_filter):
                yield {**entry, "path": prefix + entry["path"]}, lines

    def _read_index(self, path: str) -> tuple[FileKnowledgeIndex, str, str]:
        path = self._validate_relative_path(path)
        if path.startswith(PRIVATE_PREFIX):
            relative = path[len(PRIVATE_PREFIX):]
            self.private._validate_relative_path(relative)
            return self.private, relative, PRIVATE_PREFIX
        if path == "private":
            raise FileKnowledgeError("private memory path must name a file")
        if path == "private-keys" or path.startswith(OTHER_PRIVATE_PREFIX):
            if self.memory_scope != "super":
                raise FileKnowledgeError("other keys' private memory is not accessible")
            for prefix, index in self.read_sources():
                if prefix.startswith(OTHER_PRIVATE_PREFIX) and path.startswith(prefix):
                    relative = path[len(prefix):]
                    index._validate_relative_path(relative)
                    return index, relative, prefix
            raise FileKnowledgeError("private memory space does not exist")
        if self.memory_scope == "private_only":
            raise FileKnowledgeError("private_only reads require a private/ path; shared memory is not accessible")
        return self.shared, path, ""

    def read_lines(self, path: str, *, offset: int = 1, limit: int = 200) -> dict[str, Any]:
        index, relative, prefix = self._read_index(path)
        result = index.read_lines(relative, offset=offset, limit=limit)
        return {**result, "path": prefix + result["path"]}

    def read_bytes(self, path: str, *, byte_offset: int = 0, max_bytes: int = 8192) -> dict[str, Any]:
        index, relative, prefix = self._read_index(path)
        result = index.read_bytes(relative, byte_offset=byte_offset, max_bytes=max_bytes)
        return {**result, "path": prefix + result["path"]}
