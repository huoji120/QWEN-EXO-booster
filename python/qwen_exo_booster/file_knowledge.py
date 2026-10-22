"""File-backed ordinary Knowledge (the model's memory) and its SQLite index.

Document bytes stay on the filesystem and every read re-validates the source
file against its indexed identity. SQLite holds file identity, a coarse kind
and title per file, hashes of text segments (to recognise boilerplate repeated
across files) and the generated per-file summaries.
"""

from __future__ import annotations

import codecs
import fnmatch
import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from stat import S_ISREG
from typing import IO, Any, Iterable, Iterator

from qwen_exo_booster import memory_grep

SCHEMA_VERSION = 2
# Console file lists are paged. Callers that need every row (summary and dream
# workers) still use ``metadata``.
PAGE_SIZE_DEFAULT = 100
PAGE_SIZE_MAX = 200
WORK_MEMORY_ROOT = "work-memory"
MAX_READ_LIMIT = 1_048_576
MAX_SEARCH_RESULTS = 1_000
MAX_REGEX_LENGTH = 4_096
MAX_READ_LINES = 2_000
MAX_READ_BYTES = 65_536
MAX_CONTEXT_LINES = 20
MAX_PER_FILE = 50
# Longer lines (minified code, base64) are clipped so one hit cannot flood a
# model's context; the model can page through them with ``read_bytes``.
MAX_LINE_CHARS = 2_000
# Uploads have no size limit, but never fill the disk.
MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
_SNIFF_BYTES = 65_536
_COPY_CHUNK = 1024 * 1024


class FileKnowledgeError(ValueError):
    """Base class for rejected file-index operations."""


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class InvalidPathError(FileKnowledgeError):
    """The requested path is not a safe relative POSIX path."""


class InvalidEncodingError(FileKnowledgeError):
    """A file is not valid UTF-8 and therefore cannot be returned."""


class StaleFileError(FileKnowledgeError):
    """The filesystem bytes no longer match indexed metadata."""


class ReadLimitError(FileKnowledgeError):
    """A read request exceeds the bounded byte limit."""


class RegexConstraintError(FileKnowledgeError):
    """A regular expression is too large or invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_text_upload(data: bytes) -> None:
    """Accept any non-empty UTF-8 text (docs, code, logs, configs); memory is
    searched line by line, so binary content would only add noise."""
    if not data:
        raise FileKnowledgeError("file is empty")
    if b"\x00" in data:
        raise InvalidEncodingError("binary files are not supported")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidEncodingError("file is not valid UTF-8 text") from exc


def copy_validated_text(source: IO[bytes], destination: IO[bytes]) -> int:
    """Stream ``source`` into ``destination`` while checking it is non-empty
    UTF-8 text; returns the byte count. Nothing is held fully in memory."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    size = 0
    while chunk := source.read(_COPY_CHUNK):
        if b"\x00" in chunk:
            raise InvalidEncodingError("binary files are not supported")
        try:
            decoder.decode(chunk)
        except UnicodeDecodeError as exc:
            raise InvalidEncodingError("file is not valid UTF-8 text") from exc
        destination.write(chunk)
        size += len(chunk)
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise InvalidEncodingError("file is not valid UTF-8 text") from exc
    if not size:
        raise FileKnowledgeError("file is empty")
    return size


def ensure_free_space(directory: Path, incoming_bytes: int = 0) -> None:
    free = shutil.disk_usage(directory).free
    if free - max(0, incoming_bytes) < MIN_FREE_BYTES:
        raise FileKnowledgeError(
            f"not enough disk space: {free // (1024 * 1024)} MiB free, "
            f"{MIN_FREE_BYTES // (1024 ** 3)} GiB must stay free"
        )


def _archive_members(archive: Any) -> Iterator[tuple[Any, str]]:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/").strip("/")
        parts = name.split("/")
        if (
            info.is_dir()
            or not name
            or parts[0] == "__MACOSX"
            or any(part.startswith(".") for part in parts)
        ):
            continue
        yield info, name


def iter_text_archive(data: bytes) -> Iterator[tuple[str, bytes | None, str]]:
    """Yield ``(path, content, "")`` for each text member of a knowledge ZIP,
    or ``(path, None, reason)`` for a member that is skipped. Directories,
    macOS metadata and dotfiles are ignored."""
    import io
    import zipfile

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise FileKnowledgeError("not a valid ZIP archive") from exc
    with archive:
        for info, name in _archive_members(archive):
            if info.flag_bits & 0x1:
                yield name, None, "encrypted"
                continue
            with archive.open(info) as stream:
                content = stream.read()
            try:
                validate_text_upload(content)
            except FileKnowledgeError as exc:
                yield name, None, str(exc)
                continue
            yield name, content, ""


def extract_text_archive(
    data: bytes,
) -> tuple[list[tuple[str, bytes]], list[dict[str, str]]]:
    """Materialized :func:`iter_text_archive`: ``(members, skipped)``."""
    members: list[tuple[str, bytes]] = []
    skipped: list[dict[str, str]] = []
    for name, content, reason in iter_text_archive(data):
        if content is None:
            skipped.append({"path": name, "reason": reason})
        else:
            members.append((name, content))
    return members, skipped


def _aggregate_digest(items: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, source_digest in sorted(items):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _summary_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        for prefix in ("标题：", "标题:", "Title:", "title:"):
            if line.startswith(prefix):
                return line[len(prefix) :].strip()[:160]
        return line[:160]
    return ""


class FileKnowledgeIndex:
    """Index UTF-8 files below ``root`` without copying document content.

    ``state_directory`` contains ``knowledge-index.sqlite3``. Every search/read
    validates the source file against its indexed identity before returning.
    """

    table_name = "knowledge_files"

    def __init__(self, root: str | os.PathLike[str], state_directory: str | os.PathLike[str]):
        root_path = Path(root).expanduser()
        state_path = Path(state_directory).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        state_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()
        self.state_directory = state_path.resolve()
        self.database_path = self.state_directory / "knowledge-index.sqlite3"
        self.index_path = self.database_path
        self._boilerplate: tuple[tuple[int, int], frozenset[int]] | None = None
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=30, factory=_ClosingConnection
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    relative_path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    encoding TEXT NOT NULL,
                    line_count INTEGER NOT NULL,
                    indexed_ns INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'other',
                    title TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({self.table_name})")
            }
            if "kind" not in columns:
                connection.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN kind TEXT NOT NULL DEFAULT 'other'"
                )
            if "title" not in columns:
                connection.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN title TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_segments (
                    relative_path TEXT NOT NULL,
                    hash INTEGER NOT NULL,
                    PRIMARY KEY (relative_path, hash)
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_segments_hash ON knowledge_segments(hash)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_summaries (
                    relative_path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    sampled INTEGER NOT NULL,
                    origin TEXT NOT NULL,
                    updated_ns INTEGER NOT NULL
                )
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _validate_relative_path(value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value)
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        if not raw or "\x00" in raw or "\\" in raw:
            raise InvalidPathError("path must be a non-empty relative POSIX path")
        pure = PurePosixPath(raw)
        # PurePosixPath does not recognize Windows drive paths on all hosts.
        if pure.is_absolute() or PureWindowsPath(raw).drive or raw.startswith("//"):
            raise InvalidPathError("path must be relative to the knowledge root")
        if any(part in {"", ".", ".."} for part in pure.parts):
            raise InvalidPathError("path contains an unsafe component")
        normalized = "/".join(pure.parts)
        if normalized != raw:
            raise InvalidPathError("path must use normalized POSIX separators")
        return normalized

    @classmethod
    def _validate_glob(cls, value: str) -> str:
        raw = str(value)
        if not raw or "\x00" in raw or "\\" in raw:
            raise InvalidPathError("path_glob must be a relative POSIX glob")
        if raw.startswith("/") or raw.startswith("//") or PureWindowsPath(raw).drive:
            raise InvalidPathError("path_glob must be relative to the knowledge root")
        if any(part == ".." for part in raw.split("/")):
            raise InvalidPathError("path_glob contains an unsafe component")
        return raw

    def _safe_path(self, relative_path: str | os.PathLike[str]) -> tuple[str, Path]:
        normalized = self._validate_relative_path(relative_path)
        candidate = self.root.joinpath(*normalized.split("/"))
        # Check every existing component, not only the final entry: a symlinked
        # parent must not provide an alternate route into or out of the root.
        cursor = self.root
        for component in normalized.split("/"):
            cursor = cursor / component
            try:
                if cursor.is_symlink():
                    raise InvalidPathError("symlink paths are not allowed")
            except OSError as exc:
                raise InvalidPathError("path cannot be inspected") from exc
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise InvalidPathError("path escapes the knowledge root") from exc
        try:
            resolved.relative_to(self.state_directory)
        except ValueError:
            pass
        else:
            raise InvalidPathError("path points into the index state directory")
        return normalized, candidate

    def path_of(self, relative_path: str) -> Path:
        """Filesystem path of a validated relative path (may not exist)."""
        return self._safe_path(relative_path)[1]

    @staticmethod
    def _line_count(data: bytes) -> int:
        if not data:
            return 0
        return len(data.splitlines())

    @staticmethod
    def _metadata_from_bytes(
        data: bytes, mtime_ns: int, indexed_ns: int | None = None
    ) -> tuple[str, int, int, str, int, int]:
        digest = _sha256(data)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "invalid-utf-8"
            line_count = 0
        else:
            encoding = "utf-8"
            line_count = FileKnowledgeIndex._line_count(data)
        return (
            digest,
            len(data),
            int(mtime_ns),
            encoding,
            line_count,
            int(indexed_ns if indexed_ns is not None else time.time_ns()),
        )

    def _is_state_path(self, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(self.state_directory)
        except (OSError, ValueError):
            return False
        return True

    def _iter_regular_files(self) -> list[tuple[str, Path, os.stat_result]]:
        files: list[tuple[str, Path, os.stat_result]] = []
        for directory, dirnames, filenames in os.walk(self.root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not (directory_path / name).is_symlink()
                and not self._is_state_path(directory_path / name)
            )
            for name in sorted(filenames):
                if name.startswith(".knowledge-"):
                    continue
                candidate = directory_path / name
                if self._is_state_path(candidate):
                    continue
                try:
                    stat = candidate.lstat()
                except OSError:
                    continue
                if not S_ISREG(stat.st_mode):
                    continue
                if candidate.is_symlink():
                    continue
                try:
                    relative = candidate.relative_to(self.root).as_posix()
                    self._validate_relative_path(relative)
                except (InvalidPathError, ValueError):
                    continue
                files.append((relative, candidate, stat))
        return files

    def directories(self) -> list[str]:
        """Every folder below the root (including empty ones), sorted."""
        found: list[str] = []
        for directory, dirnames, _filenames in os.walk(self.root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".")
                and not (directory_path / name).is_symlink()
                and not self._is_state_path(directory_path / name)
            )
            for name in dirnames:
                found.append((directory_path / name).relative_to(self.root).as_posix())
        return sorted(found)

    def refresh(self) -> dict[str, int]:
        """Incrementally refresh metadata and remove entries for deleted files."""
        discovered = self._iter_regular_files()
        seen = {relative for relative, _path, _stat in discovered}
        now = time.time_ns()
        changed = 0
        reused = 0
        unavailable = 0
        with self._connect() as connection:
            existing = {
                row["relative_path"]: row
                for row in connection.execute(f"SELECT * FROM {self.table_name}")
            }
            connection.execute("BEGIN IMMEDIATE")
            for relative, path, stat in discovered:
                old = existing.get(relative)
                if (
                    old is not None
                    and int(old["mtime_ns"]) == int(stat.st_mtime_ns)
                    and int(old["byte_count"]) == int(stat.st_size)
                    and int(old["schema_version"]) == SCHEMA_VERSION
                ):
                    # Every read re-verifies the file (_read_checked), so an
                    # unchanged stat is enough here.
                    reused += 1
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                metadata = self._metadata_from_bytes(data, stat.st_mtime_ns, now)
                kind, title, hashes = "other", "", set()
                if metadata[3] == "utf-8":
                    text = data.decode("utf-8")
                    head = text[:_SNIFF_BYTES]
                    kind = memory_grep.detect_kind(relative, head)
                    title = memory_grep.detect_title(relative, kind, head)
                    hashes = memory_grep.file_segment_hashes(text)
                connection.execute(
                    f"""
                    INSERT INTO {self.table_name}
                      (relative_path, sha256, byte_count, mtime_ns, encoding,
                       line_count, indexed_ns, schema_version, kind, title)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                      sha256=excluded.sha256, byte_count=excluded.byte_count,
                      mtime_ns=excluded.mtime_ns, encoding=excluded.encoding,
                      line_count=excluded.line_count, indexed_ns=excluded.indexed_ns,
                      schema_version=excluded.schema_version, kind=excluded.kind,
                      title=excluded.title
                    """,
                    (relative, *metadata, SCHEMA_VERSION, kind, title),
                )
                connection.execute(
                    "DELETE FROM knowledge_segments WHERE relative_path = ?", (relative,)
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO knowledge_segments(relative_path, hash) VALUES (?, ?)",
                    ((relative, value) for value in hashes),
                )
                changed += 1
                unavailable += metadata[3] != "utf-8"
            removed = [path for path in existing if path not in seen]
            for relative in removed:
                connection.execute(
                    f"DELETE FROM {self.table_name} WHERE relative_path = ?", (relative,)
                )
                connection.execute(
                    "DELETE FROM knowledge_segments WHERE relative_path = ?", (relative,)
                )
            connection.commit()
        return {
            "files": len(seen),
            "changed": changed,
            "reused": reused,
            "unavailable": unavailable,
            "removed": len(removed),
        }

    def _row(self, relative: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                f"SELECT * FROM {self.table_name} WHERE relative_path = ?", (relative,)
            ).fetchone()

    @staticmethod
    def _read_checked(path: Path, row: sqlite3.Row, *, verify_digest: bool = True) -> bytes:
        """Read a source file, refusing it when it no longer matches the index.
        Bulk scans skip re-hashing (stat identity is checked before and after);
        reads that return bytes to a caller verify the digest too."""
        try:
            stat_before = path.stat()
            if (
                int(stat_before.st_size) != int(row["byte_count"])
                or int(stat_before.st_mtime_ns) != int(row["mtime_ns"])
            ):
                raise StaleFileError("file changed after indexing")
            data = path.read_bytes()
            stat_after = path.stat()
        except FileNotFoundError as exc:
            raise StaleFileError("file is missing") from exc
        except OSError as exc:
            raise StaleFileError("file cannot be read") from exc
        if (
            int(stat_after.st_size) != int(row["byte_count"])
            or int(stat_after.st_mtime_ns) != int(row["mtime_ns"])
            or (verify_digest and _sha256(data) != row["sha256"])
        ):
            raise StaleFileError("file changed after indexing")
        if row["encoding"] != "utf-8":
            raise InvalidEncodingError("file is not valid UTF-8")
        return data

    @staticmethod
    def _glob_match(path: str, pattern: str) -> bool:
        path_parts = path.split("/")
        pattern_parts = pattern.split("/")

        def matches(path_index: int, pattern_index: int) -> bool:
            if pattern_index == len(pattern_parts):
                return path_index == len(path_parts)
            part = pattern_parts[pattern_index]
            if part == "**":
                return matches(path_index, pattern_index + 1) or (
                    path_index < len(path_parts)
                    and matches(path_index + 1, pattern_index)
                )
            return path_index < len(path_parts) and fnmatch.fnmatchcase(
                path_parts[path_index], part
            ) and matches(path_index + 1, pattern_index + 1)

        return matches(0, 0)

    @staticmethod
    def _compile_pattern(pattern: str, case_sensitive: bool, regex: bool) -> re.Pattern[str]:
        if not isinstance(pattern, str) or not pattern:
            raise FileKnowledgeError("pattern must be a non-empty string")
        if len(pattern) > MAX_REGEX_LENGTH:
            raise RegexConstraintError("pattern is too long")
        flags = 0 if case_sensitive else re.IGNORECASE
        expression = pattern if regex else re.escape(pattern)
        try:
            return re.compile(expression, flags)
        except re.error as exc:
            raise RegexConstraintError("invalid regular expression") from exc

    def search(
        self,
        pattern: str,
        *,
        path_glob: str = "**/*",
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 50,
        context_before: int = 0,
        context_after: int = 0,
    ) -> dict[str, Any]:
        """Admin console search with exact byte offsets (not the model tool)."""
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise FileKnowledgeError("max_results must be an integer")
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise FileKnowledgeError("max_results is outside the allowed range")
        if not isinstance(context_before, int) or context_before < 0:
            raise FileKnowledgeError("context_before must be non-negative")
        if not isinstance(context_after, int) or context_after < 0:
            raise FileKnowledgeError("context_after must be non-negative")
        glob = self._validate_glob(path_glob)
        compiled = self._compile_pattern(pattern, bool(case_sensitive), bool(regex))
        self.refresh()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE encoding = 'utf-8' ORDER BY relative_path"
            ).fetchall()
        results: list[dict[str, Any]] = []
        source_items: list[tuple[str, str]] = []
        truncated = False
        for row in rows:
            relative = str(row["relative_path"])
            if not self._glob_match(relative, glob):
                continue
            _relative, path = self._safe_path(relative)
            try:
                data = self._read_checked(path, row, verify_digest=False)
            except StaleFileError:
                # Changed mid-scan; the next refresh re-indexes it.
                continue
            source_items.append((relative, str(row["sha256"])))
            raw_lines = data.splitlines(keepends=True)
            lines = [raw.decode("utf-8").rstrip("\r\n") for raw in raw_lines]
            next_line_start = 0
            for line_number, line_text in enumerate(lines, start=1):
                line_start = next_line_start
                next_line_start += len(raw_lines[line_number - 1])
                if not compiled.search(line_text):
                    continue
                for match in compiled.finditer(line_text):
                    prefix_bytes = line_text[: match.start()].encode("utf-8")
                    match_bytes = match.group(0).encode("utf-8")
                    result = {
                        "path": relative,
                        "sha256": str(row["sha256"]),
                        "source_digest": str(row["sha256"]),
                        "line": line_number,
                        "line_start_offset": line_start,
                        "match_start_offset": line_start + len(prefix_bytes),
                        "match_end_offset": line_start + len(prefix_bytes) + len(match_bytes),
                        "text": match.group(0),
                        "line_text": self._clip_line(line_text),
                        "before": lines[max(0, line_number - 1 - context_before) : line_number - 1],
                        "after": lines[line_number : line_number + context_after],
                        "truncated": False,
                    }
                    if len(results) >= max_results:
                        truncated = True
                        break
                    results.append(result)
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            for result in results:
                result["truncated"] = True
        return {
            "results": results,
            "count": len(results),
            "truncated": truncated,
            "source_digest": _aggregate_digest(source_items),
        }

    def read(self, path: str, *, offset: int = 0, limit: int = 4096) -> dict[str, Any]:
        relative, safe_path = self._safe_path(path)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise FileKnowledgeError("offset must be a non-negative byte offset")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ReadLimitError("limit must be a non-negative integer")
        if limit > MAX_READ_LIMIT:
            raise ReadLimitError(f"limit exceeds {MAX_READ_LIMIT} bytes")
        row = self._indexed_row(relative)
        data = self._read_checked(safe_path, row)
        total = len(data)
        if offset > total:
            raise FileKnowledgeError("offset is beyond end of file")
        try:
            data[:offset].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FileKnowledgeError("offset is not a UTF-8 codepoint boundary") from exc
        end = min(total, offset + limit)
        while end > offset:
            try:
                text = data[offset:end].decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            text = ""
        self._read_checked(safe_path, row)
        return {
            "path": relative,
            "sha256": str(row["sha256"]),
            "source_digest": str(row["sha256"]),
            "offset": offset,
            "end_offset": end,
            "total_bytes": total,
            "text": text,
            "eof": end >= total,
        }

    def _indexed_row(self, relative: str) -> sqlite3.Row:
        row = self._row(relative)
        if row is None:
            self.refresh()
            row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("path is not indexed")
        return row

    @staticmethod
    def _path_filter_matches(relative: str, path_filter: str | None) -> bool:
        if not path_filter:
            return True
        if any(char in path_filter for char in "*?["):
            # fnmatch's ``*`` also crosses ``/``, which is what a model expects
            # from ``docs/*.md``; a bare ``*.md`` additionally matches basenames.
            return fnmatch.fnmatch(relative, path_filter) or (
                "/" not in path_filter
                and fnmatch.fnmatch(relative.rsplit("/", 1)[-1], path_filter)
            )
        prefix = path_filter.rstrip("/")
        return relative == prefix or relative.startswith(prefix + "/")

    @staticmethod
    def _clip_line(text: str) -> str:
        if len(text) <= MAX_LINE_CHARS:
            return text
        return text[:MAX_LINE_CHARS] + f" …[clipped {len(text) - MAX_LINE_CHARS} chars]"

    def boilerplate_hashes(self) -> frozenset[int]:
        """Segment hashes present in at least ``BOILERPLATE_MIN_FILES`` files."""
        with self._connect() as connection:
            version = tuple(
                connection.execute(
                    f"SELECT count(*), coalesce(max(indexed_ns), 0) FROM {self.table_name}"
                ).fetchone()
            )
            if self._boilerplate is not None and self._boilerplate[0] == version:
                return self._boilerplate[1]
            hashes = frozenset(
                row[0]
                for row in connection.execute(
                    "SELECT hash FROM knowledge_segments GROUP BY hash HAVING count(*) >= ?",
                    (memory_grep.BOILERPLATE_MIN_FILES,),
                )
            )
        self._boilerplate = (version, hashes)
        return hashes

    def _search_documents(self, path_filter: str | None) -> Iterator[tuple[dict[str, Any], list[bytes]]]:
        """``(entry, raw_lines)`` for every searchable file and stored summary."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE encoding = 'utf-8' ORDER BY relative_path"
            ).fetchall()
            summaries = {
                row["relative_path"]: row
                for row in connection.execute("SELECT * FROM knowledge_summaries")
            }
        for path, summary in sorted(summaries.items()):
            virtual = path + memory_grep.SUMMARY_SUFFIX
            if not self._path_filter_matches(virtual, path_filter):
                continue
            yield (
                {"path": virtual, "kind": "summary", "title": summary["title"]},
                summary["summary"].encode("utf-8").splitlines(keepends=True),
            )
        for row in rows:
            relative = str(row["relative_path"])
            if not self._path_filter_matches(relative, path_filter):
                continue
            _relative, safe_path = self._safe_path(relative)
            try:
                data = self._read_checked(safe_path, row, verify_digest=False)
            except StaleFileError:
                continue
            summary = summaries.get(relative)
            yield (
                {
                    "path": relative,
                    "kind": str(row["kind"]),
                    "title": summary["title"] if summary is not None else str(row["title"]),
                },
                data.splitlines(keepends=True),
            )

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        context: int = 2,
        max_results: int = 60,
        per_file: int = 3,
        mode: str = "content",
        exclude: str = "",
        regex: bool = False,
        include_boilerplate: bool = False,
    ) -> dict[str, Any]:
        """The model's memory search (see :mod:`memory_grep` for the syntax).

        Files are ranked by kind (summaries and notes before raw trajectories)
        and match count; each shows at most ``per_file`` hits and the
        ``max_results`` budget is handed out round-robin over ranked files.
        ``mode="files"`` lists matching files only, ``mode="count"`` adds
        totals per kind. Boilerplate segments repeated across many files are
        hidden unless ``include_boilerplate``.
        """
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise FileKnowledgeError("max_results must be an integer")
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise FileKnowledgeError(f"max_results must be within 1..{MAX_SEARCH_RESULTS}")
        if not isinstance(per_file, int) or isinstance(per_file, bool) or not 1 <= per_file <= MAX_PER_FILE:
            raise FileKnowledgeError(f"per_file must be within 1..{MAX_PER_FILE}")
        if not isinstance(context, int) or isinstance(context, bool):
            raise FileKnowledgeError("context must be an integer")
        if not 0 <= context <= MAX_CONTEXT_LINES:
            raise FileKnowledgeError(f"context must be within 0..{MAX_CONTEXT_LINES}")
        if mode not in {"content", "files", "count"}:
            raise FileKnowledgeError("mode must be content, files or count")
        if path is not None:
            self._validate_glob(path)
        try:
            query = memory_grep.parse_query(pattern, is_regex=bool(regex), exclude=exclude or "")
        except ValueError as exc:
            raise RegexConstraintError(str(exc)) from exc
        deadline = time.monotonic() + memory_grep.SEARCH_SECONDS
        self.refresh()
        boilerplate = self.boilerplate_hashes()
        files: list[dict[str, Any]] = []
        hidden = 0
        stopped = False
        for entry, raw_lines in self._search_documents(path):
            scan = memory_grep.scan_lines(
                entry["path"],
                raw_lines,
                query,
                boilerplate=boilerplate,
                include_boilerplate=bool(include_boilerplate),
                context=context,
                per_file=per_file if mode == "content" else 0,
                deadline=deadline,
            )
            hidden += scan.hidden_boilerplate
            if scan.match_count:
                files.append(
                    {
                        **entry,
                        "match_count": scan.match_count,
                        "capped": scan.capped,
                        "candidates": scan.candidates,
                    }
                )
            if scan.stopped:
                stopped = True
                break
        files.sort(key=memory_grep.rank_key)
        shown = {entry["path"]: 0 for entry in files}
        budget = max_results if mode == "content" else 0
        for depth in range(per_file):
            for entry in files:
                if budget and depth < len(entry["candidates"]):
                    shown[entry["path"]] += 1
                    budget -= 1
        hits = memory_grep.fold_duplicates(
            [hit for entry in files for hit in entry["candidates"][: shown[entry["path"]]]]
        )
        total = sum(entry["match_count"] for entry in files)
        by_kind: dict[str, dict[str, int]] = {}
        for entry in files:
            bucket = by_kind.setdefault(entry["kind"], {"files": 0, "matches": 0})
            bucket["files"] += 1
            bucket["matches"] += entry["match_count"]
        return {
            "mode": mode,
            "hits": hits,
            "count": len(hits),
            "total_matches": total,
            "file_count": len(files),
            "files": [
                {
                    "path": entry["path"],
                    "kind": entry["kind"],
                    "title": entry["title"],
                    "match_count": entry["match_count"],
                    "capped": entry["capped"],
                    "shown": shown[entry["path"]],
                }
                for entry in files
            ],
            "by_kind": by_kind,
            "hidden_boilerplate": hidden,
            "per_file": per_file,
            "truncated": sum(shown.values()) < total,
            # The time budget ran out: later files were not searched.
            "stopped": stopped,
            "capped": any(entry["capped"] for entry in files),
        }

    def _summary_lines(self, relative: str) -> list[bytes] | None:
        if not relative.endswith(memory_grep.SUMMARY_SUFFIX):
            return None
        row = self.summary(relative[: -len(memory_grep.SUMMARY_SUFFIX)])
        if row is None:
            return None
        return row["summary"].encode("utf-8").splitlines(keepends=True)

    def _document_bytes(self, path: str) -> tuple[str, bytes]:
        """``(relative, bytes)`` of a file, or of a stored summary's virtual file."""
        relative = self._validate_relative_path(path)
        _relative, safe_path = self._safe_path(relative)
        if not safe_path.exists():
            summary = self._summary_lines(relative)
            if summary is not None:
                return relative, b"".join(summary)
        row = self._indexed_row(relative)
        return relative, self._read_checked(safe_path, row)

    def read_lines(self, path: str, *, offset: int = 1, limit: int = 200) -> dict[str, Any]:
        """Read ``limit`` lines starting at 1-based line ``offset``. Lines are
        split like the search (so numbers agree); long lines are clipped and
        carry their byte offset for :meth:`read_bytes`."""
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise FileKnowledgeError("offset must be a 1-based line number")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ReadLimitError("limit must be an integer")
        if not 1 <= limit <= MAX_READ_LINES:
            raise ReadLimitError(f"limit must be within 1..{MAX_READ_LINES} lines")
        relative, data = self._document_bytes(path)
        raw_lines = data.splitlines(keepends=True)
        total = len(raw_lines)
        if offset > max(total, 1):
            raise FileKnowledgeError(f"offset is beyond end of file ({total} lines)")
        end = min(total, offset - 1 + limit)
        position = sum(len(raw) for raw in raw_lines[: offset - 1])
        lines = []
        for number in range(offset - 1, end):
            text = raw_lines[number].decode("utf-8", errors="replace").rstrip("\r\n")
            lines.append(
                {
                    "line": number + 1,
                    "text": self._clip_line(text),
                    "byte_offset": position,
                    "clipped": len(text) > MAX_LINE_CHARS,
                }
            )
            position += len(raw_lines[number])
        return {
            "path": relative,
            "offset": offset,
            "end_line": end,
            "total_lines": total,
            "total_bytes": len(data),
            "lines": lines,
            "eof": end >= total,
        }

    def read_bytes(self, path: str, *, byte_offset: int = 0, max_bytes: int = 8192) -> dict[str, Any]:
        """Read a byte range (for very long lines), aligned to UTF-8 characters."""
        if not isinstance(byte_offset, int) or isinstance(byte_offset, bool) or byte_offset < 0:
            raise FileKnowledgeError("byte_offset must be a non-negative integer")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= MAX_READ_BYTES:
            raise ReadLimitError(f"max_bytes must be within 1..{MAX_READ_BYTES}")
        relative, data = self._document_bytes(path)
        total = len(data)
        if byte_offset >= max(total, 1):
            raise FileKnowledgeError(f"byte_offset is beyond end of file ({total} bytes)")
        start = byte_offset
        while start < total and data[start] & 0xC0 == 0x80:
            start += 1
        end = min(total, start + max_bytes)
        while end < total and end > start and data[end] & 0xC0 == 0x80:
            end -= 1
        return {
            "path": relative,
            "byte_offset": start,
            "end_offset": end,
            "total_bytes": total,
            "total_lines": len(data.splitlines()),
            "line": data.count(b"\n", 0, start) + 1,
            "text": data[start:end].decode("utf-8", errors="replace"),
            "eof": end >= total,
        }

    # --- writes ---------------------------------------------------------------------

    def upsert_bytes(self, path: str, data: bytes | bytearray | memoryview) -> dict[str, Any]:
        ensure_free_space(self.root, len(data))
        relative = self._write_atomic(path, bytes(data))
        self.refresh()
        row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("upserted file was not indexed")
        return {key: row[key] for key in row.keys()}

    def store_upload(self, path: str, source: IO[bytes]) -> dict[str, Any]:
        """Stream one uploaded text file into place (validated while copying)."""
        relative = self._write_stream(path, source)
        self.refresh()
        row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("uploaded file was not indexed")
        return {key: row[key] for key in row.keys()}

    def import_archive(
        self, data: bytes | IO[bytes] | os.PathLike[str] | str, *, directory: str = ""
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Store each text member of a ZIP (bytes, a file object or a path) as
        its own file under ``directory``; returns ``(imported rows, skipped)``.
        Members are streamed to disk one by one, then the index refreshes once."""
        import io
        import zipfile

        source = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        try:
            archive = zipfile.ZipFile(source)
        except zipfile.BadZipFile as exc:
            raise FileKnowledgeError("not a valid ZIP archive") from exc
        prefix = directory.strip().strip("/")
        written: list[str] = []
        skipped: list[dict[str, str]] = []
        with archive:
            for info, name in _archive_members(archive):
                relative_path = f"{prefix}/{name}" if prefix else name
                if info.flag_bits & 0x1:
                    skipped.append({"path": relative_path, "reason": "encrypted"})
                    continue
                try:
                    with archive.open(info) as stream:
                        written.append(
                            self._write_stream(relative_path, stream, expected_bytes=info.file_size)
                        )
                except FileKnowledgeError as exc:
                    skipped.append({"path": relative_path, "reason": str(exc)})
        self.refresh()
        rows = [row for relative in written if (row := self._row(relative)) is not None]
        return [{key: row[key] for key in row.keys()} for row in rows], skipped

    def _write_stream(self, path: str, source: IO[bytes], *, expected_bytes: int = 0) -> str:
        relative, destination = self._safe_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._safe_path(relative)
        ensure_free_space(self.root, expected_bytes)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".knowledge-", delete=False
            ) as stream:
                temporary = stream.name
                copy_validated_text(source, stream)
                stream.flush()
                os.fsync(stream.fileno())
            ensure_free_space(self.root)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        return relative

    def _write_atomic(self, path: str, raw: bytes) -> str:
        relative, destination = self._safe_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Re-check newly-created parent components before writing through them.
        self._safe_path(relative)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".knowledge-", delete=False) as stream:
                temporary = stream.name
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        return relative

    def upsert(self, path: str, data: bytes | bytearray | memoryview) -> dict[str, Any]:
        """Atomically replace a source file with raw bytes and refresh metadata."""
        return self.upsert_bytes(path, data)

    def _forget(self, connection: sqlite3.Connection, relative: str) -> None:
        connection.execute(f"DELETE FROM {self.table_name} WHERE relative_path = ?", (relative,))
        connection.execute("DELETE FROM knowledge_segments WHERE relative_path = ?", (relative,))
        connection.execute("DELETE FROM knowledge_summaries WHERE relative_path = ?", (relative,))

    def delete(self, path: str) -> None:
        """Delete a source file; raises ``FileNotFoundError`` when it is absent."""
        relative, target = self._safe_path(path)
        try:
            if target.is_symlink():
                raise InvalidPathError("symlink paths are not allowed")
            target.unlink()
        except FileNotFoundError:
            with self._connect() as connection:
                self._forget(connection, relative)
            raise
        except OSError as exc:
            raise FileKnowledgeError("file could not be deleted") from exc
        with self._connect() as connection:
            self._forget(connection, relative)

    def make_directory(self, path: str) -> str:
        relative, target = self._safe_path(path)
        if target.exists() and not target.is_dir():
            raise FileKnowledgeError("a file with that name already exists")
        target.mkdir(parents=True, exist_ok=True)
        self._safe_path(relative)
        return relative

    def delete_directory(self, path: str) -> int:
        """Remove a folder with everything inside; returns the files removed."""
        relative, target = self._safe_path(path)
        if not target.is_dir():
            raise FileNotFoundError(relative)
        with self._connect() as connection:
            inside = [
                row[0]
                for row in connection.execute(
                    f"SELECT relative_path FROM {self.table_name} WHERE relative_path LIKE ? ESCAPE '\\'",
                    (self._like_prefix(relative),),
                )
            ]
        shutil.rmtree(target)
        with self._connect() as connection:
            for item in inside:
                self._forget(connection, item)
        return len(inside)

    @staticmethod
    def _like_prefix(relative: str) -> str:
        escaped = relative.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return escaped + "/%"

    def move(self, source: str, destination: str) -> dict[str, Any]:
        """Rename or move a file or folder; index rows and summaries follow it."""
        source_relative, source_path = self._safe_path(source)
        destination_relative, destination_path = self._safe_path(destination)
        if not source_path.exists():
            raise FileNotFoundError(source_relative)
        if destination_path.exists():
            raise FileKnowledgeError(f"{destination_relative} already exists")
        if source_path.is_dir() and (destination_relative + "/").startswith(source_relative + "/"):
            raise FileKnowledgeError("a folder cannot be moved into itself")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._safe_path(destination_relative)
        is_directory = source_path.is_dir()
        os.rename(source_path, destination_path)
        moved = 0
        with self._connect() as connection:
            if is_directory:
                pairs = [
                    (row[0], destination_relative + row[0][len(source_relative) :])
                    for row in connection.execute(
                        f"SELECT relative_path FROM {self.table_name} WHERE relative_path LIKE ? ESCAPE '\\'",
                        (self._like_prefix(source_relative),),
                    )
                ]
            else:
                pairs = [(source_relative, destination_relative)]
            for old, new in pairs:
                for table in (self.table_name, "knowledge_segments", "knowledge_summaries"):
                    connection.execute(
                        f"UPDATE {table} SET relative_path = ? WHERE relative_path = ?",
                        (new, old),
                    )
                moved += 1
            connection.commit()
        return {"from": source_relative, "to": destination_relative, "files": moved}

    # --- metadata and summaries ---------------------------------------------------------

    _FILE_SELECT = """
        SELECT f.*, s.sha256 AS summary_sha256, s.title AS summary_title,
               s.origin AS summary_origin, s.sampled AS summary_sampled,
               s.updated_ns AS summary_updated_ns
        FROM {table} f
        LEFT JOIN knowledge_summaries s ON s.relative_path = f.relative_path
    """

    @staticmethod
    def _public_file_row(row: sqlite3.Row) -> dict[str, Any]:
        item = {key: row[key] for key in row.keys() if not str(key).startswith("summary_")}
        if row["summary_sha256"] is None:
            item["summary"] = None
            return item
        item["summary"] = {
            "title": row["summary_title"],
            "origin": row["summary_origin"],
            "sampled": bool(row["summary_sampled"]),
            "updated_ns": row["summary_updated_ns"],
            "stale": row["summary_sha256"] != row["sha256"],
        }
        return item

    def metadata(self) -> list[dict[str, Any]]:
        """Every metadata row (no file content), with each file's summary state.

        Summary generation and dream still walk this full list on the server.
        The console must use ``page_files`` instead of shipping it to the browser.
        """
        sql = self._FILE_SELECT.format(table=self.table_name) + " ORDER BY f.relative_path"
        with self._connect() as connection:
            rows = connection.execute(sql).fetchall()
        return [self._public_file_row(row) for row in rows]

    def directory_summary(self) -> dict[str, Any]:
        """Directory paths and recursive file counts. Does not return file rows.

        ``counts[""]`` is the whole corpus. ``areas`` splits imported files from
        ``work-memory/``. Empty folders are included with a zero count.
        """
        directories = self.directories()
        counts: dict[str, dict[str, int]] = {"": {"files": 0, "bytes": 0}}
        areas = {
            "imported": {"files": 0, "bytes": 0},
            "work": {"files": 0, "bytes": 0},
        }
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT relative_path, byte_count FROM {self.table_name}"
            )
            for row in rows:
                path = str(row["relative_path"])
                size = int(row["byte_count"])
                counts[""]["files"] += 1
                counts[""]["bytes"] += size
                work = path == WORK_MEMORY_ROOT or path.startswith(f"{WORK_MEMORY_ROOT}/")
                bucket = areas["work" if work else "imported"]
                bucket["files"] += 1
                bucket["bytes"] += size
                parent = path.rpartition("/")[0]
                while parent:
                    slot = counts.setdefault(parent, {"files": 0, "bytes": 0})
                    slot["files"] += 1
                    slot["bytes"] += size
                    parent = parent.rpartition("/")[0]
        for directory in directories:
            counts.setdefault(directory, {"files": 0, "bytes": 0})
        return {"directories": directories, "counts": counts, "areas": areas}

    def page_files(
        self,
        *,
        folder: str = "",
        nested: bool = True,
        query: str = "",
        area: str = "all",
        limit: int = PAGE_SIZE_DEFAULT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """One page of file metadata for ``folder`` (``""`` is the corpus root).

        A flat corpus (every file directly under the root) is the ``folder=""``
        case. ``nested`` includes files in subfolders. ``query`` matches the
        relative path or detected title. The response never includes file bytes
        or summary text.
        """
        if area not in {"all", "imported", "work"}:
            raise FileKnowledgeError("area must be all, imported or work")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise FileKnowledgeError("limit must be an integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise FileKnowledgeError("offset must be a non-negative integer")
        if not 1 <= limit <= PAGE_SIZE_MAX:
            raise FileKnowledgeError(f"limit must be between 1 and {PAGE_SIZE_MAX}")
        folder_path = self._validate_relative_path(folder) if str(folder).strip() else ""
        where: list[str] = []
        params: list[Any] = []
        work_like = self._like_prefix(WORK_MEMORY_ROOT)
        if area == "work":
            where.append("(f.relative_path = ? OR f.relative_path LIKE ? ESCAPE '\\')")
            params.extend([WORK_MEMORY_ROOT, work_like])
        elif area == "imported":
            where.append(
                "NOT (f.relative_path = ? OR f.relative_path LIKE ? ESCAPE '\\')"
            )
            params.extend([WORK_MEMORY_ROOT, work_like])
        if folder_path:
            prefix = self._like_prefix(folder_path)
            if nested:
                where.append("f.relative_path LIKE ? ESCAPE '\\'")
                params.append(prefix)
            else:
                where.append(
                    "f.relative_path LIKE ? ESCAPE '\\' "
                    "AND f.relative_path NOT LIKE ? ESCAPE '\\'"
                )
                params.extend([prefix, prefix + "/%"])
        elif not nested:
            where.append("instr(f.relative_path, '/') = 0")
        needle = str(query or "").strip().lower()
        if needle:
            escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like = f"%{escaped}%"
            where.append(
                "(lower(f.relative_path) LIKE ? ESCAPE '\\' "
                "OR lower(f.title) LIKE ? ESCAPE '\\')"
            )
            params.extend([like, like])
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        sql_from = (
            f"FROM {self.table_name} f "
            "LEFT JOIN knowledge_summaries s ON s.relative_path = f.relative_path"
        )
        with self._connect() as connection:
            total, total_bytes = connection.execute(
                f"SELECT COUNT(*), COALESCE(SUM(f.byte_count), 0) {sql_from}{clause}",
                params,
            ).fetchone()
            rows = connection.execute(
                self._FILE_SELECT.format(table=self.table_name)
                + clause
                + " ORDER BY f.relative_path LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return {
            "files": [self._public_file_row(row) for row in rows],
            "total": int(total or 0),
            "total_bytes": int(total_bytes or 0),
            "offset": offset,
            "limit": limit,
        }

    def summary(self, path: str) -> dict[str, Any] | None:
        relative = self._validate_relative_path(path)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_summaries WHERE relative_path = ?", (relative,)
            ).fetchone()
        return None if row is None else {key: row[key] for key in row.keys()}

    def store_summary(
        self, path: str, *, sha256: str, summary: str, sampled: bool, origin: str
    ) -> dict[str, Any]:
        relative = self._validate_relative_path(path)
        text = str(summary).strip()
        if not text:
            raise FileKnowledgeError("summary is empty")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_summaries
                  (relative_path, sha256, title, summary, sampled, origin, updated_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                  sha256=excluded.sha256, title=excluded.title, summary=excluded.summary,
                  sampled=excluded.sampled, origin=excluded.origin,
                  updated_ns=excluded.updated_ns
                """,
                (relative, sha256, _summary_title(text), text, int(bool(sampled)), origin, time.time_ns()),
            )
        return self.summary(relative) or {}

    def delete_summary(self, path: str) -> bool:
        relative = self._validate_relative_path(path)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM knowledge_summaries WHERE relative_path = ?", (relative,)
            )
            return cursor.rowcount > 0

    def summary_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                {key: row[key] for key in row.keys()}
                for row in connection.execute(
                    "SELECT * FROM knowledge_summaries ORDER BY relative_path"
                )
            ]

    def read_text(self, path: str) -> tuple[str, str]:
        """``(sha256, text)`` of an indexed file, digest-verified."""
        relative, safe_path = self._safe_path(path)
        row = self._indexed_row(relative)
        return str(row["sha256"]), self._read_checked(safe_path, row).decode("utf-8")


__all__ = [
    "FileKnowledgeError",
    "FileKnowledgeIndex",
    "InvalidEncodingError",
    "InvalidPathError",
    "ReadLimitError",
    "RegexConstraintError",
    "StaleFileError",
    "MAX_CONTEXT_LINES",
    "MAX_PER_FILE",
    "MAX_READ_BYTES",
    "MAX_READ_LIMIT",
    "MAX_READ_LINES",
    "MAX_SEARCH_RESULTS",
    "MIN_FREE_BYTES",
    "copy_validated_text",
    "ensure_free_space",
    "extract_text_archive",
    "iter_text_archive",
    "validate_text_upload",
]
