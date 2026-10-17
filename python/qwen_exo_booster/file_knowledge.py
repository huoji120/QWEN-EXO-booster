"""Secure, metadata-only file-backed ordinary Knowledge index.

The index deliberately keeps document bytes on the filesystem.  SQLite contains
only file identity and validation metadata; callers must use :meth:`search` or
:meth:`read` to obtain bytes from the still-authoritative source file.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from stat import S_ISREG
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1
MAX_READ_LIMIT = 1_048_576
MAX_SEARCH_RESULTS = 1_000
MAX_REGEX_LENGTH = 4_096
MAX_READ_LINES = 2_000
MAX_CONTEXT_LINES = 20
# Longer lines (minified code, base64) are clipped so one hit cannot flood a
# model's context; the model can still page through with ``read_lines``.
MAX_LINE_CHARS = 2_000


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


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
# A ZIP upload itself, and the total it may expand to.
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_000
MAX_ARCHIVE_TOTAL_BYTES = 1024 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_text_upload(data: bytes) -> None:
    """Accept any non-empty UTF-8 text (docs, code, logs, configs); memory is
    searched line by line, so binary content would only add noise."""
    if not data:
        raise FileKnowledgeError("file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileKnowledgeError(
            f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
        )
    if b"\x00" in data:
        raise InvalidEncodingError("binary files are not supported")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidEncodingError("file is not valid UTF-8 text") from exc


def iter_text_archive(data: bytes) -> Iterator[tuple[str, bytes | None, str]]:
    """Yield ``(path, content, "")`` for each text member of a knowledge ZIP,
    or ``(path, None, reason)`` for a member that is skipped.

    Members are read one at a time, so a large archive never sits fully
    expanded in memory. Each keeps its in-archive path (callers still route it
    through path validation, which rejects traversal). Directories, macOS
    metadata and dotfiles are ignored. Sizes are checked from headers *and*
    while reading, so a forged header cannot inflate past the limits.
    """
    import io
    import zipfile

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise FileKnowledgeError("not a valid ZIP archive") from exc
    accepted = 0
    total = 0
    with archive:
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
            if accepted >= MAX_ARCHIVE_MEMBERS:
                raise FileKnowledgeError(
                    f"archive has more than {MAX_ARCHIVE_MEMBERS} files"
                )
            if info.flag_bits & 0x1:
                yield name, None, "encrypted"
                continue
            if info.file_size > MAX_UPLOAD_BYTES:
                yield name, None, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
                continue
            with archive.open(info) as stream:
                content = stream.read(MAX_UPLOAD_BYTES + 1)
            total += len(content)
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise FileKnowledgeError(
                    f"archive expands beyond {MAX_ARCHIVE_TOTAL_BYTES // (1024 * 1024)} MiB"
                )
            try:
                validate_text_upload(content)
            except FileKnowledgeError as exc:
                yield name, None, str(exc)
                continue
            accepted += 1
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


class FileKnowledgeIndex:
    """Index UTF-8 files below ``root`` without copying document content.

    ``state_directory`` contains only ``knowledge-index.sqlite3``.  The class
    intentionally has no content cache: every search/read validates the source
    file's indexed digest before returning anything.
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
                    schema_version INTEGER NOT NULL
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
        return normalized, candidate

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

    def _iter_regular_files(self) -> list[tuple[str, Path, os.stat_result]]:
        files: list[tuple[str, Path, os.stat_result]] = []
        for directory, dirnames, filenames in os.walk(self.root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            dirnames[:] = sorted(
                name for name in dirnames if not (directory_path / name).is_symlink()
            )
            for name in sorted(filenames):
                candidate = directory_path / name
                try:
                    candidate.resolve().relative_to(self.state_directory)
                except (OSError, ValueError):
                    pass
                else:
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
                    # Every read re-verifies the digest (_read_checked), so an
                    # unchanged stat is enough here; re-hashing the whole tree
                    # on each search scaled with the knowledge base size.
                    reused += 1
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                metadata = self._metadata_from_bytes(data, stat.st_mtime_ns, now)
                connection.execute(
                    f"""
                    INSERT INTO {self.table_name}
                      (relative_path, sha256, byte_count, mtime_ns, encoding,
                       line_count, indexed_ns, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                      sha256=excluded.sha256, byte_count=excluded.byte_count,
                      mtime_ns=excluded.mtime_ns, encoding=excluded.encoding,
                      line_count=excluded.line_count, indexed_ns=excluded.indexed_ns,
                      schema_version=excluded.schema_version
                    """,
                    (relative, *metadata, SCHEMA_VERSION),
                )
                changed += 1
                unavailable += metadata[3] != "utf-8"
            if seen:
                placeholders = ",".join("?" for _ in seen)
                connection.execute(
                    f"DELETE FROM {self.table_name} WHERE relative_path NOT IN ({placeholders})",
                    tuple(sorted(seen)),
                )
            else:
                connection.execute(f"DELETE FROM {self.table_name}")
            connection.commit()
        return {
            "files": len(seen),
            "changed": changed,
            "reused": reused,
            "unavailable": unavailable,
        }

    def _row(self, relative: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                f"SELECT * FROM {self.table_name} WHERE relative_path = ?", (relative,)
            ).fetchone()

    @staticmethod
    def _read_checked(path: Path, row: sqlite3.Row) -> bytes:
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
            or _sha256(data) != row["sha256"]
        ):
            raise StaleFileError("file changed after indexing")
        if row["encoding"] != "utf-8":
            raise InvalidEncodingError("file is not valid UTF-8")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidEncodingError("file is not valid UTF-8") from exc
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
                data = self._read_checked(path, row)
            except StaleFileError:
                # Changed mid-scan; the next refresh re-indexes it.
                continue
            source_items.append((relative, str(row["sha256"])))
            raw_lines = data.splitlines(keepends=True)
            lines = [raw.decode("utf-8").rstrip("\r\n") for raw in raw_lines]
            # Running byte offset (re-summing all previous lines per line was
            # quadratic: a 700k-line file never finished).
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
        row = self._row(relative)
        if row is None:
            self.refresh()
            row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("path is not indexed")
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

    @staticmethod
    def _grep_expression(pattern: str) -> re.Pattern[str]:
        """Compile a grep-like pattern: ``*`` matches anything within a line,
        ``|`` separates alternatives, every other character is literal."""
        if not isinstance(pattern, str) or not pattern.strip():
            raise FileKnowledgeError("pattern must be a non-empty string")
        if len(pattern) > MAX_REGEX_LENGTH:
            raise RegexConstraintError("pattern is too long")
        alternatives = [part.strip() for part in pattern.split("|")]
        alternatives = [part for part in alternatives if part.strip("*")]
        if not alternatives:
            raise FileKnowledgeError("pattern must contain non-wildcard text")
        expression = "|".join(
            ".*?".join(re.escape(piece) for piece in part.split("*"))
            for part in alternatives
        )
        return re.compile(expression, re.IGNORECASE)

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

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        context: int = 2,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """Line-oriented grep over UTF-8 files: one hit per matching line with
        numbered context lines, ready for :meth:`read_lines` follow-ups."""
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise FileKnowledgeError("max_results must be an integer")
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise FileKnowledgeError(f"max_results must be within 1..{MAX_SEARCH_RESULTS}")
        if not isinstance(context, int) or isinstance(context, bool):
            raise FileKnowledgeError("context must be an integer")
        if not 0 <= context <= MAX_CONTEXT_LINES:
            raise FileKnowledgeError(f"context must be within 0..{MAX_CONTEXT_LINES}")
        if path is not None:
            self._validate_glob(path)
        compiled = self._grep_expression(pattern)
        self.refresh()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE encoding = 'utf-8' ORDER BY relative_path"
            ).fetchall()
        hits: list[dict[str, Any]] = []
        matched_files: set[str] = set()
        truncated = False
        for row in rows:
            relative = str(row["relative_path"])
            if not self._path_filter_matches(relative, path):
                continue
            _relative, safe_path = self._safe_path(relative)
            try:
                lines = self._read_checked(safe_path, row).decode("utf-8").splitlines()
            except StaleFileError:
                continue
            for index, line in enumerate(lines):
                if not compiled.search(line):
                    continue
                if len(hits) >= max_results:
                    truncated = True
                    break
                matched_files.add(relative)
                first = max(0, index - context)
                hits.append(
                    {
                        "path": relative,
                        "line": index + 1,
                        "text": self._clip_line(line),
                        "context": [
                            {"line": number + 1, "text": self._clip_line(lines[number])}
                            for number in range(first, min(len(lines), index + context + 1))
                            if number != index
                        ],
                    }
                )
            if truncated:
                break
        return {
            "hits": hits,
            "count": len(hits),
            "file_count": len(matched_files),
            "truncated": truncated,
        }

    def read_lines(self, path: str, *, offset: int = 1, limit: int = 200) -> dict[str, Any]:
        """Read ``limit`` lines starting at 1-based line ``offset``."""
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise FileKnowledgeError("offset must be a 1-based line number")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ReadLimitError("limit must be an integer")
        if not 1 <= limit <= MAX_READ_LINES:
            raise ReadLimitError(f"limit must be within 1..{MAX_READ_LINES} lines")
        relative, safe_path = self._safe_path(path)
        row = self._row(relative)
        if row is None:
            self.refresh()
            row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("path is not indexed")
        lines = self._read_checked(safe_path, row).decode("utf-8").splitlines()
        total = len(lines)
        if offset > max(total, 1):
            raise FileKnowledgeError(f"offset is beyond end of file ({total} lines)")
        end = min(total, offset - 1 + limit)
        return {
            "path": relative,
            "offset": offset,
            "end_line": end,
            "total_lines": total,
            "lines": [
                {"line": number + 1, "text": self._clip_line(lines[number])}
                for number in range(offset - 1, end)
            ],
            "eof": end >= total,
        }

    def upsert_bytes(self, path: str, data: bytes | bytearray | memoryview) -> dict[str, Any]:
        relative = self._write_atomic(path, bytes(data))
        self.refresh()
        row = self._row(relative)
        if row is None:
            raise FileKnowledgeError("upserted file was not indexed")
        return {key: row[key] for key in row.keys()}

    def import_archive(
        self, data: bytes, *, directory: str = ""
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Store each text member of a ZIP as its own file under
        ``directory``; returns ``(imported rows, skipped)``. Members are
        written one by one and the index is refreshed once at the end."""
        prefix = directory.strip().strip("/")
        written: list[str] = []
        skipped: list[dict[str, str]] = []
        for name, content, reason in iter_text_archive(data):
            relative_path = f"{prefix}/{name}" if prefix else name
            if content is None:
                skipped.append({"path": relative_path, "reason": reason})
                continue
            try:
                written.append(self._write_atomic(relative_path, content))
            except FileKnowledgeError as exc:
                skipped.append({"path": relative_path, "reason": str(exc)})
        self.refresh()
        rows = [row for relative in written if (row := self._row(relative)) is not None]
        return [{key: row[key] for key in row.keys()} for row in rows], skipped

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

    def delete(self, path: str) -> None:
        """Delete a source file; raises ``FileNotFoundError`` when it is absent."""
        relative, target = self._safe_path(path)
        try:
            if target.is_symlink():
                raise InvalidPathError("symlink paths are not allowed")
            target.unlink()
        except FileNotFoundError:
            with self._connect() as connection:
                connection.execute(
                    f"DELETE FROM {self.table_name} WHERE relative_path = ?", (relative,)
                )
            raise
        except OSError as exc:
            raise FileKnowledgeError("file could not be deleted") from exc
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM {self.table_name} WHERE relative_path = ?", (relative,)
            )

    def metadata(self) -> list[dict[str, Any]]:
        """Return metadata rows only; no filesystem content is included."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table_name} ORDER BY relative_path"
            ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]


__all__ = [
    "FileKnowledgeError",
    "FileKnowledgeIndex",
    "InvalidEncodingError",
    "InvalidPathError",
    "ReadLimitError",
    "RegexConstraintError",
    "StaleFileError",
    "MAX_CONTEXT_LINES",
    "MAX_READ_LIMIT",
    "MAX_READ_LINES",
    "MAX_SEARCH_RESULTS",
    "MAX_ARCHIVE_BYTES",
    "MAX_UPLOAD_BYTES",
    "extract_text_archive",
    "iter_text_archive",
    "validate_text_upload",
]
