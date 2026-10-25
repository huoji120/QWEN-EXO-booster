"""Recoverable, no-overwrite transfers between physically isolated memory roots.

Only same-filesystem atomic renames are supported. No document is copied or
unlinked: an interrupted transfer leaves the original inode at one of its two
journaled names. SQLite's attached rollback-journal transaction moves metadata
and a commit marker together; the next transfer recovers interrupted renames.
"""

from __future__ import annotations

import codecs
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import time
import uuid

from qwen_exo_booster import memory_grep
from qwen_exo_booster.file_knowledge import (
    FileKnowledgeError, FileKnowledgeIndex, MEMORY_MUTATION_LOCK, SCHEMA_VERSION,
    memory_mutation,
)
from qwen_exo_booster.scoped_memory import ensure_no_links

_CHUNK = 1024 * 1024


class TransferConflict(FileKnowledgeError):
    """A destination, changed source, or pending recovery prevents the move."""


class TransferRecoveryRequired(FileKnowledgeError):
    """Original bytes remain journaled; an occupied name prevents rollback."""


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        # MoveFileExW below uses MOVEFILE_WRITE_THROUGH on Windows.
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Never use POSIX rename's implicit destination replacement."""
    ensure_no_links(source)
    ensure_no_links(destination)
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel.MoveFileExW
        move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move.restype = ctypes.c_int
        if not move(str(source), str(destination), 8):
            error = ctypes.get_last_error()
            if error in (80, 183):
                raise TransferConflict("destination already exists")
            raise ctypes.WinError(error)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise FileKnowledgeError("atomic no-replace rename is unavailable on this host")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                raise TransferConflict("destination already exists")
            if error in (errno.EXDEV, errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
                raise FileKnowledgeError("filesystem does not support this atomic transfer")
            raise OSError(error, os.strerror(error))
    else:
        raise FileKnowledgeError("atomic no-replace transfer is unsupported on this host")


def _identity(path: Path) -> tuple[int, int]:
    status = path.lstat()
    return status.st_dev, status.st_ino


def _version(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _safe_entry(index: FileKnowledgeIndex, relative: str) -> Path:
    path = index.path_of(relative)
    ensure_no_links(path)
    if any(part.startswith(".knowledge-") for part in Path(relative).parts):
        raise FileKnowledgeError("temporary knowledge files cannot be transferred")
    if any(
        part.startswith("knowledge-index.sqlite3")
        or part in {"work-memory-trash", "memory-transfers", "private-memory"}
        for part in Path(relative).parts
    ):
        raise FileKnowledgeError("index state and trash cannot be transferred")
    return path


def _file_snapshot(path: Path) -> dict:
    """Hash and describe a file with fixed-size buffers, including long lines."""
    ensure_no_links(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise FileKnowledgeError("only regular files and directories can be transferred")
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    valid = True
    lines = 0
    trailing_cr = False
    has_tail = False
    head = ""
    with path.open("r+b" if os.name == "nt" else "rb") as stream:
        if _version(os.fstat(stream.fileno())) != _version(before):
            raise TransferConflict("source changed while opening it")
        while True:
            block = stream.read(_CHUNK)
            if not block:
                break
            digest.update(block)
            # Match FileKnowledgeIndex's bytes.splitlines(), not Unicode text
            # line boundaries, including CRLF split across read buffers.
            lines += block.count(b"\n") + block.count(b"\r") - block.count(b"\r\n")
            lines -= int(trailing_cr and block.startswith(b"\n"))
            trailing_cr = block.endswith(b"\r")
            has_tail = block[-1] not in (10, 13)
            if valid:
                try:
                    text = decoder.decode(block)
                except UnicodeDecodeError:
                    valid = False
                else:
                    if len(head) < 65536:
                        head += text[:65536 - len(head)]
        if valid:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                valid = False
        os.fsync(stream.fileno())
        if _version(os.fstat(stream.fileno())) != _version(before):
            raise TransferConflict("source changed while hashing it")
    if _version(path.lstat()) != _version(before):
        raise TransferConflict("source changed while hashing it")
    kind = memory_grep.detect_kind(path.name, head) if valid else "other"
    title = memory_grep.detect_title(path.name, kind, head) if valid else ""
    return {
        "identity": _version(before), "sha256": digest.hexdigest(),
        "byte_count": before.st_size, "mtime_ns": before.st_mtime_ns,
        "encoding": "utf-8" if valid else "invalid-utf-8",
        "line_count": lines + int(has_tail) if valid else 0,
        "kind": kind, "title": title,
    }


def _snapshot(index: FileKnowledgeIndex, target: Path) -> tuple[dict, dict]:
    """All regular files and directory identities; reserved state fails closed."""
    files, directories = {}, {}
    stack = [target]
    while stack:
        path = stack.pop()
        ensure_no_links(path)
        relative = path.relative_to(index.root).as_posix()
        if path != index.root:
            _safe_entry(index, relative)
        if index._is_state_path(path):
            raise FileKnowledgeError("index state and trash cannot be transferred")
        status = path.lstat()
        suffix = path.relative_to(target).as_posix()
        if stat.S_ISDIR(status.st_mode):
            directories[suffix] = (status.st_dev, status.st_ino)
            stack.extend(sorted(path.iterdir(), reverse=True))
        elif stat.S_ISREG(status.st_mode):
            files[suffix] = _file_snapshot(path)
        else:
            raise FileKnowledgeError("only regular files and directories can be transferred")
    return files, directories


def _same_snapshot(before: tuple[dict, dict], after: tuple[dict, dict]) -> bool:
    old_files, old_directories = before
    new_files, new_directories = after
    return (
        old_directories == new_directories and old_files.keys() == new_files.keys()
        and all(
            old_files[path]["identity"] == new_files[path]["identity"]
            and old_files[path]["sha256"] == new_files[path]["sha256"]
            for path in old_files
        )
    )


def _mapped(path: str, source: str, destination: str) -> str:
    if not source:
        return destination + "/" + path
    return destination + path[len(source):]


def _move_metadata(connection, source: str, destination: str, files: dict, target_schema: str) -> None:
    """Move only current summaries; erase stale source rows, including orphans."""
    now = time.time_ns()
    for suffix, metadata in files.items():
        old = source if suffix == "." else (source + "/" if source else "") + suffix
        new = _mapped(old, source, destination)
        existing = connection.execute(
            "SELECT * FROM main.knowledge_files WHERE relative_path = ?", (old,)
        ).fetchone()
        current = existing is not None and existing["sha256"] == metadata["sha256"]
        kind = existing["kind"] if current else metadata["kind"]
        title = existing["title"] if current else metadata["title"]
        connection.execute(
            f"INSERT INTO {target_schema}.knowledge_files "
            "(relative_path,sha256,byte_count,mtime_ns,encoding,line_count,indexed_ns,schema_version,kind,title) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (new, metadata["sha256"], metadata["byte_count"], metadata["mtime_ns"],
             metadata["encoding"], metadata["line_count"], now, SCHEMA_VERSION, kind, title),
        )
        if current:
            connection.execute(
                f"INSERT INTO {target_schema}.knowledge_segments SELECT ?, hash "
                "FROM main.knowledge_segments WHERE relative_path = ?", (new, old),
            )
        connection.execute(
            f"INSERT INTO {target_schema}.knowledge_summaries "
            "SELECT ?,sha256,title,summary,sampled,origin,updated_ns FROM main.knowledge_summaries "
            "WHERE relative_path = ? AND sha256 = ?", (new, old, metadata["sha256"]),
        )
    for table in ("knowledge_files", "knowledge_segments", "knowledge_summaries"):
        connection.execute(
            f"DELETE FROM main.{table} WHERE ? = '' OR relative_path = ? OR instr(relative_path, ?) = 1",
            (source, source, source + "/"),
        )
    # An ancestor's aggregate summary can still contain a moved child's text.
    for schema, path in (("main", source), (target_schema, destination)):
        parent = path.rpartition("/")[0]
        while parent:
            connection.execute(f"DELETE FROM {schema}.knowledge_summaries WHERE relative_path = ?", (parent,))
            parent = parent.rpartition("/")[0]


def _clear_destination_metadata(connection, destination: str, schema: str) -> None:
    # The filesystem destination is absent. Old index rows here are stale, not
    # data to merge; clearing them is part of the same rollback-able transaction.
    for table in ("knowledge_files", "knowledge_segments", "knowledge_summaries"):
        connection.execute(
            f"DELETE FROM {schema}.{table} WHERE relative_path = ? OR instr(relative_path, ?) = 1",
            (destination, destination + "/"),
        )


def _write_journal(directory: Path, record: dict) -> Path:
    path = directory / (record["id"] + ".json")
    temporary = directory / (record["id"] + ".pending")
    published = False
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-overwrite publication of the complete durable journal.
        # An incomplete .pending file is never considered a recovery record.
        os.link(temporary, path)
        published = True
        _sync_directory(directory)
        return path
    except BaseException:
        if published:
            try:
                path.unlink()
            except OSError:
                # Still a complete prepared record; recovery sees source intact.
                pass
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _finish_journal(path: Path, source_index: FileKnowledgeIndex, transfer_id: str) -> None:
    try:
        path.unlink()
        _sync_directory(path.parent)
    except OSError:
        # Committed data is complete; keep its marker so recovery never undoes it.
        return
    try:
        with source_index._connect() as connection:
            connection.execute("DELETE FROM memory_transfer_commits WHERE id = ?", (transfer_id,))
    except (OSError, sqlite3.Error):
        # An orphan commit marker is harmless and contains no memory content.
        pass


def _rollback_names(source: Path, destination: Path, record: dict) -> None:
    ensure_no_links(source)
    ensure_no_links(destination)
    identity = tuple(record["identity"])
    if source.exists() and _identity(source) == identity:
        return
    if not destination.exists() or _identity(destination) != identity:
        raise TransferRecoveryRequired(f"transfer {record['id']} requires recovery: original inode changed")
    if source.exists():
        # Index construction may recreate a whole-space root after a crash.
        if record["source"] == "" and source.is_dir() and not any(source.iterdir()):
            source.rmdir()
        else:
            raise TransferRecoveryRequired(f"transfer {record['id']} requires recovery: source was recreated; both versions retained")
    _rename_no_replace(destination, source)
    _sync_directory(source.parent)
    _sync_directory(destination.parent)


def _remove_created_parents(index: FileKnowledgeIndex, created: list) -> None:
    for relative, identity in reversed(created):
        path = _safe_entry(index, relative)
        try:
            if path.is_dir() and _identity(path) == tuple(identity) and not any(path.iterdir()):
                path.rmdir()
                _sync_directory(path.parent)
        except OSError:
            # Never remove content concurrently written into a new parent.
            continue


@memory_mutation
def recover_transfers(shared, store, directory: Path, expected: dict) -> None:
    """Replay incomplete journals conservatively; never overwrite a newer name."""
    from qwen_exo_booster.admin_memory import select_memory_index

    for path in sorted(directory.glob("*.json")):
        ensure_no_links(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if path.stem != record["id"]:
                raise ValueError("journal identity mismatch")
            if any(record.get(key) != expected[key] for key in (
                "source_key_id", "destination_key_id", "source", "destination",
            )):
                raise TransferRecoveryRequired(
                    f"transfer {record['id']} is pending; retry its original source and destination before another transfer"
                )
            source_index = select_memory_index(shared, store, record["source_key_id"]) if record["source_key_id"] is not None else shared
            destination_index = select_memory_index(shared, store, record["destination_key_id"]) if record["destination_key_id"] is not None else shared
            source = _safe_entry(source_index, record["source"]) if record["source"] else source_index.root
            destination = _safe_entry(destination_index, record["destination"])
            with source_index._connect() as connection:
                committed = connection.execute("SELECT 1 FROM memory_transfer_commits WHERE id = ?", (record["id"],)).fetchone()
            if not committed:
                _rollback_names(source, destination, record)
                _remove_created_parents(destination_index, record.get("created_parents", []))
            _finish_journal(path, source_index, record["id"])
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            raise TransferRecoveryRequired(f"transfer recovery blocked at {path.name}: {exc}") from exc


def transfer_indices(shared, store, source_index, destination_index, *, source: str,
                     destination: str, source_key_id: str | None,
                     destination_key_id: str | None) -> dict:
    """Atomic filesystem ownership change with recoverable SQLite consistency."""
    with MEMORY_MUTATION_LOCK:
        journal_directory = shared.state_directory / "memory-transfers"
        ensure_no_links(journal_directory)
        journal_directory.mkdir(exist_ok=True)
        _sync_directory(journal_directory.parent)
        recover_transfers(shared, store, journal_directory, {
            "source_key_id": source_key_id, "destination_key_id": destination_key_id,
            "source": source, "destination": destination,
        })
        ensure_no_links(source_index.root)
        ensure_no_links(destination_index.root)
        source_path = _safe_entry(source_index, source) if source else source_index.root
        destination_path = _safe_entry(destination_index, destination)
        if not source_path.exists():
            raise FileNotFoundError(source)
        if destination_path.exists():
            raise TransferConflict("destination already exists; merging is not supported")
        if destination_path == source_path or destination_path.is_relative_to(source_path):
            raise TransferConflict("source cannot be moved into itself")
        existing_parent = destination_path.parent
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        if not existing_parent.is_dir():
            raise TransferConflict("destination parent is not a directory")
        if source_path.stat().st_dev != existing_parent.stat().st_dev:
            raise FileKnowledgeError("cross-filesystem transfer is unsupported; source is unchanged")
        same_index = source_index.database_path == destination_index.database_path
        target_schema = "main" if same_index else "destination"
        with source_index._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS memory_transfer_commits (id TEXT PRIMARY KEY)")
            connection.commit()
            if not same_index:
                # Check the destination's private layout before SQLite opens it.
                with destination_index._connect():
                    pass
                connection.execute("ATTACH DATABASE ? AS destination", (str(destination_index.database_path),))
            for schema in ({"main", target_schema}):
                mode = connection.execute(f"PRAGMA {schema}.journal_mode").fetchone()[0]
                if mode not in ("delete", "truncate", "persist"):
                    raise FileKnowledgeError("transfer requires SQLite rollback-journal mode")
                connection.execute(f"PRAGMA {schema}.synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            journal = None
            record = None
            created_parents = []
            try:
                missing = []
                parent = destination_path.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    _safe_entry(destination_index, parent.relative_to(destination_index.root).as_posix())
                    try:
                        parent.mkdir()
                    except FileExistsError:
                        ensure_no_links(parent)
                        if not parent.is_dir():
                            raise TransferConflict("destination parent is not a directory")
                    else:
                        created_parents.append((parent.relative_to(destination_index.root).as_posix(), _identity(parent)))
                        _sync_directory(parent.parent)
                snapshot = _snapshot(source_index, source_path)
                # Persist source data and directory entries before moving names.
                for suffix in snapshot[1]:
                    _sync_directory(source_path if suffix == "." else source_path / suffix)
                record = {
                    "id": uuid.uuid4().hex, "source_key_id": source_key_id,
                    "destination_key_id": destination_key_id,
                    "source": source, "destination": destination,
                    "identity": _identity(source_path),
                    "created_parents": created_parents,
                }
                _clear_destination_metadata(connection, destination, target_schema)
                _move_metadata(connection, source, destination, snapshot[0], target_schema)
                journal = _write_journal(journal_directory, record)
                # Revalidate every component immediately before the ownership change.
                if source:
                    _safe_entry(source_index, source)
                _safe_entry(destination_index, destination)
                _rename_no_replace(source_path, destination_path)
                _sync_directory(source_path.parent)
                _sync_directory(destination_path.parent)
                if not _same_snapshot(snapshot, _snapshot(destination_index, destination_path)):
                    raise TransferConflict("source changed during transfer; transfer rolled back")
                if source_path.exists():
                    raise TransferConflict("source was recreated during transfer")
                if not source:
                    source_path.mkdir()
                    _sync_directory(source_path.parent)
                connection.execute("INSERT INTO memory_transfer_commits(id) VALUES (?)", (record["id"],))
                connection.commit()
            except BaseException as exc:
                try:
                    connection.rollback()
                    committed = journal is not None and connection.execute(
                        "SELECT 1 FROM memory_transfer_commits WHERE id = ?", (record["id"],)
                    ).fetchone()
                except sqlite3.Error as transaction_error:
                    if journal is not None:
                        raise TransferRecoveryRequired(
                            f"transfer {record['id']} metadata outcome requires journal recovery; no document was deleted"
                        ) from transaction_error
                    raise
                if committed:
                    raise TransferRecoveryRequired(
                        f"transfer {record['id']} committed but reported an error; retain destination and retry for journal recovery"
                    ) from exc
                if journal is not None and record is not None:
                    try:
                        _rollback_names(source_path, destination_path, record)
                    except (OSError, ValueError) as rollback_error:
                        raise TransferRecoveryRequired(
                            f"transfer {record['id']} failed; original data retained; recovery required: {rollback_error}"
                        ) from exc
                    _finish_journal(journal, source_index, record["id"])
                _remove_created_parents(destination_index, created_parents)
                raise
        if journal is not None:
            _finish_journal(journal, source_index, record["id"])
        source_index._boilerplate = destination_index._boilerplate = None
        return {"from": source, "to": destination, "files": len(snapshot[0]),
                "source_key_id": source_key_id, "destination_key_id": destination_key_id}
