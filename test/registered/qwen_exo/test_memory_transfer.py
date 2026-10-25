"""Ownership transfer regressions; real roots, bytes, SQLite and recovery journals."""

import errno
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from qwen_exo_booster.admin_memory import select_memory_index, transfer_memory
from qwen_exo_booster.api_keys import ApiKeyStore
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex, StaleFileError
from qwen_exo_booster.memory_tools import MemoryToolExecutor
import qwen_exo_booster.memory_transfer as transfers


@pytest.fixture
def spaces(tmp_path):
    shared = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    store = ApiKeyStore(tmp_path / "keys.json")
    a, b = [store.create(label, "private")["id"] for label in ("A", "B")]
    indices = {None: shared, a: select_memory_index(shared, store, a), b: select_memory_index(shared, store, b)}
    return shared, store, a, b, indices


def move(spaces, source_owner, destination_owner, source="note.md", destination="moved.md", **kwargs):
    shared, store, _a, _b, _indices = spaces
    return transfer_memory(shared, store, source_key_id=source_owner,
                           destination_key_id=destination_owner, source=source,
                           destination=destination, **kwargs)


def metadata_rows(index):
    with index._connect() as connection:
        return {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY relative_path")]
            for table in ("knowledge_files", "knowledge_segments", "knowledge_summaries")
        }


@pytest.mark.parametrize("direction", ["shared-private", "private-shared", "private-private", "same-private"])
@pytest.mark.parametrize("origin", ["manual", "auto"])
def test_transfer_bytes_summary_provenance_and_read_ownership(spaces, direction, origin):
    shared, _store, a, b, indices = spaces
    owners = {"shared-private": (None, a), "private-shared": (a, None),
              "private-private": (a, b), "same-private": (a, a)}
    source_owner, destination_owner = owners[direction]
    source, destination = indices[source_owner], indices[destination_owner]
    raw = "exact\r\nUnicode 记忆\n\x00kept".encode()
    row = source.upsert("note.md", raw)
    summary = source.store_summary("note.md", sha256=row["sha256"], summary="# Summary\nprivate-marker",
                                   sampled=True, origin=origin)
    result = move(spaces, source_owner, destination_owner, confirm_publish=True)
    assert result == {"from": "note.md", "to": "moved.md", "files": 1,
                      "source_key_id": source_owner, "destination_key_id": destination_owner}
    assert not source.path_of("note.md").exists()
    assert destination.path_of("moved.md").read_bytes() == raw
    assert destination.summary("moved.md") == {**summary, "relative_path": "moved.md"}
    assert source.summary("note.md") is None
    public = MemoryToolExecutor(shared)
    if destination_owner is None:
        assert public.read_index.grep("private-marker")["file_count"] == 1
    else:
        assert public.read_index.grep("private-marker")["file_count"] == 0
    destination.refresh()
    assert destination.read_text("moved.md")[0] == hashlib.sha256(raw).hexdigest()


def test_whole_retained_private_root_keeps_state_trash_and_empty_directories(spaces):
    shared, store, a, _b, indices = spaces
    private = indices[a]
    private.make_directory("empty/nested")
    row = private.upsert("folder/note.md", b"private")
    private.store_summary("folder/note.md", sha256=row["sha256"], summary="private summary", sampled=False, origin="manual")
    trash = private.state_directory / "work-memory-trash" / "saved"
    trash.mkdir(parents=True)
    (trash / "old.md").write_bytes(b"do not move")
    store.revoke(a)
    store.delete([a])
    registry = store.path.read_bytes()
    result = move(spaces, a, None, source="", destination="work-memory/retained", confirm_publish=True)
    assert result["files"] == 1
    assert (shared.root / "work-memory/retained/empty/nested").is_dir()
    assert shared.read_text("work-memory/retained/folder/note.md")[1] == "private"
    assert private.root.is_dir() and not list(private.root.iterdir())
    assert private.database_path.is_file() and private.metadata() == []
    assert (trash / "old.md").read_bytes() == b"do not move"
    assert not (shared.root / "work-memory/retained/state").exists()
    assert store.path.read_bytes() == registry


def test_safe_legacy_orphan_can_be_source_and_destination(spaces):
    shared, store, a, _b, _indices = spaces
    orphan = "qxk_retained-legacy-owner"
    root = shared.state_directory / "private-memory" / orphan / "knowledge"
    root.mkdir(parents=True)
    (root / "note.md").write_bytes(b"legacy exact bytes")
    assert move(spaces, orphan, a)["files"] == 1
    assert move(spaces, a, orphan, source="moved.md", destination="restored.md")["files"] == 1
    assert (root / "restored.md").read_bytes() == b"legacy exact bytes"
    with pytest.raises(FileNotFoundError):
        select_memory_index(shared, store, "qxk_unknown-owner")


def test_empty_folder_and_empty_root_transfer(spaces):
    shared, _store, a, b, indices = spaces
    indices[a].make_directory("empty/nested")
    assert move(spaces, a, b, source="empty", destination="folder")["files"] == 0
    assert (indices[b].root / "folder/nested").is_dir()
    assert move(spaces, a, None, source="", destination="empty-space", confirm_publish=True)["files"] == 0
    assert (shared.root / "empty-space").is_dir()
    assert indices[a].root.is_dir()


def test_publish_confirmation_and_destination_conflict_do_not_mutate(spaces):
    shared, _store, a, b, indices = spaces
    indices[a].upsert("note.md", b"private")
    with pytest.raises(transfers.TransferConflict, match="confirm_publish"):
        move(spaces, a, None)
    assert not (shared.root / "moved.md").exists()
    indices[b].upsert("moved.md", b"keep destination")
    before = metadata_rows(indices[a]), metadata_rows(indices[b])
    with pytest.raises(transfers.TransferConflict, match="already exists"):
        move(spaces, a, b)
    assert indices[a].path_of("note.md").read_bytes() == b"private"
    assert indices[b].path_of("moved.md").read_bytes() == b"keep destination"
    assert before == (metadata_rows(indices[a]), metadata_rows(indices[b]))


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/windows", "a/../b", "a\\b", "a//b", ".knowledge-upload"])
def test_unsafe_paths_never_move_data(spaces, path):
    _shared, _store, a, b, indices = spaces
    indices[a].upsert("note.md", b"keep")
    with pytest.raises(FileKnowledgeError):
        move(spaces, a, b, destination=path)
    assert indices[a].path_of("note.md").read_bytes() == b"keep"


def test_root_state_and_self_descendant_rejected(spaces):
    shared, _store, a, _b, indices = spaces
    indices[a].upsert("folder/note.md", b"keep")
    with pytest.raises(transfers.TransferConflict):
        move(spaces, a, a, source="folder", destination="folder/child")
    with pytest.raises(FileKnowledgeError):
        move(spaces, None, a, source="", destination="all")
    with pytest.raises(FileKnowledgeError):
        move(spaces, a, None, source="../state", destination="state", confirm_publish=True)
    nested = FileKnowledgeIndex(shared.root / "nested", shared.root / "nested/state")
    (nested.state_directory / "keep").write_bytes(b"state")
    with pytest.raises(FileKnowledgeError, match="state"):
        move(spaces, None, a, source="nested", destination="nested")
    assert (nested.state_directory / "keep").read_bytes() == b"state"


@pytest.mark.parametrize("location", ["source", "source-child", "destination-parent"])
def test_symlinks_rejected_without_following(spaces, tmp_path, location):
    _shared, _store, a, b, indices = spaces
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.md").write_bytes(b"outside")
    indices[a].make_directory("folder")
    indices[a].upsert("folder/note.md", b"inside")
    link = {"source": indices[a].root / "alias", "source-child": indices[a].root / "folder/alias",
            "destination-parent": indices[b].root / "alias"}[location]
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    source = "alias" if location == "source" else "folder"
    destination = "alias/moved" if location == "destination-parent" else "moved"
    with pytest.raises(FileKnowledgeError):
        move(spaces, a, b, source=source, destination=destination)
    assert (external / "secret.md").read_bytes() == b"outside"
    assert indices[a].path_of("folder/note.md").read_bytes() == b"inside"


@pytest.mark.parametrize("failure", ["hash", "metadata", "post-rename", "cross-device"])
def test_failures_restore_original_directory_and_sqlite(spaces, monkeypatch, failure):
    _shared, _store, a, b, indices = spaces
    source, destination = indices[a], indices[b]
    source.make_directory("folder/empty")
    row = source.upsert("folder/note.md", b"original")
    source.store_summary("folder/note.md", sha256=row["sha256"], summary="summary", sampled=False, origin="auto")
    before = metadata_rows(source), metadata_rows(destination)
    snapshot = transfers._snapshot
    metadata = transfers._move_metadata
    calls = 0

    def fail_snapshot(*args):
        nonlocal calls
        calls += 1
        if failure == "hash" or calls == 2:
            raise OSError("injected hash failure")
        return snapshot(*args)

    def fail_metadata(*args):
        metadata(*args)
        raise sqlite3.OperationalError("injected metadata failure")

    def fail_rename(*_args):
        raise OSError(errno.EXDEV, "injected unsupported cross-device move")

    if failure in ("hash", "post-rename"):
        monkeypatch.setattr(transfers, "_snapshot", fail_snapshot)
    elif failure == "metadata":
        monkeypatch.setattr(transfers, "_move_metadata", fail_metadata)
    else:
        monkeypatch.setattr(transfers, "_rename_no_replace", fail_rename)
    with pytest.raises((OSError, sqlite3.Error)):
        move(spaces, a, b, source="folder", destination="new/parent/moved")
    assert source.path_of("folder/note.md").read_bytes() == b"original"
    assert source.path_of("folder/empty").is_dir()
    assert not destination.path_of("new/parent/moved").exists()
    assert before == (metadata_rows(source), metadata_rows(destination))


def test_concurrent_source_edit_is_preserved_not_overwritten(spaces, monkeypatch):
    _shared, _store, a, b, indices = spaces
    indices[a].upsert("note.md", b"original")
    rename = transfers._rename_no_replace
    first = True

    def concurrent_edit(source, destination):
        nonlocal first
        if first:
            first = False
            source.write_bytes(b"concurrent edited version")
        rename(source, destination)

    monkeypatch.setattr(transfers, "_rename_no_replace", concurrent_edit)
    with pytest.raises(transfers.TransferConflict, match="changed"):
        move(spaces, a, b)
    assert indices[a].path_of("note.md").read_bytes() == b"concurrent edited version"
    assert not indices[b].path_of("moved.md").exists()


def test_destination_race_never_overwrites_existing_file(spaces, monkeypatch):
    _shared, _store, a, b, indices = spaces
    indices[a].upsert("note.md", b"source")
    rename = transfers._rename_no_replace

    def race(source, destination):
        destination.write_bytes(b"concurrent destination")
        rename(source, destination)

    monkeypatch.setattr(transfers, "_rename_no_replace", race)
    with pytest.raises(transfers.TransferConflict):
        move(spaces, a, b)
    assert indices[a].path_of("note.md").read_bytes() == b"source"
    assert indices[b].path_of("moved.md").read_bytes() == b"concurrent destination"


def test_recreated_source_keeps_both_versions_and_recovery_is_scoped(spaces, monkeypatch):
    shared, store, a, b, indices = spaces
    indices[a].upsert("note.md", b"original")
    rename = transfers._rename_no_replace
    first = True

    def recreate(source, destination):
        nonlocal first
        rename(source, destination)
        if first:
            first = False
            source.write_bytes(b"concurrent new source")

    monkeypatch.setattr(transfers, "_rename_no_replace", recreate)
    with pytest.raises(transfers.TransferRecoveryRequired):
        move(spaces, a, b)
    assert indices[a].path_of("note.md").read_bytes() == b"concurrent new source"
    assert indices[b].path_of("moved.md").read_bytes() == b"original"
    journals = list((shared.state_directory / "memory-transfers").glob("*.json"))
    assert len(journals) == 1
    record = json.loads(journals[0].read_text())
    with pytest.raises(transfers.TransferRecoveryRequired, match="original source"):
        move(spaces, a, None, destination="different.md", confirm_publish=True)
    # Resolve the name collision without deleting either version, then retry the
    # original operation. Journal recovery first restores the old source name.
    indices[a].path_of("note.md").rename(indices[a].root / "concurrent.md")
    assert move(spaces, a, b)["files"] == 1
    assert indices[a].path_of("concurrent.md").read_bytes() == b"concurrent new source"
    assert indices[b].path_of("moved.md").read_bytes() == b"original"
    assert not journals[0].exists()
    assert record["source_key_id"] == a and record["destination_key_id"] == b


def test_journal_unlink_failure_never_undoes_committed_data(spaces, monkeypatch):
    shared, store, a, b, indices = spaces
    indices[a].upsert("note.md", b"complete")
    unlink = Path.unlink

    def fail_journal_unlink(path, *args, **kwargs):
        if path.parent.name == "memory-transfers":
            raise OSError("injected journal cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_journal_unlink)
    result = move(spaces, a, b)
    assert result["files"] == 1
    assert not indices[a].path_of("note.md").exists()
    assert indices[b].read_text("moved.md")[1] == "complete"
    journals = list((shared.state_directory / "memory-transfers").glob("*.json"))
    assert len(journals) == 1
    monkeypatch.setattr(Path, "unlink", unlink)
    transfers.recover_transfers(shared, store, journals[0].parent, {
        "source_key_id": a, "destination_key_id": b, "source": "note.md", "destination": "moved.md",
    })
    assert not journals[0].exists()
    assert indices[b].read_text("moved.md")[1] == "complete"
    assert not indices[a].path_of("note.md").exists()


def test_inflight_summary_cannot_republish_shared_content_after_transfer(spaces, monkeypatch):
    shared, _store, a, _b, indices = spaces
    row = shared.upsert("note.md", b"shared source")
    started = threading.Event()
    rename = transfers._rename_no_replace
    futures = []

    def publish_summary():
        started.set()
        return shared.store_summary("note.md", sha256=row["sha256"], summary="must not leak", sampled=False, origin="auto")

    with ThreadPoolExecutor(max_workers=1) as pool:
        def during_transfer(source, destination):
            futures.append(pool.submit(publish_summary))
            assert started.wait(5)
            rename(source, destination)
        monkeypatch.setattr(transfers, "_rename_no_replace", during_transfer)
        move(spaces, None, a)
        with pytest.raises(StaleFileError):
            futures[0].result(timeout=5)
    assert shared.summary("note.md") is None
    assert shared.grep("must not leak")["file_count"] == 0
    assert indices[a].path_of("moved.md").read_bytes() == b"shared source"


def test_streamed_snapshot_matches_line_count_and_rejects_changed_summary(spaces, monkeypatch):
    _shared, _store, a, b, indices = spaces
    raw = "a\r\nb\r\nc\u2028tail".encode()
    indices[a].upsert("note.md", raw)
    monkeypatch.setattr(transfers, "_CHUNK", 2)
    move(spaces, a, b)
    assert indices[b].metadata()[0]["line_count"] == len(raw.splitlines())
    with pytest.raises(StaleFileError):
        indices[b].store_summary("moved.md", sha256="obsolete", summary="old", sampled=False, origin="manual")


def test_unindexed_binary_and_empty_files_move_without_body_materialization(spaces, monkeypatch):
    _shared, _store, a, b, indices = spaces
    source = indices[a]
    source.make_directory("folder")
    raw = b"\xff\x00exact\r\n" * (256 * 1024)
    source.path_of("folder/raw.bin").write_bytes(raw)
    source.path_of("folder/empty.txt").write_bytes(b"")
    read_bytes = Path.read_bytes

    def no_materialization(_path):
        raise AssertionError("transfer must stream file bodies")

    monkeypatch.setattr(Path, "read_bytes", no_materialization)
    assert move(spaces, a, b, source="folder", destination="folder")["files"] == 2
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert indices[b].path_of("folder/raw.bin").read_bytes() == raw
    assert indices[b].path_of("folder/empty.txt").read_bytes() == b""
    assert not source.path_of("folder").exists()


def test_reparse_point_is_rejected_as_a_windows_junction(spaces, monkeypatch):
    from types import SimpleNamespace

    _shared, _store, a, b, indices = spaces
    indices[a].upsert("folder/note.md", b"keep")
    junction = indices[a].root / "folder"
    lstat = Path.lstat

    def reparse_status(path, *args, **kwargs):
        status = lstat(path, *args, **kwargs)
        if path == junction:
            return SimpleNamespace(st_mode=status.st_mode, st_file_attributes=0x400)
        return status

    monkeypatch.setattr(Path, "lstat", reparse_status)
    with pytest.raises(FileKnowledgeError, match="junction"):
        move(spaces, a, b, source="folder", destination="folder")
    monkeypatch.setattr(Path, "lstat", lstat)
    assert indices[a].path_of("folder/note.md").read_bytes() == b"keep"


def test_destination_fsync_error_restores_source_and_indices(spaces, monkeypatch):
    _shared, _store, a, b, indices = spaces
    source, destination = indices[a], indices[b]
    source.upsert("note.md", b"original")
    before = metadata_rows(source), metadata_rows(destination)
    sync = transfers._sync_directory
    failed = False

    def fail_once(path):
        nonlocal failed
        if not failed and destination.path_of("moved.md").exists():
            failed = True
            raise OSError("injected destination durability failure")
        sync(path)

    monkeypatch.setattr(transfers, "_sync_directory", fail_once)
    with pytest.raises(OSError, match="durability"):
        move(spaces, a, b)
    assert source.path_of("note.md").read_bytes() == b"original"
    assert not destination.path_of("moved.md").exists()
    assert before == (metadata_rows(source), metadata_rows(destination))


@pytest.mark.parametrize("writer", ["atomic", "stream"])
def test_normal_replacing_writer_cannot_change_journaled_inode(spaces, monkeypatch, writer):
    import io
    import qwen_exo_booster.file_knowledge as file_knowledge

    shared, _store, a, b, indices = spaces
    source, destination = indices[a], indices[b]
    source.upsert("note.md", b"original transferred version")
    real_lock = file_knowledge.MEMORY_MUTATION_LOCK
    transfer_thread = threading.get_ident()
    attempted = threading.Event()
    futures = []
    rename = transfers._rename_no_replace

    class ObservedMutationLock:
        def __enter__(self):
            if threading.get_ident() != transfer_thread:
                attempted.set()
            real_lock.acquire()
            return self

        def __exit__(self, *_args):
            real_lock.release()

    monkeypatch.setattr(file_knowledge, "MEMORY_MUTATION_LOCK", ObservedMutationLock())

    def replace_source():
        if writer == "atomic":
            return source.upsert("note.md", b"later writer version")
        return source.store_upload("note.md", io.BytesIO(b"later writer version"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        def before_rename(old, new):
            futures.append(pool.submit(replace_source))
            assert attempted.wait(5), "writer must reach the shared mutation lock"
            assert not futures[0].done()
            assert old.read_bytes() == b"original transferred version"
            rename(old, new)

        monkeypatch.setattr(transfers, "_rename_no_replace", before_rename)
        assert move(spaces, a, b)["files"] == 1
        futures[0].result(timeout=5)
    assert destination.read_text("moved.md")[1] == "original transferred version"
    assert source.read_text("note.md")[1] == "later writer version"
    assert not list((shared.state_directory / "memory-transfers").glob("*.json"))


@pytest.mark.parametrize("failure", ["write", "file-fsync", "publish", "directory-fsync"])
def test_incomplete_journal_never_blocks_later_transfers(spaces, monkeypatch, failure):
    import os

    shared, _store, a, b, indices = spaces
    source, destination = indices[a], indices[b]
    source.upsert("note.md", b"unchanged until publication succeeds")
    before = metadata_rows(source), metadata_rows(destination)
    directory = shared.state_directory / "memory-transfers"
    dump, fsync, link = transfers.json.dump, transfers.os.fsync, transfers.os.link
    sync_directory = transfers._sync_directory

    def fail_write(_record, stream, **_kwargs):
        stream.write('{"partial":')
        raise OSError("injected journal write failure")

    def fail_fsync(descriptor):
        identity = os.fstat(descriptor)
        if any((path.stat().st_dev, path.stat().st_ino) == (identity.st_dev, identity.st_ino)
               for path in directory.glob("*.pending")):
            raise OSError("injected journal fsync failure")
        fsync(descriptor)

    def fail_link(*_args, **_kwargs):
        raise OSError("injected journal publication failure")

    def fail_directory_sync(path):
        if path == directory and list(directory.glob("*.json")):
            raise OSError("injected journal directory fsync failure")
        sync_directory(path)

    if failure == "write":
        monkeypatch.setattr(transfers.json, "dump", fail_write)
    elif failure == "file-fsync":
        monkeypatch.setattr(transfers.os, "fsync", fail_fsync)
    elif failure == "publish":
        monkeypatch.setattr(transfers.os, "link", fail_link)
    else:
        monkeypatch.setattr(transfers, "_sync_directory", fail_directory_sync)
    with pytest.raises(OSError, match="journal"):
        move(spaces, a, b)
    assert source.path_of("note.md").read_bytes() == b"unchanged until publication succeeds"
    assert not destination.path_of("moved.md").exists()
    assert before == (metadata_rows(source), metadata_rows(destination))
    assert not list(directory.glob("*.json"))
    assert not list(directory.glob("*.pending"))
    monkeypatch.setattr(transfers.json, "dump", dump)
    monkeypatch.setattr(transfers.os, "fsync", fsync)
    monkeypatch.setattr(transfers.os, "link", link)
    monkeypatch.setattr(transfers, "_sync_directory", sync_directory)
    assert move(spaces, a, b)["files"] == 1
    assert destination.read_text("moved.md")[1] == "unchanged until publication succeeds"
