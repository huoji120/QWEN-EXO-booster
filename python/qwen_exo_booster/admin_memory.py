"""Explicit admin-only selection of physically isolated private memory stores."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from qwen_exo_booster.api_keys import ApiKeyStore, is_valid_api_key_id
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex
from qwen_exo_booster.scoped_memory import ensure_no_links, get_private_memory_index
from qwen_exo_booster.work_memory import WorkMemoryTrash


def _private_base(shared: FileKnowledgeIndex) -> Path:
    base = shared.state_directory / "private-memory"
    ensure_no_links(base)
    resolved = base.resolve()
    if resolved.is_relative_to(shared.root) or shared.root.is_relative_to(resolved):
        raise FileKnowledgeError("private memory storage must be outside the shared knowledge root")
    return base


def _check_owner(owner: Path) -> None:
    ensure_no_links(owner / "knowledge")
    ensure_no_links(owner / "state")


def _owner_directory(shared: FileKnowledgeIndex, store: ApiKeyStore, key_id: str) -> Path:
    """Validate an owner without constructing an index or creating its root."""
    if not is_valid_api_key_id(key_id):
        raise FileKnowledgeError("invalid private memory key id")
    owner = _private_base(shared) / key_id
    _check_owner(owner)
    known = any(row["id"] == key_id for row in store.listing()["keys"])
    if not known and not owner.is_dir():
        raise FileNotFoundError("unknown private memory owner")
    return owner


def select_memory_index(
    shared: FileKnowledgeIndex, store: ApiKeyStore, key_id: str
) -> FileKnowledgeIndex:
    """Only registry owners or retained directories can be selected by admins."""
    _owner_directory(shared, store, key_id)
    return get_private_memory_index(shared.root, shared.state_directory, key_id)


def memory_spaces(shared: FileKnowledgeIndex, store: ApiKeyStore) -> dict:
    """List owners without creating directories, indices, or reading file bodies."""
    records = {row["id"]: row for row in store.listing()["keys"]}
    owners = {key_id for key_id, row in records.items() if row["memory_scope"] != "shared"}
    base = _private_base(shared)
    if base.is_dir():
        for owner in base.iterdir():
            if not is_valid_api_key_id(owner.name):
                continue
            try:
                _check_owner(owner)
            except FileKnowledgeError:
                continue
            if owner.is_dir():
                owners.add(owner.name)
    spaces = []
    for key_id in sorted(owners):
        owner = base / key_id
        try:
            _check_owner(owner)
        except FileKnowledgeError:
            continue
        files = total_bytes = 0
        for directory, dirs, names in os.walk(owner / "knowledge", followlinks=False):
            directory = Path(directory)
            safe_dirs = []
            for name in dirs:
                try:
                    ensure_no_links(directory / name)
                except FileKnowledgeError:
                    continue
                safe_dirs.append(name)
            dirs[:] = safe_dirs
            for name in names:
                if name.startswith(".knowledge-"):
                    continue
                path = directory / name
                try:
                    ensure_no_links(path)
                    FileKnowledgeIndex._validate_relative_path(
                        path.relative_to(owner / "knowledge").as_posix()
                    )
                    status = path.lstat()
                except (OSError, ValueError):
                    continue
                if stat.S_ISREG(status.st_mode):
                    files += 1
                    total_bytes += status.st_size
        record = records.get(key_id)
        spaces.append({
            "key_id": key_id,
            "label": (record.get("label") or key_id) if record else key_id,
            "memory_scope": record["memory_scope"] if record else "private",
            "revoked": bool(record and record.get("revoked_at")),
            "deleted": record is None,
            "files": files,
            "bytes": total_bytes,
        })
    return {"spaces": spaces}


def private_work_trash(index: FileKnowledgeIndex) -> WorkMemoryTrash:
    """Refuse link-bearing trash before the existing restore implementation walks it."""
    trash = WorkMemoryTrash(index)
    ensure_no_links(trash.root)
    for directory, dirs, names in os.walk(trash.root, followlinks=False):
        for name in (*dirs, *names):
            ensure_no_links(Path(directory) / name)
    return trash


def validate_memory_transfer(
    shared: FileKnowledgeIndex, store: ApiKeyStore, *, source_key_id: str | None,
    destination_key_id: str | None, source: str, destination: str,
    confirm_publish: bool = False,
) -> None:
    """Read-only preflight, before runtime guards can create scoped resources."""
    from qwen_exo_booster.memory_transfer import TransferConflict

    if source_key_id is not None and destination_key_id is None and not confirm_publish:
        raise TransferConflict("publishing private memory requires confirm_publish=true")
    if not source and source_key_id is None:
        raise FileKnowledgeError("only a private space can be transferred as a whole")
    FileKnowledgeIndex._validate_relative_path(destination)
    if source:
        FileKnowledgeIndex._validate_relative_path(source)
    for key_id, relative in ((source_key_id, source), (destination_key_id, destination)):
        if key_id is None:
            if relative:
                shared.path_of(relative)
            root = shared.root
        else:
            root = _owner_directory(shared, store, key_id) / "knowledge"
        ensure_no_links(root / relative if relative else root)


def transfer_memory(
    shared: FileKnowledgeIndex, store: ApiKeyStore, *, source_key_id: str | None,
    destination_key_id: str | None, source: str, destination: str,
    confirm_publish: bool = False,
) -> dict:
    """Explicit admin operation; request-selected/public key scope is irrelevant."""
    from qwen_exo_booster.memory_transfer import transfer_indices

    validate_memory_transfer(
        shared, store, source_key_id=source_key_id, destination_key_id=destination_key_id,
        source=source, destination=destination, confirm_publish=confirm_publish,
    )
    source_index = select_memory_index(shared, store, source_key_id) if source_key_id is not None else shared
    destination_index = select_memory_index(shared, store, destination_key_id) if destination_key_id is not None else shared
    return transfer_indices(
        shared, store, source_index, destination_index, source=source,
        destination=destination, source_key_id=source_key_id,
        destination_key_id=destination_key_id,
    )
