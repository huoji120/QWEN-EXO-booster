"""Actual-file isolation and globally bounded search for managed-key memory."""

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex
from qwen_exo_booster.memory_tools import (
    MEMORY_READ_TOOL,
    MEMORY_SEARCH_TOOL,
    MEMORY_WRITE_TOOL,
    MemoryToolExecutor,
)
from qwen_exo_booster.scoped_memory import get_private_memory_index, private_memory_key_ids

KEY_A = "qxk_" + "a" * 16
KEY_B = "qxk_" + "b" * 16


@pytest.fixture
def scopes(tmp_path):
    shared = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    a = get_private_memory_index(shared.root, shared.state_directory, KEY_A)
    b = get_private_memory_index(shared.root, shared.state_directory, KEY_B)
    return (
        MemoryToolExecutor(shared),
        MemoryToolExecutor(a, shared_index=shared, key_id=KEY_A, memory_scope="private"),
        MemoryToolExecutor(b, shared_index=shared, key_id=KEY_B, memory_scope="private"),
    )


def put(executor, path, content, mode="create"):
    return executor.run_sync(MEMORY_WRITE_TOOL, {
        "path": path, "title": "Scoped note", "content": content, "mode": mode,
    })


def test_three_scopes_keep_files_and_searches_isolated(scopes):
    shared, a, b = scopes
    for executor, text in ((shared, "union shared"), (a, "union ownerA"), (b, "union ownerB")):
        ok, result = put(executor, "same.md", text)
        assert ok, result
    assert not a.index.root.is_relative_to(shared.index.root)
    assert not b.index.root.is_relative_to(shared.index.root)
    assert a.index.state_directory != b.index.state_directory
    assert "private/work-memory/extra.md" in put(a, "extra.md", "extra")[1]
    for mode in ("content", "files", "count"):
        shared_result = shared.read_index.grep("union", mode=mode)
        a_result = a.read_index.grep("union", mode=mode)
        b_result = b.read_index.grep("union", mode=mode)
        assert shared_result["file_count"] == 1
        assert a_result["file_count"] == b_result["file_count"] == 2
        assert a_result["total_matches"] == b_result["total_matches"] == 2
        assert {entry["path"] for entry in a_result["files"]} == {
            "work-memory/same.md", "private/work-memory/same.md",
        }
        assert a.read_index.grep("ownerB", mode=mode)["file_count"] == 0
        assert b.read_index.grep("ownerA", mode=mode)["file_count"] == 0
        assert shared.read_index.grep("ownerA|ownerB", mode=mode)["file_count"] == 0
    for executor, expected in ((a, "ownerA"), (b, "ownerB")):
        ok, own = executor.run_sync(MEMORY_READ_TOOL, {"path": "private/work-memory/same.md"})
        assert ok and expected in own and "private/work-memory/same.md:" in own
        ok, public = executor.run_sync(MEMORY_READ_TOOL, {"path": "work-memory/same.md"})
        assert ok and "union shared" in public and expected not in public


def test_private_mutations_never_fall_back_to_shared(scopes):
    shared, a, b = scopes
    assert put(shared, "same.md", "shared original")[0]
    shared_path = shared.index.path_of("work-memory/same.md")
    original = shared_path.read_bytes()
    for mode in ("edit", "replace", "delete"):
        args = {"path": "work-memory/same.md", "mode": mode, "title": "changed", "content": "private replacement", "old_text": "original", "new_text": "changed"}
        ok, _ = a.run_sync(MEMORY_WRITE_TOOL, args)
        assert not ok
        assert shared_path.read_bytes() == original
    assert put(a, "private/work-memory/same.md", "private original", "append")[0]
    assert put(a, "same.md", "private appended", "append")[0]
    ok, result = a.run_sync(MEMORY_WRITE_TOOL, {
        "path": "private/work-memory/same.md", "mode": "edit", "old_text": "original", "new_text": "edited",
    })
    assert ok and "Edited private/work-memory/same.md" in result
    assert put(a, "same.md", "private replacement", "replace")[0]
    ok, result = a.run_sync(MEMORY_WRITE_TOOL, {"path": "same.md", "mode": "delete"})
    assert ok and "Deleted private/work-memory/same.md" in result
    assert shared_path.read_bytes() == original
    assert not a.index.path_of("work-memory/same.md").exists()
    assert not b.index.path_of("work-memory/same.md").exists()
    assert list(a.index.state_directory.glob("work-memory-trash/*/work-memory/same.md"))
    assert not list(shared.index.state_directory.glob("work-memory-trash/*/work-memory/same.md"))


@pytest.mark.parametrize("tool,args", [
    (MEMORY_READ_TOOL, {"path": "private/work-memory/note.md"}),
    (MEMORY_SEARCH_TOOL, {"pattern": "note", "path": "private/**"}),
    (MEMORY_WRITE_TOOL, {"path": "private/note.md", "title": "note", "content": "note"}),
    (MEMORY_WRITE_TOOL, {"path": r"private\note.md", "title": "note", "content": "note"}),
])
def test_shared_tools_reject_private_virtual_paths(scopes, tool, args):
    assert not scopes[0].run_sync(tool, args)[0]


@pytest.mark.parametrize("path", [
    "../work-memory/note.md", "private/../work-memory/note.md",
    "private/../../" + KEY_B + "/knowledge/work-memory/note.md",
    "private/" + KEY_B + "/work-memory/note.md", "/private/work-memory/note.md",
    r"private\..\work-memory\note.md",
])
def test_private_read_cannot_address_another_owner_or_traverse(scopes, path):
    shared, a, b = scopes
    assert put(b, "note.md", "other owner's fact")[0]
    ok, result = a.run_sync(MEMORY_READ_TOOL, {"path": path})
    assert not ok and "other owner's fact" not in result
    assert a.run_sync(MEMORY_SEARCH_TOOL, {"pattern": "other owner"})[0]
    assert a.read_index.grep("other owner")["file_count"] == 0


def test_client_key_arguments_cannot_select_another_owner(scopes):
    _, a, b = scopes
    ok, result = a.run_sync(MEMORY_WRITE_TOOL, {
        "path": "note.md", "title": "Note", "content": "current owner", "key_id": KEY_B,
    })
    assert ok, result
    assert a.index.path_of("work-memory/note.md").exists()
    assert not b.index.path_of("work-memory/note.md").exists()


def test_union_search_limits_filters_context_and_read_round_trip(scopes):
    shared, a, _ = scopes
    for executor, owner in ((shared, "shared"), (a, "own")):
        for number in range(3):
            assert put(executor, f"n{number}.md", f"before {owner}\nneedle {owner} {number}\nafter {owner}\nneedle skip\n")[0]
    result = a.read_index.grep("needle", max_results=2, per_file=1, context=1, exclude="skip")
    assert result["file_count"] == result["total_matches"] == 6
    assert result["count"] <= 2
    assert sum(entry["shown"] for entry in result["files"]) == 2
    assert result["truncated"]
    for hit in result["hits"]:
        assert [line["line"] for line in hit["context"]] == [hit["line"] - 1, hit["line"] + 1]
        assert hit["context"][0]["text"].startswith("before ")
        assert hit["context"][1]["text"].startswith("after ")
    for entry in result["files"]:
        page = a.read_index.read_lines(entry["path"], offset=1, limit=2)
        assert page["path"] == entry["path"] and len(page["lines"]) == 2 and not page["eof"]
        raw = a.read_index.read_bytes(entry["path"], byte_offset=0, max_bytes=16)
        assert raw["path"] == entry["path"] and raw["end_offset"] <= 16
    for path, expected in (("private/work-memory", 3), ("work-memory", 3), ("*.md", 6), ("private/**/n0.md", 1)):
        for mode in ("content", "files", "count"):
            filtered = a.read_index.grep("needle", path=path, mode=mode, exclude="skip")
            assert filtered["file_count"] == expected
            assert filtered["total_matches"] == expected
    for arguments in ({"context": 21}, {"per_file": 51}, {"max_results": 0}, {"path": "private/../*"}):
        with pytest.raises(FileKnowledgeError):
            a.read_index.grep("needle", **arguments)


def test_union_folds_duplicate_paths_and_counts_boilerplate_across_scopes(scopes):
    shared, a, _ = scopes
    banner = "repeated scoped convention: preserve the original evidence exactly"
    for executor, names in ((shared, ("s0", "s1", "s2")), (a, ("p0", "p1"))):
        for name in names:
            assert put(executor, name, banner)[0]
    hidden = a.read_index.grep("repeated scoped convention")
    assert hidden["total_matches"] == 0 and hidden["hidden_boilerplate"] == 5
    shown = a.read_index.grep("repeated scoped convention", include_boilerplate=True)
    assert shown["total_matches"] == 5
    hit, = shown["hits"]
    assert len(hit["also_in"]) == 4
    for path in (hit["path"], *hit["also_in"]):
        assert banner in a.read_index.read_bytes(path)["text"]


@pytest.mark.parametrize("key_id", ["", "a", "../x", "qxk_", "qxk_legacy:other", KEY_A + "/../" + KEY_B])
def test_private_storage_rejects_invalid_owner_ids(tmp_path, key_id):
    with pytest.raises(FileKnowledgeError):
        get_private_memory_index(tmp_path / "knowledge", tmp_path / "state", key_id)


def test_private_storage_rejects_shared_root_overlap(tmp_path):
    with pytest.raises(FileKnowledgeError):
        get_private_memory_index(tmp_path, tmp_path / "state", KEY_A)
    with pytest.raises(FileKnowledgeError):
        get_private_memory_index(tmp_path / "private-memory" / KEY_A / "knowledge", tmp_path, KEY_A)


def symlink_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_private_storage_refuses_symlinked_owner_directory(tmp_path):
    shared = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    base = shared.state_directory / "private-memory"
    base.mkdir()
    symlink_or_skip(base / KEY_A, shared.root)
    with pytest.raises(FileKnowledgeError):
        get_private_memory_index(shared.root, shared.state_directory, KEY_A)


def test_private_tools_refuse_symlink_routes_to_shared_and_other_keys(scopes):
    shared, a, b = scopes
    assert put(shared, "note.md", "shared original")[0]
    assert put(b, "note.md", "other original")[0]
    symlink_or_skip(a.index.root / "work-memory", shared.index.root / "work-memory")
    assert not put(a, "note.md", "changed", "append")[0]
    assert not a.run_sync(MEMORY_READ_TOOL, {"path": "private/work-memory/note.md"})[0]
    a.index.root.joinpath("work-memory").unlink()
    symlink_or_skip(a.index.root / "work-memory", b.index.root / "work-memory")
    assert not a.run_sync(MEMORY_READ_TOOL, {"path": "private/work-memory/note.md"})[0]
    assert "shared original" in shared.index.read_bytes("work-memory/note.md")["text"]
    assert "other original" in b.index.read_bytes("work-memory/note.md")["text"]


def test_write_window_is_independent_and_serialized(scopes, monkeypatch):
    monkeypatch.setattr("qwen_exo_booster.memory_tools.MAX_WRITES_PER_WINDOW", 2)
    shared, a, b = scopes
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: put(a, f"n{i}", "bounded own note")[0], range(4)))
    assert sum(results) == 2
    assert put(shared, "n", "shared note")[0]
    assert put(b, "n", "other note")[0]


def test_server_caches_private_executors_and_preserves_mode_switch_storage(tmp_path):
    from sglang.srt.entrypoints.openai.tool_server import FileMemoryToolServer

    server = FileMemoryToolServer(tmp_path / "knowledge", tmp_path / "state")
    shared = server.executor_for_memory_scope(None, "shared")
    with ThreadPoolExecutor(max_workers=4) as pool:
        executors = list(pool.map(lambda key: server.executor_for_memory_scope(key, "private"), [KEY_A] * 4))
    a = executors[0]
    assert all(executor is a for executor in executors)
    assert shared is server.executor
    assert put(a, "retained.md", "retained private note")[0]
    assert server.executor_for_memory_scope(None, "shared").read_index.grep("retained")["file_count"] == 0
    assert server.executor_for_memory_scope(KEY_A, "private") is a
    assert a.read_index.grep("retained")["file_count"] == 1
    assert server.executor_for_memory_scope(KEY_B, "private").read_index.grep("retained")["file_count"] == 0
    with pytest.raises(FileKnowledgeError):
        server.executor_for_memory_scope("../" + KEY_B, "private")


def test_private_summary_and_long_line_paths_round_trip(scopes):
    shared, a, b = scopes
    assert put(a, "long.md", "前" * 1000 + "needle-long" + "后" * 1000)[0]
    source = "work-memory/long.md"
    digest, _ = a.index.read_text(source)
    a.index.store_summary(source, sha256=digest, summary="needle-summary for owner A", sampled=False, origin="auto")
    result = a.read_index.grep("needle-long")
    hit, = result["hits"]
    assert hit["path"] == "private/" + source
    assert hit["byte_offset"] is not None
    ok, text = a.run_sync(MEMORY_READ_TOOL, {"path": hit["path"], "byte_offset": hit["byte_offset"], "max_bytes": 32})
    assert ok and text.startswith("needle-long")
    summary, = a.read_index.grep("needle-summary")["files"]
    assert summary["path"] == "private/" + source + ".summary.md"
    ok, text = a.run_sync(MEMORY_READ_TOOL, {"path": summary["path"]})
    assert ok and "needle-summary for owner A" in text
    assert shared.read_index.grep("needle-summary")["file_count"] == 0
    assert b.read_index.grep("needle-summary")["file_count"] == 0


@pytest.fixture
def mode_server(tmp_path):
    from sglang.srt.entrypoints.openai.tool_server import FileMemoryToolServer

    server = FileMemoryToolServer(tmp_path / "knowledge", tmp_path / "state")
    for key, scope, text in ((None, "shared", "matrix public"), (KEY_A, "private", "matrix ownerA"), (KEY_B, "private", "matrix ownerB")):
        assert put(server.executor_for_memory_scope(key, scope), "same.md", text)[0]
    return server


@pytest.mark.parametrize("scope,expected", [
    ("shared", {"work-memory/same.md"}),
    ("private", {"work-memory/same.md", "private/work-memory/same.md"}),
    ("private_only", {"private/work-memory/same.md"}),
    ("super", {"work-memory/same.md", "private/work-memory/same.md", f"private-keys/{KEY_B}/work-memory/same.md"}),
])
def test_four_mode_filesystem_read_search_count_matrix(mode_server, scope, expected):
    executor = mode_server.executor_for_memory_scope(KEY_A, scope)
    paths = {"work-memory/same.md", "private/work-memory/same.md", f"private-keys/{KEY_B}/work-memory/same.md"}
    for mode in ("content", "files", "count"):
        result = executor.read_index.grep("matrix", mode=mode, max_results=1)
        assert {entry["path"] for entry in result["files"]} == expected
        assert result["total_matches"] == result["file_count"] == len(expected)
        assert result["count"] <= 1
    for path in paths:
        for extra in ({}, {"byte_offset": 0, "max_bytes": 256}):
            ok, text = executor.run_sync(MEMORY_READ_TOOL, {"path": path, **extra})
            assert ok == (path in expected), text
            if ok:
                assert path in text
            else:
                assert "matrix" not in text
    for prefix in ("private", "private-keys", "work-memory"):
        ok, text = executor.run_sync(MEMORY_SEARCH_TOOL, {"pattern": "matrix", "path": prefix, "mode": "count"})
        if scope == "shared" and prefix in {"private", "private-keys"}:
            assert not ok
        else:
            assert ok
            if scope == "private_only" and prefix != "private":
                assert "No memory matches" in text


def test_private_only_never_consults_shared_index(mode_server, monkeypatch):
    executor = mode_server.executor_for_memory_scope(KEY_A, "private_only")

    def forbidden(*args, **kwargs):
        raise AssertionError("private_only touched shared memory")

    for name in ("refresh", "_connect", "read_lines", "read_bytes"):
        monkeypatch.setattr(mode_server.executor.index, name, forbidden)
    for mode in ("content", "files", "count"):
        assert executor.read_index.grep("matrix", mode=mode)["file_count"] == 1
    for path in ("work-memory/same.md", "shared/work-memory/same.md", f"private-keys/{KEY_B}/work-memory/same.md", f"private/../private-keys/{KEY_B}/work-memory/same.md"):
        assert not executor.run_sync(MEMORY_READ_TOOL, {"path": path})[0]
    assert executor.run_sync(MEMORY_READ_TOOL, {"path": "private/work-memory/same.md"})[0]


@pytest.mark.parametrize("scope", ["private", "private_only", "super"])
@pytest.mark.parametrize("mode", ["create", "append", "edit", "replace", "delete"])
def test_all_private_modes_reject_other_owner_write_paths(mode_server, scope, mode):
    executor = mode_server.executor_for_memory_scope(KEY_A, scope)
    other = mode_server.executor_for_memory_scope(KEY_B, "private")
    before = other.index.read_bytes("work-memory/same.md")["text"]
    shared_before = mode_server.executor.index.read_bytes("work-memory/same.md")["text"]
    for path in (
        f"private-keys/{KEY_B}/work-memory/same.md",
        f"private/private-keys/{KEY_B}/work-memory/same.md",
        f"work-memory/private-keys/{KEY_B}/same.md",
        f"private/work-memory/private-keys/{KEY_B}/same.md",
        f"private/../private-keys/{KEY_B}/work-memory/same.md",
        f"private-keys\\{KEY_B}\\work-memory\\same.md",
    ):
        ok, _ = executor.run_sync(MEMORY_WRITE_TOOL, {
            "path": path, "mode": mode, "title": "changed", "content": "changed",
            "old_text": "matrix", "new_text": "changed",
        })
        assert not ok
    assert other.index.read_bytes("work-memory/same.md")["text"] == before
    assert mode_server.executor.index.read_bytes("work-memory/same.md")["text"] == shared_before


@pytest.mark.parametrize("scope", ["private", "private_only", "super"])
def test_private_mode_mutations_use_same_owner_storage(mode_server, scope):
    executor = mode_server.executor_for_memory_scope(KEY_A, scope)
    shared_before = mode_server.executor.index.read_bytes("work-memory/same.md")["text"]
    assert put(executor, "same.md", "own appended", "append")[0]
    assert executor.run_sync(MEMORY_WRITE_TOOL, {
        "path": "private/work-memory/same.md", "mode": "edit",
        "old_text": "own appended", "new_text": "own edited",
    })[0]
    assert put(executor, "same.md", "own replaced", "replace")[0]
    assert "own replaced" in executor.read_index.read_bytes("private/work-memory/same.md")["text"]
    assert executor.run_sync(MEMORY_WRITE_TOOL, {"path": "same.md", "mode": "delete"})[0]
    assert mode_server.executor.index.read_bytes("work-memory/same.md")["text"] == shared_before


def test_super_enumerates_new_and_removed_spaces_without_creating_directories(mode_server):
    executor = mode_server.executor_for_memory_scope(KEY_A, "super")
    state = mode_server.executor.index.state_directory
    base = state / "private-memory"
    before = {str(path.relative_to(base)) for path in base.rglob("*") if path.is_dir()}
    assert private_memory_key_ids(state) == (KEY_A, KEY_B)
    assert executor.read_index.grep("matrix")["file_count"] == 3
    assert {str(path.relative_to(base)) for path in base.rglob("*") if path.is_dir()} == before
    # No registry record exists for this legacy named owner (removed/revoked
    # keys have the same persistent layout and must remain visible).
    legacy = "qxk_removed-owner"
    other = mode_server.executor_for_memory_scope(legacy, "private_only")
    assert put(other, "legacy.md", "matrix removed owner")[0]
    path = f"private-keys/{legacy}/work-memory/legacy.md"
    assert executor.read_index.grep("matrix", path=f"private-keys/{legacy}")["file_count"] == 1
    assert "matrix removed owner" in executor.read_index.read_bytes(path)["text"]
    assert legacy in private_memory_key_ids(state)
    shutil.rmtree(base / legacy)
    assert legacy not in private_memory_key_ids(state)
    assert executor.read_index.grep("matrix")["file_count"] == 3
    assert not executor.run_sync(MEMORY_READ_TOOL, {"path": path})[0]
    assert not (base / legacy).exists()


def test_super_enumeration_excludes_links_and_invalid_names(mode_server):
    state = mode_server.executor.index.state_directory
    base = state / "private-memory"
    (base / "not-a-key" / "knowledge").mkdir(parents=True)
    (base / "qxk_incomplete" / "knowledge").mkdir(parents=True)
    link = base / "qxk_linked"
    symlink_or_skip(link, base / KEY_B)
    assert private_memory_key_ids(state) == (KEY_A, KEY_B)
    executor = mode_server.executor_for_memory_scope(KEY_A, "super")
    assert executor.read_index.grep("matrix")["file_count"] == 3
    assert not executor.run_sync(MEMORY_READ_TOOL, {"path": "private-keys/qxk_linked/work-memory/same.md"})[0]


def test_mode_switch_cannot_reset_owner_write_window(mode_server, monkeypatch):
    monkeypatch.setattr("qwen_exo_booster.memory_tools.MAX_WRITES_PER_WINDOW", 3)
    executors = [mode_server.executor_for_memory_scope(KEY_A, scope) for scope in ("private", "private_only", "super")]
    assert len({id(executor) for executor in executors}) == 3
    assert len({id(executor._write_window) for executor in executors}) == 1
    # Fixture has already consumed one successful owner-A write.
    with ThreadPoolExecutor(max_workers=6) as pool:
        successes = list(pool.map(lambda i: put(executors[i % 3], f"window-{i}.md", "bounded note")[0], range(6)))
    assert sum(successes) == 2
    for executor in executors:
        assert not put(executor, "over-budget.md", "blocked note")[0]
    assert put(mode_server.executor_for_memory_scope(KEY_B, "private_only"), "independent.md", "other budget")[0]


@pytest.mark.parametrize("scope", ["shared", "private", "private_only", "super"])
def test_scoped_schemas_state_exact_read_and_write_rights(mode_server, scope):
    schemas = mode_server.chat_tool_schemas_for_memory_scope(KEY_A, scope)
    for schema in schemas:
        description = schema["function"]["description"]
        assert f"uses {scope} memory" in description
        if scope != "shared":
            assert "ONLY your own private work-memory" in description
            assert "private-keys/ write paths are rejected" in description
        if scope == "private_only":
            assert "unprefixed shared paths are denied" in description
        if scope == "super":
            assert "removed or revoked owners" in description
            assert "no key administration or conversation access" in description


@pytest.mark.parametrize("scope", ["private", "private_only", "super", "invalid"])
def test_server_rejects_ownerless_nonshared_scopes(mode_server, scope):
    with pytest.raises(FileKnowledgeError):
        mode_server.executor_for_memory_scope(None, scope)
    with pytest.raises(FileKnowledgeError):
        mode_server.chat_tool_schemas_for_memory_scope(None, scope)
