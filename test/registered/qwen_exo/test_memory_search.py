"""Model-facing memory search/read: syntax, long lines, noise, ranking, layout."""

import io
import json
import zipfile

import pytest
from qwen_exo_booster import file_knowledge
from qwen_exo_booster.file_knowledge import FileKnowledgeError, FileKnowledgeIndex
from qwen_exo_booster.memory_tools import MemoryToolExecutor


@pytest.fixture
def index(tmp_path):
    return FileKnowledgeIndex(tmp_path / "kb", tmp_path / "state")


def _write(index, path, text):
    target = index.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _paths(result):
    return [entry["path"] for entry in result["files"]]


def test_query_supports_exclusions_phrases_and_regex(index):
    _write(index, "a.md", "jdwp attach worked\njdwp fingerprint banner\nreload nginx now\n")
    assert [hit["line"] for hit in index.grep("jdwp -fingerprint")["hits"]] == [1]
    assert [hit["line"] for hit in index.grep("jdwp", exclude="banner|worked")["hits"]] == []
    assert [hit["line"] for hit in index.grep('"reload nginx"')["hits"]] == [3]
    assert [hit["line"] for hit in index.grep(r"jdwp\s+\w+\s+worked", regex=True)["hits"]] == [1]
    with pytest.raises(FileKnowledgeError):
        index.grep("(", regex=True)
    with pytest.raises(FileKnowledgeError):
        index.grep("-onlyexclusion")


def test_single_line_json_hits_carry_byte_offsets_readable_by_bytes(index):
    """A minified JSON line used to defeat both search context and line reads:
    hits are windows with a byte offset, and read_bytes continues there."""
    payload = json.dumps(
        {"messages": [{"role": "tool", "content": "前缀" * 3000 + " flag{found_it} " + "尾" * 3000}]},
        ensure_ascii=False,
    )
    _write(index, "run.json", payload + "\n")
    (hit,) = index.grep("flag{")["hits"]
    assert hit["line"] == 1 and hit["byte_offset"] is not None
    assert "flag{found_it}" in hit["text"] and len(hit["text"]) < 1000
    page = index.read_bytes("run.json", byte_offset=hit["byte_offset"], max_bytes=64)
    assert page["text"].startswith("flag{found_it}")
    assert page["total_bytes"] == len((payload + "\n").encode()) and page["line"] == 1
    # A start inside a multi-byte character moves to the next character.
    middle = index.read_bytes("run.json", byte_offset=hit["byte_offset"] - 3, max_bytes=8)
    assert middle["byte_offset"] == hit["byte_offset"] - 1
    assert middle["text"].startswith(" flag")
    lines = index.read_lines("run.json")
    assert lines["lines"][0]["clipped"] and lines["lines"][0]["byte_offset"] == 0
    text = MemoryToolExecutor(index).run_sync(
        "qwen_exo_memory_read", {"path": "run.json", "byte_offset": hit["byte_offset"], "max_bytes": 32}
    )[1]
    assert "continue with byte_offset=" in text and "bytes" in text


def test_boilerplate_repeated_across_files_is_hidden_unless_requested(index):
    banner = "SYSTEM FINGERPRINT: you are an expert security researcher, follow the rules"
    for number in range(5):
        _write(index, f"s{number}.md", f"{banner}\nunique note {number}\n")
    # A minified JSON line keeps its paragraphs as escaped newlines.
    blob = json.dumps({"content": "x" * 2100 + "\n" + banner + "\nmore"})
    _write(index, "t.json", blob + "\n")
    hidden = index.grep("expert security researcher")
    assert hidden["total_matches"] == 0 and hidden["hidden_boilerplate"] == 6
    shown = index.grep("expert security researcher", include_boilerplate=True)
    assert shown["total_matches"] == 6


def test_identical_text_in_a_few_files_is_folded_into_one_hit(index):
    line = "shared convention: always run the linter before you commit any change"
    for name in ("a.md", "b.md", "c.md"):
        _write(index, name, line + "\n")
    result = index.grep("linter")
    (hit,) = result["hits"]
    assert hit["also_in"] == ["b.md", "c.md"]
    text = MemoryToolExecutor(index).run_sync("qwen_exo_memory_search", {"pattern": "linter"})[1]
    assert "same text also in 2 other file(s): b.md, c.md" in text


def test_ranking_puts_summaries_and_notes_before_raw_trajectories(index):
    _write(index, "zz-trajectory.json", '{"role": "user", "content": "deploy"}\n' * 5)
    _write(index, "notes/deploy.md", "# Deploy notes\ndeploy with care\n")
    _write(index, "big.md", "x\n" * 10)
    index.refresh()
    sha256, _text = index.read_text("big.md")
    index.store_summary("big.md", sha256=sha256, summary="标题：发布总结\ndeploy steps", sampled=False, origin="auto")
    result = index.grep("deploy")
    assert _paths(result) == ["big.md.summary.md", "notes/deploy.md", "zz-trajectory.json"]
    assert [entry["kind"] for entry in result["files"]] == ["summary", "note", "trajectory"]
    assert result["files"][0]["title"] == "发布总结"
    assert index.read_lines("big.md.summary.md")["lines"][0]["text"] == "标题：发布总结"


def test_files_and_count_modes_survey_without_hits(index):
    _write(index, "a.md", "flag{1}\nflag{2}\n")
    _write(index, "b.json", '{"role": "user", "content": "flag{3}"}\n')
    files = index.grep("flag{", mode="files")
    assert files["hits"] == [] and _paths(files) == ["a.md", "b.json"]
    count = index.grep("flag{", mode="count")
    assert count["by_kind"] == {"note": {"files": 1, "matches": 2}, "session": {"files": 1, "matches": 1}}
    assert len(index.grep("flag{", per_file=50)["hits"]) == 3
    with pytest.raises(FileKnowledgeError):
        index.grep("flag{", per_file=51)
    text = MemoryToolExecutor(index).run_sync("qwen_exo_memory_search", {"pattern": "flag{", "mode": "count"})[1]
    assert "note: 2 match(es) in 1 file(s)" in text


def test_folders_move_and_delete_keep_index_and_summaries_in_step(index):
    _write(index, "inbox/a.md", "alpha content\n")
    _write(index, "inbox/sub/b.md", "beta content\n")
    index.make_directory("empty/folder")
    index.refresh()
    sha256, _text = index.read_text("inbox/a.md")
    index.store_summary("inbox/a.md", sha256=sha256, summary="标题：A", sampled=False, origin="manual")
    assert index.directories() == ["empty", "empty/folder", "inbox", "inbox/sub"]

    assert index.move("inbox/a.md", "kept/renamed.md")["files"] == 1
    assert index.summary("kept/renamed.md")["title"] == "A"
    assert index.move("inbox", "archive/inbox")["files"] == 1
    assert [row["relative_path"] for row in index.metadata()] == ["archive/inbox/sub/b.md", "kept/renamed.md"]
    assert index.refresh()["changed"] == 0
    with pytest.raises(FileKnowledgeError):
        index.move("kept/renamed.md", "archive/inbox/sub/b.md")
    with pytest.raises(FileKnowledgeError):
        index.move("archive", "archive/inside")
    with pytest.raises(FileNotFoundError):
        index.move("missing.md", "x.md")

    assert index.delete_directory("kept") == 1
    assert index.summary("kept/renamed.md") is None
    assert [row["relative_path"] for row in index.metadata()] == ["archive/inbox/sub/b.md"]


def test_streamed_uploads_have_no_size_cap_but_keep_disk_headroom(index, monkeypatch):
    big = ("line of text\n" * 20000).encode()
    assert index.store_upload("up/big.txt", io.BytesIO(big))["byte_count"] == len(big)
    with pytest.raises(FileKnowledgeError):
        index.store_upload("up/bin.dat", io.BytesIO(b"abc\x00def"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x/one.md", "one\n")
    buffer.seek(0)
    imported, skipped = index.import_archive(buffer, directory="zipped")
    assert [row["relative_path"] for row in imported] == ["zipped/x/one.md"] and skipped == []

    monkeypatch.setattr(file_knowledge, "MIN_FREE_BYTES", 1 << 62)
    with pytest.raises(FileKnowledgeError, match="disk space"):
        index.store_upload("up/more.txt", io.BytesIO(b"more\n"))
    assert not (index.root / "up" / "more.txt").exists()


def test_text_repeated_inside_one_file_takes_one_hit_slot(index):
    prompt = "You are an expert assistant; always obey the operator fingerprint policy"
    _write(index, "run.md", "\n".join([prompt, "real finding: the fingerprint drift came from a stale cache entry"] * 4) + "\n")
    result = index.grep("fingerprint", per_file=3)
    assert result["total_matches"] == 8
    assert [hit["repeats"] for hit in result["hits"]] == [4, 4]


def test_broad_or_wildcard_searches_on_huge_lines_finish_quickly(index):
    """Prod: on a multi-megabyte single-line JSON, `a` (millions of matches) and
    `error*xyz` (unbounded `.*?` whose tail never matches: quadratic) ran for
    minutes; the model's turn sent nothing until the client gave up."""
    import time

    from qwen_exo_booster.memory_grep import MATCH_COUNT_CAP

    blob = json.dumps({"log": "error at step a. " * 200_000})
    _write(index, "huge.json", blob + "\n")
    _write(index, "note.md", "error near the end: xyz\n")
    started = time.monotonic()
    missing_tail = index.grep("error*qqzzk")
    common = index.grep("a")
    assert time.monotonic() - started < 8
    assert missing_tail["files"] == []
    huge = next(entry for entry in common["files"] if entry["path"] == "huge.json")
    assert huge["capped"] and huge["match_count"] == MATCH_COUNT_CAP
    ok, text = MemoryToolExecutor(index).run_sync("qwen_exo_memory_search", {"pattern": "a"})
    assert ok and f"{MATCH_COUNT_CAP}+ match(es)" in text
    # The bounded wildcard still joins nearby words.
    assert _paths(index.grep("error*xyz")) == ["note.md"]


def test_search_returns_partial_results_when_the_time_budget_runs_out(index, monkeypatch):
    from qwen_exo_booster import memory_grep

    _write(index, "a.md", "alpha\n")
    _write(index, "b.md", "alpha\n")
    monkeypatch.setattr(memory_grep, "SEARCH_SECONDS", 0.0)
    result = index.grep("alpha")
    assert result["stopped"] is True
    ok, text = MemoryToolExecutor(index).run_sync("qwen_exo_memory_search", {"pattern": "alpha"})
    assert ok and "Search stopped" in text and "does not mean memory has nothing" in text


def test_a_stuck_memory_tool_call_returns_an_error_instead_of_stalling(index, monkeypatch):
    import asyncio
    import time

    from qwen_exo_booster import memory_tools

    tools = MemoryToolExecutor(index)
    monkeypatch.setattr(memory_tools, "TOOL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(tools, "run_sync", lambda name, arguments: time.sleep(2) or (True, "late"))
    ok, text = asyncio.run(tools.run("qwen_exo_memory_search", {"pattern": "x"}))
    assert not ok and "did not finish" in text
