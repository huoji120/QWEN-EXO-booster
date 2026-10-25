import hashlib

from qwen_exo_booster.memory_hints import (
    MemoryHintIndex,
    chunk_text,
    is_memory_block,
    render_memory_block,
    text_terms,
)


def _rows(root, files):
    rows = []
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        rows.append(
            {
                "relative_path": path,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "encoding": "utf-8",
            }
        )
    return rows


FILES = {
    "ops/release.md": "# 发布流程\n\n灰度发布前检查 canary 指标。\n回滚由值班表第二位负责。\n",
    "ops/nginx.md": "# Nginx\n\nReload nginx after editing the upstream config.\n",
    "notes/frontend.md": "# 前端视觉规范\n\n按钮圆角统一 8px，页面留白不少于 24px。\n",
    "noise/blob.json": "{" + "\"k\": 1, " * 5000 + "}\n",
}


def test_terms_cover_words_identifier_parts_and_cjk_bigrams():
    terms = text_terms("fileKnowledge memory_tool 灰度发布")
    assert {"fileknowledge", "file", "knowledge", "memory_tool", "memory", "tool"} <= set(terms)
    assert {"灰度", "度发", "发布"} <= set(terms)


def test_chunks_follow_headings_and_skip_one_line_blobs():
    chunks = chunk_text("# A\nx\ny\n# B\nz\n" + "q" * 30_000 + "\nlast\n")
    assert [(start, end, heading) for start, end, heading, _ in chunks] == [
        (1, 3, "A"),
        (4, 5, "B"),
        (7, 7, "B"),
    ]


def test_lookup_ranks_relevant_chunks_with_synonyms_and_threshold(tmp_path):
    index = MemoryHintIndex(tmp_path / "kb", tmp_path / "state" / "hints.sqlite3")
    assert index.refresh(_rows(tmp_path / "kb", FILES))["indexed"] == 4

    hints = index.lookup("灰度发布出问题了，怎么 rollback？谁负责值班")
    assert [hint["path"] for hint in hints] == ["ops/release.md"]
    assert hints[0]["line"] in (3, 4) and hints[0]["heading"] == "发布流程"

    assert index.lookup("今天天气怎么样") == []
    assert index.lookup("reload nginx upstream")[0]["path"] == "ops/nginx.md"
    excluded = index.lookup(
        "reload nginx upstream", exclude=frozenset({("ops/nginx.md", 3)})
    )
    assert excluded == []


def test_refresh_drops_removed_and_reindexes_changed_files(tmp_path):
    index = MemoryHintIndex(tmp_path / "kb", tmp_path / "state" / "hints.sqlite3")
    rows = _rows(tmp_path / "kb", FILES)
    index.refresh(rows)
    changed = _rows(tmp_path / "kb", {"ops/nginx.md": "# Nginx\n\nUse caddy instead.\n"})
    result = index.refresh([row for row in rows if row["relative_path"] != "ops/release.md" and row["relative_path"] != "ops/nginx.md"] + changed)
    assert result == {"removed": 1, "indexed": 1, "files": 3}
    assert index.lookup("灰度发布 回滚 值班") == []
    assert set(index.indexed_files()) == {"ops/nginx.md", "notes/frontend.md", "noise/blob.json"}


def test_transfer_invalidation_fences_captured_summary_refresh(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    index = MemoryHintIndex(tmp_path / "kb", tmp_path / "state" / "hints.sqlite3")
    background = _rows(tmp_path / "kb", {"background.md": "Unrelated gardening flowers and soil."})
    rows = [{"relative_path": "secret.md.summary.md", "sha256": "old",
             "text": "Private transport client evidence must not remain shared."}, *background]
    index.refresh(rows)
    assert index.lookup("private transport client evidence")
    generation = index.generation
    captured, resume = Event(), Event()

    def stale_refresh():
        captured.set()
        assert resume.wait(5)
        return index.refresh(rows, generation=generation)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stale_refresh)
        try:
            assert captured.wait(5)
            index.invalidate()
        finally:
            resume.set()
        assert future.result()["indexed"] == 0
    assert index.indexed_files() == {}
    assert index.lookup("private transport client evidence") == []
    index.refresh([], generation=index.generation)
    assert index.indexed_files() == {}


def test_lookup_skips_term_rows_whose_chunk_vanished(tmp_path):
    """Prod: a refresh committed between the FTS match and the chunk fetch, the
    fetch returned None and the memory block lookup raised TypeError."""
    import sqlite3

    index = MemoryHintIndex(tmp_path / "kb", tmp_path / "state" / "hints.sqlite3")
    index.refresh(_rows(tmp_path / "kb", FILES))
    with sqlite3.connect(index.database) as connection:
        connection.execute("DELETE FROM chunks WHERE path = 'ops/release.md'")
    assert index.lookup("灰度发布出问题了，怎么 rollback？谁负责值班") == []
    assert index.lookup("reload nginx upstream")[0]["path"] == "ops/nginx.md"


def test_memory_block_says_hits_are_unrelated_recollections():
    """Hits are framed as memory, not as files of the current workspace, and
    the no-hit block still prompts a memory search."""
    block = render_memory_block(
        [{"path": "ops/release.md", "line": 4, "heading": "发布流程", "excerpt": "回滚由值班表第二位负责。"}]
    )
    assert is_memory_block(block)
    assert "与当前会话无关" in block and "不要试图在当前 agent" in block
    assert "- 记忆文件 ops/release.md 第 4 行 [发布流程]：回滚由值班表第二位负责。" in block
    empty = render_memory_block([])
    assert is_memory_block(empty) and "调用 qwen_exo_memory_search 搜索记忆" in empty
    assert not is_memory_block("plain answer")
