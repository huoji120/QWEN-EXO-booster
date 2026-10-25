"""BM25 hints over the knowledge files for the server-side memory block.

A small SQLite FTS5 index of knowledge chunks. Terms are words and code
identifiers (plus their camelCase / snake_case parts) and CJK character
bigrams, so Chinese needs no segmenter. A lookup keeps the query's rarer terms,
widens a few common Chinese/English synonyms, ranks chunks by BM25 and keeps
those covering enough of the query's IDF weight. Hints only point the model at
memory (path, line, short excerpt); it reads the full text with the memory
tools itself.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


CHUNK_LINES = 40
CHUNK_CHARS = 4000
# Minified JSON and similar one-line blobs give useless line hints; the
# search tool still covers them.
MAX_INDEXED_LINE_CHARS = 20_000
MAX_QUERY_CHARS = 4000
MAX_QUERY_TERMS = 24
MIN_COVERAGE = 0.3
MIN_MATCHED_CONCEPTS = 2
EXCERPT_CHARS = 200

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]{3,}")
_WORD_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_CJK = re.compile(r"[㐀-䶿一-鿿]+")
_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")

_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have he her his how
    if in into is it its me my no not of on or our she so than that the their
    them then there these they this to too us was we were what when where which
    who why will with you your all any also just more most only other some such
    very use used using get got let make made need new now one two out up""".split()
)

# CJK bigrams with a function character mostly span word boundaries
# ("面的", "的视") and only dilute the query.
_CJK_STOP_CHARS = frozenset("的了和是在我你他她它们这那个也就都而及与或把被让给吗呢吧啊着过对从")
# Question and filler words carry no topic.
_CJK_STOP_BIGRAMS = frozenset(
    """应该 哪些 地方 同时 怎么 如何 什么 为什么 可以 需要 一下 一个 这个 那个 现在
    已经 还是 就是 不是 没有 因为 所以 如果 然后 时候 帮我 我们 你们 他们 一些 有没
    能不 是否 进行 使用""".split()
)
# Raw dumps (sessions, trajectories) match almost anything; curated notes win ties.
_RAW_SUFFIX_WEIGHT = {".json": 0.8, ".jsonl": 0.8}
# Generated summaries condense a whole file; prefer them over raw chunks.
_SUMMARY_WEIGHT = 1.2

# Each group is one concept: any member counts as a match.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("部署", "deploy", "deployment", "发布", "上线", "release"),
    ("回滚", "rollback", "revert"),
    ("报错", "错误", "异常", "error", "exception", "failure"),
    ("配置", "设置", "config", "configuration", "settings"),
    ("测试", "test", "tests", "testing"),
    ("数据库", "database", "db", "sqlite", "mysql", "postgres"),
    ("前端", "界面", "frontend", "ui"),
    ("后端", "服务端", "backend", "server"),
    ("缓存", "cache"),
    ("日志", "log", "logs", "logging"),
    ("端口", "port"),
    ("登录", "认证", "login", "auth", "authentication"),
    ("密码", "密钥", "password", "secret"),
    ("超时", "timeout"),
    ("权限", "permission", "permissions"),
    ("依赖", "dependency", "dependencies"),
    ("构建", "编译", "build", "compile"),
    ("规范", "约定", "规则", "convention", "conventions", "guideline", "guidelines", "rule", "rules"),
    ("视觉", "visual", "visuals"),
    ("质量", "quality"),
    ("页面", "page", "pages"),
    ("检查", "审查", "check", "review", "audit"),
    ("验证", "校验", "verify", "verification", "validation"),
    ("文档", "doc", "docs", "documentation"),
    ("修改", "改动", "变更", "change", "changes", "modify"),
    ("代码", "code"),
    ("文件", "file", "files"),
    ("问题", "故障", "issue", "problem", "bug"),
    ("性能", "performance", "latency"),
    ("安全", "security"),
    ("接口", "api", "endpoint"),
    ("模型", "model"),
    ("请求", "request", "requests"),
    ("环境", "environment", "env"),
    ("版本", "version"),
    ("脚本", "script"),
    ("命令", "command"),
    ("函数", "function"),
    ("安装", "install", "installation"),
    ("升级", "更新", "upgrade", "update"),
    ("调试", "debug", "debugging"),
    ("内存", "memory", "oom"),
    ("显存", "gpu", "vram", "cuda"),
)


def text_terms(text: str) -> list[str]:
    """Index/query terms in order: lowercased words, identifier parts, CJK bigrams."""
    terms: list[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        if len(lowered) >= 2 and lowered not in _STOPWORDS:
            terms.append(lowered)
        parts = [part.lower() for part in _WORD_PART.findall(word.replace("_", " "))]
        if len(parts) > 1:
            terms.extend(
                part for part in parts if len(part) >= 2 and part not in _STOPWORDS
            )
    for run in _CJK.findall(text):
        if len(run) == 1 and run not in _CJK_STOP_CHARS:
            terms.append(run)
        terms.extend(
            bigram
            for bigram in (run[index : index + 2] for index in range(len(run) - 1))
            if not (_CJK_STOP_CHARS & set(bigram)) and bigram not in _CJK_STOP_BIGRAMS
        )
    return terms


def _member_terms(member: str) -> tuple[str, ...]:
    return tuple(text_terms(member)) or (member.lower(),)


_SYNONYMS: dict[str, tuple[tuple[str, ...], ...]] = {}
for _group in SYNONYM_GROUPS:
    _members = tuple(_member_terms(member) for member in _group)
    for _member in _members:
        if len(_member) == 1:
            _SYNONYMS[_member[0]] = _members


def _fts_phrase(terms: tuple[str, ...]) -> str:
    return '"' + " ".join(term.replace('"', '""') for term in terms) + '"'


def chunk_text(text: str) -> list[tuple[int, int, str, str]]:
    """``(start_line, end_line, heading, text)`` chunks of a file, 1-based."""
    chunks: list[tuple[int, int, str, str]] = []
    heading = ""
    start = 0
    buffer: list[str] = []
    size = 0

    def flush(end_line: int) -> None:
        nonlocal buffer, size
        body = "\n".join(buffer).strip()
        if body:
            chunks.append((start + 1, end_line, heading, body))
        buffer, size = [], 0

    for index, line in enumerate(text.splitlines()):
        if len(line) > MAX_INDEXED_LINE_CHARS:
            flush(index)
            continue
        match = _HEADING.match(line)
        if match and buffer:
            flush(index)
        if match:
            heading = match.group(1)[:120]
        if not buffer:
            start = index
        buffer.append(line)
        size += len(line) + 1
        if len(buffer) >= CHUNK_LINES or size >= CHUNK_CHARS:
            flush(index + 1)
    flush(start + len(buffer))
    return chunks


class MemoryHintIndex:
    def __init__(self, knowledge_directory: Path, database: Path):
        self.knowledge_directory = Path(knowledge_directory)
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._generation = 0
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY, sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY, path TEXT NOT NULL,
                    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                    heading TEXT NOT NULL, text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_terms USING fts5(terms);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def indexed_files(self) -> dict[str, str]:
        with closing(self._connect()) as connection:
            return dict(connection.execute("SELECT path, sha256 FROM files"))

    @property
    def generation(self) -> int:
        return self._generation

    def invalidate(self) -> None:
        """Fence captured rows, including synthetic summaries, and remove old hints."""
        with self._write_lock, closing(self._connect()) as connection, connection:
            self._generation += 1
            connection.execute("DELETE FROM chunk_terms")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM files")

    def refresh(
        self, files: Iterable[dict[str, Any]], *, generation: int | None = None
    ) -> dict[str, int]:
        """Sync with ``FileKnowledgeIndex.metadata()`` rows (path, sha256, encoding)."""
        wanted = {str(row["relative_path"]): row for row in files}
        with self._write_lock, closing(self._connect()) as connection:
            if generation is not None and generation != self._generation:
                return {"removed": 0, "indexed": 0, "files": 0}
            current = dict(connection.execute("SELECT path, sha256 FROM files"))
            removed = [path for path in current if path not in wanted]
            changed = [
                path
                for path, row in wanted.items()
                if current.get(path) != str(row["sha256"])
            ]
            for path in removed + changed:
                self._drop(connection, path)
            for path in changed:
                row = wanted[path]
                text = row.get("text")
                if text is None:
                    try:
                        text = (self.knowledge_directory / path).read_text(
                            encoding=str(row.get("encoding") or "utf-8"), errors="replace"
                        )
                    except OSError:
                        continue
                for start, end, heading, body in chunk_text(text):
                    cursor = connection.execute(
                        "INSERT INTO chunks(path, start_line, end_line, heading, text)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (path, start, end, heading, body),
                    )
                    terms = text_terms(f"{path} {heading} {body}")
                    connection.execute(
                        "INSERT INTO chunk_terms(rowid, terms) VALUES (?, ?)",
                        (cursor.lastrowid, " ".join(terms)),
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO files(path, sha256) VALUES (?, ?)",
                    (path, str(row["sha256"])),
                )
                connection.commit()
            connection.commit()
        return {"removed": len(removed), "indexed": len(changed), "files": len(wanted)}

    @staticmethod
    def _drop(connection: sqlite3.Connection, path: str) -> None:
        ids = [row[0] for row in connection.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
        connection.executemany("DELETE FROM chunk_terms WHERE rowid = ?", ((i,) for i in ids))
        connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
        connection.execute("DELETE FROM files WHERE path = ?", (path,))

    def lookup(
        self,
        text: str,
        *,
        exclude: frozenset[tuple[str, int]] = frozenset(),
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Up to ``limit`` hints (one per file) relevant to ``text``."""
        query_terms = list(dict.fromkeys(text_terms(str(text)[-MAX_QUERY_CHARS:])))
        if not query_terms:
            return []
        with closing(self._connect()) as connection:
            # One WAL snapshot: refresh() commits per file, so separate reads
            # could see an FTS row whose chunk was deleted in between.
            connection.execute("BEGIN")
            total = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if not total:
                return []
            concepts = self._concepts(connection, query_terms, total)
            if len(concepts) < MIN_MATCHED_CONCEPTS:
                return []
            match = " OR ".join(
                _fts_phrase(member) for members, _weight in concepts for member in members
            )
            candidates = connection.execute(
                "SELECT rowid, bm25(chunk_terms), terms FROM chunk_terms"
                " WHERE chunk_terms MATCH ? ORDER BY bm25(chunk_terms) LIMIT 40",
                (match,),
            ).fetchall()
            total_weight = sum(weight for _members, weight in concepts)
            scored = []
            paths = dict(
                connection.execute(
                    "SELECT id, path FROM chunks WHERE id IN (%s)"
                    % ",".join(str(int(rowid)) for rowid, _score, _terms in candidates)
                )
            ) if candidates else {}
            for rowid, bm25_score, chunk_terms in candidates:
                present = set(chunk_terms.split())
                matched = [
                    (members, weight)
                    for members, weight in concepts
                    if any(all(term in present for term in member) for member in members)
                ]
                coverage = sum(weight for _members, weight in matched) / total_weight
                coverage *= _path_weight(paths.get(rowid, ""))
                if len(matched) >= MIN_MATCHED_CONCEPTS and coverage >= MIN_COVERAGE:
                    scored.append((coverage, -bm25_score, rowid, matched))
            scored.sort(reverse=True)
            hints: list[dict[str, Any]] = []
            seen_paths: set[str] = set()
            for coverage, _score, rowid, matched in scored:
                row = connection.execute(
                    "SELECT path, start_line, heading, text FROM chunks WHERE id = ?",
                    (rowid,),
                ).fetchone()
                if row is None:
                    continue
                path, start, heading, body = row
                if path in seen_paths:
                    continue
                line, excerpt = _best_line(body, start, matched)
                if (path, line) in exclude or _looks_garbled(excerpt):
                    continue
                seen_paths.add(path)
                hints.append(
                    {
                        "path": path,
                        "line": line,
                        "heading": heading,
                        "excerpt": excerpt,
                        "coverage": round(coverage, 3),
                    }
                )
                if len(hints) >= limit:
                    break
            return hints

    @staticmethod
    def _concepts(
        connection: sqlite3.Connection, query_terms: list[str], total: int
    ) -> list[tuple[tuple[tuple[str, ...], ...], float]]:
        """Rarest query terms (df > 0) as synonym-widened, IDF-weighted concepts."""
        weighted = []
        for term in query_terms[:200]:
            df = connection.execute(
                "SELECT count(*) FROM chunk_terms WHERE chunk_terms MATCH ?",
                (_fts_phrase((term,)),),
            ).fetchone()[0]
            if 0 < df <= total * 0.5:
                weighted.append((math.log((total - df + 0.5) / (df + 0.5) + 1.0), term))
        weighted.sort(reverse=True)
        concepts = []
        seen: set[tuple[str, ...]] = set()
        for weight, term in weighted[:MAX_QUERY_TERMS]:
            members = _SYNONYMS.get(term, ((term,),))
            if members in seen:
                continue
            seen.add(members)
            concepts.append((members, weight))
        return concepts


def _path_weight(path: str) -> float:
    if path.endswith(".summary.md"):
        return _SUMMARY_WEIGHT
    return _RAW_SUFFIX_WEIGHT.get(Path(path).suffix, 1.0)


def _looks_garbled(text: str) -> bool:
    """UTF-8 read as Latin-1 (mojibake) is unreadable as a hint."""
    latin = sum(1 for char in text if "\x80" <= char <= "\xff")
    return latin * 5 > max(len(text), 1)


def _best_line(
    body: str, start_line: int, matched: list[tuple[tuple[tuple[str, ...], ...], float]]
) -> tuple[int, str]:
    best = (0.0, 0, "")
    for offset, line in enumerate(body.splitlines()):
        present = set(text_terms(line))
        score = sum(
            weight
            for members, weight in matched
            if any(all(term in present for term in member) for member in members)
        )
        if score > best[0]:
            best = (score, offset, line)
    _score, offset, _line = best
    # The best line plus what follows it, up to the excerpt budget.
    parts: list[str] = []
    size = 0
    for line in body.splitlines()[offset:]:
        text = " ".join(line.split())
        if not text:
            continue
        parts.append(text)
        size += len(text) + 3
        if size >= EXCERPT_CHARS:
            break
    excerpt = " / ".join(parts)
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[: EXCERPT_CHARS - 1] + "…"
    return start_line + offset, excerpt


MEMORY_BLOCK_TAG = "server_side_qwen_exo_memory_block"
_BLOCK_OPEN = f"<{MEMORY_BLOCK_TAG}>"
_BLOCK_CLOSE = f"</{MEMORY_BLOCK_TAG}>"
_PREAMBLE = (
    "以下内容由推理服务端（QWEN-EXO）附加，不是用户写的，也不是你说的话。",
    "记忆：来自用户知识库和以往会话的回忆，与当前会话无关。记忆里出现的文件、"
    "路径、key、命令、账号等都属于记忆库或过去的环境，不要试图在当前 agent 的"
    "工作环境里查找、打开或使用它们，只能作为行动参考。记忆可能混乱、过时或互相"
    "冲突，只做参考，以你现在观察到的为准。",
    "记忆工具 qwen_exo_memory_search、qwen_exo_memory_read、qwen_exo_memory_write 由服务端执行"
    "（不在客户端，不碰用户的机器），调用后结果在本轮直接返回给你，不需要用户批准。",
)
_PATH_RULE = (
    "用记忆少走重复的路：\n"
    "- 动手前先查：准备走一条没把握的路（换方案、改配置、部署、排查某个报错、用不熟的"
    "工具或命令）之前，先用 1–2 个关键词搜一下这条路有没有记忆。以前走通过就照着走；"
    "以前失败过，先看清失败原因，别再踩同一个坑。\n"
    "- 失败了先查：命令报错、测试不过、方案行不通时，先用报错里最有辨识度的关键字"
    "（错误信息原文、组件名、命令名）搜记忆，看以前有没有遇到过、当时怎么解决的，再"
    "决定下一步，不要盲目重试。\n"
    "- 走完了就记：值得记的路，成功和失败都用 qwen_exo_memory_write 记下来。成功的路标"
    "【成功】，写清怎么走通的；失败的路标【失败】，写清为什么失败、后来改用了什么办法。"
    "一眼就能看出来的小错不用记。\n"
    "- 记得能被搜到：标题写清楚场景；正文按“场景/目标、做法（关键命令、配置、文件）、"
    "结果（成功/失败，附报错原文）、原因、下次怎么做/什么情况不适用”来写；最后加一行"
    "“关键词：”，写上中英文同义词、报错关键字、组件名，以后换个说法也能搜到。\n"
    "- 记忆也会写错，不要完全相信：曾经出现过路本来是对的，记忆却写成了错的，结果一直"
    "绕开正确做法、迟迟完成不了任务目标。记忆说“这条路不通”或“必须这样做”，而它又"
    "决定你下一步怎么走时，先多看几条相关记忆互相印证，再用当前环境里的实际结果（运行、"
    "测试、查看文件）验证；和事实冲突就以事实为准，并立刻用 qwen_exo_memory_write 的 "
    "edit / replace / delete 改正那条记忆，别让它继续误导以后的任务。"
)
_STOP_RULE = (
    "同一个方向最多搜两次（第一次 + 换词或换语言再试一次）。两次都搜不到，就认定"
    "记忆里没有这方面的内容，别再换着词反复搜，专心用自己的工具和推理把任务做完；"
    "也不要把“记忆里没有”当成回答。之后遇到新的问题或新的方向再搜。这件事完成或有"
    "重大发现后，用 qwen_exo_memory_write 写下操作经验永久保存，下次就能搜到；发现"
    "工作记忆（work-memory/）写错了或过时了，用它的 edit / replace / delete 直接改正。"
)


def is_memory_block(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(_BLOCK_OPEN)


def render_memory_block(hints: list[dict[str, Any]]) -> str:
    lead, memory, tools = _PREAMBLE
    lines = [
        _BLOCK_OPEN,
        lead,
        ("这是你的" if hints else "你有") + memory,
        tools,
        "",
    ]
    if hints:
        lines.append("根据当前内容自动匹配到的记忆片段（仅供参考，未必相关）：")
        for hint in hints:
            heading = f" [{hint['heading']}]" if hint["heading"] else ""
            lines.append(
                f"- 记忆文件 {hint['path']} 第 {hint['line']} 行{heading}：{hint['excerpt']}"
            )
        lines.append(
            "需要完整上下文时，调用 qwen_exo_memory_read 读取（path 填上面的记忆"
            "文件名）；也可以调用 qwen_exo_memory_search 换关键词搜索更多记忆。"
        )
    else:
        lines.append(
            "当前内容没有自动匹配到新的记忆片段。如果任务可能用到用户的项目约定、"
            "环境配置、以前遇到过的问题或经验，调用 qwen_exo_memory_search 搜索记忆。"
        )
    lines.append(_PATH_RULE)
    lines.append(_STOP_RULE)
    lines.append(_BLOCK_CLOSE)
    return "\n".join(lines)
