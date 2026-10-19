"""Grep engine behind ``qwen_exo_memory_search``.

Pure functions over decoded file text:

* query syntax: ``*`` wildcard and ``|`` alternatives (as before), ``-word``
  exclusions, ``"quoted phrases"``, or a regular expression;
* very long lines (minified JSON) yield a window around each match plus its
  byte offset, so ``qwen_exo_memory_read`` can continue there by bytes;
* boilerplate: text segments repeated across many files (fingerprints, system
  prompts, shared conventions) are hidden from hits unless asked for;
* each file gets a kind (summary, note, trajectory, ...) and a title, and
  files are ranked by kind before match count.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import PurePosixPath
from typing import Any, Iterable

import msgspec
# ``regex`` (a transformers/tiktoken dependency) supports per-call timeouts, so
# a catastrophic user regex cannot pin a server thread.
import regex

# Lines longer than this are shown as windows around each match.
LONG_LINE_CHARS = 2000
WINDOW_CHARS = 300
CONTEXT_LINE_CHARS = 300
# A segment at least this long that occurs in this many files is boilerplate.
SEGMENT_MIN_CHARS = 40
# Longer segments (a JSON blob without escaped newlines) are never boilerplate.
SEGMENT_MAX_CHARS = 4000
BOILERPLATE_MIN_FILES = 5
MAX_PATTERN_CHARS = 1024
SUMMARY_SUFFIX = ".summary.md"
# A ``*`` wildcard spans at most this many characters: unbounded ``.*?`` on a
# multi-megabyte JSON line is quadratic when the tail never matches.
WILDCARD_SPAN = 200
# After a file has shown its hits, matches are counted up to this cap and the
# rest of the file is skipped (a single common letter matches millions of times).
MATCH_COUNT_CAP = 2000
# Whole-search wall clock; past it the search returns what it found so far.
SEARCH_SECONDS = 10.0

KIND_WEIGHT = {
    "summary": 6,
    "note": 5,
    "work": 5,
    "code": 4,
    "other": 3,
    "data": 2,
    "log": 2,
    "session": 1,
    "trajectory": 1,
}
_NOTE_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org", ".adoc"}
_CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".sh", ".ps1", ".sql", ".lua",
    ".kt", ".swift", ".toml", ".yaml", ".yml", ".ini", ".cfg",
}
_DATA_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".xml"}
_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_JSON_TITLE = re.compile(r'"(?:title|task|name|subject|challenge)"\s*:\s*"([^"\\]{3,160})"')
_EXCLUSION = re.compile(r'(?:^|\s)-("(?:[^"]+)"|\S+)')


class Query(msgspec.Struct, frozen=True):
    positive: regex.Pattern[str]
    exclude: tuple[regex.Pattern[str], ...] = ()


def _literal_expression(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    pieces = [regex.escape(piece) for piece in text.split("*")]
    return f".{{0,{WILDCARD_SPAN}}}?".join(pieces) if text.strip("*") else ""


def _alternatives(text: str) -> str:
    parts = [_literal_expression(part) for part in text.split("|")]
    return "|".join(part for part in parts if part)


def _compile(expression: str) -> regex.Pattern[str]:
    return regex.compile(expression, regex.IGNORECASE | regex.VERSION0)


def parse_query(pattern: str, *, is_regex: bool = False, exclude: str = "") -> Query:
    """Compile the search syntax; raises ``ValueError`` with a usable message."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    if len(pattern) > MAX_PATTERN_CHARS or len(exclude or "") > MAX_PATTERN_CHARS:
        raise ValueError(f"pattern is longer than {MAX_PATTERN_CHARS} characters")
    exclusions = [part for part in (exclude or "").split("|") if part.strip()]
    if is_regex:
        positive_source = pattern
    else:
        exclusions += [match.group(1) for match in _EXCLUSION.finditer(pattern)]
        positive_source = _EXCLUSION.sub(" ", pattern).strip()
    try:
        positive = _compile(positive_source if is_regex else _alternatives(positive_source))
    except regex.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
    if not positive.pattern:
        raise ValueError("pattern must contain text to find besides -exclusions")
    excluded = tuple(
        _compile(expression)
        for expression in (_alternatives(part) for part in exclusions)
        if expression
    )
    return Query(positive=positive, exclude=excluded)


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def segment_hash(text: str) -> int:
    """Signed 64-bit hash of a whitespace/case-normalized segment (SQLite INTEGER)."""
    digest = hashlib.blake2b(_normalized(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _line_segments(line: str) -> Iterable[str]:
    if len(line) <= LONG_LINE_CHARS:
        yield line
        return
    # JSON-escaped text keeps its paragraphs as literal "\n" sequences.
    yield from line.split("\\n")


def file_segment_hashes(text: str) -> set[int]:
    """Distinct hashes of the file's segments long enough to be boilerplate."""
    hashes: set[int] = set()
    for line in text.splitlines():
        for segment in _line_segments(line):
            if SEGMENT_MIN_CHARS <= len(segment.strip()) <= SEGMENT_MAX_CHARS:
                hashes.add(segment_hash(segment))
    return hashes


def detect_kind(path: str, head: str) -> str:
    """Coarse file kind from the path and the first ~64 KiB of text."""
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(name).suffix
    if name.endswith(SUMMARY_SUFFIX):
        return "summary"
    if path.startswith("work-memory/"):
        return "work"
    if "trajectory" in name:
        return "trajectory"
    if suffix in _NOTE_SUFFIXES:
        return "note"
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix == ".log":
        return "log"
    if suffix in _DATA_SUFFIXES:
        conversational = '"role"' in head and '"content"' in head
        if "session" in name or conversational:
            return "session"
        return "data"
    return "other"


def detect_title(path: str, kind: str, head: str) -> str:
    stem = PurePosixPath(path).name
    if kind in {"session", "trajectory", "data"}:
        match = _JSON_TITLE.search(head)
        return (match.group(1) if match else stem)[:120]
    title = re.search(r"\A---\n(?:.*\n)*?title:\s*(.+)\n", head)
    if title:
        return title.group(1).strip()[:120]
    heading = _HEADING.search(head)
    if heading:
        return heading.group(1)[:120]
    for line in head.splitlines():
        if line.strip():
            return line.strip()[:120]
    return stem


class FileScan(msgspec.Struct):
    match_count: int = 0
    hidden_boilerplate: int = 0
    candidates: list[dict[str, Any]] = []
    # match_count stopped at MATCH_COUNT_CAP; the rest of the file was skipped.
    capped: bool = False
    # The search deadline passed (or the regex timed out) inside this file.
    stopped: bool = False


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit} chars]"


def _segment_around(line: str, start: int, end: int) -> str:
    """The escaped-newline paragraph holding a match in a long line, or just
    the window around it when that paragraph is itself huge."""
    left = line.rfind("\\n", max(0, start - SEGMENT_MAX_CHARS), start)
    right = line.find("\\n", end, end + SEGMENT_MAX_CHARS)
    if (left < 0 and start > SEGMENT_MAX_CHARS) or (
        right < 0 and len(line) - end > SEGMENT_MAX_CHARS
    ):
        return line[max(0, start - WINDOW_CHARS) : end + WINDOW_CHARS]
    return line[left + 2 if left >= 0 else 0 : right if right >= 0 else len(line)]


def scan_lines(
    path: str,
    raw_lines: list[bytes],
    query: Query,
    *,
    boilerplate: frozenset[int],
    include_boilerplate: bool,
    context: int,
    per_file: int,
    deadline: float,
) -> FileScan:
    """Matches in one file. ``raw_lines`` come from ``bytes.splitlines(keepends=True)``
    so line numbers and byte offsets agree with ``read_lines``/``read_bytes``.
    Stops at ``deadline`` (``time.monotonic``) or after MATCH_COUNT_CAP matches."""
    lines = [raw.decode("utf-8", errors="replace").rstrip("\r\n") for raw in raw_lines]
    scan = FileScan(candidates=[])
    # Text repeated inside one file (a system prompt per turn) is one hit.
    repeated: dict[int, dict[str, Any]] = {}
    line_start = 0
    try:
        for index, line in enumerate(lines):
            start_of_line = line_start
            line_start += len(raw_lines[index])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                scan.stopped = True
                break
            first = query.positive.search(line, timeout=remaining)
            if first is None:
                continue
            long_line = len(line) > LONG_LINE_CHARS
            matches = query.positive.finditer(line, timeout=remaining) if long_line else (first,)
            for examined, match in enumerate(matches, 1):
                if examined % 256 == 0 and time.monotonic() > deadline:
                    scan.stopped = True
                    break
                _record_match(
                    scan,
                    repeated,
                    path=path,
                    lines=lines,
                    index=index,
                    start_of_line=start_of_line,
                    match=match,
                    long_line=long_line,
                    query=query,
                    boilerplate=boilerplate,
                    include_boilerplate=include_boilerplate,
                    context=context,
                    per_file=per_file,
                )
                if scan.match_count + scan.hidden_boilerplate >= MATCH_COUNT_CAP:
                    scan.capped = True
                    break
            if scan.stopped or scan.capped:
                break
    except TimeoutError:
        scan.stopped = True
    return scan


def _record_match(
    scan: FileScan,
    repeated: dict[int, dict[str, Any]],
    *,
    path: str,
    lines: list[str],
    index: int,
    start_of_line: int,
    match: Any,
    long_line: bool,
    query: Query,
    boilerplate: frozenset[int],
    include_boilerplate: bool,
    context: int,
    per_file: int,
) -> None:
    line = lines[index]
    segment = _segment_around(line, match.start(), match.end()) if long_line else line
    if any(pattern.search(segment) for pattern in query.exclude):
        return
    fold_key = (
        segment_hash(segment)
        if SEGMENT_MIN_CHARS <= len(segment.strip()) <= SEGMENT_MAX_CHARS
        else None
    )
    if fold_key is not None and fold_key in boilerplate and not include_boilerplate:
        scan.hidden_boilerplate += 1
        return
    scan.match_count += 1
    if fold_key is not None and fold_key in repeated:
        repeated[fold_key]["repeats"] += 1
        return
    if len(scan.candidates) >= per_file:
        return
    if long_line:
        left = max(0, match.start() - WINDOW_CHARS)
        right = min(len(line), match.end() + WINDOW_CHARS)
        window = ("…" if left else "") + line[left:right] + ("…" if right < len(line) else "")
        scan.candidates.append(
            {
                "path": path,
                "line": index + 1,
                "byte_offset": start_of_line + len(line[: match.start()].encode("utf-8")),
                "text": window,
                "context": [],
                "fold_key": fold_key,
                "repeats": 1,
            }
        )
    else:
        first = max(0, index - context)
        scan.candidates.append(
            {
                "path": path,
                "line": index + 1,
                "byte_offset": None,
                "text": _clip(line, LONG_LINE_CHARS),
                "context": [
                    {"line": number + 1, "text": _clip(lines[number], CONTEXT_LINE_CHARS)}
                    for number in range(first, min(len(lines), index + context + 1))
                    if number != index
                ],
                "fold_key": fold_key,
                "repeats": 1,
            }
        )
    if fold_key is not None:
        repeated[fold_key] = scan.candidates[-1]


def rank_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    """Summaries and notes before raw trajectories, then by match count."""
    return (-KIND_WEIGHT.get(entry["kind"], 3), -entry["match_count"], entry["path"])


def fold_duplicates(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse hits with identical text across files into the first one,
    which lists the other paths in ``also_in``."""
    first_by_key: dict[int, dict[str, Any]] = {}
    folded: list[dict[str, Any]] = []
    for hit in hits:
        key = hit.get("fold_key")
        if key is not None and key in first_by_key:
            owner = first_by_key[key]
            if hit["path"] != owner["path"] and hit["path"] not in owner["also_in"]:
                owner["also_in"].append(hit["path"])
            continue
        hit = {**hit, "also_in": []}
        if key is not None:
            first_by_key[key] = hit
        folded.append(hit)
    return folded


__all__ = [
    "BOILERPLATE_MIN_FILES",
    "KIND_WEIGHT",
    "LONG_LINE_CHARS",
    "MATCH_COUNT_CAP",
    "Query",
    "SEARCH_SECONDS",
    "SUMMARY_SUFFIX",
    "detect_kind",
    "detect_title",
    "file_segment_hashes",
    "fold_duplicates",
    "parse_query",
    "rank_key",
    "scan_lines",
    "segment_hash",
]
