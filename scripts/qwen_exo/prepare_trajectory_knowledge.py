from __future__ import annotations

import argparse
import json
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from qwen_exo_booster.knowledge import normalize_markdown


@dataclass(frozen=True)
class Chunk:
    source_line: int
    source_record: int
    chunk_index: int
    messages: tuple[dict[str, Any], ...]
    token_count: int
    prefix_included: bool


@dataclass
class BuildReport:
    input: str
    output_dir: str
    tokenizer: str
    max_tokens: int
    source_rows: int = 0
    parse_errors: int = 0
    empty_rows: int = 0
    chunks_written: int = 0
    chunks_skipped_oversize: int = 0
    rows_with_oversize_system: int = 0
    oversize_system_tokens: int = 0
    trailing_messages_dropped: int = 0
    chunk_token_min: int | None = None
    chunk_token_p50: int | None = None
    chunk_token_max: int | None = None
    row_chunk_counts: dict[str, int] | None = None



def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "")
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    parts.append(f"[{item_type}] {_text(text)}".strip())
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)



def normalize_message(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or "").strip().lower()
    if role == "developer":
        role = "system"
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    content = raw.get("content")
    if content is None:
        content = raw.get("text")
    if content is None:
        content = raw.get("output")
    result: dict[str, Any] = {"role": role, "content": _text(content)}
    tool_calls = raw.get("tool_calls")
    if tool_calls:
        result["tool_calls"] = tool_calls
    if raw.get("tool_call_id") is not None:
        result["tool_call_id"] = str(raw["tool_call_id"])
    return result



def render_message(message: dict[str, Any]) -> str:
    role = str(message["role"]).upper()
    content = str(message.get("content") or "")
    extras: list[str] = []
    if message.get("tool_call_id"):
        extras.append(f"tool_call_id: {message['tool_call_id']}")
    if message.get("tool_calls"):
        extras.append(
            "tool_calls: "
            + json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
        )
    if extras:
        content = (content + "\n" if content else "") + "\n".join(extras)
    return f"### {role}\n\n{content}\n"



def render_messages(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(render_message(message) for message in messages).strip() + "\n"



def token_count(tokenizer: Any, messages: Iterable[dict[str, Any]]) -> int:
    content = normalize_markdown(render_messages(messages))
    return len(tokenizer.encode(content, add_special_tokens=False))



def split_row(
    tokenizer: Any,
    messages: tuple[dict[str, Any], ...],
    *,
    max_tokens: int,
) -> tuple[list[tuple[tuple[dict[str, Any], ...], bool, int]], int, int, int]:
    """Split only at assistant-completed message boundaries.

    Returns chunks, dropped trailing message count, oversize-system token count,
    and oversize body-unit count. A system prefix is included only in the first
    chunk when it fits; later chunks never duplicate it.
    """
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end]["role"] == "system":
        prefix_end += 1
    prefix = messages[:prefix_end]
    prefix_tokens = token_count(tokenizer, prefix) if prefix else 0
    prefix_fits = not prefix or prefix_tokens <= max_tokens
    oversize_system_tokens = prefix_tokens if prefix and not prefix_fits else 0

    units: list[tuple[dict[str, Any], ...]] = []
    current: list[dict[str, Any]] = []
    for message in messages[prefix_end:]:
        current.append(message)
        if message["role"] == "assistant":
            units.append(tuple(current))
            current = []
    trailing_messages = len(current)

    chunks: list[tuple[tuple[dict[str, Any], ...], bool, int]] = []
    pending: list[dict[str, Any]] = []
    first_chunk = True
    oversize_body_units = 0

    for unit in units:
        candidate_prefix = prefix if first_chunk and prefix_fits else ()
        candidate = tuple(candidate_prefix) + tuple(pending) + unit
        candidate_tokens = token_count(tokenizer, candidate)
        if candidate_tokens <= max_tokens:
            pending.extend(unit)
            continue

        if pending:
            final_messages = tuple(candidate_prefix) + tuple(pending)
            final_tokens = token_count(tokenizer, final_messages)
            if final_tokens <= max_tokens:
                chunks.append((final_messages, bool(candidate_prefix), final_tokens))
                first_chunk = False
                pending = list(unit)
            else:
                # This should only occur when a prefix itself is too large.
                pending = list(unit)
        else:
            unit_without_prefix = tuple(unit)
            unit_tokens = token_count(tokenizer, unit_without_prefix)
            if unit_tokens > max_tokens:
                oversize_body_units += 1
                continue
            chunks.append((unit_without_prefix, False, unit_tokens))
            first_chunk = False
            pending = []

    if pending:
        final_prefix = prefix if first_chunk and prefix_fits else ()
        final_messages = tuple(final_prefix) + tuple(pending)
        final_tokens = token_count(tokenizer, final_messages)
        if final_tokens <= max_tokens:
            chunks.append((final_messages, bool(final_prefix), final_tokens))
        else:
            body_tokens = token_count(tokenizer, tuple(pending))
            if body_tokens <= max_tokens:
                chunks.append((tuple(pending), False, body_tokens))
            else:
                oversize_body_units += 1

    return chunks, trailing_messages, oversize_system_tokens, oversize_body_units



def write_chunk(output_dir: Path, chunk: Chunk) -> None:
    path = output_dir / f"trajectory-{chunk.source_record:06d}-{chunk.chunk_index:03d}.md"
    front_matter = "\n".join(
        (
            "---",
            "canonical: false",
            "quality: 0.25",
            "source_kind: imported_other_model_trajectory",
            "document_group: train-side-windows-4k-test",
            f"source_record: {chunk.source_record}",
            f"source_line: {chunk.source_line}",
            f"chunk_index: {chunk.chunk_index}",
            f"chunk_tokens: {chunk.token_count}",
            f"prefix_included: {str(chunk.prefix_included).lower()}",
            "---",
            "",
        )
    )
    path.write_text(front_matter + render_messages(chunk.messages), encoding="utf-8")



def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]



def build(args: argparse.Namespace) -> BuildReport:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(
                f"Refusing to reuse output directory without --replace: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    report = BuildReport(
        input=str(args.input),
        output_dir=str(output_dir),
        tokenizer=str(args.tokenizer),
        max_tokens=args.max_tokens,
        row_chunk_counts={},
    )
    token_counts: list[int] = []

    opener = gzip.open if args.input.suffix.lower() == ".gz" else Path.open
    with opener(args.input, "rt", encoding="utf-8", errors="replace") as source:
        for line_no, line in enumerate(source, 1):
            if args.limit and report.source_rows >= args.limit:
                break
            report.source_rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                report.parse_errors += 1
                continue
            raw_messages = raw.get("messages") if isinstance(raw, dict) else None
            messages = tuple(
                message
                for item in (raw_messages or [])
                if (message := normalize_message(item)) is not None
            )
            if not messages:
                report.empty_rows += 1
                continue
            pieces, trailing, oversize_system, oversize_units = split_row(
                tokenizer, messages, max_tokens=args.max_tokens
            )
            report.trailing_messages_dropped += trailing
            if oversize_system:
                report.rows_with_oversize_system += 1
                report.oversize_system_tokens += oversize_system
            report.chunks_skipped_oversize += oversize_units
            for chunk_index, (piece, prefix_included, count) in enumerate(pieces):
                chunk = Chunk(
                    source_line=line_no,
                    source_record=report.source_rows,
                    chunk_index=chunk_index,
                    messages=piece,
                    token_count=count,
                    prefix_included=prefix_included,
                )
                write_chunk(output_dir, chunk)
                token_counts.append(count)
                report.chunks_written += 1
            report.row_chunk_counts[str(report.source_rows)] = len(pieces)

    report.chunk_token_min = min(token_counts) if token_counts else None
    report.chunk_token_p50 = percentile(token_counts, 0.50)
    report.chunk_token_max = max(token_counts) if token_counts else None
    (output_dir / "_build_report.json").write_text(
        json.dumps(report.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report



def verify_output(output_dir: Path, tokenizer: Any, max_tokens: int) -> dict[str, Any]:
    counts: list[tuple[str, int]] = []
    for path in sorted(output_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        count = len(tokenizer.encode(normalize_markdown(content), add_special_tokens=False))
        counts.append((path.name, count))
    offenders = [
        {"file": name, "tokens": count}
        for name, count in counts
        if count > max_tokens
    ]
    return {
        "documents": len(counts),
        "max_tokens": max((count for _name, count in counts), default=0),
        "min_tokens": min((count for _name, count in counts), default=0),
        "over_limit": offenders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping, message-boundary 4K QWEN-EXO Knowledge docs."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 64:
        raise SystemExit("--max-tokens must be at least 64")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    if args.verify_only:
        if args.input is not None:
            raise SystemExit("--verify-only does not accept --input")
        print(json.dumps(verify_output(args.output_dir, tokenizer, args.max_tokens), indent=2))
        return
    if args.input is None:
        raise SystemExit("--input is required unless --verify-only is used")
    report = build(args)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
