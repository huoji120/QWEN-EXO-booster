from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_exo_booster.knowledge import markdown_metadata, normalize_markdown

_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_FILES = 20
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENT_PARTS_PER_FILE = 64
_MAX_DOCUMENT_PARTS_PER_BATCH = 128
SUPPORTED_KNOWLEDGE_SUFFIXES = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".csv",
    }
)
_SUPPORTED_SUFFIXES = SUPPORTED_KNOWLEDGE_SUFFIXES
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_STRUCTURED_LANGUAGES = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
}
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def is_supported_knowledge_filename(filename: str) -> bool:
    return Path(str(filename)).suffix.lower() in _SUPPORTED_SUFFIXES


def validate_knowledge_source_bytes(filename: str, raw: bytes) -> None:
    safe_name = Path(str(filename)).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise KnowledgeIngestError("invalid_filename", "文件名无效", filename=filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise KnowledgeIngestError(
            "unsupported_file_type",
            "仅支持 Markdown、TXT、RST、JSON、JSONL、YAML 和 CSV 文本文件",
            filename=safe_name,
            details={"suffix": suffix or None},
        )
    if not raw:
        raise KnowledgeIngestError("empty_file", "文件内容为空", filename=safe_name)
    if len(raw) > _MAX_FILE_BYTES:
        raise KnowledgeIngestError(
            "file_too_large",
            f"单个文件不得超过 {_MAX_FILE_BYTES // (1024 * 1024)} MiB",
            filename=safe_name,
            details={"byte_count": len(raw), "maximum_bytes": _MAX_FILE_BYTES},
        )


class KnowledgeIngestError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        filename: str | None = None,
        details: dict[str, object] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.filename = filename
        self.details = details or {}

    def public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
        if self.filename is not None:
            payload["filename"] = self.filename
        return payload


@dataclass(frozen=True, slots=True)
class PreparedKnowledgeDocument:
    original_filename: str
    relative_path: str
    document_group: str
    retrieval_category: str
    content: str
    token_count: int
    byte_count: int
    sha256: str
    changes: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "original_filename": self.original_filename,
            "relative_path": self.relative_path,
            "document_group": self.document_group,
            "retrieval_category": self.retrieval_category,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "changes": list(self.changes),
        }


@dataclass(frozen=True, slots=True)
class KnowledgeUploadPreview:
    original_filename: str
    suggested_path: str
    content: str
    tags: tuple[str, ...]
    source_kind: str
    retrieval_category: str
    byte_count: int
    changes: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "original_filename": self.original_filename,
            "suggested_path": self.suggested_path,
            "content": self.content,
            "tags": list(self.tags),
            "source_kind": self.source_kind,
            "retrieval_category": self.retrieval_category,
            "byte_count": self.byte_count,
            "changes": list(self.changes),
        }


def preview_knowledge_upload(
    filename: str, content_base64: str
) -> KnowledgeUploadPreview:
    safe_name = Path(str(filename)).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise KnowledgeIngestError("invalid_filename", "文件名无效", filename=filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise KnowledgeIngestError(
            "unsupported_file_type",
            "仅支持 Markdown、TXT、RST、JSON、JSONL、YAML 和 CSV 文本文件",
            filename=safe_name,
            details={"suffix": suffix or None},
        )
    try:
        raw = base64.b64decode(str(content_base64), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise KnowledgeIngestError(
            "invalid_base64", "文件内容不是有效的 Base64", filename=safe_name
        ) from exc
    if not raw:
        raise KnowledgeIngestError("empty_file", "文件内容为空", filename=safe_name)
    if len(raw) > _MAX_FILE_BYTES:
        raise KnowledgeIngestError(
            "file_too_large",
            f"单个文件不得超过 {_MAX_FILE_BYTES // (1024 * 1024)} MiB",
            filename=safe_name,
            details={"byte_count": len(raw), "maximum_bytes": _MAX_FILE_BYTES},
        )
    text, decoding_change = _decode_text(raw, safe_name)
    source_tags = tuple(str(tag) for tag in markdown_metadata(text).get("tags", ()))
    body, changes, source_kind = _clean_body(text, suffix, safe_name)
    retrieval_category = source_kind
    if decoding_change is not None:
        changes.insert(0, decoding_change)
    slug = _slug(Path(safe_name).stem, raw)
    metadata = (
        "---\n"
        "canonical: false\n"
        "quality: 0.65\n"
        f"source_kind: {source_kind}\n"
        f"retrieval_category: {retrieval_category}\n"
        "---\n\n"
    )
    content = metadata + body.strip() + "\n"
    return KnowledgeUploadPreview(
        original_filename=safe_name,
        suggested_path=f"uploads/{slug}.md",
        content=content,
        tags=source_tags,
        source_kind=source_kind,
        retrieval_category=retrieval_category,
        byte_count=len(content.encode("utf-8")),
        changes=tuple(changes),
    )


def validate_upload_batch(files: list[dict[str, object]]) -> None:
    if not files:
        raise KnowledgeIngestError("empty_upload", "至少选择一个文件")
    if len(files) > _MAX_FILES:
        raise KnowledgeIngestError(
            "too_many_files",
            f"一次最多上传 {_MAX_FILES} 个文件",
            details={"file_count": len(files), "maximum": _MAX_FILES},
        )
    encoded_bytes = sum(len(str(item.get("content_base64") or "")) for item in files)
    if encoded_bytes > (_MAX_TOTAL_BYTES * 4 // 3) + 4096:
        raise KnowledgeIngestError(
            "batch_too_large",
            f"一次上传的原始文件总量不得超过 {_MAX_TOTAL_BYTES // (1024 * 1024)} MiB",
            details={"maximum_bytes": _MAX_TOTAL_BYTES},
        )
    names = [Path(str(item.get("filename") or "")).name.casefold() for item in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise KnowledgeIngestError(
            "duplicate_filename",
            f"同一批次存在重复文件名：{', '.join(duplicates)}",
            filename=duplicates[0],
        )


def validate_prepared_batch(
    documents: tuple[PreparedKnowledgeDocument, ...],
) -> None:
    if len(documents) > _MAX_DOCUMENT_PARTS_PER_BATCH:
        raise KnowledgeIngestError(
            "too_many_document_parts",
            f"本批次生成 {len(documents)} 个原生片段，超过上限 {_MAX_DOCUMENT_PARTS_PER_BATCH}",
            details={
                "document_count": len(documents),
                "maximum": _MAX_DOCUMENT_PARTS_PER_BATCH,
            },
        )


def prepare_knowledge_bytes(
    filename: str,
    raw: bytes,
    *,
    tokenizer: Any,
    max_source_tokens: int,
    relative_path_prefix: str = "uploads",
    document_group_prefix: str = "upload",
    retrieval_category: str | None = None,
) -> tuple[PreparedKnowledgeDocument, ...]:
    safe_name = Path(str(filename)).name.strip()
    validate_knowledge_source_bytes(safe_name, raw)
    suffix = Path(safe_name).suffix.lower()
    text, decoding_change = _decode_text(raw, safe_name)
    body, changes, source_kind = _clean_body(text, suffix, safe_name)
    if decoding_change is not None:
        changes.insert(0, decoding_change)
    title = _document_title(body, Path(safe_name).stem)
    slug = _slug(Path(safe_name).stem, raw)
    return _split_documents(
        body,
        title=title,
        slug=slug,
        group=f"{document_group_prefix}_{slug}",
        source_kind=source_kind,
        retrieval_category=(retrieval_category or source_kind),
        original_filename=safe_name,
        tokenizer=tokenizer,
        max_source_tokens=max_source_tokens,
        changes=tuple(changes),
        relative_path_prefix=relative_path_prefix,
    )


def prepare_knowledge_upload(
    filename: str,
    content_base64: str,
    *,
    tokenizer: Any,
    max_source_tokens: int,
    retrieval_category: str | None = None,
) -> tuple[PreparedKnowledgeDocument, ...]:
    safe_name = Path(str(filename)).name.strip()
    try:
        raw = base64.b64decode(str(content_base64), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise KnowledgeIngestError(
            "invalid_base64", "文件内容不是有效的 Base64", filename=safe_name
        ) from exc
    return prepare_knowledge_bytes(
        safe_name,
        raw,
        tokenizer=tokenizer,
        max_source_tokens=max_source_tokens,
        retrieval_category=retrieval_category,
    )


def _decode_text(raw: bytes, filename: str) -> tuple[str, str | None]:
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16"), "utf16_decoded"
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig"), "utf8_bom_removed"
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        raise KnowledgeIngestError(
            "invalid_text_encoding",
            "文件必须使用 UTF-8，或带 BOM 的 UTF-16",
            filename=filename,
            details={"offset": exc.start},
        ) from exc


def _clean_body(text: str, suffix: str, filename: str) -> tuple[str, list[str], str]:
    changes: list[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized != text:
        changes.append("line_endings_normalized")
    normalized = unicodedata.normalize("NFC", normalized)
    filtered = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    if filtered != normalized:
        changes.append("control_characters_removed")

    if suffix == ".json":
        try:
            parsed = json.loads(filtered)
        except json.JSONDecodeError as exc:
            raise KnowledgeIngestError(
                "invalid_json",
                f"JSON 解析失败：第 {exc.lineno} 行，第 {exc.colno} 列",
                filename=filename,
            ) from exc
        filtered = json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
        changes.append("json_formatted")
    elif suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(filtered.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise KnowledgeIngestError(
                    "invalid_jsonl",
                    f"JSONL 第 {line_number} 行解析失败：第 {exc.colno} 列",
                    filename=filename,
                ) from exc
        if not records:
            raise KnowledgeIngestError(
                "empty_file", "JSONL 不包含有效记录", filename=filename
            )
        filtered = "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
        )
        changes.append("jsonl_formatted")

    title = _display_title(Path(filename).stem)
    if suffix in _MARKDOWN_SUFFIXES:
        body = normalize_markdown(filtered)
        source_kind = "uploaded_markdown"
        changes.append("markdown_normalized")
    elif suffix in _STRUCTURED_LANGUAGES:
        escaped = filtered.replace("```", "\\`\\`\\`")
        body = (
            f"# {title}\n\n```{_STRUCTURED_LANGUAGES[suffix]}\n{escaped.strip()}\n```"
        )
        source_kind = "uploaded_structured_text"
        changes.append("wrapped_as_markdown")
    else:
        body = normalize_markdown(filtered)
        if not _HEADING.search(body):
            body = f"# {title}\n\n{body}"
            changes.append("title_added")
        source_kind = "uploaded_text"
    if not body.strip():
        raise KnowledgeIngestError(
            "empty_file", "清洗后没有有效文本", filename=filename
        )
    changes.append("metadata_replaced")
    return body, changes, source_kind


def _document_title(body: str, stem: str) -> str:
    match = _HEADING.search(body)
    return match.group(1).strip() if match else _display_title(stem)


def _display_title(stem: str) -> str:
    title = re.sub(r"[-_]+", " ", stem).strip()
    return title or "Uploaded Knowledge"


def _slug(stem: str, raw: bytes) -> str:
    normalized = unicodedata.normalize("NFKC", stem).casefold()
    slug = "".join(
        character if character.isalnum() else "-" for character in normalized
    )
    slug = re.sub(r"-+", "-", slug).strip("-")[:64]
    return slug or f"document-{hashlib.sha256(raw).hexdigest()[:12]}"


def _split_documents(
    body: str,
    *,
    title: str,
    slug: str,
    group: str,
    source_kind: str,
    retrieval_category: str,
    original_filename: str,
    tokenizer: Any,
    max_source_tokens: int,
    changes: tuple[str, ...],
    relative_path_prefix: str,
) -> tuple[PreparedKnowledgeDocument, ...]:
    if max_source_tokens < 128:
        raise KnowledgeIngestError(
            "token_budget_too_small",
            "Tensor Bank 可用的文档 token 预算不足",
            filename=original_filename,
            details={"max_source_tokens": max_source_tokens},
        )
    body_tokens = _encode(tokenizer, normalize_markdown(body))
    if not body_tokens:
        raise KnowledgeIngestError(
            "empty_file", "清洗后未编码出任何 token", filename=original_filename
        )
    if len(body_tokens) <= max_source_tokens:
        return (
            _prepared_document(
                relative_path=f"{relative_path_prefix}/{slug}.md",
                body=body,
                group=group,
                source_kind=source_kind,
                retrieval_category=retrieval_category,
                original_filename=original_filename,
                tokenizer=tokenizer,
                max_source_tokens=max_source_tokens,
                changes=changes,
            ),
        )

    payload_budget = max(64, max_source_tokens - 96)
    minimum_part_count = math.ceil(len(body_tokens) / payload_budget)
    if minimum_part_count > _MAX_DOCUMENT_PARTS_PER_FILE:
        raise KnowledgeIngestError(
            "too_many_document_parts",
            f"文件至少需要 {minimum_part_count} 个原生片段，超过单文件上限 {_MAX_DOCUMENT_PARTS_PER_FILE}",
            filename=original_filename,
            details={
                "minimum_part_count": minimum_part_count,
                "maximum": _MAX_DOCUMENT_PARTS_PER_FILE,
            },
        )
    parts: list[str] = []
    offset = 0
    while offset < len(body_tokens):
        if len(parts) >= _MAX_DOCUMENT_PARTS_PER_FILE:
            raise KnowledgeIngestError(
                "too_many_document_parts",
                f"文件切分结果超过单文件上限 {_MAX_DOCUMENT_PARTS_PER_FILE}",
                filename=original_filename,
                details={"maximum": _MAX_DOCUMENT_PARTS_PER_FILE},
            )
        upper = min(len(body_tokens), offset + payload_budget)
        low = offset + 1
        best: tuple[int, str] | None = None
        while low <= upper:
            middle = (low + upper) // 2
            decoded = _decode_tokens(tokenizer, body_tokens[offset:middle]).strip()
            candidate = f"# {title} — Part {len(parts) + 1}\n\n{decoded}"
            if (
                len(_encode(tokenizer, normalize_markdown(candidate)))
                <= max_source_tokens
            ):
                best = (middle, candidate)
                low = middle + 1
            else:
                upper = middle - 1
        if best is None:
            raise KnowledgeIngestError(
                "document_split_failed",
                "无法在 Tensor Bank token 上限内切分文档",
                filename=original_filename,
            )
        offset, part_body = best
        parts.append(part_body)

    part_changes = (*changes, f"split_into_{len(parts)}_parts")
    return tuple(
        _prepared_document(
            relative_path=f"{relative_path_prefix}/{slug}-part-{index:02d}.md",
            body=part,
            group=group,
            source_kind=source_kind,
            retrieval_category=retrieval_category,
            original_filename=original_filename,
            tokenizer=tokenizer,
            max_source_tokens=max_source_tokens,
            changes=part_changes,
        )
        for index, part in enumerate(parts, start=1)
    )


def _prepared_document(
    *,
    relative_path: str,
    body: str,
    group: str,
    source_kind: str,
    retrieval_category: str,
    original_filename: str,
    tokenizer: Any,
    max_source_tokens: int,
    changes: tuple[str, ...],
) -> PreparedKnowledgeDocument:
    metadata = (
        "---\n"
        "canonical: false\n"
        "quality: 0.65\n"
        f"source_kind: {source_kind}\n"
        f"document_group: {group}\n"
        f"retrieval_category: {retrieval_category}\n"
        f"original_filename: {json.dumps(original_filename, ensure_ascii=False)}\n"
        "---\n\n"
    )
    content = metadata + body.strip() + "\n"
    token_count = len(_encode(tokenizer, normalize_markdown(content)))
    if token_count > max_source_tokens:
        raise KnowledgeIngestError(
            "document_token_limit_exceeded",
            f"清洗后的片段仍有 {token_count} tokens，超过上限 {max_source_tokens}",
            filename=original_filename,
            details={"token_count": token_count, "maximum": max_source_tokens},
        )
    encoded = content.encode("utf-8")
    return PreparedKnowledgeDocument(
        original_filename=original_filename,
        relative_path=relative_path,
        document_group=group,
        retrieval_category=retrieval_category,
        content=content,
        token_count=token_count,
        byte_count=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        changes=changes,
    )


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(token) for token in tokenizer.encode(text, add_special_tokens=False)
    )


def _decode_tokens(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))
