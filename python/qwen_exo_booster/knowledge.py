from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from qwen_exo_booster.tags import TagValidationError, normalize_tags

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_TERM_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_YAML_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_scalar(value: str):
    normalized = value.strip().strip("\"'")
    lowered = normalized.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return float(normalized)
    except ValueError:
        return normalized


def markdown_metadata(text: str) -> dict[str, object]:
    match = _YAML_FRONT_MATTER.match(text)
    raw = {}
    if match:
        for line in match.group(1).splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            raw[key.strip().lower()] = _parse_scalar(value)

    canonical = bool(raw.get("canonical", False))
    quality_value = raw.get("quality", 1.0 if canonical else 0.5)
    try:
        quality = float(quality_value)
    except (TypeError, ValueError):
        quality = 1.0 if canonical else 0.5
    quality = min(max(quality, 0.0), 1.0)
    source_kind = str(raw.get("source_kind", "unclassified")).strip()
    document_group = str(raw.get("document_group", "")).strip() or None
    title = str(raw.get("title", "")).strip()
    if not title:
        body = text[match.end() :] if match else text
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", body)
        title = heading.group(1).strip() if heading else ""
    try:
        tags = normalize_tags(raw.get("tags"))
    except TagValidationError:
        tags = ()
    return {
        "canonical": canonical,
        "quality": quality,
        "source_kind": source_kind or "unclassified",
        "document_group": document_group,
        "title": title or None,
        "tags": tags,
    }


def set_markdown_tags(text: str, tags: object) -> str:
    normalized_tags = normalize_tags(tags)
    match = _YAML_FRONT_MATTER.match(text)
    tag_line = (
        f"tags: {json.dumps(list(normalized_tags), ensure_ascii=False)}"
        if normalized_tags
        else None
    )
    if match:
        lines = []
        replaced = False
        for line in match.group(1).splitlines():
            key = line.split(":", 1)[0].strip().lower() if ":" in line else ""
            if key == "tags":
                if tag_line is not None and not replaced:
                    lines.append(tag_line)
                replaced = True
                continue
            lines.append(line)
        if tag_line is not None and not replaced:
            lines.append(tag_line)
        body = text[match.end() :]
        frontmatter = "\n".join(lines)
        return f"---\n{frontmatter}\n---\n\n{body.lstrip()}"
    if tag_line is None:
        return text
    return f"---\n{tag_line}\n---\n\n{text.lstrip()}"


def normalize_markdown(text: str) -> str:
    match = _YAML_FRONT_MATTER.match(text)
    if match:
        text = text[match.end() :]
    text = _HTML_COMMENT.sub("", text).replace("\r\n", "\n").replace("\r", "\n")

    normalized = []
    blank = False
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if normalized and not blank:
                normalized.append("")
            blank = True
            continue
        normalized.append(line)
        blank = False
    return "\n".join(normalized).strip()


def lexical_terms(text: str) -> tuple[str, ...]:
    normalized = normalize_markdown(str(text)).casefold()
    return tuple(term for term in _TERM_PATTERN.findall(normalized) if len(term) > 1)


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    relative_path: str
    sha256: str
    byte_count: int
    modified_ns: int
    canonical: bool
    quality: float
    source_kind: str
    document_group: str | None
    title: str | None
    content: str
    normalized_content: str
    tags: tuple[str, ...] = ()

    def public_dict(self, include_content: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "modified_ns": self.modified_ns,
            "canonical": self.canonical,
            "quality": self.quality,
            "source_kind": self.source_kind,
            "document_group": self.document_group,
            "title": self.title,
            "tags": list(self.tags),
        }
        if include_content:
            payload["content"] = self.content
        return payload


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    source_digest: str
    documents: tuple[KnowledgeDocument, ...]

    def by_id(self) -> dict[str, KnowledgeDocument]:
        return {document.document_id: document for document in self.documents}


@dataclass(frozen=True, slots=True)
class NativePrefixSelection:
    source_digest: str
    page_id: int
    document_id: str
    local_positions: tuple[int, ...]
    source_positions: tuple[int, ...]
    token_ids: tuple[int, ...] = field(repr=False)
    prefix_identity: str
    radix_namespace: str

    def scheduler_payload(self) -> dict[str, object]:
        return {
            "source_digest": self.source_digest,
            "page_id": self.page_id,
            "local_positions": list(self.local_positions),
            "prefix_identity": self.prefix_identity,
        }

    def public_dict(self) -> dict[str, object]:
        return {
            "source_digest": self.source_digest,
            "page_id": self.page_id,
            "document_id": self.document_id,
            "source_positions": list(self.source_positions),
            "prefix_identity": self.prefix_identity,
            "radix_namespace": self.radix_namespace,
            "tokens": len(self.token_ids),
        }


@dataclass(frozen=True, slots=True)
class KnowledgeCandidate:
    candidate_id: str
    document_id: str
    relative_path: str
    score: float
    lexical_score: float
    quality_prior: float
    canonical: bool
    reference_digest: str
    reference_content: str = field(repr=False)
    normalized_reference_content: str = field(repr=False)
    lane: str = "knowledge"
    tensor_score: float | None = None
    page_ids: tuple[int, ...] = ()
    source_positions: tuple[int, ...] = ()
    virtual_positions: tuple[int, ...] = ()
    token_attributions: tuple[tuple[int, int, float], ...] = ()
    candidate_origin: str = "lexical"
    native_prefix: NativePrefixSelection | None = field(default=None, repr=False)

    def public_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "score": self.score,
            "lexical_score": self.lexical_score,
            "quality_prior": self.quality_prior,
            "canonical": self.canonical,
            "reference_digest": self.reference_digest,
            "lane": self.lane,
            "policy": self.lane == "policydata",
            "tensor_score": self.tensor_score,
            "page_ids": list(self.page_ids),
            "source_positions": list(self.source_positions),
            "virtual_positions": list(self.virtual_positions),
            "token_attributions": [
                {
                    "query_token_offset": query_token_offset,
                    "page_id": page_id,
                    "score": score,
                }
                for query_token_offset, page_id, score in self.token_attributions
            ],
            "native_prefix": (
                self.native_prefix.public_dict()
                if self.native_prefix is not None
                else None
            ),
            "candidate_origin": self.candidate_origin,
        }


class KnowledgeRepository:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.RLock()
        self._snapshot = KnowledgeSnapshot(
            source_digest=_digest_bytes(b""), documents=()
        )
        self._term_counts: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_document_length = 0.0

    @property
    def snapshot(self) -> KnowledgeSnapshot:
        with self._lock:
            return self._snapshot

    def refresh(self) -> KnowledgeSnapshot:
        self.root.mkdir(parents=True, exist_ok=True)
        documents = []
        for path in sorted(self.root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in _MARKDOWN_SUFFIXES
                or self._contains_symlink(path)
            ):
                continue
            documents.append(self._read_document(path))

        digest = hashlib.sha256()
        for document in documents:
            digest.update(document.relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.sha256.encode("ascii"))
            digest.update(b"\0")
        snapshot = KnowledgeSnapshot(
            source_digest=digest.hexdigest(), documents=tuple(documents)
        )

        term_counts = {
            document.document_id: Counter(lexical_terms(document.normalized_content))
            for document in documents
        }
        document_frequency: Counter[str] = Counter()
        for counts in term_counts.values():
            document_frequency.update(counts.keys())
        average_length = (
            sum(sum(counts.values()) for counts in term_counts.values())
            / len(term_counts)
            if term_counts
            else 0.0
        )

        with self._lock:
            self._snapshot = snapshot
            self._term_counts = term_counts
            self._document_frequency = document_frequency
            self._average_document_length = average_length
            return snapshot

    def upsert(
        self, relative_path: str, content: str, *, tags: object = None
    ) -> KnowledgeDocument:
        if not str(content).strip():
            raise ValueError("Knowledge document content cannot be empty")
        if Path(relative_path).suffix.lower() not in _MARKDOWN_SUFFIXES:
            raise ValueError("Knowledge sources must use a .md or .markdown suffix")
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored_content = str(content)
        if tags is not None:
            stored_content = set_markdown_tags(stored_content, tags)
        encoded = stored_content.encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.refresh()
        return self._read_document(path)

    def delete(self, relative_path: str) -> None:
        self.delete_many((relative_path,))

    def delete_many(self, relative_paths: tuple[str, ...] | list[str]) -> None:
        paths = []
        seen = set()
        for relative_path in relative_paths:
            normalized = str(relative_path).replace("\\", "/").strip()
            if normalized in seen:
                continue
            if Path(normalized).suffix.lower() not in _MARKDOWN_SUFFIXES:
                raise ValueError("Knowledge sources must use a .md or .markdown suffix")
            path = self._safe_path(normalized)
            if not path.is_file():
                raise FileNotFoundError(normalized)
            paths.append(path)
            seen.add(normalized)
        for path in paths:
            path.unlink()
        if paths:
            self.refresh()

    def get(self, document_id: str) -> KnowledgeDocument:
        document = self.snapshot.by_id().get(document_id)
        if document is None:
            raise KeyError(document_id)
        return document

    def candidate_for_document(
        self, document_id: str, query: str
    ) -> KnowledgeCandidate:
        document = self.get(document_id)
        snapshot = self.snapshot
        quality_prior = document.quality * 0.1 + (0.05 if document.canonical else 0.0)
        return KnowledgeCandidate(
            candidate_id=_digest_bytes(
                f"{snapshot.source_digest}\0{document.document_id}\0{query}".encode(
                    "utf-8"
                )
            ),
            document_id=document.document_id,
            relative_path=document.relative_path,
            score=quality_prior,
            lexical_score=0.0,
            quality_prior=quality_prior,
            canonical=document.canonical,
            reference_digest=document.sha256,
            reference_content=document.content,
            normalized_reference_content=document.normalized_content,
        )

    def rank(self, query: str, limit: int = 8) -> tuple[KnowledgeCandidate, ...]:
        query_counts = Counter(lexical_terms(query))
        if not query_counts or limit < 1:
            return ()

        with self._lock:
            snapshot = self._snapshot
            term_counts = self._term_counts
            document_frequency = self._document_frequency
            average_length = self._average_document_length

        document_count = len(snapshot.documents)
        if document_count == 0:
            return ()

        candidates = []
        k1 = 1.5
        b = 0.75
        for document in snapshot.documents:
            counts = term_counts[document.document_id]
            document_length = sum(counts.values())
            lexical_score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency_for_term = document_frequency[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency_for_term + 0.5)
                    / (document_frequency_for_term + 0.5)
                )
                length_ratio = (
                    document_length / average_length if average_length > 0 else 1.0
                )
                denominator = frequency + k1 * (1 - b + b * length_ratio)
                lexical_score += (
                    inverse_frequency
                    * frequency
                    * (k1 + 1)
                    / denominator
                    * min(query_frequency, 3)
                )
            if lexical_score <= 0:
                continue
            quality_prior = document.quality * 0.1 + (
                0.05 if document.canonical else 0.0
            )
            total = lexical_score + quality_prior
            candidates.append(
                KnowledgeCandidate(
                    candidate_id=_digest_bytes(
                        f"{snapshot.source_digest}\0{document.document_id}\0{query}".encode(
                            "utf-8"
                        )
                    ),
                    document_id=document.document_id,
                    relative_path=document.relative_path,
                    score=total,
                    lexical_score=lexical_score,
                    quality_prior=quality_prior,
                    canonical=document.canonical,
                    reference_digest=document.sha256,
                    reference_content=document.content,
                    normalized_reference_content=document.normalized_content,
                )
            )
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.document_id))
        return tuple(candidates[:limit])

    def _contains_symlink(self, path: Path) -> bool:
        current = path
        while current != self.root:
            if current.is_symlink():
                return True
            if self.root not in current.parents:
                return True
            current = current.parent
        return False

    def _safe_path(self, relative_path: str) -> Path:
        normalized = str(relative_path).replace("\\", "/").strip()
        pure = PurePosixPath(normalized)
        if not normalized or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(
                "Knowledge path must be relative and cannot traverse parents"
            )
        if pure.suffix.lower() not in _MARKDOWN_SUFFIXES:
            raise ValueError("Knowledge files must use .md or .markdown")
        unresolved = self.root / Path(*pure.parts)
        if self._contains_symlink(unresolved):
            raise ValueError("Knowledge path cannot traverse symbolic links")
        candidate = unresolved.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Knowledge path escapes the configured repository")
        return candidate

    def _read_document(self, path: Path) -> KnowledgeDocument:
        encoded = path.read_bytes()
        text = encoded.decode("utf-8")
        digest = _digest_bytes(encoded)
        metadata = markdown_metadata(text)
        relative_path = path.relative_to(self.root).as_posix()
        stat = path.stat()
        return KnowledgeDocument(
            document_id=_digest_bytes(relative_path.encode("utf-8"))[:24],
            relative_path=relative_path,
            sha256=digest,
            byte_count=len(encoded),
            modified_ns=stat.st_mtime_ns,
            canonical=bool(metadata["canonical"]),
            quality=float(metadata["quality"]),
            source_kind=str(metadata["source_kind"]),
            document_group=(
                str(metadata["document_group"])
                if metadata["document_group"] is not None
                else None
            ),
            title=(str(metadata["title"]) if metadata["title"] is not None else None),
            tags=tuple(str(tag) for tag in metadata["tags"]),
            content=text,
            normalized_content=normalize_markdown(text),
        )
