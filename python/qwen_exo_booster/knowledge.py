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
from qwen_exo_booster.contracts import stable_digest

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_TERM_PATTERN = re.compile(r"[\w]+", re.UNICODE)
# Han, Hiragana/Katakana and Hangul runs carry no word boundaries, so a run is
# indexed as overlapping character bigrams (single characters stay as-is).
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
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
    retrieval_category = str(raw.get("retrieval_category", "")).strip() or None
    reflection_memory_schema = raw.get("reflection_memory_schema")
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
        "retrieval_category": retrieval_category,
        "reflection_memory_schema": reflection_memory_schema,
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


def set_markdown_retrieval_category(text: str, category: object) -> str:
    normalized_category = str(category or "").strip()
    if not normalized_category:
        raise ValueError("Retrieval category cannot be empty")
    match = _YAML_FRONT_MATTER.match(text)
    category_line = (
        f"retrieval_category: {json.dumps(normalized_category, ensure_ascii=False)}"
    )
    if match:
        lines = [
            line
            for line in match.group(1).splitlines()
            if line.split(":", 1)[0].strip().lower() != "retrieval_category"
        ]
        lines.append(category_line)
        frontmatter = "\n".join(lines)
        body = text[match.end() :]
        return f"---\n{frontmatter}\n---\n\n{body.lstrip()}"
    return f"---\n{category_line}\n---\n\n{text.lstrip()}"


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
    """Tokenize for BM25: word tokens, with CJK runs split into bigrams.

    ``\\w+`` treats a whole Chinese clause as one token, so a question and a
    document that share a phrase never shared a term and the lexical channel
    was silent for Chinese content.
    """
    normalized = normalize_markdown(str(text)).casefold()
    terms: list[str] = []
    for token in _TERM_PATTERN.findall(normalized):
        for piece in _CJK_RUN.split(token):
            if len(piece) > 1:
                terms.append(piece)
        for run in _CJK_RUN.findall(token):
            if len(run) == 1:
                terms.append(run)
            else:
                terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(terms)


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
    retrieval_category: str | None
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
            "retrieval_category": self.retrieval_category,
            "title": self.title,
            "tags": list(self.tags),
            "retrieval_diversity_bucket": retrieval_diversity_bucket(self),
        }
        if include_content:
            payload["content"] = self.content
        return payload


def semantic_document_group(document: KnowledgeDocument) -> str:
    """Return only groups that represent one semantic document."""

    group = str(document.document_group or "").strip()
    if document.source_kind == "trajectory_reflection" or group == "reflection_memory":
        return document.document_id
    return group or document.document_id


def retrieval_diversity_bucket(document: KnowledgeDocument) -> str:
    """Return a user-extensible retrieval category with a source fallback."""

    category = str(document.retrieval_category or "").strip()
    if category:
        return category
    source_kind = str(document.source_kind or "unclassified").strip()
    return source_kind or "unclassified"


def is_reflection_memory_document(document: KnowledgeDocument) -> bool:
    return bool(
        document.source_kind == "trajectory_reflection"
        or document.document_group == "reflection_memory"
        or "reflection-memory" in document.tags
    )


def reflection_task_category(original_task: str) -> str:
    normalized = " ".join(str(original_task).split()).casefold()
    digest = stable_digest("reflection-memory-task-category-v1", normalized)[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    for prefix in ("please-solve-this-issue-", "solve-this-issue-"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            break
    slug = slug[:64].rstrip("-")
    return f"reflection-task-{slug + '-' if slug else ''}{digest}"


CROSS_TASK_REFLECTION_NOTE = (
    "This reflection memory was distilled from a different task than the current "
    "one. Exact task wording or domain need not match. Select it only when its "
    "reusable rule, observed failure, evidence-backed hypothesis, or diagnostic "
    "next check materially helps with a specific shared problem, mechanism, or "
    "diagnostic pattern of symptoms and conditions. A verified root cause is not "
    "required; preserve supported/unresolved uncertainty, missing evidence, and "
    "applicability boundaries. Do not promote hypotheses to established causation "
    "or automatically transfer original-task facts. Shared topic or tool name "
    "alone is insufficient."
)
_TITLE_SEGMENT_SPLIT = re.compile(r"[：:｜|—\-–,，、;；()（）\[\]【】/]+")


def question_names_document(question: str, document: KnowledgeDocument) -> bool:
    """True when the question quotes a whole segment of the document title.

    Naming a memory makes its original evidence the subject of the question,
    so no cross-task scope note is needed. It still requires normal admission.
    """
    question_terms = set(lexical_terms(question))
    if not question_terms:
        return False
    for segment in _TITLE_SEGMENT_SPLIT.split(str(document.title or "")):
        terms = lexical_terms(segment)
        if len(terms) >= 2 and all(term in question_terms for term in terms):
            return True
    return False


def reflection_memory_matches_task(
    document: KnowledgeDocument, original_task: str
) -> bool:
    if not is_reflection_memory_document(document):
        return True
    category = str(document.retrieval_category or "").strip()
    if not category.startswith("reflection-task-"):
        return True
    return category == reflection_task_category(original_task)


def is_compatible_reflection_memory(document: KnowledgeDocument) -> bool:
    """Accept only bounded, schema-versioned reflection evidence cards."""
    if not is_reflection_memory_document(document):
        return False
    value = markdown_metadata(document.content).get("reflection_memory_schema")
    try:
        return int(float(value)) == 3
    except (TypeError, ValueError):
        return False


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
class QueryQKAttribution:
    query_index: int
    query_role: str
    query_prompt_start: int
    query_prompt_end: int
    query_source_start: int
    query_source_end: int
    page_id: int
    score: float
    support_score: float
    source_positions: tuple[int, ...]
    window_start: int
    window_end: int
    relative_score: float = 0.0
    head_group_count: int = 0

    def public_dict(self) -> dict[str, object]:
        return {
            "query_index": self.query_index,
            "query_role": self.query_role,
            "query_prompt_start": self.query_prompt_start,
            "query_prompt_end": self.query_prompt_end,
            "query_source_start": self.query_source_start,
            "query_source_end": self.query_source_end,
            "page_id": self.page_id,
            "score": self.score,
            "support_score": self.support_score,
            "source_positions": list(self.source_positions),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "relative_score": self.relative_score,
            "head_group_count": self.head_group_count,
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
    relative_tensor_score: float | None = None
    score_percentile: float | None = None
    anchor_support_count: int = 0
    anchor_role_count: int = 0
    head_group_count: int = 0

    page_ids: tuple[int, ...] = ()
    source_positions: tuple[int, ...] = ()
    virtual_positions: tuple[int, ...] = ()
    token_attributions: tuple[tuple[int, int, float], ...] = ()
    qk_attributions: tuple[QueryQKAttribution, ...] = ()
    candidate_origin: str = "lexical"
    native_prefix: NativePrefixSelection | None = field(default=None, repr=False)
    # Provenance shown to the judge next to the reference, e.g. that a
    # reflection was distilled from a different task than the current one.
    scope_note: str | None = None

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
            "relative_tensor_score": self.relative_tensor_score,
            "score_percentile": self.score_percentile,
            "anchor_support_count": self.anchor_support_count,
            "anchor_role_count": self.anchor_role_count,
            "head_group_count": self.head_group_count,
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
            "qk_attributions": [
                attribution.public_dict() for attribution in self.qk_attributions
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

    def lexical_document_scores(self, query: str) -> dict[str, float]:
        """Return BM25 scores by document id for documents sharing a query term."""
        query_counts = Counter(lexical_terms(query))
        if not query_counts:
            return {}
        with self._lock:
            snapshot = self._snapshot
            term_counts = self._term_counts
            document_frequency = self._document_frequency
            average_length = self._average_document_length
        document_count = len(snapshot.documents)
        if document_count == 0:
            return {}
        k1 = 1.5
        b = 0.75
        scores: dict[str, float] = {}
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
            if lexical_score > 0:
                scores[document.document_id] = lexical_score
        return scores

    def rank(self, query: str, limit: int = 8) -> tuple[KnowledgeCandidate, ...]:
        if limit < 1:
            return ()
        lexical_scores = self.lexical_document_scores(query)
        if not lexical_scores:
            return ()
        snapshot = self.snapshot

        candidates = []
        for document in snapshot.documents:
            lexical_score = lexical_scores.get(document.document_id, 0.0)
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
            retrieval_category=(
                str(metadata["retrieval_category"])
                if metadata["retrieval_category"] is not None
                else None
            ),
            tags=tuple(str(tag) for tag in metadata["tags"]),
            content=text,
            normalized_content=normalize_markdown(text),
        )
