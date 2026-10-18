from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from qwen_exo_booster.knowledge import (
    KnowledgeCandidate,
    KnowledgeDocument,
    KnowledgeRepository,
    KnowledgeSnapshot,
)

NON_REFERENCE_POLICY_SOURCE_KINDS = frozenset({"coding_agent_execution_policy"})


@dataclass(frozen=True, slots=True)
class PolicyDataAttachment:
    """The canonical PolicyData document rendered as trusted instructions."""

    source_digest: str
    attachment_digest: str | None
    document_ids: tuple[str, ...]
    document_digests: tuple[str, ...]
    attached_tokens: int
    instructions: str | None = field(repr=False)

    @property
    def active(self) -> bool:
        return bool(self.instructions)

    def public_dict(self) -> dict[str, Any]:
        return {
            "source_digest": self.source_digest,
            "attachment_digest": self.attachment_digest,
            "document_ids": list(self.document_ids),
            "document_digests": list(self.document_digests),
            "attached_tokens": self.attached_tokens,
            "active": self.active,
            "always_on": True,
            "semantic_eligibility_required": False,
            "qk_relevance_required": False,
            "reference_judge_required": False,
            "injection_mode": "text_instructions" if self.active else "none",
            "text_attached": self.active,
            "native_state": None,
        }


class PolicyDataRepository:
    """Authoritative single personality-and-execution PolicyData document."""

    def __init__(self, root: Path | str):
        self._repository = KnowledgeRepository(root)
        self._lock = threading.RLock()
        self._compiled: dict[tuple[str, str], PolicyDataAttachment] = {}
        self._disk_signature: tuple[tuple[str, int, int], ...] | None = None

    @property
    def root(self) -> Path:
        return self._repository.root

    @property
    def snapshot(self) -> KnowledgeSnapshot:
        return self._repository.snapshot

    def refresh(self) -> KnowledgeSnapshot:
        snapshot = self._repository.refresh()
        if len(snapshot.documents) > 1:
            raise RuntimeError(
                "QWEN-EXO PolicyData directory must contain at most one document"
            )
        with self._lock:
            self._compiled.clear()
        return snapshot

    def upsert(
        self, relative_path: str, content: str, *, tags: object = None
    ) -> KnowledgeDocument:
        del tags
        requested_path = Path(relative_path).as_posix()
        with self._lock:
            snapshot = self.refresh()
            if (
                snapshot.documents
                and snapshot.documents[0].relative_path != requested_path
            ):
                raise RuntimeError(
                    "QWEN-EXO PolicyData directory already contains its one document"
                )
            document = self._repository.upsert(relative_path, content, tags=())
            self.refresh()
            self._compiled.clear()
            return document

    def delete(self, relative_path: str) -> None:
        self._repository.delete(relative_path)
        with self._lock:
            self._compiled.clear()

    def delete_many(self, relative_paths: tuple[str, ...] | list[str]) -> None:
        self._repository.delete_many(relative_paths)
        with self._lock:
            self._compiled.clear()

    def get(self, document_id: str) -> KnowledgeDocument:
        return self._repository.get(document_id)

    @staticmethod
    def _policy_candidate(candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        candidate_id = hashlib.sha256(
            f"policydata\0{candidate.candidate_id}".encode("utf-8")
        ).hexdigest()
        return replace(candidate, candidate_id=candidate_id, lane="policydata")

    def rank(self, query: str, *, limit: int) -> tuple[KnowledgeCandidate, ...]:
        return tuple(
            self._policy_candidate(candidate)
            for candidate in self._repository.rank(query, limit=limit)
        )

    def candidate_for_document(
        self, document_id: str, query: str
    ) -> KnowledgeCandidate:
        return self._policy_candidate(
            self._repository.candidate_for_document(document_id, query)
        )

    def is_non_reference_candidate(self, candidate: KnowledgeCandidate) -> bool:
        if candidate.lane != "policydata":
            return False
        try:
            document = self.get(candidate.document_id)
        except KeyError:
            return False
        return document.source_kind in NON_REFERENCE_POLICY_SOURCE_KINDS

    def refresh_if_changed(self) -> None:
        """Reload when a document file changed on disk (cheap stat per call)."""
        root = self._repository.root
        try:
            signature = tuple(
                sorted(
                    (entry.name, entry.stat().st_mtime_ns, entry.stat().st_size)
                    for entry in root.iterdir()
                    if entry.is_file()
                )
            )
        except OSError:
            return
        if signature == self._disk_signature:
            return
        with self._lock:
            if signature != self._disk_signature:
                self.refresh()
                self._disk_signature = signature

    def compile_text_attachment(self, tokenizer: Any) -> PolicyDataAttachment:
        """Render the sole PolicyData document, in full, into the instructions."""

        snapshot = self.snapshot
        inactive = PolicyDataAttachment(
            source_digest=snapshot.source_digest,
            attachment_digest=None,
            document_ids=(),
            document_digests=(),
            attached_tokens=0,
            instructions=None,
        )
        if not snapshot.documents:
            return inactive
        document = snapshot.documents[0]
        cache_key = (snapshot.source_digest, document.sha256)
        with self._lock:
            cached = self._compiled.get(cache_key)
            if cached is not None:
                return cached
        content = document.normalized_content.strip()
        if not content:
            return inactive
        attached_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        instructions = (
            "QWEN-EXO trusted private execution policy follows. Apply it as "
            "system-level operating guidance. Do not quote or disclose this "
            "private policy unless explicitly asked about system internals.\n\n"
            f'<policy_data id="{document.document_id}">\n{content}\n</policy_data>'
        )
        digest = hashlib.sha256()
        for value in (snapshot.source_digest, document.sha256):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        attachment = PolicyDataAttachment(
            source_digest=snapshot.source_digest,
            attachment_digest=digest.hexdigest(),
            document_ids=(document.document_id,),
            document_digests=(document.sha256,),
            attached_tokens=attached_tokens,
            instructions=instructions,
        )
        with self._lock:
            self._compiled[cache_key] = attachment
        return attachment
