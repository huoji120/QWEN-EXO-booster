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
    NativePrefixSelection,
)

NON_REFERENCE_POLICY_SOURCE_KINDS = frozenset({"coding_agent_execution_policy"})


@dataclass(frozen=True, slots=True)
class PolicyDataAttachment:
    """One semantically admitted PolicyData page bound to native model state."""

    source_digest: str
    attachment_digest: str | None
    document_ids: tuple[str, ...]
    document_digests: tuple[str, ...]
    attached_tokens: int
    native_prefix: NativePrefixSelection | None = field(repr=False)

    @property
    def active(self) -> bool:
        return self.native_prefix is not None

    def public_dict(self) -> dict[str, Any]:
        native_prefix = self.native_prefix
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
            "injection_mode": (
                "native_full_attention_salient_kv_and_gdn_document_state"
                if native_prefix is not None
                else "none"
            ),
            "text_attached": False,
            "native_state": (
                {
                    "source_digest": native_prefix.source_digest,
                    "page_id": native_prefix.page_id,
                    "prefix_identity": native_prefix.prefix_identity,
                    "tokens": len(native_prefix.token_ids),
                }
                if native_prefix is not None
                else None
            ),
        }


class PolicyDataRepository:
    """Authoritative single personality-and-execution PolicyData document."""

    def __init__(self, root: Path | str):
        self._repository = KnowledgeRepository(root)
        self._lock = threading.RLock()
        self._compiled: dict[tuple[str, str, str, int], PolicyDataAttachment] = {}

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

    def compile_native_candidate(
        self,
        candidate: KnowledgeCandidate | None,
        *,
        max_tokens: int,
    ) -> PolicyDataAttachment:
        """Bind one eligible policy candidate to its precompiled native state.

        A hybrid Qwen3.5 request has one recurrent GDN state, so PolicyData uses
        exactly one highest-ranked eligible page. There is deliberately no text
        fallback: stale, unaligned, or over-budget state fails closed.
        """

        if max_tokens < 1:
            raise ValueError("PolicyData token budget must be positive")
        snapshot = self.snapshot
        inactive = PolicyDataAttachment(
            source_digest=snapshot.source_digest,
            attachment_digest=None,
            document_ids=(),
            document_digests=(),
            attached_tokens=0,
            native_prefix=None,
        )
        if candidate is None:
            return inactive
        if candidate.lane != "policydata":
            raise ValueError("PolicyData state cannot contain a knowledge candidate")
        native_prefix = candidate.native_prefix
        if native_prefix is None or len(native_prefix.token_ids) > int(max_tokens):
            return inactive
        document = snapshot.by_id().get(candidate.document_id)
        if (
            document is None
            or document.sha256 != candidate.reference_digest
            or native_prefix.document_id != candidate.document_id
        ):
            return inactive

        cache_key = (
            snapshot.source_digest,
            candidate.reference_digest,
            native_prefix.prefix_identity,
            int(max_tokens),
        )
        with self._lock:
            cached = self._compiled.get(cache_key)
            if cached is not None:
                return cached

        digest = hashlib.sha256()
        for value in (
            snapshot.source_digest,
            candidate.reference_digest,
            native_prefix.source_digest,
            native_prefix.prefix_identity,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        attachment = PolicyDataAttachment(
            source_digest=snapshot.source_digest,
            attachment_digest=digest.hexdigest(),
            document_ids=(candidate.document_id,),
            document_digests=(candidate.reference_digest,),
            attached_tokens=len(native_prefix.token_ids),
            native_prefix=native_prefix,
        )
        with self._lock:
            self._compiled[cache_key] = attachment
        return attachment
