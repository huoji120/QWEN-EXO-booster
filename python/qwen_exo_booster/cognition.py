from __future__ import annotations

from qwen_exo_booster.knowledge import (
    KnowledgeDocument,
    KnowledgeRepository,
    KnowledgeSnapshot,
)

COGNITION_SOURCE_KIND = "gpt_cognition_identity_card"


class CognitionRepository(KnowledgeRepository):
    """Legacy single-card lane retained only for existing deployments."""

    def refresh(self) -> KnowledgeSnapshot:
        snapshot = super().refresh()
        if len(snapshot.documents) > 1:
            raise RuntimeError(
                "QWEN-EXO Cognition directory must contain at most one card"
            )
        if snapshot.documents:
            document = snapshot.documents[0]
            if document.source_kind != COGNITION_SOURCE_KIND:
                raise RuntimeError(
                    "QWEN-EXO Cognition card source_kind must be "
                    f"{COGNITION_SOURCE_KIND!r}"
                )
        return snapshot

    @property
    def card(self) -> KnowledgeDocument | None:
        documents = self.snapshot.documents
        return documents[0] if documents else None
