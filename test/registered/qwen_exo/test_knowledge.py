import base64
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.document_categories import DocumentCategoryStore
from qwen_exo_booster.knowledge import (
    KnowledgeRepository,
    lexical_terms,
    markdown_metadata,
    normalize_markdown,
    set_markdown_retrieval_category,
)
from qwen_exo_booster.document_ingest import (
    KnowledgeIngestError,
    prepare_knowledge_bytes,
    prepare_knowledge_upload,
)
from qwen_exo_booster.reflection_memory import ReflectionMemory
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.runtime import QwenExoRuntime
from qwen_exo_booster.query_probe import QueryStateSpan


def test_markdown_normalization_removes_metadata_and_comments():
    source = """---
canonical: true
quality: 0.9
source_kind: local_verified
---
<!-- hidden -->
# Title

WFP_ALE_AUTH_CONNECT   permits identifiers.
"""

    assert markdown_metadata(source) == {
        "canonical": True,
        "quality": 0.9,
        "source_kind": "local_verified",
        "document_group": None,
        "retrieval_category": None,
        "tags": (),
        "title": "Title",
    }
    normalized = normalize_markdown(source)
    assert "canonical:" not in normalized
    assert "hidden" not in normalized
    assert "WFP_ALE_AUTH_CONNECT" in normalized
    assert "wfp_ale_auth_connect" in lexical_terms(source)


def test_retrieval_category_update_preserves_existing_front_matter():
    updated = set_markdown_retrieval_category(
        "---\ntitle: WFP\nsource_kind: local_sdk_verified\n---\n\n# WFP\n",
        "windows-networking",
    )

    assert markdown_metadata(updated)["retrieval_category"] == "windows-networking"
    assert markdown_metadata(updated)["source_kind"] == "local_sdk_verified"
    assert updated.count("retrieval_category:") == 1


def test_repository_upsert_refresh_and_delete(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    first = repository.upsert(
        "network/wfp.md",
        "---\nretrieval_category: windows-networking\n---\n# WFP\nAppID filter guidance",
    )
    digest = repository.snapshot.source_digest

    assert first.relative_path == "network/wfp.md"
    assert first.title == "WFP"
    assert first.public_dict()["title"] == "WFP"
    assert first.public_dict()["retrieval_category"] == "windows-networking"
    assert first.public_dict()["retrieval_diversity_bucket"] == "windows-networking"
    assert repository.get(first.document_id).normalized_content.startswith("# WFP")
    assert repository.refresh().source_digest == digest

    repository.delete("network/wfp.md")
    assert repository.snapshot.documents == ()


def test_repository_tags_persist_and_legacy_documents_use_empty_tags(tmp_path):
    (tmp_path / "legacy.md").write_text("# Legacy\nNo tag metadata.", encoding="utf-8")
    repository = KnowledgeRepository(tmp_path)
    legacy_snapshot = repository.refresh()
    assert legacy_snapshot.documents[0].tags == ()

    tagged = repository.upsert(
        "network/wfp.md",
        "# WFP\nAppID filter guidance",
        tags=["Windows", "WFP", "wfp"],
    )

    assert tagged.tags == ("Windows", "WFP")
    assert 'tags: ["Windows", "WFP"]' in tagged.content
    refreshed = repository.refresh()
    assert next(
        document
        for document in refreshed.documents
        if document.relative_path == "network/wfp.md"
    ).tags == ("Windows", "WFP")

    updated = repository.upsert(
        "network/wfp.md",
        tagged.content.replace("guidance", "reference"),
        tags=["reviewed"],
    )
    assert updated.tags == ("reviewed",)
    assert 'tags: ["reviewed"]' in updated.content
    assert "Windows" not in updated.tags


def test_runtime_source_listing_includes_tokens_compile_status_time_and_content_search(
    tmp_path,
):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    document = runtime.knowledge.upsert(
        "reflection-memory/one.md", "# Reflection\nUNIQUE-CONTENT-EVIDENCE"
    )
    bank_path = tmp_path / "tensor-bank.pt"
    bank_path.write_bytes(b"bank")
    runtime.tensor_bank = SimpleNamespace(
        path=bank_path,
        snapshot=SimpleNamespace(
            pages=(
                SimpleNamespace(
                    lane="knowledge",
                    document_id=document.document_id,
                    reference_digest=document.sha256,
                    token_end=44,
                    cognition_token_count=4,
                ),
            )
        ),
    )
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=_CharacterTokenizer())
    runtime.reflection_memory_store = SimpleNamespace(
        list=lambda: [
            {
                "document_path": "reflection-memory/one.md",
                "created_at": 1234.5,
            }
        ]
    )

    listed = runtime.knowledge_documents(query="content-evidence")

    assert len(listed) == 1
    assert listed[0]["token_count"] == 40
    assert listed[0]["compiled"] is True
    assert listed[0]["compile_status"] == "compiled"
    assert listed[0]["compiled_at"] is not None
    assert listed[0]["ingested_at"] == 1234.5
    assert runtime.knowledge_documents(query="not-present") == []

    updated = runtime.knowledge.upsert(
        document.relative_path, "# Reflection\nmodified source"
    )
    stale = runtime.knowledge_documents()[0]
    assert stale["compiled"] is False
    assert stale["compile_status"] == "uncompiled"
    assert stale["token_count"] == len(updated.normalized_content)


@pytest.mark.asyncio
async def test_selected_document_compile_retains_unchanged_compiled_documents(tmp_path):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.policy_data = KnowledgeRepository(tmp_path / "policydata")
    first = runtime.knowledge.upsert("first.md", "# First\ncompiled")
    second = runtime.knowledge.upsert("second.md", "# Second\npending")
    first_page = SimpleNamespace(
        page_id=0,
        lane="knowledge",
        document_id=first.document_id,
        relative_path=first.relative_path,
        reference_digest=first.sha256,
    )

    class SelectiveTensorBank:
        def __init__(self):
            self.repositories = {
                "knowledge": runtime.knowledge,
                "policydata": runtime.policy_data,
            }
            self.snapshot = SimpleNamespace(pages=(first_page,))
            self.included = None
            self.resident = None

        async def ensure_ready(self, *, included_documents):
            self.included = included_documents
            pages = tuple(
                SimpleNamespace(page_id=index)
                for index, _key in enumerate(sorted(included_documents))
            )
            return SimpleNamespace(
                pages=pages,
                public_dict=lambda: {"document_state_count": len(pages)},
            )

        async def ensure_resident(self, page_ids):
            self.resident = tuple(page_ids)

    runtime.tensor_bank = SelectiveTensorBank()
    runtime._tensor_bank_admin_lock = asyncio.Lock()
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)

    result = await runtime.compile_source_documents("knowledge", ["second.md"])

    assert runtime.tensor_bank.included == {
        ("knowledge", first.relative_path),
        ("knowledge", second.relative_path),
    }
    assert result["requested_paths"] == ["second.md"]
    assert result["compiled_document_count"] == 2


def test_repository_delete_many_validates_before_removing_any_document(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert("one.md", "# One")
    repository.upsert("two.md", "# Two")

    with pytest.raises(FileNotFoundError):
        repository.delete_many(["one.md", "missing.md"])

    assert {document.relative_path for document in repository.snapshot.documents} == {
        "one.md",
        "two.md",
    }
    repository.delete_many(["one.md", "two.md"])
    assert repository.snapshot.documents == ()


def test_policydata_ignores_tags_and_supports_delete(tmp_path):
    repository = PolicyDataRepository(tmp_path / "policydata")
    document = repository.upsert(
        "personality.md",
        "# Personality\nGPT identity",
        tags=["personality", "policydata"],
    )

    assert document.tags == ()
    assert "tags:" not in (repository.root / "personality.md").read_text(
        encoding="utf-8"
    )
    repository.delete_many(["personality.md"])
    assert repository.snapshot.documents == ()


def test_repository_rejects_path_traversal_and_non_markdown(tmp_path):
    repository = KnowledgeRepository(tmp_path)

    with pytest.raises(ValueError, match="traverse"):
        repository.upsert("../secret.md", "secret")
    with pytest.raises(ValueError, match=".md"):
        repository.upsert("knowledge.txt", "text")


def test_repository_ignores_and_rejects_symbolic_links(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("sensitive host content", encoding="utf-8")
    link = root / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are not available on this host")

    repository = KnowledgeRepository(root)

    assert repository.refresh().documents == ()
    with pytest.raises(ValueError, match="symbolic links"):
        repository.upsert("linked.md", "replacement")


def test_rank_prefers_relevant_canonical_reference(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert(
        "wfp.md",
        """---
canonical: true
quality: 1.0
---
# WFP
Use FWPM_LAYER_ALE_AUTH_CONNECT_V4 and an AppID condition.
""",
    )
    repository.upsert(
        "ctf.md",
        "# CTF\nHeap exploitation, canaries, and return oriented programming.",
    )

    candidates = repository.rank("How does WFP AppID ALE_AUTH_CONNECT work?")

    assert candidates
    assert candidates[0].relative_path == "wfp.md"
    assert candidates[0].canonical
    assert (
        candidates[0].reference_digest
        == repository.get(candidates[0].document_id).sha256
    )


def test_rank_fails_closed_without_lexical_evidence(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert("wfp.md", "Windows filtering platform guidance")

    assert repository.rank("totally_unrelated_unique_term") == ()
    assert repository.rank("") == ()


def test_public_document_redacts_content_by_default(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    document = repository.upsert("private.md", "sensitive reference text")

    assert "content" not in document.public_dict()
    assert document.public_dict(include_content=True)["content"] == document.content


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


def _encoded(text):
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_uploaded_text_is_cleaned_and_converted_to_managed_markdown():
    (document,) = prepare_knowledge_upload(
        "SDK_notes.txt",
        _encoded("first line\r\nsecond\x00 line"),
        tokenizer=_CharacterTokenizer(),
        max_source_tokens=256,
    )

    assert document.relative_path == "uploads/sdk-notes.md"
    assert document.document_group == "upload_sdk-notes"
    assert "# SDK notes" in document.content
    assert document.retrieval_category == "uploaded_text"
    assert "retrieval_category: uploaded_text" in document.content
    assert "\x00" not in document.content
    assert "line_endings_normalized" in document.changes
    assert "control_characters_removed" in document.changes
    assert "metadata_replaced" in document.changes
    assert document.token_count <= 256


def test_oversized_upload_is_split_below_model_token_limit():
    documents = prepare_knowledge_upload(
        "large-reference.md",
        _encoded("# Large reference\n\n" + "reliable evidence " * 80),
        tokenizer=_CharacterTokenizer(),
        max_source_tokens=192,
        retrieval_category="api-guides",
    )
    assert {document.retrieval_category for document in documents} == {"api-guides"}
    assert all(
        "retrieval_category: api-guides" in document.content for document in documents
    )

    assert len(documents) > 1
    assert {document.document_group for document in documents} == {
        "upload_large-reference"
    }
    assert [document.relative_path for document in documents] == [
        f"uploads/large-reference-part-{index:02d}.md"
        for index in range(1, len(documents) + 1)
    ]
    assert all(document.token_count <= 192 for document in documents)
    assert all(
        f"split_into_{len(documents)}_parts" in document.changes
        for document in documents
    )


def test_upload_rejects_a_file_that_would_create_too_many_native_parts():
    with pytest.raises(KnowledgeIngestError) as error:
        prepare_knowledge_upload(
            "unbounded.txt",
            _encoded("evidence " * 1000),
            tokenizer=_CharacterTokenizer(),
            max_source_tokens=128,
        )

    assert error.value.code == "too_many_document_parts"
    assert error.value.details["maximum"] == 64


def test_upload_rejects_unsupported_or_malformed_files_with_actionable_codes():
    with pytest.raises(KnowledgeIngestError) as unsupported:
        prepare_knowledge_upload(
            "archive.pdf",
            _encoded("not a supported source"),
            tokenizer=_CharacterTokenizer(),
            max_source_tokens=256,
        )
    assert unsupported.value.code == "unsupported_file_type"
    assert unsupported.value.filename == "archive.pdf"

    with pytest.raises(KnowledgeIngestError) as malformed:
        prepare_knowledge_upload(
            "broken.json",
            _encoded('{"missing":'),
            tokenizer=_CharacterTokenizer(),
            max_source_tokens=256,
        )
    assert malformed.value.code == "invalid_json"
    assert "第 1 行" in str(malformed.value)


def test_pre_complete_bytes_use_isolated_paths_and_automatic_paging():
    documents = prepare_knowledge_bytes(
        "private-reference.txt",
        ("private model facts " * 80).encode("utf-8"),
        tokenizer=_CharacterTokenizer(),
        max_source_tokens=192,
        relative_path_prefix="pre-complete",
        document_group_prefix="pre_complete",
    )

    assert len(documents) > 1
    assert {document.document_group for document in documents} == {
        "pre_complete_private-reference"
    }
    assert [document.relative_path for document in documents] == [
        f"pre-complete/private-reference-part-{index:02d}.md"
        for index in range(1, len(documents) + 1)
    ]
    assert all(document.token_count <= 192 for document in documents)


def test_pre_complete_bytes_reject_files_above_upload_safety_limit():
    with pytest.raises(KnowledgeIngestError) as error:
        prepare_knowledge_bytes(
            "huge-reference.txt",
            b"x" * (4 * 1024 * 1024 + 1),
            tokenizer=_CharacterTokenizer(),
            max_source_tokens=192,
            relative_path_prefix="pre-complete",
            document_group_prefix="pre_complete",
        )

    assert error.value.code == "file_too_large"
    assert error.value.details["maximum_bytes"] == 4 * 1024 * 1024


def test_runtime_consumes_pre_complete_sources_once_after_materialization(tmp_path):
    source_root = tmp_path / "pre-complete"
    source_root.mkdir()
    (source_root / ".gitkeep").write_text("", encoding="utf-8")
    source = source_root / "private-reference.txt"
    source.write_text("private model facts " * 80, encoding="utf-8")
    events = []
    runtime = object.__new__(QwenExoRuntime)
    runtime._pre_complete_directory = source_root
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.config = SimpleNamespace(tensor_bank_max_document_tokens=192)
    runtime.tensor_bank = SimpleNamespace(cognition_token_ids=lambda: ())
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )

    staged = runtime._stage_pre_complete_knowledge(_CharacterTokenizer())

    assert staged is not None
    assert staged["source_files"] == ["private-reference.txt"]
    assert staged["document_count"] > 1
    assert staged["split_document_count"] == 1
    assert source.is_file()
    assert events == []
    assert len(runtime.knowledge.snapshot.documents) == staged["document_count"]
    assert all(
        document.relative_path.startswith("pre-complete/")
        for document in runtime.knowledge.snapshot.documents
    )

    payload = runtime._commit_pre_complete_knowledge()

    assert payload == staged
    assert not source.exists()
    assert events[0][1] == "knowledge.pre_complete_consumed"
    assert (source_root / ".gitkeep").is_file()
    assert runtime._stage_pre_complete_knowledge(_CharacterTokenizer()) is None
    assert runtime._commit_pre_complete_knowledge() is None


def test_runtime_rolls_back_staged_pre_complete_documents_without_consuming_source(
    tmp_path,
):
    source_root = tmp_path / "pre-complete"
    source_root.mkdir()
    source = source_root / "private-reference.txt"
    source.write_text("private model facts " * 80, encoding="utf-8")
    runtime = object.__new__(QwenExoRuntime)
    runtime._pre_complete_directory = source_root
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.config = SimpleNamespace(tensor_bank_max_document_tokens=192)
    runtime.tensor_bank = SimpleNamespace(cognition_token_ids=lambda: ())
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)

    assert runtime._stage_pre_complete_knowledge(_CharacterTokenizer()) is not None
    assert runtime.knowledge.snapshot.documents

    runtime._rollback_staged_pre_complete_knowledge()

    assert source.is_file()
    assert runtime.knowledge.snapshot.documents == ()
    assert runtime._pending_pre_complete_sources == ()
    assert runtime._pending_pre_complete_payload is None


def test_runtime_keeps_pre_complete_source_when_materialization_fails(tmp_path):
    source_root = tmp_path / "pre-complete"
    source_root.mkdir()
    source = source_root / "broken.json"
    source.write_text('{"missing":', encoding="utf-8")
    runtime = object.__new__(QwenExoRuntime)
    runtime._pre_complete_directory = source_root
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.config = SimpleNamespace(tensor_bank_max_document_tokens=256)
    runtime.tensor_bank = SimpleNamespace(cognition_token_ids=lambda: ())
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)

    with pytest.raises(KnowledgeIngestError) as error:
        runtime._stage_pre_complete_knowledge(_CharacterTokenizer())
    assert error.value.code == "invalid_json"
    assert source.is_file()
    assert runtime.knowledge.snapshot.documents == ()


class _RuntimeTensorBank:
    def __init__(self, runtime):
        self.runtime = runtime
        self.fail = False
        self.resident_page_ids = ()

    def cognition_token_ids(self):
        return ()

    async def ensure_ready(self):
        if self.fail:
            raise RuntimeError("compile failed")
        pages = tuple(
            SimpleNamespace(page_id=document.document_id)
            for document in self.runtime.knowledge.snapshot.documents
        )
        return SimpleNamespace(
            pages=pages,
            public_dict=lambda: {"document_state_count": len(pages)},
        )

    async def ensure_resident(self, page_ids):
        self.resident_page_ids = page_ids


@pytest.mark.asyncio
async def test_runtime_hot_ingest_replaces_groups_and_rolls_back_compile_failure(
    tmp_path,
):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.config = SimpleNamespace(tensor_bank_max_document_tokens=256)
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=_CharacterTokenizer())
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._tensor_bank_admin_lock = asyncio.Lock()
    runtime.tensor_bank = _RuntimeTensorBank(runtime)

    first = await runtime.ingest_knowledge_files(
        [{"filename": "guide.txt", "content_base64": _encoded("first version")}]
    )
    first_digest = runtime.knowledge.snapshot.source_digest

    assert first["hot_updated"] is True
    assert first["restart_required"] is False
    assert first["document_count"] == 1
    assert runtime.knowledge.snapshot.documents[0].relative_path == ("uploads/guide.md")

    second = await runtime.ingest_knowledge_files(
        [{"filename": "guide.txt", "content_base64": _encoded("second version")}]
    )
    stable_digest = runtime.knowledge.snapshot.source_digest
    stable_content = runtime.knowledge.snapshot.documents[0].content

    assert second["document_count"] == 1
    assert second["replaced_document_count"] == 1
    assert stable_digest != first_digest
    assert "second version" in stable_content

    runtime.tensor_bank.fail = True
    with pytest.raises(RuntimeError, match="compile failed"):
        await runtime.ingest_knowledge_files(
            [{"filename": "guide.txt", "content_base64": _encoded("bad version")}]
        )

    assert runtime.knowledge.snapshot.source_digest == stable_digest
    assert runtime.knowledge.snapshot.documents[0].content == stable_content


class _ReflectionQueryProbe:
    async def probe(self, _parent_id, _query):
        return SimpleNamespace(
            status="ready",
            query_heads=(((1.0, 0.0),),),
            query_states=(QueryStateSpan("current_user", 0, 1, 0, 1),),
            public_dict=lambda: {
                "status": "ready",
                "query_count": 1,
                "query_head_count": 1,
                "head_dim": 2,
                "query_states": [
                    QueryStateSpan("current_user", 0, 1, 0, 1).public_dict()
                ],
            },
        )


class _ReflectionRankBank:
    def __init__(self, candidate):
        self.candidate = candidate
        self.eligible_documents = None

    async def ensure_ready(self):
        return SimpleNamespace(ready=True)

    def rank(self, _query_heads, **kwargs):
        self.eligible_documents = kwargs["eligible_documents"]
        kwargs["audit"].update({"status": "ready", "candidate_count": 1})
        return (self.candidate,)


@pytest.mark.asyncio
async def test_runtime_reflection_qk_search_is_scoped_to_reflection_documents(tmp_path):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    reflection_document = runtime.knowledge.upsert(
        "reflection-memory/network.md",
        """---
source_kind: trajectory_reflection
document_group: reflection_memory
title: Network probe
tags: [reflection-memory]
---

Observe the transport error class before retrying.
""",
    )
    unrelated_document = runtime.knowledge.upsert(
        "reference/general.md",
        """---
source_kind: local_verified
title: General reference
---

Generic network reference.
""",
    )
    candidate = runtime.knowledge.candidate_for_document(
        reflection_document.document_id, "network probe"
    )
    candidate = replace(candidate, tensor_score=0.88)
    runtime.tensor_bank = _ReflectionRankBank(candidate)
    runtime.query_probe = _ReflectionQueryProbe()
    events = []
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )

    candidates = await runtime._retrieve_reflection_memory_candidates(
        "reflection-memory:test", "network transport failure"
    )

    assert len(candidates) == 1
    assert candidates[0].document_path == reflection_document.relative_path
    assert runtime.tensor_bank.eligible_documents == frozenset(
        {("knowledge", reflection_document.document_id)}
    )
    assert ("knowledge", unrelated_document.document_id) not in (
        runtime.tensor_bank.eligible_documents
    )
    completed = [payload for _, event, payload in events if event.endswith("completed")]
    assert completed[-1]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_runtime_reflection_organizer_builds_model_reviews_from_high_qk_pairs(
    tmp_path,
):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    documents = tuple(
        runtime.knowledge.upsert(
            f"reflection-memory/{name}.md",
            f"""---
source_kind: trajectory_reflection
document_group: reflection_memory
title: {title}
tags: [reflection-memory]
---

{content}
""",
        )
        for name, title, content in (
            ("a", "探针甲", "用一次匹配请求区分网络环境。"),
            ("b", "探针乙", "重复错误后改用等价客户端探针。"),
            ("c", "探针丙", "依据错误类别停止无信息枚举。"),
        )
    )
    scopes = []

    class _OrganizerBank:
        async def ensure_ready(self):
            return SimpleNamespace(ready=True)

        def rank(self, _query_heads, **kwargs):
            eligible = kwargs["eligible_documents"]
            scopes.append(eligible)
            return tuple(
                replace(
                    runtime.knowledge.candidate_for_document(document_id, ""),
                    tensor_score=0.81,
                )
                for _lane, document_id in sorted(eligible)
            )[: kwargs["limit"]]

    reviews = []

    async def organize_candidates(**kwargs):
        reviews.append(kwargs)
        return None

    events = []
    runtime.tensor_bank = _OrganizerBank()
    runtime.query_probe = _ReflectionQueryProbe()
    runtime.reflection_memory_service = SimpleNamespace(
        organize_candidates=organize_candidates
    )
    runtime._reflection_memory_organize_lock = asyncio.Lock()
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )
    progress_events = []

    result = await runtime.organize_reflection_memories(
        progress=lambda stage, progress, message, details: progress_events.append(
            (stage, progress, message, details)
        )
    )

    assert result["status"] == "kept_distinct"
    assert result["high_qk_pair_count"] == 3
    assert len(reviews) == 1
    assert {candidate.document_path for candidate in reviews[0]["candidates"]} == {
        document.relative_path for document in documents
    }
    assert all(score >= 0.55 for _left, _right, score in reviews[0]["qk_pairs"])
    assert len(scopes) == 3
    for document, scope in zip(documents, scopes):
        assert ("knowledge", document.document_id) not in scope
    assert any(
        event == "reflection_memory.organization.qk_completed" for _, event, _ in events
    )
    assert {stage for stage, *_rest in progress_events} >= {
        "scanning",
        "qk_retrieval",
        "model_review",
    }
    assert all(0 <= progress <= 100 for _stage, progress, *_rest in progress_events)


@pytest.mark.asyncio
async def test_runtime_reflection_organization_runs_in_background_with_status():
    runtime = object.__new__(QwenExoRuntime)
    runtime.reflection_memory_service = object()
    runtime.tensor_bank = object()
    runtime.query_probe = object()
    runtime.knowledge = SimpleNamespace(
        snapshot=SimpleNamespace(source_digest="knowledge-digest")
    )
    events = []
    runtime.telemetry = SimpleNamespace(
        emit=lambda request_id, event_type, payload: events.append(
            (request_id, event_type, payload)
        )
    )
    runtime._reflection_memory_organization_task = None
    runtime._reflection_memory_organization_state = {
        "job_id": None,
        "status": "idle",
        "stage": "idle",
        "progress": 0,
        "message": "尚未开始整理",
        "queued_at": None,
        "started_at": None,
        "updated_at": None,
        "finished_at": None,
        "details": {},
        "result": None,
        "error": None,
    }
    started = asyncio.Event()
    release = asyncio.Event()

    async def organize_reflection_memories(*, progress):
        progress(
            "model_review",
            48,
            "模型正在审查候选",
            {"pass_index": 1, "review_count": 1},
        )
        started.set()
        await release.wait()
        return {
            "status": "kept_distinct",
            "document_count": 3,
            "high_qk_pair_count": 2,
            "review_count": 1,
            "merged_document_count": 0,
        }

    runtime.organize_reflection_memories = organize_reflection_memories

    accepted = runtime.start_reflection_memory_organization()
    await started.wait()
    running = runtime.reflection_memory_organization_status()

    assert accepted["status"] == "queued"
    assert accepted["job_id"].startswith("reflection-organization-")
    assert running["status"] == "running"
    assert running["stage"] == "model_review"
    assert running["progress"] == 48
    assert running["details"]["review_count"] == 1
    with pytest.raises(RuntimeError, match="already running"):
        runtime.start_reflection_memory_organization()

    release.set()
    await runtime._reflection_memory_organization_task
    completed = runtime.reflection_memory_organization_status()

    assert completed["status"] == "succeeded"
    assert completed["stage"] == "completed"
    assert completed["progress"] == 100
    assert completed["result"]["status"] == "kept_distinct"
    assert any(
        event_type == "reflection_memory.organization.job_completed"
        for _request_id, event_type, _payload in events
    )


@pytest.mark.asyncio
async def test_runtime_reflection_update_replaces_existing_document_atomically(
    tmp_path,
):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.document_categories = DocumentCategoryStore(
        tmp_path / "state" / "document-categories.sqlite3"
    )
    runtime.policy_data = SimpleNamespace(snapshot=SimpleNamespace(documents=()))
    previous = runtime.knowledge.upsert(
        "reflection-memory/network.md",
        """---
source_kind: trajectory_reflection
document_group: reflection_memory
retrieval_category: network-debugging
title: Old network probe
tags: [reflection-memory]
---

Old process lesson.
""",
    )
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._tensor_bank_admin_lock = asyncio.Lock()
    runtime.tensor_bank = _RuntimeTensorBank(runtime)
    runtime.query_probe = object()
    reflection = ReflectionMemory(
        trajectory_id="resp-update",
        conversation_key="conversation-update",
        source_digest="source-update",
        title="Observe before changing HTTP clients",
        outcome="mixed",
        reflection="The first controllable error was repeating requests without classifying the transport signal.",
        evidence="The same WinError 10106 repeated while a curl request returned a concrete HTTP response.",
        causal_analysis="The repeated loop added no information; a one-request client comparison would discriminate environment from target behavior.",
        reusable_experience="When clients disagree on one endpoint, compare one matched request and inspect error classes before choosing a bulk strategy.",
        avoid="Do not enumerate hundreds of payloads after identical client-level failures; pivot after the repeated result class is stable.",
        next_time="Observe one baseline, form transport and target hypotheses, run one matched probe, choose the working path, then verify one modified request.",
        memory_action="update",
        target_document_path=previous.relative_path,
        target_document_sha256=previous.sha256,
        source_event_count=4,
        source_token_count=1024,
        attempts=1,
        created_at=1.0,
        retrieval_category="reflection-task-new-run-deadbeef",
    )

    payload = await runtime._publish_reflection_memory(reflection)

    documents = runtime.knowledge.snapshot.documents
    assert len(documents) == 1
    assert documents[0].relative_path == previous.relative_path
    assert "Observe before changing HTTP clients" in documents[0].content
    assert documents[0].retrieval_category == "network-debugging"
    categories = {
        category["category_id"]: category
        for category in runtime.document_categories.categories()
    }
    assert categories["network-debugging"]["parent_id"] == "reflection-memory"
    assert payload["memory_action"] == "update"
    assert payload["replaced_document_sha256"] == previous.sha256

    with pytest.raises(RuntimeError, match="target is stale"):
        await runtime._publish_reflection_memory(
            replace(reflection, target_document_sha256="stale-sha")
        )
    assert runtime.knowledge.snapshot.documents[0].sha256 == payload["document_sha256"]


@pytest.mark.asyncio
async def test_runtime_reflection_merge_is_atomic_and_deletes_superseded_documents(
    tmp_path,
):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    left = runtime.knowledge.upsert(
        "reflection-memory/left.md",
        """---
source_kind: trajectory_reflection
document_group: reflection_memory
title: 左侧旧记忆
tags: [reflection-memory]
---

左侧旧经验。
""",
    )
    right = runtime.knowledge.upsert(
        "reflection-memory/right.md",
        """---
source_kind: trajectory_reflection
document_group: reflection_memory
title: 右侧旧记忆
tags: [reflection-memory]
---

右侧旧经验。
""",
    )
    original_digest = runtime.knowledge.snapshot.source_digest
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._tensor_bank_admin_lock = asyncio.Lock()
    runtime.tensor_bank = _RuntimeTensorBank(runtime)
    runtime.query_probe = object()
    reflection = ReflectionMemory(
        trajectory_id="reflection-memory-organization:test",
        conversation_key="reflection-memory-organization",
        source_digest="source-merged",
        title="合并后的网络探测经验",
        outcome="mixed",
        reflection="两条记忆描述同一网络探测失败，合并后保留工具响应和转向时机。",
        evidence="left.md 与 right.md 均记录 WinError 10106，curl 的 HTTP 响应提供区分证据。",
        causal_analysis="重复请求没有增加信息，单次匹配客户端探针才能区分环境失败和目标失败。",
        conflict_resolution="旧记录对失败阶段表述不同；按工具时间戳保留为请求前与请求后两个边界。",
        reusable_experience="客户端结果不一致时先比较一次等价请求，再依据错误类别选择后续路径。",
        avoid="不要在相同错误类别稳定后继续枚举请求，也不要静默丢弃版本边界。",
        next_time="先跑一个基线和一个等价探针，记录状态与时间戳，再验证一次修改后的请求。",
        memory_action="update",
        target_document_path=left.relative_path,
        target_document_sha256=left.sha256,
        merge_document_paths=(left.relative_path, right.relative_path),
        merge_document_sha256s=(
            (left.relative_path, left.sha256),
            (right.relative_path, right.sha256),
        ),
        source_event_count=0,
        source_token_count=1024,
        attempts=1,
        created_at=1.0,
    )

    runtime.tensor_bank.fail = True
    with pytest.raises(RuntimeError, match="compile failed"):
        await runtime._publish_reflection_memory(reflection)
    assert runtime.knowledge.snapshot.source_digest == original_digest
    assert {
        document.relative_path for document in runtime.knowledge.snapshot.documents
    } == {
        left.relative_path,
        right.relative_path,
    }

    runtime.tensor_bank.fail = False
    payload = await runtime._publish_reflection_memory(reflection)

    documents = runtime.knowledge.snapshot.documents
    assert len(documents) == 1
    assert documents[0].relative_path == left.relative_path
    assert "合并后的网络探测经验" in documents[0].content
    assert payload["merge_document_paths"] == [left.relative_path, right.relative_path]
    assert payload["removed_document_paths"] == [right.relative_path]
    assert payload["hot_updated"] is True
