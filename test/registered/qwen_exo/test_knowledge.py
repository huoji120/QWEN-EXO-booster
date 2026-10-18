import asyncio
import base64
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.document_categories import DocumentCategoryStore
from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
from qwen_exo_booster.document_ingest import (
    KnowledgeIngestError,
    prepare_knowledge_bytes,
    prepare_knowledge_upload,
)
from qwen_exo_booster.knowledge import (
    KnowledgeRepository,
    is_compatible_reflection_memory,
    lexical_terms,
    markdown_metadata,
    normalize_markdown,
    reflection_memory_matches_task,
    reflection_task_category,
    set_markdown_retrieval_category,
)
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.reflection_memory import ReflectionMemory, ReflectionMemoryStore
from qwen_exo_booster.runtime import QwenExoRuntime, QwenExoRuntimeState
from qwen_exo_booster.task_scheduling import current_internal_work_class


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
        "reflection_memory_schema": None,
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


def test_reflection_memory_schema_gate_rejects_legacy_documents(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    legacy = repository.upsert(
        "reflection-memory/legacy.md",
        "---\nsource_kind: trajectory_reflection\ndocument_group: reflection_memory\n"
        "tags: [reflection-memory]\n---\n\n# Legacy\nLong legacy reflection.",
    )
    current = repository.upsert(
        "reflection-memory/current.md",
        "---\nsource_kind: trajectory_reflection\ndocument_group: reflection_memory\n"
        "reflection_memory_schema: 3\ntags: [reflection-memory]\n---\n\n"
        "# Current\n可执行规则：先观察证据。",
    )

    assert is_compatible_reflection_memory(legacy) is False
    assert is_compatible_reflection_memory(current) is True


def test_task_scoped_reflection_matches_only_its_original_task(tmp_path):
    task = "Please solve this issue: add implicit HEAD and OPTIONS routing"
    category = reflection_task_category(task)
    repository = KnowledgeRepository(tmp_path)
    scoped = repository.upsert(
        "reflection-memory/scoped.md",
        "---\nsource_kind: trajectory_reflection\n"
        "document_group: reflection_memory\nreflection_memory_schema: 3\n"
        f"retrieval_category: {category}\n---\n\n# Scoped\nRule.",
    )
    shared = repository.upsert(
        "reflection-memory/shared.md",
        "---\nsource_kind: trajectory_reflection\n"
        "document_group: reflection_memory\nreflection_memory_schema: 3\n"
        "retrieval_category: shared-reflection\n---\n\n# Shared\nRule.",
    )

    assert reflection_memory_matches_task(scoped, task) is True
    assert (
        reflection_memory_matches_task(scoped, "Fix deprecated response headers")
        is False
    )
    assert reflection_memory_matches_task(shared, "Any unrelated task") is True


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


def test_lexical_terms_index_cjk_runs_as_bigrams():
    """Chinese clauses must share terms with the documents they name.

    ``\\w+`` yielded one token for an entire Chinese clause, so the BM25
    channel produced no score for a question that quoted a document title
    verbatim. Bigrams over each CJK run make the phrase overlap visible while
    Latin words keep whole-word matching.
    """
    from qwen_exo_booster.knowledge import lexical_terms

    question = lexical_terms("在之前做笔记资料整理的时候 我们遇到什么问题了？")
    title = lexical_terms("笔记资料整理：交付物观测与验收边界 pytest")

    assert {"笔记", "记资", "资料", "料整", "整理"} <= set(question)
    assert {"笔记", "记资", "资料", "料整", "整理"} <= set(title)
    assert "pytest" in title
    assert all(len(term) <= 2 or not _is_cjk(term) for term in question)


def _is_cjk(term: str) -> bool:
    return all("\u4e00" <= character <= "\u9fff" for character in term)


def test_question_names_document_requires_a_whole_title_segment():
    """Naming a memory lifts the cross-task scope gate; topical overlap does not.

    Live case: a question quoting "笔记资料整理" was blocked because the
    reflection carried another task's category even though the judge picked
    it. A single shared bigram must not count as naming the document.
    """
    from types import SimpleNamespace

    from qwen_exo_booster.knowledge import question_names_document

    notes = SimpleNamespace(title="笔记资料整理：交付物观测与验收边界")
    lint = SimpleNamespace(title="TS 规则提交前门禁误判：tsc 错误计数未闭环")
    latin = SimpleNamespace(title="Mashumaro flatten task contract")

    assert question_names_document(
        "在之前做笔记资料整理的时候 我们遇到什么问题了？", notes
    )
    assert not question_names_document(
        "在之前做笔记资料整理的时候 我们遇到什么问题了？", lint
    )
    assert not question_names_document("整理一下今天的资料", notes)
    assert question_names_document(
        "what did the mashumaro flatten task contract say?", latin
    )
    assert not question_names_document("", notes)


def _reflection_text_runtime(tmp_path):
    runtime = object.__new__(QwenExoRuntime)
    runtime.knowledge = KnowledgeRepository(tmp_path / "knowledge")
    runtime.file_knowledge = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    runtime.document_categories = DocumentCategoryStore(
        tmp_path / "state" / "document-categories.sqlite3"
    )
    runtime.policy_data = SimpleNamespace(snapshot=SimpleNamespace(documents=()))
    runtime.reflection_memory_store = SimpleNamespace(list=lambda: [])
    runtime.telemetry = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._reflection_publish_lock = asyncio.Lock()
    return runtime


def _network_reflection(**overrides):
    values = dict(
        trajectory_id="resp-network",
        conversation_key="conversation-network",
        source_digest="source-network",
        title="Observe before changing HTTP clients",
        outcome="mixed",
        reflection="The first controllable error was repeating requests without classifying the transport signal.",
        evidence="The same WinError 10106 repeated while a curl request returned a concrete HTTP response.",
        causal_analysis="The repeated loop added no information; a one-request client comparison would discriminate environment from target behavior.",
        reusable_experience="When clients disagree on one endpoint, compare one matched request and inspect error classes before choosing a bulk strategy.",
        avoid="Do not enumerate hundreds of payloads after identical client-level failures.",
        next_time="Observe one baseline, run one matched probe, then verify one modified request.",
        memory_action="insert",
        target_document_path=None,
        target_document_sha256=None,
        source_event_count=4,
        source_token_count=1024,
        attempts=1,
        created_at=1.0,
    )
    values.update(overrides)
    return ReflectionMemory(**values)


@pytest.mark.asyncio
async def test_runtime_reflection_publishes_text_memory_for_memory_tools(tmp_path):
    runtime = _reflection_text_runtime(tmp_path)

    payload = await runtime._publish_reflection_memory(_network_reflection())

    path = payload["document_path"]
    assert path.startswith("reflection-memory/") and path.endswith(".md")
    assert (tmp_path / "knowledge" / path).is_file()
    assert payload["native_source_digest"] == runtime._file_knowledge_digest()
    hits = runtime.file_knowledge.grep("WinError 10106")
    assert [hit["path"] for hit in hits["hits"]] == [path]


@pytest.mark.asyncio
async def test_runtime_reflection_update_replaces_text_memory_and_rejects_stale(
    tmp_path,
):
    runtime = _reflection_text_runtime(tmp_path)
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
    reflection = _network_reflection(
        memory_action="update",
        target_document_path=previous.relative_path,
        target_document_sha256=previous.sha256,
        retrieval_category="reflection-task-new-run-deadbeef",
    )

    payload = await runtime._publish_reflection_memory(reflection)

    documents = runtime.knowledge.snapshot.documents
    assert [document.relative_path for document in documents] == [previous.relative_path]
    assert "Observe before changing HTTP clients" in documents[0].content
    assert documents[0].retrieval_category == "network-debugging"
    assert payload["replaced_document_sha256"] == previous.sha256
    assert runtime.file_knowledge.grep("Old process lesson")["hits"] == []
    with pytest.raises(RuntimeError, match="target is stale"):
        await runtime._publish_reflection_memory(
            replace(reflection, target_document_sha256="stale-sha")
        )


@pytest.mark.asyncio
async def test_similar_reflection_lookup_is_lexical_and_skips_unrelated(tmp_path):
    runtime = _reflection_text_runtime(tmp_path)
    related = await runtime._publish_reflection_memory(_network_reflection())
    await runtime._publish_reflection_memory(
        _network_reflection(
            source_digest="source-css",
            title="Pin the stylesheet build before diffing screenshots",
            reflection="Screenshots drifted because the CSS bundle was rebuilt.",
            evidence="Two snapshots differed only in font hinting.",
            causal_analysis="Unpinned bundler output changed between runs.",
            reusable_experience="Freeze the stylesheet build before visual diffs.",
            avoid="Do not compare screenshots across bundler versions.",
            next_time="Pin the bundler, rebuild once, then diff.",
        )
    )

    candidates = await runtime._retrieve_reflection_memory_candidates(
        "parent", "HTTP client WinError transport request comparison"
    )

    assert [candidate.document_path for candidate in candidates] == [
        related["document_path"]
    ]
    assert 0 < candidates[0].tensor_score <= 1
