from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import Counter, OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.document_categories import DocumentCategoryStore
from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
from qwen_exo_booster.initial_gdn_request import (
    INITIAL_GDN_PARAM,
    bind_initial_gdn_request,
)
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.memory_hints import MemoryHintIndex
from qwen_exo_booster.reflection_evidence import ReflectionEvidenceStore
from qwen_exo_booster.reflection_memory import (
    ReflectionMemory,
    ReflectionMemoryStore,
    ReflectionSourceSnapshot,
    ReflectionSourceStore,
)
from qwen_exo_booster.runtime import (
    MemoryScopeError,
    QwenExoRuntime,
    _memory_identity_from_conversation,
    _memory_key_from_conversation,
    _scope_conversation_key,
)


KEY_A = "qxk_aaaaaaaaaaaaaaaa"
KEY_B = "qxk_bbbbbbbbbbbbbbbb"


class _Tokenizer:
    def encode(self, text, **kwargs):
        return list(str(text).encode())

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


def _runtime(tmp_path):
    runtime = object.__new__(QwenExoRuntime)
    runtime.config = SimpleNamespace(
        knowledge_directory=tmp_path / "knowledge",
        state_directory=tmp_path / "state",
        context_length=32768,
    )
    runtime.knowledge = KnowledgeRepository(runtime.config.knowledge_directory)
    runtime.file_knowledge = FileKnowledgeIndex(
        runtime.config.knowledge_directory, runtime.config.state_directory
    )
    runtime.document_categories = DocumentCategoryStore(tmp_path / "state" / "categories.sqlite3")
    runtime.policy_data = SimpleNamespace(snapshot=SimpleNamespace(documents=()))
    runtime.reflection_memory_store = ReflectionMemoryStore(tmp_path / "state" / "reflections.json")
    runtime.reflection_source_store = ReflectionSourceStore(tmp_path / "state" / "sources.sqlite3")
    runtime.reflection_evidence_store = ReflectionEvidenceStore(tmp_path / "state" / "evidence.sqlite3")
    runtime.telemetry = SimpleNamespace(emit=lambda *args, **kwargs: None)
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=_Tokenizer())
    runtime.model_identity = SimpleNamespace(fingerprint="scope-test-model")
    runtime._reflection_publish_lock = asyncio.Lock()
    runtime._session_initial_gdn_value_lock = threading.RLock()
    runtime._session_initial_gdn_pins = OrderedDict()
    runtime._session_initial_gdn_value = {"source_digest": "a" * 64, "state_identity": "b" * 64}
    runtime._request_conversation_keys = {}
    runtime._conversation_keys_by_response_id = OrderedDict()
    runtime._compaction_summaries = OrderedDict()
    runtime._max_conversation_keys = 32
    runtime._memory_tool_stats = Counter()
    runtime._memory_block_hints = OrderedDict()
    runtime._memory_block_turns = OrderedDict()
    runtime.memory_hints = MemoryHintIndex(tmp_path / "knowledge", tmp_path / "state" / "hints.sqlite3")
    runtime._memory_hints_fingerprint = None
    runtime._memory_hints_refresh = None
    return runtime


def _request(request_id, owner, *, cache_key=None, text="Identical task", mode=None):
    return SimpleNamespace(
        request_id=request_id, input=text, prompt_cache_key=cache_key,
        _qwen_exo_memory_key_id=owner, user="same-user", session_id="same-session",
        _qwen_exo_memory_scope=mode or ("private" if owner else "shared"),
    )


def _resolve(runtime, request, previous=None, calls=()):
    return runtime._response_conversation_key(
        request_id=request.request_id, previous_response_id=previous,
        request=request, canonical_identity=runtime._canonical_response_identity(request),
        call_ids=calls,
    )


def _reflection(owner=None, source="source", marker="TRANSPORT_EVIDENCE", mode=None):
    return ReflectionMemory(
        trajectory_id="response", conversation_key=_scope_conversation_key("conversation", owner, mode),
        source_digest=source, title="Compare transport clients", outcome="mixed",
        reflection=f"Transport client comparison {marker}",
        evidence="Matched HTTP requests isolate transport failures.",
        causal_analysis="One controlled transport comparison discriminates client failure.",
        reusable_experience="Compare HTTP client transport observations before changes.",
        avoid="Avoid repeating identical failing requests.", next_time="Compare then verify.",
        memory_action="insert", target_document_path=None, target_document_sha256=None,
        source_event_count=1, source_token_count=20, attempts=1, created_at=1.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, KEY_A])
async def test_transfer_detaches_reflection_without_reassigning_transcript(tmp_path, owner):
    runtime = _runtime(tmp_path)
    runtime._session_initial_gdn_status = {"status": "ready"}
    runtime._schedule_session_initial_gdn_refresh = lambda **kwargs: None
    reflection = _reflection(owner)
    published = await runtime._publish_reflection_memory(reflection)
    stored = replace(reflection, document_path=published["document_path"],
                     document_sha256=published["document_sha256"], publication_status="published")
    runtime.reflection_memory_store.append(stored)
    index = runtime.file_knowledge if owner is None else runtime._private_memory_resources(owner)[0]
    prefix = f"private/{owner}/" if owner is not None else ""
    relative = published["document_path"][len(prefix):]
    destination = runtime._private_memory_resources(KEY_B)[0]
    async with runtime.memory_transfer_guard({"source_key_id": owner, "destination_key_id": KEY_B,
                                             "source": relative, "destination": "received.md"}):
        (index.root / relative).rename(destination.root / "received.md")
    if runtime._memory_hints_refresh is not None:
        await runtime._memory_hints_refresh
    record = runtime.reflection_memory_store.get(stored.source_digest)
    assert record["memory_transfer_detached"] is True
    assert record["conversation_key"] == stored.conversation_key
    assert record["document_path"] == stored.document_path
    assert (destination.root / "received.md").is_file()
    with pytest.raises(RuntimeError, match="detached"):
        runtime._reflection_regeneration_target(record, stored.document_sha256)
    with pytest.raises(RuntimeError, match="detached"):
        await runtime._publish_reflection_memory(reflection)
    with pytest.raises(RuntimeError, match="stale"):
        await runtime._publish_reflection_memory(replace(
            reflection, source_digest="in-flight-new-source", memory_action="update",
            target_document_path=stored.document_path,
            target_document_sha256=stored.document_sha256,
        ))
    with pytest.raises(RuntimeError, match="detached"):
        runtime.reflection_memory_store.append(stored)
    assert not (index.root / relative).exists()
    reloaded = ReflectionMemoryStore(runtime.reflection_memory_store.path)
    assert reloaded.get(stored.source_digest)["memory_transfer_detached"] is True


@pytest.mark.asyncio
async def test_transfer_revokes_new_gdn_bindings_and_stale_generation(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._session_initial_gdn_status = {"status": "ready"}
    runtime._schedule_session_initial_gdn_refresh = lambda **kwargs: None
    reflection = _reflection(marker="SHARED_NOW_PRIVATE")
    published = await runtime._publish_reflection_memory(reflection)
    runtime.reflection_memory_store.append(replace(
        reflection, document_path=published["document_path"],
        document_sha256=published["document_sha256"], publication_status="published",
    ))
    old_prompt = runtime._build_session_initial_gdn_prompt()
    assert "SHARED_NOW_PRIVATE" in old_prompt["prompt"]
    assert runtime.initial_gdn_selection(response_id="before") is not None
    destination = runtime._private_memory_resources(KEY_A)[0]
    async with runtime.memory_transfer_guard({"source_key_id": None, "destination_key_id": KEY_A,
                                             "source": published["document_path"], "destination": "received.md"}):
        assert runtime.initial_gdn_selection(response_id="during") is None
        (runtime.file_knowledge.root / published["document_path"]).rename(destination.root / "received.md")
    await runtime._memory_hints_refresh
    assert runtime.initial_gdn_selection(response_id="after") is None
    assert runtime.initial_gdn_selection(previous_response_id="before", response_id="continuing") is None
    assert not runtime._session_initial_gdn_pins
    assert runtime._build_session_initial_gdn_prompt() is None
    with pytest.raises(RuntimeError, match="source changed"):
        runtime._adopt_session_initial_gdn(old_prompt, {}, reason="late_worker", loaded_from_artifact=True)
    assert runtime._session_initial_gdn_value is None


@pytest.mark.asyncio
async def test_transfer_drops_lookup_already_waiting_on_old_hint_snapshot(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._session_initial_gdn_status = {"status": "ready"}
    runtime._schedule_session_initial_gdn_refresh = lambda **kwargs: None
    runtime.file_knowledge.upsert("secret.md", b"Private transport client evidence")
    runtime.file_knowledge.upsert("background.md", b"Unrelated gardening flowers and soil")
    runtime.memory_hints.refresh(runtime.file_knowledge.metadata())
    runtime._refresh_memory_hints_if_stale = lambda: None
    entered, resume = threading.Event(), threading.Event()
    lookup = runtime.memory_hints.lookup
    assert lookup("private transport client evidence")

    def waiting_lookup(*args, **kwargs):
        old_hints = lookup(*args, **kwargs)
        entered.set()
        assert resume.wait(5)
        return old_hints

    runtime.memory_hints.lookup = waiting_lookup
    task = asyncio.create_task(runtime.memory_block_for_turn(
        "waiting", blocked_reason=None, query_text="private transport client evidence",
    ))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        destination = runtime._private_memory_resources(KEY_A)[0]
        async with runtime.memory_transfer_guard({"source_key_id": None, "destination_key_id": KEY_A,
                                                 "source": "secret.md", "destination": "received.md"}):
            (runtime.file_knowledge.root / "secret.md").rename(destination.root / "received.md")
    finally:
        resume.set()
    assert "secret.md" not in await task
    runtime.memory_hints.lookup = lookup
    assert runtime.memory_hints.lookup("private transport client evidence") == []


@pytest.mark.asyncio
async def test_transfer_cancellation_drains_finalization_before_unlock(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._session_initial_gdn_status = {"status": "ready"}
    scheduled = []
    runtime._schedule_session_initial_gdn_refresh = lambda **kwargs: scheduled.append(kwargs)
    reflection = _reflection()
    published = await runtime._publish_reflection_memory(reflection)
    runtime.reflection_memory_store.append(replace(
        reflection, document_path=published["document_path"],
        document_sha256=published["document_sha256"], publication_status="published",
    ))
    destination = runtime._private_memory_resources(KEY_A)[0]
    entered, resume = threading.Event(), threading.Event()
    invalidate = runtime.memory_hints.invalidate
    calls = 0

    def delayed_finalize_invalidation():
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert resume.wait(5)
        invalidate()

    runtime.memory_hints.invalidate = delayed_finalize_invalidation

    async def transfer():
        async with runtime.memory_transfer_guard({"source_key_id": None, "destination_key_id": KEY_A,
                                                 "source": published["document_path"], "destination": "received.md"}):
            (runtime.file_knowledge.root / published["document_path"]).rename(destination.root / "received.md")

    task = asyncio.create_task(transfer())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert runtime._reflection_publish_lock.locked()
            assert runtime._memory_transfer_owners == {None, KEY_A}
        assert (destination.root / "received.md").is_file()
    finally:
        resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime._memory_hints_refresh
    assert not runtime._reflection_publish_lock.locked()
    assert runtime._memory_transfer_owners == set()
    assert runtime.reflection_memory_store.get(reflection.source_digest)["memory_transfer_detached"]
    assert runtime.initial_gdn_selection(response_id="after-cancel") is None
    assert scheduled == [{"reason": "memory_transfer"}]
    with pytest.raises(RuntimeError, match="detached"):
        await runtime._publish_reflection_memory(reflection)


@pytest.mark.parametrize("cache_key", [None, "identical-client-cache"])
def test_identical_heads_cache_keys_and_calls_are_isolated(tmp_path, cache_key):
    runtime = _runtime(tmp_path)
    keys = {}
    for number, owner in enumerate((None, KEY_A, KEY_B)):
        request = _request(f"head-{number}", owner, cache_key=cache_key)
        key = _resolve(runtime, request, calls=("same-call",))
        keys[owner] = key
        assert _memory_key_from_conversation(key) == owner
        continuation = _request(f"tool-{number}", owner, text=[])
        assert _resolve(runtime, continuation, calls=("same-call",)) == key
        assert _resolve(runtime, _request(f"lineage-{number}", owner), previous=request.request_id) == key
    assert len(set(keys.values())) == 3
    assert runtime._prompt_cache_conversation_key(_request("r", None, cache_key="x")) != runtime._prompt_cache_conversation_key(_request("r", KEY_A, cache_key="x"))


def test_foreign_lineage_and_request_id_reuse_fail_closed(tmp_path):
    runtime = _runtime(tmp_path)
    _resolve(runtime, _request("owned", KEY_A))
    for owner in (None, KEY_B):
        with pytest.raises(MemoryScopeError):
            _resolve(runtime, _request("new", owner), previous="owned")
        with pytest.raises(MemoryScopeError):
            _resolve(runtime, _request("owned", owner))
    assert runtime.response_memory_scope("owned") == (True, KEY_A, "private")
    assert runtime.response_memory_scope("missing") == (False, None, "shared")
    # Switching a key to shared does not make its former conversation shared.
    assert _resolve(runtime, _request("new-shared", None)) != _resolve(runtime, _request("new-private", KEY_A))


def test_compaction_scope_is_checked_even_without_previous_response_id(tmp_path):
    runtime = _runtime(tmp_path)
    payload = dict(schema="qwen-exo-response-compaction-v1", response_id="compact", source_digest="digest", summary="PRIVATE SUMMARY")
    encoded = "qwen-exo-v1." + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    items = [{"type": "compaction", "encrypted_content": encoded}]
    runtime._compaction_summaries["compact"] = {**payload, "memory_key_id": KEY_A}
    assert runtime._verified_response_compaction_envelope(items, memory_key_id=KEY_A) == payload
    for owner in (None, KEY_B):
        with pytest.raises(MemoryScopeError):
            runtime._verified_response_compaction_envelope(items, memory_key_id=owner)
    assert runtime.response_memory_scope("compact") == (True, KEY_A, "private")
    runtime._compaction_summaries["compact"].pop("memory_key_id")
    assert runtime._verified_response_compaction_envelope(items) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private", "private_only", "super"])
async def test_private_publication_retrieval_and_regeneration_stay_owned(tmp_path, mode):
    runtime = _runtime(tmp_path)
    published = {}
    for owner, marker in ((None, "SHARED_MARKER"), (KEY_A, "PRIVATE_A_MARKER"), (KEY_B, "PRIVATE_B_MARKER")):
        reflection = _reflection(owner, marker=marker, mode=mode if owner else "shared")
        result = await runtime._publish_reflection_memory(reflection)
        stored = replace(reflection, document_path=result["document_path"], document_sha256=result["document_sha256"], publication_status="published")
        runtime.reflection_memory_store.append(stored)
        published[owner] = (stored, result)
    assert len(runtime.reflection_memory_store.list()) == 3
    assert len(runtime.knowledge.snapshot.documents) == 1
    assert not runtime.file_knowledge.grep("PRIVATE_A_MARKER")['hits']
    assert not runtime.file_knowledge.grep("PRIVATE_B_MARKER")['hits']
    private_index, _, _ = runtime._private_memory_resources(KEY_A)
    assert private_index.grep("PRIVATE_A_MARKER")['hits']
    assert not private_index.grep("PRIVATE_B_MARKER")['hits']
    assert all("private" not in row["relative_path"] for row in runtime.file_knowledge.metadata())
    for owner in (None, KEY_A, KEY_B):
        candidates = await runtime._retrieve_reflection_memory_candidates("parent", "transport client comparison", memory_key_id=owner)
        assert [item.document_path for item in candidates] == [published[owner][1]["document_path"]]
    for target_owner in (None, KEY_B):
        target = published[target_owner][1]
        with pytest.raises(MemoryScopeError):
            await runtime._publish_reflection_memory(replace(
                _reflection(KEY_A, mode=mode), memory_action="update", target_document_path=target["document_path"],
                target_document_sha256=target["document_sha256"],
            ))
    stored, result = published[KEY_A]
    updated = await runtime._publish_reflection_memory(replace(
        stored, memory_action="update", source_digest="replacement",
        target_document_path=result["document_path"], target_document_sha256=result["document_sha256"],
        reflection="Owned update remains private after key setting changes.",
    ))
    target = runtime._reflection_regeneration_target(
        {**stored.public_dict(), "document_sha256": updated["document_sha256"]}, updated["document_sha256"]
    )
    assert target.document_path == result["document_path"]
    # Ownership is persisted, not held only in mutable runtime/key settings.
    reloaded = ReflectionMemoryStore(runtime.reflection_memory_store.path)
    assert {_memory_key_from_conversation(record["conversation_key"]) for record in reloaded.list()} == {None, KEY_A, KEY_B}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private", "private_only", "super"])
async def test_private_records_never_feed_global_gdn_or_refresh(tmp_path, mode):
    runtime = _runtime(tmp_path)
    shared = _reflection(marker="PUBLIC_GDN_MARKER")
    private = _reflection(KEY_A, marker="PRIVATE_GDN_MARKER", mode=mode)
    runtime.reflection_memory_store.append(shared)
    runtime.reflection_memory_store.append(private)
    prompt = runtime._build_session_initial_gdn_prompt()
    assert "PUBLIC_GDN_MARKER" in prompt["prompt"]
    assert "PRIVATE_GDN_MARKER" not in prompt["prompt"]
    assert prompt["memory_count"] == 1
    refreshes = []
    runtime._schedule_session_initial_gdn_refresh = lambda **kwargs: refreshes.append(kwargs)
    await runtime._on_reflection_memory_stored(private)
    assert not refreshes
    await runtime._on_reflection_memory_stored(shared)
    assert refreshes == [{"reason": "new_memory"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private", "private_only", "super"])
async def test_private_hints_read_shared_and_own_with_executor_paths(tmp_path, mode):
    runtime = _runtime(tmp_path)
    for owner, marker in ((None, "SHARED_HINT"), (KEY_A, "PRIVATE_A_HINT"), (KEY_B, "PRIVATE_B_HINT")):
        index = runtime.file_knowledge if owner is None else runtime._private_memory_resources(owner)[0]
        (index.root / "target.md").write_text(
            f"# Transport debugging\nHTTP client transport comparison {marker}.\n", encoding="utf-8"
        )
        for number in range(3):
            (index.root / f"unrelated-{number}.md").write_text(
                f"# Botanical inventory {number}\nOrchids blossoms petals and stems.\n", encoding="utf-8"
            )
        index.refresh()
    runtime.memory_hints.refresh(runtime.file_knowledge.metadata())
    for number, owner in enumerate((None, KEY_A, KEY_B)):
        request_id = f"hints-{number}"
        scope = mode if owner else "shared"
        runtime._request_conversation_keys[request_id] = _scope_conversation_key("hint-conversation", owner, scope)
        block = await runtime.memory_block_for_turn(request_id, blocked_reason=None, query_text="HTTP client transport comparison", memory_key_id=owner, memory_scope=scope)
        assert ("SHARED_HINT" in block) == (scope != "private_only")
        assert ("PRIVATE_A_HINT" in block) == (owner == KEY_A or scope == "super")
        assert ("PRIVATE_B_HINT" in block) == (owner == KEY_B or scope == "super")
        if owner is not None:
            assert "private/target.md" in block
            assert owner not in block
            if scope == "super":
                other = KEY_B if owner == KEY_A else KEY_A
                assert f"private-keys/{other}/target.md" in block
    with pytest.raises(MemoryScopeError):
        await runtime.memory_block_for_turn("hints-1", blocked_reason=None, query_text="transport", memory_key_id=KEY_B)
    if runtime._memory_hints_refresh is not None:
        await runtime._memory_hints_refresh


def test_regeneration_rejects_cross_owner_source(tmp_path):
    runtime = _runtime(tmp_path)
    snapshot = ReflectionSourceSnapshot(
        source_digest="snapshot", trajectory_id="r", conversation_key=_scope_conversation_key("c", KEY_A),
        original_task="private", trajectory_history=(), capsule_history=(), verifier_feedback="",
        source_event_count=0, source_token_count=1, source_audit={}, captured_at=1.0,
    )
    runtime.reflection_source_store.save(snapshot)
    with pytest.raises(MemoryScopeError):
        runtime._load_reflection_source({"source_digest": "snapshot", "conversation_key": "shared"})
    assert runtime._load_reflection_source({"source_digest": "snapshot", "conversation_key": snapshot.conversation_key}) == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private", "private_only", "super"])
async def test_background_retrieval_callback_captures_original_owner(tmp_path, mode):
    runtime = _runtime(tmp_path)
    runtime.reflection_memory_service = SimpleNamespace(retrieve_similar=runtime._retrieve_reflection_memory_candidates)
    for owner in (None, KEY_A, KEY_B):
        await runtime._publish_reflection_memory(_reflection(owner))
    service_a = runtime._reflection_service_for_conversation(_scope_conversation_key("c", KEY_A, mode))
    service_b = runtime._reflection_service_for_conversation(_scope_conversation_key("c", KEY_B, mode))
    shared_service = runtime._reflection_service_for_conversation("c")
    for service, prefix in ((service_a, f"private/{KEY_A}/"), (service_b, f"private/{KEY_B}/"), (shared_service, "reflection-memory/")):
        candidates = await service.retrieve_similar("same-parent", "transport client comparison")
        assert candidates and all(candidate.document_path.startswith(prefix) for candidate in candidates)


@pytest.mark.parametrize("cache_key", [None, "same-cache"])
def test_mode_change_isolates_canonical_cache_calls_and_retained_responses(tmp_path, cache_key):
    runtime = _runtime(tmp_path)
    conversations = {}
    for mode in ("private", "private_only", "super"):
        request = _request(mode, KEY_A, mode=mode, cache_key=cache_key)
        key = _resolve(runtime, request, calls=("same-call",))
        conversations[mode] = key
        assert _memory_identity_from_conversation(key) == (KEY_A, mode)
        assert runtime.response_memory_scope(mode) == (True, KEY_A, mode)
        tool = _request(mode + "-tool", KEY_A, mode=mode, text=[])
        assert _resolve(runtime, tool, calls=("same-call",)) == key
    assert len(set(conversations.values())) == 3
    for previous in ("private", "super"):
        with pytest.raises(MemoryScopeError):
            _resolve(runtime, _request("downgrade", KEY_A, mode="private_only"), previous=previous)
        with pytest.raises(MemoryScopeError):
            _resolve(runtime, _request(previous, KEY_A, mode="private_only"))
    assert _memory_identity_from_conversation(f"memory-private:{KEY_A}:old") == (KEY_A, "private")


@pytest.mark.parametrize("previous_mode", ["private", "super"])
def test_compaction_mode_downgrade_fails_without_explicit_lineage(tmp_path, previous_mode):
    runtime = _runtime(tmp_path)
    payload = dict(schema="qwen-exo-response-compaction-v1", response_id="compact", source_digest="digest", summary="SHARED OR OTHER MEMORY")
    encoded = "qwen-exo-v1." + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    items = [{"type": "compaction", "encrypted_content": encoded}]
    runtime._compaction_summaries["compact"] = {
        **payload, "memory_key_id": KEY_A, "memory_scope": previous_mode,
    }
    assert runtime._verified_response_compaction_envelope(items, memory_key_id=KEY_A, memory_scope=previous_mode) == payload
    with pytest.raises(MemoryScopeError):
        runtime._verified_response_compaction_envelope(items, memory_key_id=KEY_A, memory_scope="private_only")


def test_private_only_suppresses_shared_native_and_latent_memory(tmp_path):
    runtime = _runtime(tmp_path)
    namespaces = set()
    for owner, mode in ((None, "shared"), (KEY_A, "private"), (KEY_A, "private_only"), (KEY_A, "super"), (KEY_B, "private_only")):
        params, namespace = bind_initial_gdn_request(
            runtime, custom_params={INITIAL_GDN_PARAM: {"state_identity": "forged"}},
            extra_key="same-client-key", response_id=f"{owner}-{mode}",
            memory_key_id=owner, memory_scope=mode,
        )
        namespaces.add(namespace)
        assert (INITIAL_GDN_PARAM in params) == (mode != "private_only")
    assert len(namespaces) == 5
    assert runtime.initial_gdn_selection(memory_key_id=KEY_A, memory_scope="private_only") is None
    request = _request("latent", KEY_A, mode="private_only")
    request.metadata = {"qwen_exo_latent_transplant": "shared-artifact"}
    assert runtime.latent_transplant_payload(request) is None
    for mode in ("private", "super"):
        with pytest.raises(MemoryScopeError):
            bind_initial_gdn_request(
                runtime, custom_params={}, extra_key=None,
                previous_response_id=f"{KEY_A}-{mode}",
                memory_key_id=KEY_A, memory_scope="private_only",
            )


def test_regeneration_rejects_same_owner_cross_mode_source(tmp_path):
    runtime = _runtime(tmp_path)
    snapshot = ReflectionSourceSnapshot(
        source_digest="snapshot", trajectory_id="r",
        conversation_key=_scope_conversation_key("c", KEY_A, "super"),
        original_task="private", trajectory_history=(), capsule_history=(), verifier_feedback="",
        source_event_count=0, source_token_count=1, source_audit={}, captured_at=1.0,
    )
    runtime.reflection_source_store.save(snapshot)
    with pytest.raises(MemoryScopeError):
        runtime._load_reflection_source({
            "source_digest": "snapshot",
            "conversation_key": _scope_conversation_key("c", KEY_A, "private_only"),
        })


@pytest.mark.asyncio
async def test_private_only_does_not_query_or_refresh_shared_hints(tmp_path):
    runtime = _runtime(tmp_path)
    accesses = []

    def shared_access(*args, **kwargs):
        accesses.append(True)
        raise AssertionError("Shared hints must not be accessed")

    runtime.memory_hints.lookup = shared_access
    runtime._refresh_memory_hints_if_stale = shared_access
    runtime._request_conversation_keys["only"] = _scope_conversation_key("c", KEY_A, "private_only")
    block = await runtime.memory_block_for_turn(
        "only", blocked_reason=None, query_text="transport comparison",
        memory_key_id=KEY_A, memory_scope="private_only",
    )
    assert block
    assert not accesses


def test_private_only_disables_explicit_and_default_worker_editor(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    result = runtime.activation_editor_request(
        {"editor": "shared-trajectory"}, memory_scope="private_only"
    )
    assert result["spec"] is None
    params, _ = bind_initial_gdn_request(
        runtime,
        custom_params={
            "qwen_exo_activation_editor": {"editor": "shared-trajectory"},
            "qwen_exo_latent_transplant": {"artifact": "shared-latent"},
            INITIAL_GDN_PARAM: {"state_identity": "forged"},
        },
        extra_key=None, memory_key_id=KEY_A, memory_scope="private_only",
    )
    from sglang.srt.model_executor import forward_batch_info as worker

    active = tmp_path / "activation-editors" / "active.json"
    active.parent.mkdir()
    active.write_text(json.dumps({"editor": "shared-trajectory", "mode": "active"}))
    monkeypatch.setattr(worker, "get_server_args", lambda: SimpleNamespace(
        qwen_exo_state_dir=str(tmp_path), qwen_exo_activation_editor_enabled=True,
        qwen_exo_activation_editor_strength=1.0,
    ))
    normal = SimpleNamespace(sampling_params=SimpleNamespace(custom_params={"qwen_exo_kind": "user"}))
    assert worker._qwen_exo_activation_editor(normal)["editor"] == "shared-trajectory"
    strict = SimpleNamespace(sampling_params=SimpleNamespace(custom_params={
        **params, "qwen_exo_kind": "user",
    }))
    assert worker._qwen_exo_activation_editor(strict) is None
    assert worker._qwen_exo_latent_transplant(strict) is None
    strict.sampling_params.custom_params["qwen_exo_activation_editor"] = {"editor": "shared-trajectory"}
    assert worker._qwen_exo_activation_editor(strict) is None
