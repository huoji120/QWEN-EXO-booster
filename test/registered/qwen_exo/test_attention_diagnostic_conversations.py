from types import SimpleNamespace


from qwen_exo_booster.attention_diagnostic import parse_attention_upload
from qwen_exo_booster.attention_diagnostic_conversations import (
    AttentionDiagnosticConversations,
)
from qwen_exo_booster.reflection_evidence import ReflectionEvidenceStore


def test_recent_import_uses_activity_not_per_conversation_ordinal(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "journal.sqlite3")
    store.append_rows(
        "long-old",
        "r-old",
        [{"kind": "user_context", "content": str(i)} for i in range(8)],
    )
    store.append_rows(
        "short-new", "r-new", [{"kind": "user_context", "content": "fresh"}]
    )
    with store._lock, store._connect() as db:
        db.execute(
            "UPDATE conversations SET last_seen=10 WHERE conversation_key='long-old'"
        )
        db.execute(
            "UPDATE conversations SET last_seen=20 WHERE conversation_key='short-new'"
        )
    service = AttentionDiagnosticConversations(
        SimpleNamespace(reflection_evidence_store=store)
    )
    first = service.list(limit=1, offset=0)
    assert first["conversations"][0]["title"] == "short-new"
    assert first["total"] == 2
    assert service.list(limit=1, offset=1)["conversations"][0]["title"] == "long-old"


def test_import_retains_tool_identity_and_is_read_only(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "journal.sqlite3")
    store.append_rows(
        "conv",
        "r",
        [
            {"kind": "user_context", "content": "Read the source."},
            {
                "kind": "tool_action",
                "tool_name": "read",
                "call_id": "c-1",
                "content": '{"path":"source.py"}',
            },
            {
                "kind": "tool_observation",
                "tool_name": "read",
                "call_id": "c-1",
                "content": "中文😀",
            },
        ],
    )
    before = store.rows("conv")
    service = AttentionDiagnosticConversations(
        SimpleNamespace(reflection_evidence_store=store)
    )
    detail = service.detail(service.list()["conversations"][0]["id"])
    parsed = parse_attention_upload(detail["content"])
    assert [m["role"] for m in parsed.messages] == ["user", "assistant", "tool"]
    assert "source.py" in parsed.messages[1]["content"]
    assert "c-1" in parsed.messages[1]["content"]
    assert "中文😀" in parsed.messages[2]["content"]
    assert parsed.messages[2]["tool_call_id"] == "c-1"
    assert store.rows("conv") == before
    assert detail["warnings"]
    assert service.detail("journal:missing") is None
