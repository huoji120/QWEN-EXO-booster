import base64
import gzip
import json

import pytest

from qwen_exo_booster.trajectory_store import (
    TrajectoryStore,
    TrajectoryStoreError,
    normalize_chatml,
    parse_trajectory_upload,
)

MESSAGES = [
    {"role": "system", "content": "你是助手。"},
    {"role": "user", "content": "开始任务。"},
    {"role": "assistant", "content": "收到，开始。"},
]


def test_normalize_chatml_accepts_session_and_raw_list():
    wrapped = normalize_chatml({"session": {"messages": MESSAGES}})
    assert len(wrapped["session"]["messages"]) == 3
    raw = normalize_chatml(MESSAGES)
    assert len(raw["session"]["messages"]) == 3


def test_normalize_chatml_rejects_missing_assistant():
    with pytest.raises(TrajectoryStoreError, match="助手"):
        normalize_chatml(MESSAGES[:2])


def test_parse_jsonl_upload():
    data = "\n".join(json.dumps(m, ensure_ascii=False) for m in MESSAGES).encode()
    parsed = parse_trajectory_upload("trace.jsonl", data)
    assert len(parsed["session"]["messages"]) == 3


def test_parse_gzip_upload():
    data = gzip.compress(
        json.dumps({"session": {"messages": MESSAGES}}, ensure_ascii=False).encode()
    )
    parsed = parse_trajectory_upload("trace.json.gz", data)
    assert len(parsed["session"]["messages"]) == 3


def test_parse_rejects_bad_role_and_bad_json():
    with pytest.raises(TrajectoryStoreError):
        normalize_chatml([{"role": "hacker", "content": "x"}, *MESSAGES])
    with pytest.raises(TrajectoryStoreError):
        parse_trajectory_upload("trace.json", b"not json")


def test_store_roundtrip_and_delete(tmp_path):
    store = TrajectoryStore(tmp_path)
    result = store.save("ctf-run", {"session": {"messages": MESSAGES}})
    assert result["messages"] == 3
    assert store.list()[0]["name"] == "ctf-run"
    assert store.get("ctf-run")["messages"] == 3
    store.delete("ctf-run")
    assert store.list() == []
    with pytest.raises(TrajectoryStoreError):
        store.get("ctf-run")


def test_store_requires_custom_name_and_tags_then_allows_tagged_edits(tmp_path):
    store = TrajectoryStore(tmp_path)
    payload = {"session": {"messages": MESSAGES}}

    legacy = store.save("legacy-run", payload)
    assert legacy["tags"] == []
    assert store.list()[0]["tags"] == []

    with pytest.raises(TrajectoryStoreError, match="至少填写一个标签"):
        store.create("custom-run", payload, tags=[])
    with pytest.raises(TrajectoryStoreError, match="轨迹名称"):
        store.create("", payload, tags=["coding"])

    created = store.create(
        "custom-run",
        payload,
        tags=["coding", "success", "Coding"],
    )
    assert created["name"] == "custom-run"
    assert created["tags"] == ["coding", "success"]
    assert store.get("custom-run")["tags"] == ["coding", "success"]

    edited_messages = [*MESSAGES, {"role": "user", "content": "复查结果。"}]
    edited = store.save(
        "custom-run",
        {"session": {"messages": edited_messages}},
        tags=["reviewed"],
    )
    assert edited["messages"] == 4
    assert edited["tags"] == ["reviewed"]
    assert store.get("custom-run")["tags"] == ["reviewed"]


def test_store_rejects_bad_names(tmp_path):
    store = TrajectoryStore(tmp_path)
    with pytest.raises(TrajectoryStoreError):
        store.save("../escape", {"session": {"messages": MESSAGES}})


def test_upload_base64_flow(tmp_path):
    payload = json.dumps({"session": {"messages": MESSAGES}}, ensure_ascii=False)
    data = base64.b64encode(payload.encode()).decode()
    parsed = parse_trajectory_upload("trace.json", base64.b64decode(data))
    assert TrajectoryStore(tmp_path).save("t", parsed)["messages"] == 3
