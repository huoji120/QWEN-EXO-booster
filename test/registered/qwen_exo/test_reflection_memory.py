from __future__ import annotations
import asyncio
import json
from dataclasses import replace

import pytest
from qwen_exo_booster.internal_jobs import InternalJobResult

from qwen_exo_booster.reflection_memory import (
    REFLECTION_MEMORY_TOOL_NAME,
    ReflectionMemory,
    ReflectionMemoryCandidate,
    ReflectionMemoryService,
    ReflectionMemoryStore,
)
from qwen_exo_booster.telemetry import TelemetryStore


class _CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        return list(str(value))

    def decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(values)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(str(message["content"]) for message in messages)


class _ReflectionRunner:
    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []
        self.jobs = []

    async def run_batch(self, jobs, prompts, _sampling_params, **_kwargs):
        self.jobs.extend(jobs)
        self.prompts.extend(str(prompt) for prompt in prompts)
        return (
            InternalJobResult(
                job=jobs[0],
                text=self.text,
                prompt_tokens=256,
                completion_tokens=512,
                finish_reason="stop",
                latency_seconds=0.01,
            ),
        )


def _service(tmp_path, *, max_history_tokens=8192):
    return ReflectionMemoryService(
        runner=None,
        tokenizer=_CharacterTokenizer(),
        telemetry=TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_history_tokens=max_history_tokens,
    )


def _fields(
    *,
    memory_action: str = "insert",
    target_document_path: str = "none",
    merge_document_paths: str = "[]",
) -> dict[str, str]:
    return {
        "title": "WFP 出站授权层判定与验证",
        "outcome": "success",
        "memory_action": memory_action,
        "target_document_path": target_document_path,
        "merge_document_paths": merge_document_paths,
        "reflection": (
            "工具输出直接证明当前生产配置使用 WFP_LAYER_ALE_AUTH_CONNECT_V4 处理出站授权。"
            "先前沿用 ALE_AUTH_RECV_ACCEPT 的判断没有核对流量方向，导致排查入口错误；读取 policy.py 后"
            "立即转向 connect 路径，并用定向探针验证结果，最终确认配置和运行行为一致。"
        ),
        "evidence": (
            "read_repository_file 返回 policy.py 中精确的 WFP_LAYER_ALE_AUTH_CONNECT_V4 标识；"
            "随后 focused-connect-probe 成功命中同一层，响应状态和日志方向均为 outbound。"
            "这两项直接证据比旧 capsule 中的入站假设更新，且未观察到 ALE_AUTH_RECV_ACCEPT 命中。"
        ),
        "causal_analysis": (
            "根因是把旧环境的入站规则当成当前环境事实，未在第一次工具读取后检查 direction 字段。"
            "WFP wrapper 会根据流量方向映射不同 layer constant，因此错误入口让后续参数和过滤路径全部偏移。"
            "定向探针能够区分两个竞争假设，并在修改前形成最小、可复现的判据。"
        ),
        "conflict_resolution": (
            "旧记录声称 ALE_AUTH_RECV_ACCEPT 生效，但它没有当前版本的文件或运行日志支持；"
            "本次以 policy.py 和 focused-connect-probe 的直接证据替换该结论。若未来切回 inbound 场景，"
            "仍需重新检查 layer constant，不能把本结论外推到所有 WFP 事件。"
        ),
        "reusable_experience": (
            "当 wrapper 将 WFP 事件映射到层常量时，先保留工具返回的精确标识，再核对 inbound/outbound"
            "方向，最后运行只覆盖目标路径的探针。只有文件配置与运行命中同时一致，才修改过滤逻辑；"
            "该规则适用于存在多层映射且历史文档可能滞后的环境。"
        ),
        "avoid": (
            "不要在最新源码明确写出 WFP_LAYER_ALE_AUTH_CONNECT_V4 时，仍把助手先验中的"
            " ALE_AUTH_RECV_ACCEPT 当成事实；也不要通过重复宽泛请求来掩盖方向未核对的问题。"
            "缺少直接命中证据时应保留不确定性，而不是静默选择旧结论。"
        ),
        "next_time": (
            "下一次先读取具体 policy source，记录版本和 direction，再运行 targeted connect probe；"
            "若配置与运行结果冲突，分别保存两者的时间戳和环境边界，停止修改并定位加载缓存。"
            "完成后复查 filter path，确保结论只作用于已验证的出站授权场景。"
        ),
    }


def _tool_call(name: str, fields: dict[str, str]) -> str:
    return '<tool_call name="{}">{}</tool_call>'.format(
        name,
        "".join(f"<{key}>{value}</{key}>" for key, value in fields.items()),
    )


def test_reflection_memory_uses_renamed_tool_and_knowledge_metadata():
    fields = _fields()
    parsed = ReflectionMemoryService.parse_tool_call(
        _tool_call(REFLECTION_MEMORY_TOOL_NAME, fields)
    )
    record = ReflectionMemory(
        trajectory_id="resp-1",
        conversation_key="conversation-1",
        source_digest="source-digest",
        source_event_count=3,
        source_token_count=512,
        attempts=1,
        created_at=0.0,
        target_document_sha256=None,
        **parsed,
    )

    markdown = record.markdown()

    assert "source_kind: trajectory_reflection" in markdown
    assert "document_group: reflection_memory" in markdown
    assert '"reflection-memory"' in markdown
    assert "WFP_LAYER_ALE_AUTH_CONNECT_V4" in markdown
    with pytest.raises(ValueError, match="unexpected tool"):
        ReflectionMemoryService.parse_tool_call(
            _tool_call("save_trajectory_reflection", fields)
        )


def test_reflection_memory_rejects_english_human_fields():
    fields = _fields()
    fields["reflection"] = (
        "The tool output identified the active layer, but this field has no Chinese. "
        * 12
    )
    with pytest.raises(ValueError, match="must be Chinese"):
        ReflectionMemoryService.parse_tool_call(
            _tool_call(REFLECTION_MEMORY_TOOL_NAME, fields)
        )


def test_reflection_memory_rejects_removed_shadow_mode(tmp_path):
    with pytest.raises(ValueError, match="off/active"):
        ReflectionMemoryService(
            runner=None,
            tokenizer=None,
            telemetry=TelemetryStore(tmp_path / "trace.jsonl"),
            model_fingerprint="model",
            mode="shadow",
        )


def test_reflection_prompt_contains_full_structured_trajectory_and_title_contract(
    tmp_path,
):
    service = _service(tmp_path)
    history = (
        {"kind": "user_context", "content": "inspect the current implementation"},
        {
            "kind": "assistant_trajectory",
            "content": "I should inspect the wrapper before deciding.",
        },
        {
            "kind": "tool_action",
            "tool_name": "read_repository_file",
            "call_id": "call-1",
            "content": '{"path":"policy.py"}',
        },
        {
            "kind": "tool_observation",
            "tool_name": "read_repository_file",
            "call_id": "call-1",
            "content": "WFP_LAYER_ALE_AUTH_CONNECT_V4",
        },
        {
            "kind": "assistant_output",
            "content": "The outbound authorization layer is active.",
        },
    )

    prompt, audit = service._prompt(
        original_task="Inspect the WFP wrapper",
        tool_ledger=({"observation": "WFP_LAYER_ALE_AUTH_CONNECT_V4"},),
        trajectory_history=history,
        capsule_history=({"phase": "verified", "event": "targeted probe passed"},),
        previous_failure="",
    )

    assert "inspect the current implementation" in prompt
    assert "I should inspect the wrapper before deciding" in prompt
    assert "read_repository_file" in prompt
    assert "WFP_LAYER_ALE_AUTH_CONNECT_V4" in prompt
    assert "The outbound authorization layer is active" in prompt
    assert "中文标题" in prompt
    assert "不得使用文件名、哈希、请求 ID 或照抄任务" in prompt
    assert "禁止按时间线复述" in prompt
    assert "暴力猜测" in prompt
    assert "观察→假设→最小验证→决策→复核" in prompt
    assert "哪一条精确观察能区分竞争假设" in prompt
    assert "哪一条观察本应触发停止或转向" in prompt
    assert "批判决策质量和信息增益" in prompt
    assert "<memory_action>insert|update</memory_action>" in prompt
    assert audit["provided_history_rows"] == 5
    assert audit["retained_history_rows"] == 5
    assert audit["source_tokens"] <= 8192


def test_reflection_source_window_bounds_long_history_and_keeps_recent_evidence(
    tmp_path,
):
    service = _service(tmp_path, max_history_tokens=2048)
    history = tuple(
        {
            "kind": "tool_observation",
            "content": f"event-{index}-" + ("x" * 420),
            "source_digest": f"digest-{index}",
        }
        for index in range(12)
    )

    source, audit = service._source_payload(
        original_task="Long agent trajectory",
        tool_ledger=({"observation": "latest"},),
        trajectory_history=history,
        capsule_history=(),
    )

    retained = source["trajectory_history"]
    assert retained
    assert retained[-1]["source_digest"] == "digest-11"
    assert audit["omitted_history_rows"] > 0
    assert audit["omitted_history_digest"]
    assert audit["source_tokens"] <= 2048


def test_reflection_prompt_includes_bounded_qk_memory_candidates(tmp_path):
    service = _service(tmp_path)
    candidate = ReflectionMemoryCandidate(
        document_path="reflection-memory/network-probe.md",
        document_sha256="sha-old",
        title="Differentiate transport failures before retrying",
        content=(
            "Observe status, response body, and client error class before changing "
            "the request strategy; repeated identical failures require a pivot."
        ),
        tensor_score=0.82,
    )

    prompt, audit = service._prompt(
        original_task="Diagnose repeated localhost request failures",
        tool_ledger=({"observation": "WinError 10106"},),
        trajectory_history=(
            {
                "kind": "tool_observation",
                "content": "WinError 10106 repeated 256 times",
            },
        ),
        capsule_history=(),
        previous_failure="",
        existing_memories=(candidate,),
    )

    assert candidate.document_path in prompt
    assert candidate.document_sha256 in prompt
    assert "repeated identical failures require a pivot" in prompt
    assert "底层问题、因果机制、决策点和可复用决策规则" in prompt
    assert audit["provided_memory_candidates"] == 1
    assert audit["retained_memory_candidates"] == 1


def test_reflection_update_must_target_a_qk_candidate():
    candidate = ReflectionMemoryCandidate(
        document_path="reflection-memory/proposed.md",
        document_sha256="sha-proposed",
        title="Proposed",
        content="Concrete prior reflection content",
        tensor_score=0.7,
    )
    parsed = ReflectionMemoryService.parse_tool_call(
        _tool_call(
            REFLECTION_MEMORY_TOOL_NAME,
            _fields(
                memory_action="update",
                target_document_path=candidate.document_path,
            ),
        )
    )

    target, merged = ReflectionMemoryService._validate_memory_action(
        parsed, (candidate,)
    )
    assert target == candidate
    assert merged == (candidate,)
    with pytest.raises(ValueError, match="not proposed by QK"):
        ReflectionMemoryService._validate_memory_action(parsed, ())
    with pytest.raises(ValueError, match="insert target must be none"):
        ReflectionMemoryService.parse_tool_call(
            _tool_call(
                REFLECTION_MEMORY_TOOL_NAME,
                _fields(target_document_path=candidate.document_path),
            )
        )


def test_reflection_generation_updates_retrieved_memory_instead_of_inserting(tmp_path):
    candidate = ReflectionMemoryCandidate(
        document_path="reflection-memory/network-probe.md",
        document_sha256="sha-old",
        title="Differentiate transport failures before retrying",
        content="The prior record requires a discriminating transport probe.",
        tensor_score=0.91,
    )
    runner = _ReflectionRunner(
        _tool_call(
            REFLECTION_MEMORY_TOOL_NAME,
            _fields(
                memory_action="update",
                target_document_path=candidate.document_path,
            ),
        )
    )
    retrieval_queries: list[str] = []
    published: list[ReflectionMemory] = []

    async def retrieve(_parent_id: str, query: str):
        retrieval_queries.append(query)
        return (candidate,)

    async def publish(reflection: ReflectionMemory):
        published.append(reflection)
        return {
            "document_path": candidate.document_path,
            "document_sha256": "sha-updated",
            "native_source_digest": "bank-updated",
            "hot_updated": True,
            "restart_required": False,
            "publication_status": "published",
        }

    store = ReflectionMemoryStore(tmp_path / "reflection-memory.json")
    service = ReflectionMemoryService(
        runner,
        _CharacterTokenizer(),
        TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_attempts=1,
        store=store,
        publish=publish,
        retrieve_similar=retrieve,
    )

    result = asyncio.run(
        service.reflect(
            trajectory_id="resp-update",
            conversation_key="conversation-update",
            original_task="Diagnose repeated localhost failures",
            tool_ledger=({"observation": "WinError 10106 repeated 256 times"},),
            trajectory_history=(
                {"kind": "tool_observation", "content": "curl returned a response"},
            ),
            capsule_history=(),
            source_token_count=512,
        )
    )

    assert result is not None
    assert result.memory_action == "update"
    assert result.target_document_path == candidate.document_path
    assert result.target_document_sha256 == candidate.document_sha256
    assert result.document_path == candidate.document_path
    assert retrieval_queries and "WinError 10106" in retrieval_queries[0]
    assert candidate.content in runner.prompts[0]
    assert runner.jobs[0].token_budget == service.max_output_tokens - 1
    assert len(published) == 1
    assert published[0].memory_action == "update"
    assert published[0].target_document_path == candidate.document_path
    assert published[0].target_document_sha256 == candidate.document_sha256
    assert published[0].document_path is None
    assert len(store.list()) == 1


def test_reflection_organizer_merges_model_selected_qk_candidates(tmp_path):
    left = ReflectionMemoryCandidate(
        document_path="reflection-memory/left.md",
        document_sha256="sha-left",
        title="左侧记忆",
        content="相同因果经验的左侧完整反思。",
        tensor_score=0.91,
    )
    right = ReflectionMemoryCandidate(
        document_path="reflection-memory/right.md",
        document_sha256="sha-right",
        title="右侧记忆",
        content="相同因果经验的右侧完整反思。",
        tensor_score=0.89,
    )
    runner = _ReflectionRunner(
        _tool_call(
            REFLECTION_MEMORY_TOOL_NAME,
            _fields(
                memory_action="update",
                target_document_path=left.document_path,
                merge_document_paths=json.dumps(
                    [left.document_path, right.document_path]
                ),
            ),
        )
    )
    service = ReflectionMemoryService(
        runner,
        _CharacterTokenizer(),
        TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_attempts=1,
    )

    result = asyncio.run(
        service.organize_candidates(
            organization_id="manual-1",
            candidates=(left, right),
            qk_pairs=((left.document_path, right.document_path, 0.91),),
        )
    )

    assert result is not None
    assert result.target_document_path == left.document_path
    assert result.target_document_sha256 == left.document_sha256
    assert result.merge_document_paths == (left.document_path, right.document_path)
    assert dict(result.merge_document_sha256s) == {
        left.document_path: left.document_sha256,
        right.document_path: right.document_sha256,
    }
    assert "模型原生 Q×K 高分检索" in runner.prompts[0]
    assert "冲突整理" in runner.prompts[0]


def test_reflection_organizer_keeps_distinct_memories_on_skip(tmp_path):
    candidates = (
        ReflectionMemoryCandidate("reflection-memory/a.md", "sha-a", "甲", "甲", 0.8),
        ReflectionMemoryCandidate("reflection-memory/b.md", "sha-b", "乙", "乙", 0.8),
    )
    runner = _ReflectionRunner(
        '<tool_call name="skip_reflection_memory">'
        "<reason>两条记忆只有主题相似，因果机制不同，应保持分开。</reason>"
        "</tool_call>"
    )
    service = ReflectionMemoryService(
        runner,
        _CharacterTokenizer(),
        TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_attempts=1,
    )

    result = asyncio.run(
        service.organize_candidates(
            organization_id="manual-2",
            candidates=candidates,
            qk_pairs=((candidates[0].document_path, candidates[1].document_path, 0.8),),
        )
    )

    assert result is None


def test_reflection_generation_fails_closed_when_qk_retrieval_fails(tmp_path):
    runner = _ReflectionRunner(_tool_call(REFLECTION_MEMORY_TOOL_NAME, _fields()))
    published: list[ReflectionMemory] = []

    async def retrieve(_parent_id: str, _query: str):
        raise RuntimeError("query probe unavailable")

    async def publish(reflection: ReflectionMemory):
        published.append(reflection)
        return {}

    service = ReflectionMemoryService(
        runner,
        _CharacterTokenizer(),
        TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_attempts=1,
        publish=publish,
        retrieve_similar=retrieve,
    )

    result = asyncio.run(
        service.reflect(
            trajectory_id="resp-failed-qk",
            conversation_key="conversation-failed-qk",
            original_task="Diagnose transport behavior",
            tool_ledger=({"observation": "one concrete result"},),
            trajectory_history=(),
            capsule_history=(),
            source_token_count=512,
        )
    )

    assert result is None
    assert runner.prompts == []
    assert published == []


def test_reflection_store_replaces_records_for_the_same_document(tmp_path):
    parsed = ReflectionMemoryService.parse_tool_call(
        _tool_call(REFLECTION_MEMORY_TOOL_NAME, _fields())
    )
    first = ReflectionMemory(
        trajectory_id="resp-1",
        conversation_key="conversation-1",
        source_digest="source-1",
        source_event_count=3,
        source_token_count=512,
        attempts=1,
        created_at=1.0,
        target_document_sha256=None,
        document_path="reflection-memory/shared.md",
        document_sha256="sha-1",
        **parsed,
    )
    second = replace(
        first,
        trajectory_id="resp-2",
        conversation_key="conversation-2",
        source_digest="source-2",
        document_sha256="sha-2",
    )
    store = ReflectionMemoryStore(tmp_path / "reflection-memory.json")

    store.append(first)
    store.append(second)

    records = store.list()
    assert len(records) == 1
    assert records[0]["source_digest"] == "source-2"


def test_reflection_store_removes_every_merged_document_record(tmp_path):
    parsed = ReflectionMemoryService.parse_tool_call(
        _tool_call(REFLECTION_MEMORY_TOOL_NAME, _fields())
    )
    left = ReflectionMemory(
        trajectory_id="resp-left",
        conversation_key="conversation",
        source_digest="source-left",
        source_event_count=3,
        source_token_count=512,
        attempts=1,
        created_at=1.0,
        target_document_sha256=None,
        document_path="reflection-memory/left.md",
        document_sha256="sha-left",
        **parsed,
    )
    right = replace(
        left,
        trajectory_id="resp-right",
        source_digest="source-right",
        document_path="reflection-memory/right.md",
        document_sha256="sha-right",
    )
    merged = replace(
        left,
        trajectory_id="resp-merged",
        source_digest="source-merged",
        memory_action="update",
        target_document_path=left.document_path,
        target_document_sha256=left.document_sha256,
        merge_document_paths=(left.document_path, right.document_path),
        merge_document_sha256s=(
            (left.document_path, left.document_sha256),
            (right.document_path, right.document_sha256),
        ),
        document_sha256="sha-merged",
    )
    store = ReflectionMemoryStore(tmp_path / "reflection-memory.json")

    store.append(left)
    store.append(right)
    store.append(merged)

    records = store.list()
    assert len(records) == 1
    assert records[0]["source_digest"] == "source-merged"




def test_precompaction_checkpoint_can_reflect_without_tool_events(tmp_path):
    runner = _ReflectionRunner(
        _tool_call(REFLECTION_MEMORY_TOOL_NAME, _fields())
    )
    service = ReflectionMemoryService(
        runner,
        _CharacterTokenizer(),
        TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="model",
        mode="active",
        max_attempts=1,
    )

    reflection = asyncio.run(
        service.reflect(
            trajectory_id="resp_compact_checkpoint",
            conversation_key="conversation-checkpoint",
            original_task="Preserve the pre-compaction trajectory",
            tool_ledger=(),
            trajectory_history=(
                {
                    "kind": "assistant_trajectory",
                    "content": "PRECOMPACTION-EVIDENCE",
                    "source_digest": "checkpoint-row",
                },
            ),
            capsule_history=(),
            source_token_count=256,
            allow_without_tool_events=True,
        )
    )

    assert reflection is not None
    assert "PRECOMPACTION-EVIDENCE" in runner.prompts[0]
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
