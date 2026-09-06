from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.internal_jobs import InternalJobResult, InternalJobRunner
from qwen_exo_booster.reflection_evidence import ReflectionEvidenceStore
from qwen_exo_booster.reflection_memory import (
    ReflectionMemory,
    ReflectionMemoryCandidate,
    ReflectionMemoryService,
    ReflectionMemoryStore,
    ReflectionSourceStore,
)
from qwen_exo_booster.telemetry import TelemetryStore


class _CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        return list(str(value))

    def decode(self, values, skip_special_tokens=True):
        return "</think>" if tuple(values) == (999,) else "".join(values)

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False) + "\n"


def _call(payload):
    return (
        "<tool_call>"
        + json.dumps(
            {"name": "record_causal_analysis", "arguments": payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


def _ref(event):
    return {"event_id": event["event_id"], "quote": event["content"]}


def _issue(events, title="缓存变更需隔离验证"):
    return {
        "title": title,
        "scope": "相同输入和环境下的缓存读取路径。",
        "problem": "旧缓存可能导致读取过期结果。",
        "action": "固定输入，只切换缓存启用状态。",
        "observation": "保留切换前后返回值，避免用助手完成声明代替结果。",
        "mechanism": "缓存是否导致结果差异仍需与真实观察对照。",
        "rule": "同一输入出现旧值时，先隔离缓存变量再决定是否失效缓存。",
        "next_check": "恢复缓存设置并重放同一输入；若结果未复现则重新排查。",
        "alternatives": ["后台数据也可能发生变化。"],
        "counterevidence": [],
        "missing_evidence": [],
        "evidence_refs": [_ref(event) for event in events],
    }


def _review(payload, *, verified=False, method="controlled_comparison", target=None):
    return {
        "causal_status": "verified" if verified else "supported",
        "method": method if verified else "none",
        "evidence_refs": payload["issue"]["evidence_refs"],
        "reason": "比较实际观测与干预，限定结论只适用于已观察范围。",
        "missing_evidence": [] if verified else ["尚未取得隔离变量的重复结果。"],
        "confounders_resolved": verified,
        "scope_supported": verified,
        "rule_supported": verified,
        "target": target,
        "retire": False,
    }


class _Runner:
    """Model stand-in uses real prompt event IDs, never invented evidence IDs."""

    def __init__(self, respond, *, two_phase=False):
        self.respond = respond
        self.two_phase = two_phase
        self.payloads = []
        self.jobs = []
        self.finished_parents = []

    async def run_batch(self, jobs, prompts, sampling_params, **kwargs):
        job = jobs[0]
        self.jobs.append(job)
        messages, _ = json.JSONDecoder().raw_decode(prompts[0])
        payload = json.loads(messages[-1]["content"])
        if self.two_phase and sampling_params.get("stop_token_ids"):
            text, count, finish = "正在核对证据。", job.token_budget, "length"
        else:
            self.payloads.append(payload)
            response = self.respond(payload)
            text = response if isinstance(response, str) else _call(response)
            prefix = '<tool_call>{"name":"record_causal_analysis","arguments":'
            if prompts[0].endswith(prefix) and text.startswith(prefix):
                text = text[len(prefix) :]
            count, finish = 32, "stop"
        return (
            InternalJobResult(
                job=job,
                text=text,
                prompt_tokens=len(prompts[0]),
                completion_tokens=count,
                finish_reason=finish,
                latency_seconds=0.01,
            ),
        )

    async def finish_parent(self, parent):
        self.finished_parents.append(parent)


def _standard_response(payload, *, verified=False):
    if "issue" in payload:
        return _review(payload, verified=verified)
    return {
        "issues": [_issue(payload["events"])],
        "read_event_ids": [],
        "no_lesson_reason": "",
    }


def _service(tmp_path, runner, **options):
    return ReflectionMemoryService(
        runner=runner,
        tokenizer=_CharacterTokenizer(),
        telemetry=TelemetryStore(tmp_path / "trace.jsonl"),
        model_fingerprint="cpu-test-model",
        mode="active",
        max_attempts=1,
        max_history_tokens=12288,
        store=ReflectionMemoryStore(tmp_path / "memories.json"),
        source_store=ReflectionSourceStore(tmp_path / "sources.sqlite3"),
        evidence_store=ReflectionEvidenceStore(tmp_path / "evidence.sqlite3"),
        **options,
    )


def _reflect(service, rows, **options):
    return asyncio.run(
        service.reflect(
            trajectory_id="checkpoint",
            conversation_key="conversation",
            original_task="判断缓存是否导致旧值，不将相关性当成原因。",
            tool_ledger=({"tool_name": "probe", "observation": "已记录"},),
            trajectory_history=tuple(rows),
            capsule_history=(),
            **options,
        )
    )


def _controlled_rows():
    return (
        {
            "kind": "tool_observation",
            "call_id": "before",
            "content": "相同输入 x，缓存开启，返回旧值 1。",
        },
        {
            "kind": "tool_action",
            "call_id": "toggle",
            "content": "只关闭缓存；输入 x、数据、进程版本均保持不变。",
        },
        {
            "kind": "tool_observation",
            "call_id": "after",
            "content": "相同输入 x，缓存关闭，返回当前值 2；恢复缓存后再次返回 1。",
        },
    )


def test_ambiguous_tool_envelopes_cannot_publish_a_memory(tmp_path):
    def respond(payload):
        call = _call(_standard_response(payload))
        return call + "\n" + call

    service = _service(tmp_path, _Runner(respond))
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "failed_closed"
    assert result.coverage["pending_events"] == 3
    assert result.causal_entries == ()
    with pytest.raises(ValueError, match="Duplicate causal field"):
        service.parse_tool_call(
            '<tool_call>{"name":"record_causal_analysis",'
            '"arguments":{},"arguments":{"issues":[]}}</tool_call>'
        )


def test_full_source_segments_preserve_early_middle_and_tail(tmp_path):
    runner = _Runner(
        lambda p: {
            "issues": [],
            "read_event_ids": [],
            "no_lesson_reason": "本段仅有原始采样，尚无可复用决策。",
        }
    )
    service = _service(tmp_path, runner)
    text = "EARLY-OBSERVATION\n" + "原始采样数据\n" * 2400 + "TAIL-OBSERVATION"
    result = _reflect(service, [{"kind": "tool_observation", "content": text}])
    fragments = [event for p in runner.payloads for event in p.get("events", [])]
    assert "".join(event["content"] for event in fragments) == text
    assert len(fragments) > 1
    assert [(event["start"], event["end"]) for event in fragments] == [
        (
            sum(len(x["content"]) for x in fragments[:i]),
            sum(len(x["content"]) for x in fragments[: i + 1]),
        )
        for i in range(len(fragments))
    ]
    assert result.analysis_status == "no_lesson"
    assert result.coverage["provided_events"] == result.coverage["analyzed_events"] == 1
    assert result.coverage["pending_events"] == 0
    snapshot = ReflectionSourceStore(tmp_path / "sources.sqlite3").get(
        result.source_digest
    )
    assert snapshot.trajectory_history[0]["content"] == text


def test_later_issue_explicitly_rereads_earlier_evidence(tmp_path):
    reread = []

    def respond(payload):
        if "issue" in payload:
            return _review(payload)
        requested = payload.get("requested_evidence", [])
        if requested:
            reread.extend(requested)
            return {
                "issues": [
                    _issue(
                        [requested[0], payload["events"][-1]], "前后结果需要跨段对照"
                    )
                ],
                "read_event_ids": [],
                "no_lesson_reason": "",
            }
        if any("LATER" in event["content"] for event in payload["events"]):
            earlier = payload["earlier_issue_index"][0]["evidence_refs"][0]["event_id"]
            return {"issues": [], "read_event_ids": [earlier], "no_lesson_reason": ""}
        early = [event for event in payload["events"] if "EARLY" in event["content"]]
        return {
            "issues": [_issue(early)] if early else [],
            "read_event_ids": [],
            "no_lesson_reason": "本段仅有重复采样。",
        }

    runner = _Runner(respond)
    service = _service(tmp_path, runner)
    rows = [
        {"kind": "tool_observation", "content": "EARLY 原始失败：返回旧值。"},
        {"kind": "tool_observation", "content": "采样中。" * 1800},
        {"kind": "tool_observation", "content": "LATER 切换缓存后返回当前值。"},
    ]
    result = _reflect(service, rows)
    assert result.analysis_status == "complete"
    assert reread[0]["content"] == rows[0]["content"]
    assert {ref["event_id"] for ref in result.causal_entries[-1]["evidence_refs"]} == {
        reread[0]["event_id"],
        service.evidence_store.rows("conversation")[-1]["event_id"],
    }
    # A replay must retain the re-read proof, not validate it only against the later segment.
    replay = _reflect(service, rows)
    assert replay.coverage["pending_events"] == 0


def test_partial_retry_resumes_successful_issues_without_republishing(tmp_path):
    fail_late = True
    extractions = Counter()
    reviews = Counter()
    in_progress = []

    def respond(payload):
        if "issue" in payload:
            reviews[payload["issue"]["title"]] += 1
            return _review(payload)
        events = payload["events"]
        label = "后段" if any("LATE" in x["content"] for x in events) else "前段"
        extractions[label] += 1
        if label == "后段" and fail_late:
            in_progress.extend(
                record["analysis_status"]
                for record in service.store.list()
                if record.get("causal_entries")
            )
            return "not a closed tool call"
        interesting = [
            x for x in events if "EARLY" in x["content"] or "LATE" in x["content"]
        ]
        return {
            "issues": [_issue(interesting, label + "缓存现象")] if interesting else [],
            "read_event_ids": [],
            "no_lesson_reason": "中间采样没有新信息。",
        }

    runner = _Runner(respond)
    service = _service(tmp_path, runner)
    rows = [
        {"kind": "tool_observation", "content": "EARLY 旧值现象。"},
        {"kind": "tool_observation", "content": "采样。" * 2200},
        {"kind": "tool_observation", "content": "LATE 新值现象。"},
    ]
    first = _reflect(service, rows)
    assert first.analysis_status == "partial"
    assert first.coverage["failed_segments"] == 1
    assert first.coverage["pending_events"] > 0
    assert set(in_progress) == {"partial"}
    completed_extractions = extractions["前段"]
    completed_reviews = reviews["前段缓存现象"]
    fail_late = False
    second = _reflect(service, rows)
    assert second.analysis_status == "complete"
    assert second.coverage["pending_events"] == 0
    assert extractions["前段"] == completed_extractions
    assert reviews["前段缓存现象"] == completed_reviews
    assert all(
        record["analysis_status"] == "complete"
        for record in service.store.list()
        if record.get("causal_entries")
    )
    entries = [
        entry
        for record in service.store.list()
        for entry in record.get("causal_entries", [])
    ]
    assert Counter(entry["title"] for entry in entries) == {
        "前段缓存现象": 1,
        "后段缓存现象": 1,
    }


@pytest.mark.parametrize("causal_status", ["supported", "unresolved"])
def test_evidence_backed_memory_is_publishable_without_verified_cause(
    tmp_path, causal_status
):
    published = []

    async def publish(record):
        published.append(record)
        return {"publication_status": "published"}

    def respond(payload):
        value = _standard_response(payload)
        if "issue" in payload:
            value["causal_status"] = causal_status
        return value

    service = _service(tmp_path, _Runner(respond), publish=publish)
    result = _reflect(service, _controlled_rows())
    assert result.publication_status == "published"
    entry = result.active_entries[0]
    assert entry["causal_status"] == causal_status
    assert entry["verification"]["admitted"] is False
    assert causal_status in result.compact_content
    assert entry["missing_evidence"][0] in result.compact_content
    for ref in entry["verification"]["evidence_refs"]:
        assert ref["quote"] in result.compact_content
    assert len(published) == 1
    replay = _reflect(service, _controlled_rows())
    assert replay.source_digest == result.source_digest
    assert len(published) == 1
    assert service.store.get(result.source_digest)["publication_status"] == "published"


def test_assistant_success_claim_cannot_become_verified(tmp_path):
    service = _service(
        tmp_path, _Runner(lambda p: _standard_response(p, verified=True))
    )
    result = _reflect(
        service,
        [{"kind": "assistant", "content": "我已验证所有测试通过，关闭缓存就是根因。"}],
    )
    entry = result.causal_entries[0]
    assert entry["causal_status"] != "verified"
    assert entry["admission_status"] == "candidate"
    assert entry["missing_evidence"]
    assert result.active_entries == ()


def test_controlled_comparison_is_stored_before_gdn_refresh(tmp_path):
    callbacks = []
    service = None

    async def publish(record):
        callbacks.append("publish")
        return {
            "document_path": "reflection-memory/cache.md",
            "document_sha256": "new-sha",
            "publication_status": "published",
            "hot_updated": True,
        }

    async def refresh(record):
        stored = ReflectionMemoryStore(tmp_path / "memories.json").get(
            record.source_digest
        )
        assert stored["publication_status"] == "published"
        assert stored["document_sha256"] == "new-sha"
        callbacks.append("gdn")

    service = _service(
        tmp_path,
        _Runner(lambda p: _standard_response(p, verified=True)),
        publish=publish,
        on_memory_stored=refresh,
    )
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "complete"
    assert result.publication_status == "published"
    assert result.active_entries[0]["verification"]["method"] == "controlled_comparison"
    assert result.active_entries[0]["rule"] in result.markdown()
    assert callbacks == ["publish", "gdn"]


def test_fabricated_quote_leaves_failed_coverage_without_memory(tmp_path):
    def respond(payload):
        issue = _issue(payload["events"])
        issue["evidence_refs"][0]["quote"] = "从未出现的成功记录"
        return {"issues": [issue], "read_event_ids": [], "no_lesson_reason": ""}

    service = _service(tmp_path, _Runner(respond))
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "failed_closed"
    assert result.coverage["pending_events"] == 3
    assert not any(record["causal_entries"] for record in service.store.list())


def test_stale_review_version_does_not_replace_existing_rule(tmp_path):
    runner = _Runner(lambda p: _standard_response(p, verified=True))
    service = _service(tmp_path, runner)
    original = _reflect(service, _controlled_rows())
    original = replace(
        original,
        document_path="reflection-memory/existing.md",
        document_sha256="original-sha",
        publication_status="published",
    )
    service.store.append(original)
    candidate = ReflectionMemoryCandidate(
        original.document_path,
        original.document_sha256,
        original.title,
        original.markdown(),
        0.99,
        original.causal_entries,
    )

    async def retrieve(parent, query):
        return (candidate,)

    def stale_response(payload):
        if "issue" not in payload:
            return _standard_response(payload)
        return _review(
            payload,
            verified=True,
            target={
                "document_path": candidate.document_path,
                "entry_id": candidate.causal_entries[0]["entry_id"],
                "version": candidate.causal_entries[0]["version"] + 1,
                "relation": "same_mechanism_and_rule",
            },
        )

    service.retrieve_similar = retrieve
    service.runner = _Runner(stale_response)
    rows = list(_controlled_rows()) + [
        {"kind": "tool_observation", "content": "第二次受控重复仍然出现同样差异。"}
    ]
    result = _reflect(service, rows)
    assert result.analysis_status == "failed_closed"
    saved = service.store.get(original.source_digest)
    assert saved["document_sha256"] == "original-sha"
    assert saved["causal_entries"] == original.public_dict()["causal_entries"]


def test_uncertain_recollection_cannot_replace_a_verified_rule(tmp_path):
    service = _service(
        tmp_path, _Runner(lambda p: _standard_response(p, verified=True))
    )
    original = _reflect(service, _controlled_rows())
    original = replace(
        original,
        document_path="reflection-memory/proven.md",
        document_sha256="proven-sha",
        publication_status="published",
    )
    service.store.append(original)
    candidate = ReflectionMemoryCandidate(
        original.document_path,
        original.document_sha256,
        original.title,
        original.markdown(),
        0.99,
        original.causal_entries,
    )

    async def retrieve(parent, query):
        return (candidate,)

    async def publish(record):
        assert record.memory_action == "insert"
        assert record.target_document_path is None
        return {
            "publication_status": "published",
            "document_path": "reflection-memory/tentative.md",
        }

    def respond(payload):
        if "issue" not in payload:
            return _standard_response(payload)
        return _review(
            payload,
            target={
                "document_path": candidate.document_path,
                "entry_id": candidate.causal_entries[0]["entry_id"],
                "version": candidate.causal_entries[0]["version"],
                "relation": "same_mechanism_and_rule",
            },
        )

    service.retrieve_similar = retrieve
    service.publish = publish
    service.runner = _Runner(respond)
    rows = list(_controlled_rows()) + [
        {"kind": "tool_observation", "content": "新一次出现旧值，但尚未隔离缓存变量。"}
    ]
    result = _reflect(service, rows)
    assert result.publication_status == "published"
    assert result.causal_entries[0]["causal_status"] == "supported"
    assert (
        service.store.get(original.source_digest)["causal_entries"]
        == original.public_dict()["causal_entries"]
    )


def test_reasoning_exhaustion_preserves_tool_phase_and_releases_parents(tmp_path):
    runner = _Runner(lambda p: _standard_response(p, verified=True), two_phase=True)
    service = _service(
        tmp_path,
        runner,
        max_output_tokens=2048,
        max_reasoning_tokens=128,
        reasoning_end_token_id=999,
    )
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "complete"
    assert result.active_entries
    reasoning = [job for job in runner.jobs if job.job_id.endswith(":reasoning")]
    tools = [job for job in runner.jobs if job.job_id.endswith(":tool")]
    assert len(reasoning) == len(tools) == 2
    assert all(job.token_budget == 128 for job in reasoning)
    assert all(job.token_budget == 1920 for job in tools)
    assert all(job.deadline_monotonic is None for job in runner.jobs)
    assert set(runner.finished_parents) == {
        job.parent_request_id for job in runner.jobs
    }


def test_complete_tool_example_in_reasoning_is_never_executed(tmp_path):
    class ReasoningExampleRunner(_Runner):
        async def run_batch(self, jobs, prompts, sampling_params, **kwargs):
            result = await super().run_batch(jobs, prompts, sampling_params, **kwargs)
            if sampling_params.get("stop_token_ids"):
                return (
                    replace(
                        result[0],
                        text=_call(
                            {
                                "issues": [],
                                "read_event_ids": [],
                                "no_lesson_reason": "思考中的格式示例，不是实际调用。",
                            }
                        ),
                    ),
                )
            return result

    service = _service(
        tmp_path,
        ReasoningExampleRunner(
            lambda p: _standard_response(p, verified=True), two_phase=True
        ),
        reasoning_end_token_id=999,
    )
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "complete"
    assert result.active_entries[0]["verification"]["method"] == "controlled_comparison"


def test_legacy_narrative_remains_visible_but_is_not_merged(tmp_path):
    record = ReflectionMemory(
        trajectory_id="legacy",
        conversation_key="conversation",
        source_digest="legacy-source",
        title="历史缓存经验",
        outcome="uncertain",
        reflection="旧轨迹尚未形成逐条因果验证。",
        evidence="旧工具输出摘要。",
        causal_analysis="未知原因。",
        reusable_experience="旧版本保留的限定规则。",
        avoid="不要扩大结论。",
        next_time="重新取得证据。",
        memory_action="insert",
        target_document_path=None,
        target_document_sha256=None,
        source_event_count=1,
        source_token_count=100,
        attempts=1,
        created_at=1.0,
    )
    service = _service(tmp_path, _Runner(_standard_response))
    service.store.append(record)
    candidates = tuple(
        ReflectionMemoryCandidate(
            f"reflection-memory/{name}.md", name, record.title, record.markdown(), 0.99
        )
        for name in ("left", "right")
    )
    result = asyncio.run(
        service.organize_candidates(
            organization_id="organize",
            candidates=candidates,
            qk_pairs=(
                (candidates[0].document_path, candidates[1].document_path, 0.99),
            ),
        )
    )
    assert result is None
    restored = ReflectionMemoryStore(tmp_path / "memories.json").get(
        record.source_digest
    )
    assert restored["reflection"] == record.reflection
    assert restored["analysis_status"] == "legacy"
    assert service.runner.payloads == []


def test_reflection_jobs_pass_real_internal_admission(tmp_path):
    class Manager:
        async def generate_request(self, request, raw_request):
            messages = json.loads(request.text[0])
            payload = json.loads(messages[-1]["content"])
            yield [
                {
                    "text": _call(_standard_response(payload)),
                    "meta_info": {
                        "prompt_tokens": 100,
                        "completion_tokens": 32,
                        "finish_reason": "stop",
                    },
                }
            ]

        def abort_request(self, rid):
            raise AssertionError("Accepted causal work must not be aborted")

    runner = InternalJobRunner(
        Manager(),
        max_fanout=4,
        max_tokens_per_parent=8192,
        request_factory=SimpleNamespace,
    )
    service = _service(tmp_path, runner)
    result = _reflect(service, _controlled_rows())
    assert result.analysis_status == "complete"
    assert result.causal_entries[0]["admission_status"] == "active"
    assert service.store.get(result.source_digest)["coverage"]["pending_events"] == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
