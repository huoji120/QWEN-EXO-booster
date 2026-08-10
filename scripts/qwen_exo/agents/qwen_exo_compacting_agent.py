from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from pydantic import Field


class CompactingAgentConfig(InteractiveAgentConfig):
    """Configuration for deterministic context compaction."""

    compaction_enabled: bool = Field(
        default_factory=lambda: os.getenv("QWEN_EXO_COMPACTION_ENABLED", "1").lower()
        not in {"0", "false", "no"}
    )
    context_window_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MSWEA_CONTEXT_WINDOW_TOKENS", "102400")),
        gt=0,
    )
    compaction_threshold: float = Field(default=0.70, gt=0.0, lt=1.0)
    compaction_keep_model_turns: int = Field(default=6, ge=2)
    compaction_summary_chars: int = Field(default=16000, ge=4000)
    compaction_min_messages: int = Field(default=16, ge=6)
    compaction_cooldown_model_turns: int = Field(default=4, ge=1)


class CompactingInteractiveAgent(InteractiveAgent):
    """InteractiveAgent with bounded, auditable message-history compaction.

    The original system/task messages and the most recent model turns remain
    verbatim. Older interactions become a deterministic execution digest once
    the observed prompt reaches the configured fraction of the context window.
    Full, uncompressed messages remain in the serialized trajectory.
    """

    def __init__(
        self,
        *args: Any,
        config_class: type[CompactingAgentConfig] = CompactingAgentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, config_class=config_class, **kwargs)
        self._trajectory_messages: list[dict[str, Any]] = []
        self._compaction_events: list[dict[str, Any]] = []
        self._last_compacted_response_key: str | None = None
        self._last_compaction_call = -1

    @property
    def _compaction_config(self) -> CompactingAgentConfig:
        return self.config  # type: ignore[return-value]

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        self._trajectory_messages = []
        self._compaction_events = []
        self._last_compacted_response_key = None
        self._last_compaction_call = -1
        return super().run(task, **kwargs)

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        added = super().add_messages(*messages)
        self._trajectory_messages.extend(messages)
        return added

    def query(self) -> dict[str, Any]:
        self._compact_if_needed()
        return super().query()

    def serialize(self, *extra_dicts: dict[str, Any]) -> dict[str, Any]:
        data = super().serialize(*extra_dicts)
        data["messages"] = self._trajectory_messages
        data.setdefault("info", {})["compaction"] = {
            "enabled": self._compaction_config.compaction_enabled,
            "threshold": self._compaction_config.compaction_threshold,
            "context_window_tokens": self._compaction_config.context_window_tokens,
            "cooldown_model_turns": self._compaction_config.compaction_cooldown_model_turns,
            "event_count": len(self._compaction_events),
            "events": self._compaction_events,
        }
        return data

    def _compact_if_needed(self) -> bool:
        config = self._compaction_config
        if not config.compaction_enabled:
            return False
        if len(self.messages) < config.compaction_min_messages:
            return False
        if (
            self._compaction_events
            and self.n_calls - self._last_compaction_call
            < config.compaction_cooldown_model_turns
        ):
            return False

        response_index, response = self._latest_model_response()
        if response is None:
            return False
        response_key = self._response_key(response)
        if response_key == self._last_compacted_response_key:
            return False

        observed_tokens = self._observed_context_tokens(response_index, response)
        threshold_tokens = math.floor(
            config.context_window_tokens * config.compaction_threshold
        )
        if observed_tokens < threshold_tokens:
            return False

        response_indices = [
            index
            for index, message in enumerate(self.messages[2:], start=2)
            if self._is_model_response(message)
        ]
        if len(response_indices) <= config.compaction_keep_model_turns:
            return False

        recent_start = response_indices[-config.compaction_keep_model_turns]
        prefix = self.messages[:2]
        old_messages = self.messages[2:recent_start]
        recent_messages = self.messages[recent_start:]
        summary = self._build_execution_digest(old_messages)

        event: dict[str, Any] = {
            "model_call": self.n_calls,
            "observed_context_tokens": observed_tokens,
            "threshold_tokens": threshold_tokens,
            "before_message_count": len(self.messages),
            "compacted_message_count": len(old_messages),
            "kept_recent_model_turns": config.compaction_keep_model_turns,
            "summary_chars": len(summary),
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
        }
        summary_message: dict[str, Any] = {
            "role": "user",
            "content": summary,
            "extra": {"context_compaction": event},
        }
        continuation_message: dict[str, Any] = {
            "role": "user",
            "content": (
                "继续（Continue）。请基于压缩摘要、保留的最近交互和当前仓库状态"
                "继续完成原任务。立即执行下一项必要的 bash 工具调用；不要只输出分析或总结。"
            ),
            "extra": {"context_compaction_continue": True},
        }
        self.messages = [
            *prefix,
            summary_message,
            *recent_messages,
            continuation_message,
        ]
        event["after_message_count"] = len(self.messages)
        self._trajectory_messages.extend((summary_message, continuation_message))
        self._compaction_events.append(event)
        self._last_compacted_response_key = response_key
        self._last_compaction_call = self.n_calls

        print(
            "[context-compaction] "
            f"call={self.n_calls} observed={observed_tokens} "
            f"threshold={threshold_tokens} messages="
            f"{event['before_message_count']}->{event['after_message_count']}"
        )
        return True

    def _latest_model_response(
        self,
    ) -> tuple[int, dict[str, Any] | None]:
        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if self._is_model_response(message):
                return index, message
        return -1, None

    @staticmethod
    def _is_model_response(message: dict[str, Any]) -> bool:
        return message.get("object") == "response" or message.get("role") == "assistant"

    @staticmethod
    def _response_key(response: dict[str, Any]) -> str:
        response_id = response.get("id")
        if response_id:
            return str(response_id)
        payload = json.dumps(response, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _observed_context_tokens(
        self, response_index: int, response: dict[str, Any]
    ) -> int:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            nested_response = response.get("extra", {}).get("response", {})
            usage = (
                nested_response.get("usage")
                if isinstance(nested_response, dict)
                else None
            )
        usage = usage if isinstance(usage, dict) else {}

        prompt_tokens = self._first_int(
            usage,
            "prompt_tokens",
            "input_tokens",
            "total_tokens",
        )
        completion_tokens = self._first_int(
            usage,
            "completion_tokens",
            "output_tokens",
        )
        if prompt_tokens:
            trailing_chars = sum(
                len(json.dumps(message, default=str, separators=(",", ":")))
                for message in self.messages[response_index + 1 :]
            )
            return prompt_tokens + completion_tokens + math.ceil(trailing_chars / 4)

        serialized_chars = sum(
            len(json.dumps(message, default=str, separators=(",", ":")))
            for message in self.messages
        )
        return math.ceil(serialized_chars / 4)

    @staticmethod
    def _first_int(mapping: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        return 0

    def _build_execution_digest(self, messages: list[dict[str, Any]]) -> str:
        entries = [
            entry for message in messages if (entry := self._digest_entry(message))
        ]
        budget = self._compaction_config.compaction_summary_chars
        header = [
            "<context_compaction>",
            "Earlier tool history was compacted automatically at 70% of the context window.",
            "The original task and system instructions remain authoritative.",
            "Repository files and git state are the source of truth.",
            "The digest preserves recent observations plus high-salience decisions, failures, edits, and checks from across the compacted history.",
            f"Compacted {len(messages)} messages into an execution digest:",
        ]
        footer = [
            "Continue from the latest preserved evidence without discarding earlier causal decisions or disproved hypotheses.",
            "</context_compaction>",
        ]
        fixed_chars = sum(len(line) + 1 for line in [*header, *footer])
        fixed_chars += 120
        entry_budget = max(1000, budget - fixed_chars)

        selected: dict[int, str] = {}
        used = 0

        def admit(index: int, *, limit: int = 800) -> bool:
            nonlocal used
            if index in selected or entry_budget - used < 120:
                return False
            label = f"[history {index + 1}/{len(entries)}] "
            available = min(limit, entry_budget - used - len(label) - 1)
            if available < 80:
                return False
            row = label + self._truncate(entries[index], available)
            selected[index] = row
            used += len(row) + 1
            return True

        recent_target = max(1000, entry_budget // 3)
        for index in range(len(entries) - 1, -1, -1):
            if used >= recent_target:
                break
            admit(index)

        ranked = sorted(
            (
                (self._entry_salience(entry), index)
                for index, entry in enumerate(entries)
                if index not in selected
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for score, index in ranked:
            if score <= 0 or not admit(index):
                continue

        if entries:
            landmarks = {
                0,
                len(entries) // 4,
                len(entries) // 2,
                (len(entries) * 3) // 4,
            }
            for index in sorted(landmarks):
                admit(min(index, len(entries) - 1), limit=600)

        rows = [selected[index] for index in sorted(selected)]
        omitted = len(entries) - len(rows)
        header.append(
            f"Selected {len(rows)} entries by recency and causal salience; omitted {omitted} routine entries."
        )
        return "\n".join([*header, *rows, *footer])[:budget]

    @staticmethod
    def _entry_salience(entry: str) -> int:
        text = entry.casefold()
        score = 0
        if "tool_call:" in text and any(
            marker in text
            for marker in (
                "cat >",
                "sed -i",
                "patch",
                "write(",
                "git commit",
                "git diff",
            )
        ):
            score += 10
        if any(
            marker in text
            for marker in (
                "traceback",
                "assertionerror",
                "returncode=1",
                "returncode=2",
                "failed",
                "unexpected",
                "contradict",
                "duplicate",
                "timeout",
            )
        ):
            score += 9
        if any(
            marker in text
            for marker in (
                "root cause",
                "architectural decision",
                "better approach",
                "the issue is",
                "the problem is",
                "because",
                "invariant",
                "precedence",
                "lifecycle",
            )
        ):
            score += 7
        if any(
            marker in text
            for marker in (
                "pytest",
                "test_",
                "passed",
                "syntax ok",
                "import successful",
            )
        ):
            score += 5
        if "reasoning:" in text:
            score += 1
        return score

    def _digest_entry(self, message: dict[str, Any]) -> str:
        if message.get("object") == "response":
            parts: list[str] = []
            for item in message.get("output", []):
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "function_call":
                    arguments = item.get("arguments", "")
                    try:
                        parsed = (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else arguments
                        )
                    except json.JSONDecodeError:
                        parsed = arguments
                    command = (
                        parsed.get("command", parsed)
                        if isinstance(parsed, dict)
                        else parsed
                    )
                    parts.append(f"tool_call: {self._truncate(str(command), 1000)}")
                elif item_type == "message":
                    text = self._content_text(item.get("content"))
                    if text:
                        parts.append(f"assistant: {self._truncate(text, 700)}")
                elif item_type == "reasoning":
                    text = self._content_text(
                        item.get("summary") or item.get("content")
                    )
                    if text:
                        parts.append(f"reasoning: {self._truncate(text, 500)}")
            return "\n".join(parts)

        if message.get("type") == "function_call_output":
            output = message.get("output", "")
            try:
                parsed_output = (
                    json.loads(output) if isinstance(output, str) else output
                )
            except json.JSONDecodeError:
                parsed_output = output
            if isinstance(parsed_output, dict):
                returncode = parsed_output.get("returncode")
                text = (
                    parsed_output.get("output")
                    or parsed_output.get("output_tail")
                    or parsed_output.get("output_head")
                    or ""
                )
                return f"tool_result returncode={returncode}: {self._truncate(str(text), 1200)}"
            return f"tool_result: {self._truncate(str(parsed_output), 1200)}"

        role = message.get("role")
        content = self._content_text(message.get("content"))
        if role and content:
            return f"{role}: {self._truncate(content, 1000)}"
        return ""

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        texts.append(str(text))
                elif part:
                    texts.append(str(part))
            return "\n".join(texts)
        return "" if content is None else str(content)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        head = max(1, limit * 2 // 3)
        tail = max(1, limit - head - 25)
        return f"{text[:head]}\n...[compacted]...\n{text[-tail:]}"
