"""Divergent-question generation for mid-think uncertainty.

When the observer confirms — through sustained surprisal drift, not a single
spike — that the model is stuck in a reasoning trap, the old path ran a Q/K
retrieval + Semantic Judge round to find a relevant memory. On live agentic CTF
traffic that path recalled nothing in an entire hour (every refresh returned
``no_eligible_reference``) while burning 5-35s of GPU per event, because the
long tool-call context makes every rule card look lexically relevant and the
judge rejects them all.

This service replaces that with a single teacher-forced generation: given the
task and the stuck reasoning, produce a few short, deliberately *divergent*
questions that approach the problem from angles the current reasoning is not
exploring. They are injected at the reasoning boundary so the model reconsiders
before committing to an answer. No repository, no judge, one generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from qwen_exo_booster.contracts import (
    CancellationToken,
    InternalJob,
    InternalJobType,
    stable_digest,
)

_MAX_QUESTION_CHARS = 200
_SYSTEM_PROMPT = (
    "A reasoning model has spent its entire thinking budget on one task and is "
    "about to be cut off, so its current line of thought is stuck or looping. "
    "Read the task and the reasoning excerpt, then write a few sharp, concrete "
    "check questions that redirect it toward finishing. Make every question "
    "actionable and specific to THIS task, each pointing at a different lever:\n"
    "- the one load-bearing assumption it should verify before committing;\n"
    "- a more direct path or shortcut it has not tried;\n"
    "- a specific concrete fact, value, or edge case it has not yet checked.\n"
    "Rules: each question is a single concrete sentence naming the actual thing "
    "to check (not a vague 'have you considered alternatives'); write them in the "
    "same language as the task; do NOT restate its plan, summarize, answer the "
    "task, or ask whether some tool call merely succeeded. Return only the JSON "
    "object for the requested tool."
)


def _schema(question_count: int) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": question_count,
                    "maxItems": question_count,
                    "items": {"type": "string", "maxLength": _MAX_QUESTION_CHARS},
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class MidThinkQuestions:
    turn_id: str
    event_id: str | None
    questions: tuple[str, ...]
    latency_seconds: float

    @property
    def injection_text(self) -> str:
        """Rendered block appended at the reasoning boundary."""
        lines = "\n".join(f"{i}. {q}" for i, q in enumerate(self.questions, 1))
        return (
            "\n\n[自检] 思考预算已到，收束前先快速核对几个关键点：\n"
            f"{lines}\n"
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "question_count": len(self.questions),
            "latency_seconds": self.latency_seconds,
            "question_digests": [stable_digest(q)[:12] for q in self.questions],
        }


class MidThinkQuestionService:
    def __init__(
        self,
        runner: Any,
        tokenizer: Any,
        telemetry: Any,
        *,
        question_count: int = 3,
        max_output_tokens: int = 256,
        timeout_seconds: float = 45.0,
    ) -> None:
        if question_count < 1:
            raise ValueError("question_count must be positive")
        if max_output_tokens < 32:
            raise ValueError("max_output_tokens must be at least 32")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.runner = runner
        self.tokenizer = tokenizer
        self.telemetry = telemetry
        self.question_count = int(question_count)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self._schema_json = _schema(self.question_count)

    def _prompt(self, *, user_question: str, partial_output: str) -> str:
        user_content = (
            f"TASK:\n{str(user_question)[-6000:]}\n\n"
            f"REASONING SO FAR (budget exhausted, likely stuck or looping):\n"
            f"{str(partial_output)[-6000:]}\n\n"
            f"Return exactly {self.question_count} concrete, actionable check "
            f"questions that redirect this reasoning toward finishing the task."
        )
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _parse(self, text: str) -> tuple[str, ...]:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return ()
        raw = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return ()
        questions: list[str] = []
        for item in raw:
            cleaned = self._safe(item)
            if cleaned:
                questions.append(cleaned)
        return tuple(questions[: self.question_count])

    @staticmethod
    def _safe(value: Any) -> str:
        text = str(value or "").strip().replace("<|im_start|>", "").replace(
            "<|im_end|>", ""
        )
        text = text.replace("<think>", "").replace("</think>", "")
        return text[:_MAX_QUESTION_CHARS].strip()

    async def generate(
        self,
        *,
        parent_request_id: str,
        turn_id: str,
        event_id: str | None,
        user_question: str,
        partial_output: str,
    ) -> MidThinkQuestions | None:
        started = time.perf_counter()
        parent_request_id = str(parent_request_id)
        prompt = self._prompt(
            user_question=user_question, partial_output=partial_output
        )
        job_id = "qwen-exo-midq-" + stable_digest(parent_request_id, turn_id)[:32]
        job = InternalJob(
            parent_request_id=parent_request_id,
            turn_id=str(turn_id),
            job_id=job_id,
            job_type=InternalJobType.SELF_ASK,
            priority=-15,
            # Every event's prompt embeds a distinct reasoning excerpt, so the
            # prefix is never shared; key on the job to avoid false cache reuse.
            shared_prefix_key="qwen-exo:v1:mid-think-questions:" + job_id[-24:],
            token_budget=self.max_output_tokens,
            state_budget_bytes=0,
            deadline_monotonic=time.monotonic() + self.timeout_seconds,
            cancellation_token=CancellationToken(f"cancel-{job_id}"),
            telemetry_correlation_id=f"{parent_request_id}:mid-think-questions",
            max_fanout=1,
        )
        self.telemetry.emit(
            parent_request_id,
            "mid_think_questions.started",
            {"event_id": event_id, "turn_id": str(turn_id)},
        )
        result = (
            await self.runner.run_batch(
                (job,),
                (prompt,),
                {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": -1,
                    "skip_special_tokens": True,
                    "json_schema": self._schema_json,
                },
            )
        )[0]
        questions = self._parse(result.text)
        latency = time.perf_counter() - started
        if not questions:
            self.telemetry.emit(
                parent_request_id,
                "mid_think_questions.empty",
                {"event_id": event_id, "latency_seconds": latency},
            )
            return None
        record = MidThinkQuestions(
            turn_id=str(turn_id),
            event_id=event_id,
            questions=questions,
            latency_seconds=latency,
        )
        self.telemetry.emit(
            parent_request_id, "mid_think_questions.completed", record.public_dict()
        )
        return record
