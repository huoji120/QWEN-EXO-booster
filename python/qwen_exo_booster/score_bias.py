from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

# The first implementation uses fixed token buckets. The attention score_mod
# kernel receives key positions, so a fixed bucket size keeps the auxiliary
# tensor compact while preserving smooth block-level weighting.
SCORE_BIAS_BLOCK_SIZE = 128
SCORE_BIAS_MAX_BLOCKS = 1024
SCORE_BIAS_KERNEL_MAX_BLOCKS = 32
SCORE_BIAS_SKETCH_DIMENSIONS = 32


@dataclass(frozen=True, slots=True)
class ScoreBiasRecord:
    """A trajectory block eligible for model-selected attention recovery."""

    token_ids: tuple[int, ...]
    mean_surprisal: float
    step: int
    source: str = "tool_output"
    key_sketch: tuple[float, ...] = ()
    tool_name: str = ""
    observation_kind: str = "tool_output"

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("score-bias records require token ids")
        if not math.isfinite(float(self.mean_surprisal)) or self.mean_surprisal < 0:
            raise ValueError("score-bias surprisal must be finite and non-negative")
        if int(self.step) < 0:
            raise ValueError("score-bias record step cannot be negative")
        if self.key_sketch and (
            len(self.key_sketch) != SCORE_BIAS_SKETCH_DIMENSIONS
            or not all(math.isfinite(float(value)) for value in self.key_sketch)
        ):
            raise ValueError("score-bias key sketch is invalid")


def mean_surprisal(values: Iterable[float]) -> float:
    """Return a finite non-negative mean, ignoring no values silently."""

    values_tuple = tuple(float(value) for value in values)
    if not values_tuple:
        raise ValueError("cannot average an empty surprisal block")
    if not all(math.isfinite(value) and value >= 0 for value in values_tuple):
        raise ValueError("surprisal values must be finite and non-negative")
    return sum(values_tuple) / len(values_tuple)


def decay_score(
    score: float,
    *,
    age_steps: int,
    half_life_steps: float,
    max_bias: float,
) -> float:
    """Apply step-based half-life decay and cap the resulting score."""

    score = float(score)
    age_steps = int(age_steps)
    half_life_steps = float(half_life_steps)
    max_bias = float(max_bias)
    if not math.isfinite(score) or score < 0:
        raise ValueError("score must be finite and non-negative")
    if age_steps < 0:
        raise ValueError("score age cannot be negative")
    if not math.isfinite(half_life_steps) or half_life_steps <= 0:
        raise ValueError("score-bias half-life must be positive")
    if not math.isfinite(max_bias) or max_bias < 0:
        raise ValueError("score-bias maximum must be non-negative")
    decayed = score * (2.0 ** (-age_steps / half_life_steps))
    return min(max_bias, decayed)


def find_first_token_span(
    prompt_ids: Sequence[int], needle_ids: Sequence[int]
) -> tuple[int, int] | None:
    """Find the first exact occurrence of a token block in a prompt."""

    prompt = tuple(int(token) for token in prompt_ids)
    needle = tuple(int(token) for token in needle_ids)
    if not needle or len(needle) > len(prompt):
        return None
    first = needle[0]
    for start in range(0, len(prompt) - len(needle) + 1):
        if prompt[start] == first and prompt[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def find_last_token_span(
    prompt_ids: Sequence[int], needle_ids: Sequence[int]
) -> tuple[int, int] | None:
    """Find the last exact occurrence of a token block in a prompt."""

    prompt = tuple(int(token) for token in prompt_ids)
    needle = tuple(int(token) for token in needle_ids)
    if not needle or len(needle) > len(prompt):
        return None
    first = needle[0]
    last_start = len(prompt) - len(needle)
    for start in range(last_start, -1, -1):
        if prompt[start] != first:
            continue
        if prompt[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def build_score_bias_payload(
    prompt_ids: Sequence[int],
    records: Iterable[ScoreBiasRecord],
    *,
    current_step: int,
    half_life_steps: float,
    min_surprisal: float,
    max_bias: float,
    max_blocks: int,
    min_age_steps: int = 2,
    max_age_steps: int = 16,
    tail_exclusion_tokens: int = 4096,
    tail_exclusion_ratio: float = 0.15,
) -> tuple[dict[str, int | float | str | tuple[float, ...]], ...]:
    """Map model-indexed trajectory records onto the current prompt middle."""

    current_step = int(current_step)
    min_surprisal = float(min_surprisal)
    max_blocks = int(max_blocks)
    min_age_steps = int(min_age_steps)
    max_age_steps = int(max_age_steps)
    tail_exclusion_tokens = int(tail_exclusion_tokens)
    tail_exclusion_ratio = float(tail_exclusion_ratio)
    if current_step < 0:
        raise ValueError("current score-bias step cannot be negative")
    if not math.isfinite(min_surprisal) or min_surprisal < 0:
        raise ValueError("minimum score-bias surprisal must be non-negative")
    if max_blocks < 1:
        raise ValueError("score-bias block limit must be positive")
    if min_age_steps < 1 or max_age_steps < min_age_steps:
        raise ValueError("score-bias trajectory age window is invalid")
    if tail_exclusion_tokens < 0 or not 0 <= tail_exclusion_ratio < 1:
        raise ValueError("score-bias tail exclusion is invalid")

    prompt_length = len(prompt_ids)
    tail_tokens = max(
        tail_exclusion_tokens,
        int(math.ceil(prompt_length * tail_exclusion_ratio)),
    )
    middle_end = max(0, prompt_length - tail_tokens)
    candidates: list[dict[str, int | float | str | tuple[float, ...]]] = []
    seen_spans: set[tuple[int, int]] = set()
    for record in records:
        age_steps = current_step - int(record.step)
        if (
            age_steps < min_age_steps
            or age_steps > max_age_steps
            or float(record.mean_surprisal) < min_surprisal
            or not record.key_sketch
        ):
            continue
        span = find_last_token_span(prompt_ids, record.token_ids)
        if span is None or span in seen_spans or span[1] > middle_end:
            continue
        # Preserve evidence while it crosses the natural-recency boundary, then
        # decay only after four eligible turns instead of amplifying age zero.
        decay_age = max(0, age_steps - min_age_steps - 4)
        score = decay_score(
            record.mean_surprisal,
            age_steps=decay_age,
            half_life_steps=half_life_steps,
            max_bias=max_bias,
        )
        if score <= 0:
            continue
        seen_spans.add(span)
        candidates.append(
            {
                "start": int(span[0]),
                "end": int(span[1]),
                "score": float(score),
                "source": str(record.source),
                "key_sketch": tuple(float(value) for value in record.key_sketch),
                "age_steps": int(age_steps),
                "tool_name": str(record.tool_name),
                "observation_kind": str(record.observation_kind),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), int(item["start"])))
    candidates = candidates[:max_blocks]
    candidates.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return tuple(candidates)


def block_surprise_records(
    token_ids: Sequence[int],
    surprisals: Sequence[float],
    *,
    block_size: int = SCORE_BIAS_BLOCK_SIZE,
    step: int,
    source: str,
    key_sketches: Sequence[Sequence[float]] = (),
    tool_name: str = "",
    observation_kind: str = "tool_output",
) -> tuple[ScoreBiasRecord, ...]:
    """Aggregate aligned surprisals and optional model K sketches into blocks."""

    token_ids = tuple(int(token) for token in token_ids)
    surprisals = tuple(float(value) for value in surprisals)
    sketches = tuple(tuple(float(value) for value in item) for item in key_sketches)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("score-bias block size must be positive")
    if len(token_ids) != len(surprisals):
        raise ValueError("token ids and surprisals must have equal lengths")
    block_count = math.ceil(len(token_ids) / block_size) if token_ids else 0
    if sketches and len(sketches) != block_count:
        raise ValueError("score-bias key sketches must align with token blocks")
    records: list[ScoreBiasRecord] = []
    for block_index, start in enumerate(range(0, len(token_ids), block_size)):
        end = min(start + block_size, len(token_ids))
        records.append(
            ScoreBiasRecord(
                token_ids=token_ids[start:end],
                mean_surprisal=mean_surprisal(surprisals[start:end]),
                step=step,
                source=source,
                key_sketch=sketches[block_index] if sketches else (),
                tool_name=tool_name,
                observation_kind=observation_kind,
            )
        )
    return tuple(records)
