from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx
import torch


@dataclass(slots=True)
class MlxSamplingResult:
    token_ids: mx.array
    token_logprobs: mx.array
    top_logprobs_val: list[mx.array | None]
    top_logprobs_idx: list[mx.array | None]
    token_ids_logprobs_val: list[mx.array | None]
    token_ids_logprobs_idx: list[list[int] | None]


def _log_softmax(logits: mx.array) -> mx.array:
    values = logits.astype(mx.float32)
    return values - mx.logsumexp(values, axis=-1, keepdims=True)


def _request_seed(req: Any, generated_count: int) -> int:
    configured = getattr(req.sampling_params, "sampling_seed", None)
    if configured is not None:
        return (int(configured) + int(generated_count)) & 0xFFFFFFFF
    digest = hashlib.blake2b(
        f"{getattr(req, 'rid', '')}:{generated_count}".encode(), digest_size=4
    ).digest()
    return int.from_bytes(digest, "little")


def _apply_penalties(logits: mx.array, req: Any, history: Sequence[int]) -> mx.array:
    params = req.sampling_params
    repetition = float(getattr(params, "repetition_penalty", 1.0) or 1.0)
    frequency = float(getattr(params, "frequency_penalty", 0.0) or 0.0)
    presence = float(getattr(params, "presence_penalty", 0.0) or 0.0)
    logit_bias = dict(getattr(params, "logit_bias", None) or {})
    if repetition == 1.0 and frequency == 0.0 and presence == 0.0 and not logit_bias:
        return logits

    adjusted = mx.array(logits)
    counts: dict[int, int] = {}
    vocab_size = int(logits.shape[-1])
    for token in history:
        token_id = int(token)
        if 0 <= token_id < vocab_size:
            counts[token_id] = counts.get(token_id, 0) + 1
    if counts:
        token_ids = tuple(counts)
        indices = mx.array(token_ids, dtype=mx.int32)
        selected = adjusted[indices]
        if repetition != 1.0:
            selected = mx.where(
                selected < 0,
                selected * repetition,
                selected / repetition,
            )
        if frequency != 0.0 or presence != 0.0:
            token_counts = mx.array(
                [counts[token_id] for token_id in token_ids], dtype=mx.float32
            )
            selected = (
                selected - token_counts * frequency - (token_counts > 0) * presence
            )
        adjusted[indices] = selected

    if logit_bias:
        ids: list[int] = []
        values: list[float] = []
        for raw_token, raw_bias in logit_bias.items():
            try:
                token_id = int(raw_token)
                bias = float(raw_bias)
            except (TypeError, ValueError):
                continue
            if 0 <= token_id < vocab_size and math.isfinite(bias):
                ids.append(token_id)
                values.append(bias)
        if ids:
            indices = mx.array(ids, dtype=mx.int32)
            adjusted[indices] = adjusted[indices] + mx.array(values, dtype=mx.float32)
    return adjusted


def _apply_grammar(logits: mx.array, req: Any) -> mx.array:
    grammar = getattr(req, "grammar", None)
    if grammar is None or bool(getattr(grammar, "finished", False)):
        return logits
    is_terminated = getattr(grammar, "is_terminated", None)
    if callable(is_terminated) and is_terminated():
        return logits

    vocab_size = int(logits.shape[-1])
    mask = grammar.allocate_vocab_mask(vocab_size, 1, "cpu")
    if mask is None:
        return logits
    grammar.fill_vocab_mask(mask, 0)
    probe = torch.zeros((1, vocab_size), dtype=torch.float32)
    grammar.apply_vocab_mask(probe, mask)
    blocked = torch.isneginf(probe[0]).tolist()
    return mx.where(
        mx.array(blocked, dtype=mx.bool_),
        mx.array(float("-inf"), dtype=logits.dtype),
        logits,
    )


def _apply_min_new_tokens(logits: mx.array, req: Any, generated_count: int) -> mx.array:
    params = req.sampling_params
    minimum = int(getattr(params, "min_new_tokens", 0) or 0)
    if generated_count >= minimum:
        return logits
    stop_ids = tuple(
        int(value) for value in (getattr(params, "stop_token_ids", None) or ())
    )
    if not stop_ids:
        return logits
    vocab_size = int(logits.shape[-1])
    valid = tuple(value for value in stop_ids if 0 <= value < vocab_size)
    if not valid:
        return logits
    adjusted = mx.array(logits)
    adjusted[mx.array(valid, dtype=mx.int32)] = float("-inf")
    return adjusted


def _truncate(logits: mx.array, req: Any) -> mx.array:
    params = req.sampling_params
    top_k = int(getattr(params, "top_k", -1) or -1)
    top_p = float(getattr(params, "top_p", 1.0) or 1.0)
    min_p = float(getattr(params, "min_p", 0.0) or 0.0)
    filtered = logits
    vocab_size = int(filtered.shape[-1])

    if 0 < top_k < vocab_size:
        threshold = mx.min(mx.topk(filtered, k=top_k))
        filtered = mx.where(filtered >= threshold, filtered, float("-inf"))

    if min_p > 0.0:
        threshold = mx.max(filtered) + math.log(min_p)
        filtered = mx.where(filtered >= threshold, filtered, float("-inf"))

    if top_p < 1.0:
        order = mx.argsort(filtered)[::-1]
        sorted_logits = filtered[order]
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        keep_count = mx.maximum(mx.sum(cumulative < top_p) + 1, 1)
        ranks = mx.argsort(order)
        filtered = mx.where(ranks < keep_count, filtered, float("-inf"))
    return filtered


def sample_batch(
    logits: mx.array,
    reqs: Sequence[Any],
    histories: Sequence[Sequence[int]],
) -> MlxSamplingResult:
    if logits.ndim != 2 or int(logits.shape[0]) != len(reqs):
        raise ValueError("MLX sampler requires one logits row per request")
    if len(histories) != len(reqs):
        raise ValueError("MLX sampler history count does not match its requests")

    tokens: list[mx.array] = []
    token_logprobs: list[mx.array] = []
    top_values: list[mx.array | None] = []
    top_indices: list[mx.array | None] = []
    requested_values: list[mx.array | None] = []
    requested_indices: list[list[int] | None] = []

    for row_index, (req, history) in enumerate(zip(reqs, histories)):
        origin_length = len(getattr(req, "origin_input_ids", ()) or ())
        generated_count = max(0, len(history) - origin_length)
        row = logits[row_index].astype(mx.float32)
        row = _apply_penalties(row, req, history)
        row = _apply_min_new_tokens(row, req, generated_count)
        row = _apply_grammar(row, req)
        logprobs = _log_softmax(row)

        temperature = float(getattr(req.sampling_params, "temperature", 1.0) or 0.0)
        if temperature <= 1e-6:
            token = mx.argmax(row)
        else:
            filtered = _truncate(row / temperature, req)
            key = mx.random.key(_request_seed(req, generated_count))
            token = mx.random.categorical(filtered, key=key)
        tokens.append(token.astype(mx.int32))
        token_logprobs.append(logprobs[token])

        top_n = int(getattr(getattr(req, "logprob", None), "top_logprobs_num", 0) or 0)
        if top_n > 0:
            top_n = min(top_n, int(row.shape[-1]))
            indices = mx.argsort(logprobs)[-top_n:][::-1]
            top_values.append(logprobs[indices])
            top_indices.append(indices)
        else:
            top_values.append(None)
            top_indices.append(None)

        requested = getattr(getattr(req, "logprob", None), "token_ids_logprob", None)
        if requested:
            ids = [
                int(value)
                for value in requested
                if 0 <= int(value) < int(row.shape[-1])
            ]
            requested_indices.append(ids)
            requested_values.append(
                logprobs[mx.array(ids, dtype=mx.int32)] if ids else mx.array([])
            )
        else:
            requested_indices.append(None)
            requested_values.append(None)

    return MlxSamplingResult(
        token_ids=mx.stack(tokens),
        token_logprobs=mx.stack(token_logprobs),
        top_logprobs_val=top_values,
        top_logprobs_idx=top_indices,
        token_ids_logprobs_val=requested_values,
        token_ids_logprobs_idx=requested_indices,
    )
