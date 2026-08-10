from __future__ import annotations

from typing import Any


def find_token_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    """Return the first exact token-subsequence offset in linear time."""
    if not needle or len(needle) > len(haystack):
        return None
    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix[index] = matched
    matched = 0
    for index, token_id in enumerate(haystack):
        while matched and token_id != needle[matched]:
            matched = prefix[matched - 1]
        if token_id == needle[matched]:
            matched += 1
            if matched == len(needle):
                return index - len(needle) + 1
    return None


def locate_memory_span(
    engine_prompt: Any, tokenizer: Any, attachment: str | None
) -> tuple[int, int] | None:
    if not attachment:
        return None
    prompt_ids = (
        tokenizer.encode(engine_prompt, add_special_tokens=False)
        if isinstance(engine_prompt, str)
        else list(engine_prompt)
    )
    attachment_ids = tokenizer.encode(attachment, add_special_tokens=False)
    start = find_token_subsequence(prompt_ids, attachment_ids)
    return None if start is None else (start, len(attachment_ids))


def parse_private_memory_span(
    custom_params: Any, *, request_id: str, prompt_tokens: int
) -> tuple[int, int, str] | None:
    if not isinstance(custom_params, dict):
        return None
    if custom_params.get("qwen_exo_kind") != "user":
        return None
    start = custom_params.get("qwen_exo_memory_start")
    length = custom_params.get("qwen_exo_memory_length")
    key = custom_params.get("qwen_exo_memory_key")
    if type(start) is not int or type(length) is not int:
        return None
    if not isinstance(key, str) or not key or len(key) > 256:
        return None
    if start < 0 or length < 1 or start + length > int(prompt_tokens):
        return None
    return start, length, f"{request_id}:{key}"
