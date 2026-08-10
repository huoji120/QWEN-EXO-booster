from qwen_exo_booster.memory_span import (
    find_token_subsequence,
    locate_memory_span,
    parse_private_memory_span,
)


class CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        return [ord(character) for character in value]


def test_memory_span_finds_exact_attachment_in_rendered_prompt():
    tokenizer = CharacterTokenizer()
    prompt = "system prefix\nPRIVATE MEMORY\nuser question"

    assert locate_memory_span(prompt, tokenizer, "PRIVATE MEMORY") == (14, 14)


def test_token_subsequence_returns_none_without_partial_match():
    assert find_token_subsequence([1, 2, 1, 2, 3], [1, 2, 4]) is None
    assert find_token_subsequence([1, 2, 1, 2, 3], [1, 2, 3]) == 2


def test_private_memory_span_is_strict_and_request_scoped():
    params = {
        "qwen_exo_kind": "user",
        "qwen_exo_memory_start": 4,
        "qwen_exo_memory_length": 3,
        "qwen_exo_memory_key": "digest:4:3",
    }

    first = parse_private_memory_span(params, request_id="r1", prompt_tokens=10)
    second = parse_private_memory_span(params, request_id="r2", prompt_tokens=10)

    assert first == (4, 3, "r1:digest:4:3")
    assert second == (4, 3, "r2:digest:4:3")
    assert (
        parse_private_memory_span(
            {**params, "qwen_exo_memory_start": "bad"},
            request_id="r1",
            prompt_tokens=10,
        )
        is None
    )
    assert (
        parse_private_memory_span(
            {**params, "qwen_exo_memory_length": 30},
            request_id="r1",
            prompt_tokens=10,
        )
        is None
    )
    assert (
        parse_private_memory_span(
            {**params, "qwen_exo_kind": "internal"},
            request_id="r1",
            prompt_tokens=10,
        )
        is None
    )
