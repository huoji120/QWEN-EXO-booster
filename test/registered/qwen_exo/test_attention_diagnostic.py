import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from qwen_exo_booster.attention_diagnostic import (
    AttentionDiagnosticError,
    _render,
    _prepare_run,
    prepare_attention_preview,
    _weights,
    parse_attention_upload,
)
from qwen_exo_booster.attention_diagnostic_capture import (
    attention_diagnostic_specs,
    diagnostic_position_matches,
    mean_cached_attention,
)


def test_cached_attention_matches_gqa_sdpa_with_permuted_slots():
    generator = torch.Generator().manual_seed(73)
    query = torch.randn(8, 16, generator=generator)
    keys = torch.randn(48, 2, 16, generator=generator)
    mapping = torch.randperm(47, generator=generator)[:23] + 1
    actual = mean_cached_attention(query, keys, mapping, scaling=0.25, key_descale=0.7)
    selected = keys[mapping].repeat_interleave(4, dim=1).transpose(0, 1) * 0.7
    reference = (
        F.scaled_dot_product_attention(
            query[:, None, :], selected, torch.eye(23).expand(8, -1, -1), scale=0.25
        )
        .squeeze(1)
        .mean(0)
    )
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_text_mrope_prefix_positions_accept_axes_without_out_of_bounds():
    chunk = torch.arange(80, 87)
    axes = chunk.repeat(3, 1)
    assert diagnostic_position_matches(chunk, 7, 87)
    assert diagnostic_position_matches(axes, 7, 87)
    assert not diagnostic_position_matches(axes, 87, 87)
    assert not diagnostic_position_matches(axes, 0, 87)
    assert not diagnostic_position_matches(axes[:2], 7, 87)
    axes[1, -1] = 85
    assert not diagnostic_position_matches(axes, 7, 87)


def test_chatml_crop_preserves_exact_prefix_and_open_header():
    prefix = (
        "<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\n中文😀<|im_end|>\n"
    )
    suffix = "<|im_start|>assistant\n"
    parsed = parse_attention_upload(prefix + suffix)
    rendered, messages = _render(parsed, 2, None)
    assert rendered == prefix
    assert messages[0]["start"] == len("<|im_start|>system\n")
    assert rendered[messages[1]["start"] : messages[1]["end"]] == "中文😀"
    assert _render(parsed, 3, None)[0] == prefix + suffix
    assert parsed.messages[1]["content"] == "中文😀"


def test_responses_tool_evidence_survives_preview_without_execution():
    source = {
        "instructions": "Only inspect.",
        "input": [
            {"role": "user", "content": "Read the file."},
            {
                "type": "function_call",
                "call_id": "call-x",
                "name": "read",
                "arguments": '{"path":"a.py"}',
            },
            {"type": "function_call_output", "call_id": "call-x", "output": "print(1)"},
        ],
    }
    parsed = parse_attention_upload(json.dumps(source))
    assert parsed.messages[0]["content"] == "Only inspect."
    assert parsed.messages[-1]["tool_call_id"] == "call-x"
    assert "print(1)" in parsed.messages[-1]["content"]
    assert "a.py" in parsed.messages[-2]["content"]
    assert "call-x" in parsed.messages[-2]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "http://example.invalid/image.png",
                        }
                    ],
                }
            ]
        },
        {"previous_response_id": "opaque", "input": "continue"},
        {"prompt": [1, 2, 3]},
    ],
)
def test_unsupported_payload_is_not_silently_projected(payload):
    with pytest.raises(AttentionDiagnosticError):
        parse_attention_upload(json.dumps(payload))


def test_capture_transport_rejects_invalid_or_ambiguous_distributions():
    assert _weights({"qwen_exo_attention_layer_3": [[0, 0], [0.25, 0.75]]}, 3, 2) == [
        0.25,
        0.75,
    ]
    assert _weights({"qwen_exo_attention_layer_3": 1}, 3, 1) == [1]
    for value in (
        [0.2, 0.3],
        [-0.1, 1.1],
        [float("nan"), 1],
        [[0.25, 0.75], [0.25, 0.75]],
    ):
        with pytest.raises(AttentionDiagnosticError):
            _weights({"qwen_exo_attention_layer_3": value}, 3, 2)
    with pytest.raises(AttentionDiagnosticError):
        _weights(
            {
                "qwen_exo_attention_error": [1],
                "qwen_exo_attention_layer_3": [0.25, 0.75],
            },
            3,
            2,
        )


def test_capture_requires_isolated_internal_job_and_rejects_memory():
    custom = {
        "qwen_exo_kind": "internal",
        "qwen_exo_job_type": "attention_diagnostic",
        "qwen_exo_dflash": "target_only",
        "qwen_exo_attention_diagnostic": {"layer_ids": [3], "token_count": 2},
    }
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=custom, max_new_tokens=1),
        origin_input_ids=[1, 2],
        extra_key="isolated",
    )
    assert attention_diagnostic_specs([req])[0]["error"] == 0
    custom["qwen_exo_kind"] = "user"
    assert attention_diagnostic_specs([req])[0]["error"] == 2
    custom["qwen_exo_kind"] = "internal"
    custom["qwen_exo_session_initial_gdn"] = {"identity": "old"}
    assert attention_diagnostic_specs([req])[0]["error"] == 2


class _DiagnosticCharacterTokenizer:
    def __call__(self, text, **kwargs):
        # Distinct token IDs make any boundary re-encoding or silent skip observable.
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
            + "<assistant>"
        )


def _diagnostic_runtime(limit=64):
    return SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            tokenizer=_DiagnosticCharacterTokenizer(),
            model_config=SimpleNamespace(context_len=limit + 1),
        )
    )


def test_oversized_single_message_previews_and_requires_explicit_prefix():
    runtime = _diagnostic_runtime()
    content = json.dumps({"messages": [{"role": "user", "content": "abcdef" * 30}]})
    preview = prepare_attention_preview(runtime, content)
    assert preview["token_budget"]["prompt_tokens"] > 64
    assert preview["token_budget"]["max_prompt_tokens"] == 64
    assert preview["messages"][0]["content"] == "abcdef" * 30
    with pytest.raises(AttentionDiagnosticError):
        _prepare_run(runtime, content, None, 1, None)
    rendered, messages, ids, tokens, warnings = _prepare_run(
        runtime, content, None, 1, 64
    )
    original = _render(
        parse_attention_upload(content), 1, runtime.tokenizer_manager.tokenizer
    )[0]
    assert ids == [ord(char) for char in original[:64]]
    assert rendered == original[:64]
    assert tokens[-1]["end"] == 64
    assert messages[0]["content"] == original[6:64]
    assert messages[0]["truncated"] is True
    assert warnings


def test_token_crop_excludes_later_messages_and_invalid_boundaries():
    runtime = _diagnostic_runtime()
    content = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "abc"},
                {"role": "assistant", "content": "PRIVATE_LATER_SUFFIX"},
            ]
        }
    )
    prefix, messages, ids, tokens, _ = _prepare_run(runtime, content, None, 2, 9)
    assert prefix == "<user>abc"
    assert len(ids) == 9
    assert [m["content"] for m in messages] == ["abc"]
    assert "PRIVATE_LATER_SUFFIX" not in prefix
    assert all(t["end"] <= len(prefix) for t in tokens)
    first = prepare_attention_preview(runtime, content, end_message=1)
    full = prepare_attention_preview(runtime, content)
    assert (
        first["token_budget"]["prompt_tokens"] < full["token_budget"]["prompt_tokens"]
    )
    assert len(first["messages"]) == 2  # Changing the crop must not lose the source.
    for end in (0, 65, True):
        with pytest.raises(AttentionDiagnosticError):
            _prepare_run(runtime, content, None, 2, end)


def test_explicit_prefix_keeps_partial_unicode_token_ids_without_reencoding():
    class ByteSplitTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "😀Z"
            return {
                "input_ids": [101, 102, 103],
                "offset_mapping": [(0, 1), (0, 1), (1, 2)],
            }

    runtime = _diagnostic_runtime()
    runtime.tokenizer_manager.tokenizer = ByteSplitTokenizer()
    rendered, messages, ids, tokens, warnings = _prepare_run(runtime, "😀Z", None, 1, 1)
    assert ids == [
        101
    ]  # Encoding the displayed emoji would instead produce both 101 and 102.
    assert rendered == "😀"
    assert [t["id"] for t in tokens] == [101]
    assert messages[0]["content"] == "😀"
    assert messages[0]["truncated"] is True
    assert warnings


def test_multimodal_control_markers_are_projected_without_claiming_replay():
    parsed = parse_attention_upload(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "image <|vision_start|><|image_pad|><|vision_end|>",
                    }
                ]
            }
        )
    )
    assert (
        parsed.messages[0]["content"]
        == "image [multimodal:vision-start][multimodal:image-placeholder][multimodal:vision-end]"
    )
    assert any(
        "not an exact multimodal replay" in warning for warning in parsed.warnings
    )


@pytest.mark.parametrize(
    "source",
    [
        "image <|image_pad|>",
        "<|im_start|>user\n<|image_pad|><|im_end|>\n",
        json.dumps({"prompt": "image <|image_pad|>"}),
    ],
)
def test_raw_multimodal_prompts_require_explicit_text_projection(source):
    with pytest.raises(AttentionDiagnosticError):
        parse_attention_upload(source)


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "function_call",
            "call_id": "call-x",
            "name": "inspect",
            "arguments": '{"text":"<|image_pad|>"}',
        },
        {"type": "function_call_output", "call_id": "<|image_pad|>", "output": "done"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "inspect",
                        "arguments": '{"text":"<|image_pad|>"}',
                    }
                }
            ],
        },
    ],
)
def test_structured_tool_markers_are_inert_in_rendered_content(item):
    parsed = parse_attention_upload(json.dumps({"input": [item]}))
    assert "<|image_pad|>" not in parsed.messages[0]["content"]
    assert "[multimodal:image-placeholder]" in parsed.messages[0]["content"]
    assert any(
        "not an exact multimodal replay" in warning for warning in parsed.warnings
    )
