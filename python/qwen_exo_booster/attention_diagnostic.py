from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from qwen_exo_booster.contracts import (
    ATTENTION_DIAGNOSTIC_MAX_LAYERS,
    CancellationToken,
    InternalJob,
    InternalJobType,
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 512
MAX_PROMPT_TOKENS = 32768
DIAGNOSTIC_TIMEOUT_SECONDS = 180.0
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function"})
_CHATML_START = "<|im_start|>"
_CHATML_END = re.compile(r"<\|im_end\|>|<\|endoftext\|>")
_METHOD = "reconstructed_attention_estimate_mean_all_query_heads"


DEPENDENCY_BLOCK_DEFAULT = 64
DEPENDENCY_BLOCK_MAX = 256
DEPENDENCY_MAX_BLOCKS = 32

class AttentionDiagnosticError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ParsedAttentionUpload:
    messages: list[dict[str, Any]]
    format: str
    warnings: list[str] = field(default_factory=list)
    raw_prompt: str | None = None
    raw_ends: list[int] = field(default_factory=list)
    raw_spans: list[tuple[int, int]] = field(default_factory=list)

    def preview(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "format": self.format,
            "warnings": self.warnings,
        }


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise AttentionDiagnosticError(
            "Structured tool data must be finite, serializable JSON"
        ) from exc


_MULTIMODAL_MARKERS = {
    "<|vision_start|>": "[multimodal:vision-start]",
    "<|vision_end|>": "[multimodal:vision-end]",
    "<|image_pad|>": "[multimodal:image-placeholder]",
    "<|video_pad|>": "[multimodal:video-placeholder]",
    "<|audio_pad|>": "[multimodal:audio-placeholder]",
    "<|audio_start|>": "[multimodal:audio-start]",
}


def _has_multimodal_marker(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in _MULTIMODAL_MARKERS)
    if isinstance(value, list):
        return any(_has_multimodal_marker(part) for part in value)
    if isinstance(value, dict):
        return any(_has_multimodal_marker(part) for part in value.values())
    return False


def _project_multimodal_markers(value: str) -> str:
    for marker, replacement in _MULTIMODAL_MARKERS.items():
        value = value.replace(marker, replacement)
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _project_multimodal_markers(value)
    if not isinstance(value, list):
        raise AttentionDiagnosticError(
            "Message content must be text or text-only content parts"
        )
    parts = []
    for part in value:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("type"), str)
            or part["type"] not in {"text", "input_text", "output_text", "refusal"}
        ):
            raise AttentionDiagnosticError(
                "Images, audio, files and non-text content parts are unsupported"
            )
        key = "refusal" if part["type"] == "refusal" else "text"
        text = part.get(key)
        if not isinstance(text, str):
            raise AttentionDiagnosticError("Text content part has no string text")
        if part.get("annotations"):
            raise AttentionDiagnosticError(
                "Annotated content must be exported as explicit text first"
            )
        parts.append(_text(text))
    return "".join(parts)


def _identity(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is not None and (not isinstance(value, str) or len(value) > 1024):
        raise AttentionDiagnosticError(
            f"{key} must be a string of at most 1024 characters"
        )
    return value


def _normalize_message(item: Any, warnings: list[str]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AttentionDiagnosticError("Each message/input item must be an object")
    if _has_multimodal_marker(item):
        warnings.append(
            "Multimodal control markers were replaced with inert text placeholders; image/audio/video content is not analyzed and this is not an exact multimodal replay."
        )
    kind = item.get("type", "message")
    if not isinstance(kind, str):
        raise AttentionDiagnosticError("Message/input type must be a string")
    if kind in {"function_call", "function_call_output"}:
        call_id = _identity(item, "call_id")
        if not call_id:
            raise AttentionDiagnosticError("Function items require call_id")
        if kind == "function_call":
            if not isinstance(item.get("name"), str) or not isinstance(
                item.get("arguments"), str
            ):
                raise AttentionDiagnosticError(
                    "Function calls require string name and arguments"
                )
            result = {
                "role": "assistant",
                "content": "<tool_call>" + _json(item) + "</tool_call>",
                "name": item["name"],
                "tool_call_id": call_id,
            }
        else:
            output = _text(item.get("output"))
            identity = {key: value for key, value in item.items() if key != "output"}
            result = {
                "role": "tool",
                "content": "<tool_identity>"
                + _json(identity)
                + "</tool_identity>\n"
                + output,
                "tool_call_id": call_id,
            }
        warnings.append(
            "Structured tool calls and identities are preserved as inert literal text; no tools are executed."
        )
        result["content"] = _project_multimodal_markers(result["content"])
        return result
    if kind != "message":
        raise AttentionDiagnosticError(
            f"Unsupported Responses item type: {kind}; opaque reasoning/compaction cannot be reconstructed"
        )
    role = item.get("role")
    if not isinstance(role, str) or role not in _ROLES:
        raise AttentionDiagnosticError(f"Unsupported message role: {role}")
    for key in ("audio", "images", "image", "encrypted_content", "reasoning"):
        if item.get(key) is not None:
            raise AttentionDiagnosticError(f"Unsupported message field: {key}")
    content = _text(item.get("content"))
    for key in ("reasoning_content", "refusal"):
        if item.get(key) is not None:
            value = item[key]
            if not isinstance(value, str):
                raise AttentionDiagnosticError(f"{key} must be text")
            content += (
                ("\n" if content else "") + f"<{key}>" + _text(value) + f"</{key}>"
            )
            warnings.append(
                f"{key} is included as literal text in the diagnostic projection."
            )
    for key in ("tool_calls", "function_call"):
        if item.get(key) is not None:
            content += (
                ("\n" if content else "")
                + "<tool_call>"
                + _json(item[key])
                + "</tool_call>"
            )
            warnings.append(
                "Structured tool calls and identities are preserved as inert literal text; no tools are executed."
            )
    result = {"role": role, "content": content}
    identity = {}
    for key in ("name", "tool_call_id"):
        value = _identity(item, key)
        if value is not None:
            result[key] = value
            identity[key] = value
    if identity:
        result["content"] = (
            "<message_identity>" + _json(identity) + "</message_identity>\n" + content
        )
        warnings.append(
            "Message names/call IDs are also included in literal text so templates cannot silently drop them."
        )
    result["content"] = _project_multimodal_markers(result["content"])
    return result


def _parse_raw(content: str) -> ParsedAttentionUpload:
    if _has_multimodal_marker(content):
        raise AttentionDiagnosticError(
            "Raw multimodal prompts cannot be preserved as pure text; import messages JSON for an explicit text-only projection"
        )
    if not content.strip():
        raise AttentionDiagnosticError("Completion prompt is empty")
    if _CHATML_START not in content:
        return ParsedAttentionUpload(
            [{"index": 0, "role": "user", "content": content}],
            "completion_text",
            raw_prompt=content,
            raw_ends=[len(content)],
            raw_spans=[(0, len(content))],
        )
    starts = [match.start() for match in re.finditer(re.escape(_CHATML_START), content)]
    if len(starts) > MAX_MESSAGES:
        raise AttentionDiagnosticError(f"Upload exceeds {MAX_MESSAGES} messages")
    if content[: starts[0]].strip():
        raise AttentionDiagnosticError(
            "Text before the first ChatML message is ambiguous; export a completion prompt without mixed framing"
        )
    messages = []
    ends = []
    spans = []
    warnings = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(content)
        segment = content[start + len(_CHATML_START) : end]
        header, newline, body = segment.partition("\n")
        role, _, identity = header.strip().partition(" ")
        if role not in _ROLES:
            raise AttentionDiagnosticError(f"Unsupported ChatML header: {header[:128]}")
        closing = _CHATML_END.search(body)
        if closing:
            if body[closing.end() :].strip():
                raise AttentionDiagnosticError(
                    "Non-whitespace text after a ChatML closing marker is ambiguous"
                )
            text = body[: closing.start()]
        else:
            if index + 1 < len(starts):
                raise AttentionDiagnosticError(
                    "Only the final ChatML message may be open-ended"
                )
            text = body if newline else ""
            warnings.append(
                "The final ChatML message is open-ended; its raw prefix is preserved exactly."
            )
        message = {"index": index, "role": role, "content": text}
        if identity:
            message["name"] = identity
        messages.append(message)
        ends.append(end)
        body_start = start + len(_CHATML_START) + len(header) + len(newline)
        spans.append((body_start, body_start + len(text)))
    return ParsedAttentionUpload(messages, "chatml", warnings, content, ends, spans)


def parse_attention_upload(
    content: str, filename: str | None = None
) -> ParsedAttentionUpload:
    if not isinstance(content, str) or not content.strip():
        raise AttentionDiagnosticError("Upload is empty")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise AttentionDiagnosticError("Upload contains invalid Unicode") from exc
    if size > MAX_UPLOAD_BYTES:
        raise AttentionDiagnosticError(
            f"Upload exceeds {MAX_UPLOAD_BYTES} UTF-8 bytes", 413
        )
    stripped = content.lstrip()
    is_json = stripped.startswith(("{", "[")) or str(filename or "").lower().endswith(
        ".json"
    )
    if not is_json:
        return _parse_raw(content)
    try:
        document = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise AttentionDiagnosticError("Invalid JSON upload") from exc
    warnings = []
    if isinstance(document, list):
        items = document
        format_name = "messages_json"
    elif isinstance(document, dict):
        for key in (
            "images",
            "image_data",
            "audio",
            "input_audio",
            "encrypted_content",
            "conversation",
        ):
            if document.get(key) is not None:
                raise AttentionDiagnosticError(f"Unsupported request field: {key}")
        sources = [key for key in ("messages", "input", "prompt") if key in document]
        if len(sources) != 1:
            raise AttentionDiagnosticError(
                "JSON requires exactly one of messages, input, or prompt"
            )
        source = sources[0]
        value = document[source]
        if document.get("previous_response_id"):
            if not isinstance(value, list) or not any(
                isinstance(item, dict) and item.get("role") == "user" for item in value
            ):
                raise AttentionDiagnosticError(
                    "previous_response_id is opaque: upload actual message history, not a continuation ID"
                )
            warnings.append(
                "previous_response_id is not resolved; only the explicitly uploaded history is analyzed. Include the complete history yourself."
            )
        if source == "prompt":
            if not isinstance(value, str):
                raise AttentionDiagnosticError(
                    "Completion prompt must be one string, not token IDs or a batch"
                )
            if (
                document.get("instructions")
                or document.get("tools")
                or document.get("functions")
            ):
                raise AttentionDiagnosticError(
                    "Completion prompt cannot be combined with instructions/tool definitions"
                )
            parsed = _parse_raw(value)
            parsed.warnings.extend(warnings)
            parsed.warnings.append(
                "Completion prompt is preserved verbatim; generation parameters are not replayed."
            )
            return parsed
        format_name = "responses_json" if source == "input" else "messages_json"
        items = (
            [{"role": "user", "content": value}]
            if isinstance(value, str) and source == "input"
            else value
        )
        if not isinstance(items, list):
            raise AttentionDiagnosticError(
                "messages/input must be a message array (Responses also accepts a text input)"
            )
        items = list(items)
        instructions = document.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise AttentionDiagnosticError("instructions must be text")
        definitions = {
            key: document[key]
            for key in (
                "tools",
                "functions",
                "tool_choice",
                "parallel_tool_calls",
                "function_call",
            )
            if key in document
        }
        if definitions:
            instructions = (
                (instructions + "\n" if instructions else "")
                + "<tool_definitions>"
                + _json(definitions)
                + "</tool_definitions>"
            )
            warnings.append(
                "Uploaded tool definitions/settings are preserved as literal system text, not executable tools or an exact endpoint replay."
            )
        if instructions is not None:
            items.insert(0, {"role": "system", "content": instructions})
        warnings.append(
            "The active tokenizer chat template renders this text-only projection; generation settings and request metadata are not replayed."
        )
    else:
        raise AttentionDiagnosticError("JSON upload must be an array or request object")
    if not items or len(items) > MAX_MESSAGES:
        raise AttentionDiagnosticError(
            f"Upload must contain 1..{MAX_MESSAGES} messages"
        )
    messages = [
        {"index": index, **_normalize_message(item, warnings)}
        for index, item in enumerate(items)
    ]
    return ParsedAttentionUpload(messages, format_name, list(dict.fromkeys(warnings)))


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    return (
        config.get(key, default)
        if isinstance(config, dict)
        else getattr(config, key, default)
    )


def _available_layers(manager: Any) -> list[int]:
    config = getattr(manager, "model_config", None)
    text = _config_value(config, "hf_text_config") or _config_value(
        config, "hf_config", config
    )
    text = _config_value(text, "text_config", text)
    types = _config_value(text, "layer_types") or _config_value(
        text, "layers_block_type", ()
    )
    available = [
        index
        for index, kind in enumerate(types)
        if str(kind).lower() in {"full_attention", "attention", "full"}
    ]
    if not available:
        raise AttentionDiagnosticError(
            "Active model exposes no supported Full Attention layer structure", 503
        )
    return available


def _default_layers(available: list[int]) -> list[int]:
    count = min(ATTENTION_DIAGNOSTIC_MAX_LAYERS, len(available))
    if count == 1:
        return available[:]
    return [
        available[round(i * (len(available) - 1) / (count - 1))] for i in range(count)
    ]


def _layer_selection(manager: Any, requested: list[int] | None) -> list[int]:
    available = _available_layers(manager)
    selected = _default_layers(available) if requested is None else requested
    if (
        not 1 <= len(selected) <= ATTENTION_DIAGNOSTIC_MAX_LAYERS
        or len(set(selected)) != len(selected)
        or any(type(layer) is not int or layer not in available for layer in selected)
    ):
        raise AttentionDiagnosticError(
            f"Select 1..{ATTENTION_DIAGNOSTIC_MAX_LAYERS} distinct Full Attention layer IDs from {available}"
        )
    return selected


def _preflight_worker(manager: Any) -> None:
    args = getattr(manager, "server_args", None)
    if args is None:
        raise AttentionDiagnosticError(
            "Worker configuration is unavailable for diagnostic capability checks", 503
        )
    for key in ("pp_size", "dp_size", "attn_cp_size", "dcp_size"):
        if int(_config_value(args, key, 1) or 1) != 1:
            raise AttentionDiagnosticError(
                f"Attention diagnostics do not support {key} > 1", 503
            )
    for key in (
        "enable_dp_attention",
        "enable_prefill_cp",
        "enable_prefill_context_parallel",
        "enable_dsa_prefill_context_parallel",
        "enable_two_batch_overlap",
        "enable_mis",
    ):
        if _config_value(args, key, False):
            raise AttentionDiagnosticError(
                f"Attention diagnostics do not support {key}", 503
            )
    backend = _config_value(args, "prefill_attention_backend") or _config_value(
        args, "attention_backend"
    )
    if backend and backend not in {"triton", "flashinfer"}:
        raise AttentionDiagnosticError(
            "Attention diagnostics currently require the Triton or FlashInfer Full Attention prefill backend",
            503,
        )
    graph = _config_value(_config_value(args, "cuda_graph_config"), "prefill")
    graph_backend = _config_value(graph, "backend")
    graph_backend = getattr(graph_backend, "value", graph_backend)
    if graph_backend and graph_backend != "disabled":
        raise AttentionDiagnosticError(
            "Attention diagnostics require eager prefill; the active prefill CUDA graph mode is unsupported",
            503,
        )


def _render(
    parsed: ParsedAttentionUpload, count: int, tokenizer: Any
) -> tuple[str, list[dict[str, Any]]]:
    mapped = [dict(message) for message in parsed.messages[:count]]
    if parsed.raw_prompt is not None:
        rendered = parsed.raw_prompt[: parsed.raw_ends[count - 1]]
        for message, (start, end) in zip(mapped, parsed.raw_spans[:count]):
            if rendered[start:end] != message["content"]:
                raise AttentionDiagnosticError(
                    "Raw prompt body offsets do not match their source", 503
                )
            message.update(start=start, end=end)
        return rendered, mapped
    messages = [
        {key: value for key, value in message.items() if key != "index"}
        for message in mapped
    ]
    marker_prefix = "QWEN_EXO_SPAN_" + uuid.uuid4().hex + "_"
    marked_messages = []
    markers = []
    for index, message in enumerate(messages):
        marked = dict(message)
        if message["content"]:
            opening, closing = (
                f"[{marker_prefix}{index}_START]",
                f"[{marker_prefix}{index}_END]",
            )
            marked["content"] = opening + message["content"] + closing
            markers.append((index, opening, closing))
        marked_messages.append(marked)
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        marked_rendered = tokenizer.apply_chat_template(
            marked_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise AttentionDiagnosticError(
            "Active tokenizer cannot render this message sequence", 422
        ) from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or not isinstance(marked_rendered, str)
    ):
        raise AttentionDiagnosticError(
            "Active tokenizer returned no rendered prompt", 503
        )
    if marker_prefix in rendered:
        raise AttentionDiagnosticError(
            "Cannot establish collision-free source markers", 503
        )
    chunks = []
    cursor = 0
    source_length = 0
    for index, opening, closing in markers:
        if marked_rendered.count(opening) != 1 or marked_rendered.count(closing) != 1:
            raise AttentionDiagnosticError(
                "Active chat template omitted or duplicated message text; upload its exact raw ChatML completion prompt instead"
            )
        start = marked_rendered.find(opening, cursor)
        end = marked_rendered.find(closing, start + len(opening))
        if (
            start < cursor
            or end < start
            or marked_rendered[start + len(opening) : end] != messages[index]["content"]
        ):
            raise AttentionDiagnosticError(
                "Active chat template reordered or transformed message text; upload its exact raw ChatML completion prompt instead"
            )
        prefix = marked_rendered[cursor:start]
        text = messages[index]["content"]
        chunks.extend((prefix, text))
        source_length += len(prefix)
        mapped[index].update(start=source_length, end=source_length + len(text))
        source_length += len(text)
        cursor = end + len(closing)
    chunks.append(marked_rendered[cursor:])
    if "".join(chunks) != rendered:
        raise AttentionDiagnosticError(
            "Cannot prove message source spans under the active template; upload its exact raw ChatML completion prompt instead"
        )
    return rendered, mapped


def _encode_prompt(
    tokenizer: Any, rendered: str
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except Exception as exc:
        raise AttentionDiagnosticError(
            "Active tokenizer cannot provide exact character offsets; a fast tokenizer is required",
            503,
        ) from exc
    if not ids:
        raise AttentionDiagnosticError(
            "Rendered prompt contains 0 tokens; select a nonempty prefix", 413
        )
    if len(offsets) != len(ids):
        raise AttentionDiagnosticError(
            "Tokenizer returned misaligned token offsets", 503
        )
    return ids, offsets


def _tokens(
    rendered: str, ids: list[int], offsets: list[tuple[int, int]]
) -> tuple[list[dict[str, Any]], list[str]]:
    tokens = []
    warnings = []
    previous_start = 0
    previous_end = 0
    for token_id, span in zip(ids, offsets):
        if type(token_id) is not int or len(span) != 2:
            raise AttentionDiagnosticError(
                "Tokenizer returned invalid tokens/offsets", 503
            )
        start, end = span
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= len(rendered)
        ):
            raise AttentionDiagnosticError(
                "Tokenizer offsets are not valid Unicode character ranges", 503
            )
        if start == end:
            warnings.append(
                "Some special tokens have no source span; their empty text is not assigned a fabricated character range."
            )
        else:
            if start < previous_start:
                raise AttentionDiagnosticError(
                    "Tokenizer offsets are not ordered Unicode character ranges", 503
                )
            if start < previous_end:
                warnings.append(
                    "Some tokens share Unicode character spans (byte-level tokenization); overlapping spans are retained honestly."
                )
            previous_start, previous_end = start, end
        tokens.append(
            {"id": token_id, "text": rendered[start:end], "start": start, "end": end}
        )
    return tokens, list(dict.fromkeys(warnings))


def _prepare_prompt(
    runtime: Any, content: str, filename: str | None, end_message: int | None
):
    parsed = parse_attention_upload(content, filename)
    count = len(parsed.messages) if end_message is None else end_message
    if type(count) is not int or not 1 <= count <= len(parsed.messages):
        raise AttentionDiagnosticError(
            "end_message must select a nonempty exact message prefix"
        )
    manager = runtime.tokenizer_manager
    tokenizer = getattr(manager, "tokenizer", None)
    if tokenizer is None:
        raise AttentionDiagnosticError("Active tokenizer is unavailable", 503)
    config_limit = _config_value(
        getattr(manager, "model_config", None), "context_len", MAX_PROMPT_TOKENS + 1
    )
    limit = min(MAX_PROMPT_TOKENS, int(config_limit or MAX_PROMPT_TOKENS + 1) - 1)
    if limit < 1:
        raise AttentionDiagnosticError(
            "Active context has no capacity for diagnostic input plus one output token",
            503,
        )
    rendered, messages = _render(parsed, count, tokenizer)
    ids, offsets = _encode_prompt(tokenizer, rendered)
    return parsed, count, rendered, messages, ids, offsets, limit


def prepare_attention_preview(
    runtime: Any,
    content: str,
    filename: str | None = None,
    end_message: int | None = None,
) -> dict[str, Any]:
    parsed, count, _, _, ids, _, limit = _prepare_prompt(
        runtime, content, filename, end_message
    )
    available = _available_layers(runtime.tokenizer_manager)
    return {
        **parsed.preview(),
        "available_layer_ids": available,
        "default_layer_ids": _default_layers(available),
        "max_layers": ATTENTION_DIAGNOSTIC_MAX_LAYERS,
        "token_budget": {
            "end_message": count,
            "prompt_tokens": len(ids),
            "max_prompt_tokens": limit,
        },
    }


def _prepare_run(
    runtime: Any,
    content: str,
    filename: str | None,
    end_message: int,
    end_token: int | None,
):
    if type(end_message) is not int:
        raise AttentionDiagnosticError(
            "end_message must select a nonempty exact message prefix"
        )
    parsed, _, rendered, messages, ids, offsets, limit = _prepare_prompt(
        runtime, content, filename, end_message
    )
    original_count = len(ids)
    if end_token is None:
        if original_count > limit:
            raise AttentionDiagnosticError(
                f"Rendered prompt contains {original_count} tokens; diagnostic limit is {limit}. Select fewer messages or explicitly select a token prefix with end_token.",
                413,
            )
    elif type(end_token) is not int or not 1 <= end_token <= min(original_count, limit):
        raise AttentionDiagnosticError(
            f"end_token must select 1..{min(original_count, limit)} tokens from the {original_count}-token rendered prompt"
        )
    cropped = end_token is not None and end_token < original_count
    if cropped:
        # Slice the original encoding, never a decoded/re-encoded string. The
        # model must see precisely these IDs even at a byte-level Unicode split.
        ids = ids[:end_token]
        offsets = offsets[:end_token]
    tokens, warnings = _tokens(rendered, ids, offsets)
    if cropped:
        source_end = max(token["end"] for token in tokens)
        rendered = rendered[:source_end]
        visible_messages = []
        for message in messages:
            start, end = message.get("start"), message.get("end")
            # Empty template messages without proven spans cannot be located
            # relative to a token cut; do not claim their metadata was sent.
            if start is None or end is None or start >= source_end:
                continue
            clipped = dict(message)
            clipped["end"] = min(end, source_end)
            clipped["content"] = rendered[start : clipped["end"]]
            if end > source_end:
                clipped["truncated"] = True
                clipped.pop("name", None)
                clipped.pop("tool_call_id", None)
            visible_messages.append(clipped)
        messages = visible_messages
        warnings.append(
            f"Explicit token prefix: sampled the first {len(ids)} of {original_count} rendered tokens; the remaining {original_count - len(ids)}-token suffix was not sent or sampled. The final message or generation frame may be incomplete; no closing markers were added."
        )
        warnings.append(
            "Cropped source text shows only character spans touched by selected tokens. A boundary Unicode character may also share excluded byte-level tokens; its display does not mean all its bytes were sampled. Unlocated empty messages are omitted."
        )
    warnings.extend(parsed.warnings)
    return rendered, messages, ids, tokens, warnings


def _weights(metadata: dict[str, Any], layer: int, count: int) -> list[float]:
    errors = metadata.get("qwen_exo_attention_error", [])
    if not isinstance(errors, (list, tuple)):
        errors = [errors]
    for error in errors:
        if isinstance(error, (list, tuple)) and len(error) == 1:
            error = error[0]
        if type(error) not in (int, float) or not math.isfinite(error) or error != 0:
            reason = (
                {
                    1: "unsupported backend/cache/topology",
                    2: "invalid diagnostic request",
                    3: "unavailable prefix mapping or geometry",
                }.get(error, "invalid capture status")
                if type(error) in (int, float)
                else "invalid capture status"
            )
            raise AttentionDiagnosticError(
                "Worker attention diagnostic unavailable: " + reason, 503
            )
    value = metadata.get(f"qwen_exo_attention_layer_{layer}")
    if value is None:
        raise AttentionDiagnosticError(
            "Attention capture unavailable: worker returned no diagnostic tensor; the worker must support this feature",
            503,
        )
    # A one-token row becomes a scalar in the scheduler; other rows flatten.
    # Tokenizer accumulates rows. Ignore only explicit all-zero chunk sentinels,
    # never average several captures or repair a malformed distribution.
    if type(value) in (int, float):
        rows = [[value]]
    elif isinstance(value, (list, tuple)) and all(
        type(item) in (int, float) for item in value
    ):
        if not value or len(value) % count:
            raise AttentionDiagnosticError(
                "Worker returned an invalid attention tensor shape", 503
            )
        rows = [value[start : start + count] for start in range(0, len(value), count)]
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise AttentionDiagnosticError(
            "Worker returned an invalid attention tensor shape", 503
        )
    captures = []
    for row in rows:
        if (
            isinstance(row, (list, tuple))
            and len(row) == 1
            and isinstance(row[0], (list, tuple))
        ):
            row = row[0]
        if not isinstance(row, (list, tuple)) or len(row) < count:
            raise AttentionDiagnosticError(
                "Worker returned an invalid attention tensor shape", 503
            )
        if any(
            type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
            for weight in row
        ):
            raise AttentionDiagnosticError(
                "Worker returned non-finite or negative attention probabilities", 503
            )
        if any(row[count:]):
            raise AttentionDiagnosticError(
                "Worker returned nonzero attention padding", 503
            )
        if not any(row):
            continue
        weights = [float(weight) for weight in row[:count]]
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.005, abs_tol=0.005):
            raise AttentionDiagnosticError(
                "Worker attention probabilities are not normalized", 503
            )
        captures.append(weights)
    if len(captures) != 1:
        raise AttentionDiagnosticError(
            "Worker must return exactly one final-prefix attention capture", 503
        )
    return captures[0]


async def run_attention_diagnostic(
    runtime: Any,
    content: str,
    filename: str | None,
    end_message: int,
    sample_count: int = 1,
    layer_ids: list[int] | None = None,
    end_token: int | None = None,
) -> dict[str, Any]:
    if type(sample_count) is not int or not 1 <= sample_count <= 4:
        raise AttentionDiagnosticError("sample_count must be 1..4")
    manager = runtime.tokenizer_manager
    rendered, messages, ids, tokens, warnings = await asyncio.to_thread(
        _prepare_run, runtime, content, filename, end_message, end_token
    )
    runner = runtime.internal_jobs
    _preflight_worker(manager)
    layers = _layer_selection(manager, layer_ids)
    # Find the last message in the actual rendered source. Sample its visible
    # suffix, including generation framing; one sample is always the last input.
    last_start = messages[-1].get("start", len(rendered)) if messages else len(rendered)
    first_position = next(
        (index for index, token in enumerate(tokens) if token["end"] > last_start),
        len(ids) - 1,
    )
    positions = sorted(
        {len(ids) - 1}
        if sample_count == 1
        else {
            first_position
            + round((len(ids) - 1 - first_position) * index / (sample_count - 1))
            for index in range(sample_count)
        }
    )
    if len(positions) < sample_count:
        warnings.append(
            "The final rendered message suffix has fewer distinct token positions than requested samples."
        )
    parent = "attention-diagnostic-" + uuid.uuid4().hex
    namespace = "qwen-exo:v1:attention-diagnostic:" + parent
    deadline = time.monotonic() + DIAGNOSTIC_TIMEOUT_SECONDS
    samples = []
    try:
        for sample_index, position in enumerate(positions):
            job = InternalJob(
                parent_request_id=parent,
                turn_id=parent,
                job_id=f"{parent}-{sample_index}",
                job_type=InternalJobType.ATTENTION_DIAGNOSTIC,
                priority=-20,
                shared_prefix_key=namespace,
                token_budget=1,
                state_budget_bytes=0,
                deadline_monotonic=deadline,
                cancellation_token=CancellationToken("cancel:" + parent),
                telemetry_correlation_id=parent,
                max_fanout=1,
            )
            count = position + 1
            results = await runner.run_batch(
                (job,),
                (ids[:count],),
                {"temperature": 0.0, "max_new_tokens": 1},
                custom_params_per_job=(
                    {
                        "qwen_exo_dflash": "target_only",
                        "qwen_exo_attention_diagnostic": {
                            "layer_ids": layers,
                            "token_count": count,
                        },
                    },
                ),
                extra_keys=(namespace + f":{sample_index}",),
            )
            if len(results) != 1 or results[0].prompt_tokens != count:
                raise AttentionDiagnosticError(
                    "Worker returned mismatched diagnostic prompt length", 503
                )
            samples.append(
                {
                    "query_position": position,
                    "query_text": tokens[position]["text"],
                    "layers": [
                        {
                            "layer_id": layer,
                            "weights": _weights(results[0].metadata, layer, count),
                        }
                        for layer in layers
                    ],
                }
            )
    except asyncio.CancelledError:
        await runner.cancel_parent(parent)
        raise
    except asyncio.TimeoutError as exc:
        await runner.cancel_parent(parent)
        raise AttentionDiagnosticError(
            "Attention diagnostic timed out; internal model work was aborted", 504
        ) from exc
    except AttentionDiagnosticError:
        raise
    except Exception as exc:
        raise AttentionDiagnosticError(
            "Attention diagnostic worker failed; no attention result is available", 503
        ) from exc
    finally:
        await runner.finish_parent(parent)
    warnings.append(
        "Reconstructed Full Attention estimate from cached keys and post-RoPE queries, averaged over all query heads. Not causal importance, GDN attribution, or a replay of production memory/bias."
    )
    return {
        "schema": "qwen-exo-attention-diagnostic-v1",
        "method": _METHOD,
        "model": str(
            getattr(manager, "served_model_name", None)
            or _config_value(
                getattr(manager, "model_config", None), "model_path", "unknown"
            )
        ),
        "prompt_tokens": len(ids),
        "rendered_prompt": rendered,
        "tokens": tokens,
        "samples": samples,
        "messages": messages,
        "warnings": list(dict.fromkeys(warnings)),
    }

async def run_attention_dependency_probe(
    runtime: Any,
    content: str,
    filename: str | None,
    end_message: int,
    end_token: int | None = None,
    probe_token: str | None = None,
    block_size: int = DEPENDENCY_BLOCK_DEFAULT,
) -> dict[str, Any]:
    if type(block_size) is not int or not 1 <= block_size <= DEPENDENCY_BLOCK_MAX:
        raise AttentionDiagnosticError("block_size must be 1..256")
    manager = runtime.tokenizer_manager
    tokenizer = getattr(manager, "tokenizer", None)
    if tokenizer is None:
        raise AttentionDiagnosticError("Active tokenizer is unavailable", 503)
    rendered, messages, ids, tokens, warnings = await asyncio.to_thread(
        _prepare_run, runtime, content, filename, end_message, end_token
    )
    if len(ids) < 1:
        raise AttentionDiagnosticError("Rendered prompt contains no scoreable prefix")
    if probe_token is not None and (not isinstance(probe_token, str) or not probe_token):
        raise AttentionDiagnosticError("probe_token must be a nonempty string")
    probe_ids = None
    if probe_token is not None:
        try:
            probe_ids = list(tokenizer(probe_token, add_special_tokens=False)["input_ids"])
        except Exception as exc:
            raise AttentionDiagnosticError("probe_token could not be tokenized") from exc
        if len(probe_ids) != 1 or type(probe_ids[0]) is not int:
            raise AttentionDiagnosticError("probe_token must tokenize to exactly one token")
    _preflight_worker(manager)
    parent = "attention-dependency-" + uuid.uuid4().hex
    namespace = "qwen-exo:v1:attention-dependency:" + parent
    deadline = time.monotonic() + DIAGNOSTIC_TIMEOUT_SECONDS

    def make_job(suffix: str) -> InternalJob:
        return InternalJob(
            parent_request_id=parent, turn_id=parent, job_id=f"{parent}-{suffix}",
            job_type=InternalJobType.CAUSAL_REPLAY, priority=-12,
            shared_prefix_key=namespace, token_budget=1, state_budget_bytes=0,
            deadline_monotonic=deadline,
            cancellation_token=CancellationToken("cancel:" + parent),
            telemetry_correlation_id=parent,
            max_fanout=min(32, runtime.internal_jobs.max_fanout),
        )
    async def score(prefix: list[int], token_id: int, suffix: str) -> float:
        result = await runtime.internal_jobs.run_score_batch(
            (make_job(suffix),), ((tuple(prefix) + (token_id,)),), (max(0, len(prefix) - 1),),
            {"temperature": 0, "top_p": 1, "top_k": 1, "skip_special_tokens": True},
            extra_keys=(namespace + ":" + suffix,),
        )
        if len(result) != 1 or result[0].prompt_tokens != len(prefix) + 1:
            raise AttentionDiagnosticError("Worker returned mismatched dependency score", 503)
        values = result[0].token_logprobs
        if len(values) != 1 or not math.isfinite(values[0]):
            raise AttentionDiagnosticError("Worker returned no fixed-token logprob", 503)
        return float(values[0])

    try:
        if probe_ids is None:
            generated = await runtime.internal_jobs.run_batch(
                (make_job("baseline-generate"),), (ids,),
                {"temperature": 0, "top_p": 1, "top_k": 1, "skip_special_tokens": True},
                extra_keys=(namespace + ":baseline-generate",),
            )
            output_ids = generated[0].metadata.get("output_ids") if len(generated) == 1 else None
            if not isinstance(output_ids, (tuple, list)) or len(output_ids) != 1:
                raise AttentionDiagnosticError("Worker did not return a scoreable baseline token", 503)
            probe_ids = [int(output_ids[0])]
            warnings.append("probe_token omitted; the target model's greedy baseline next token was selected explicitly.")
        base = await score(ids, probe_ids[0], "base")
        blocks = []
        for start in range(0, len(ids), block_size):
            if len(blocks) >= DEPENDENCY_MAX_BLOCKS:
                warnings.append("Dependency blocks were capped at 32; remaining tokens were not probed.")
                break
            end = min(len(ids), start + block_size)
            prefix = ids[:start] + ids[end:]
            value = await score(prefix, probe_ids[0], f"block-{start}")
            blocks.append({"start": start, "end": end, "token_start": start,
                           "token_end": end, "text": rendered[tokens[start]["start"]:tokens[end - 1]["end"]],
                           "ablated_logprob": value, "delta": base - value})
    except asyncio.CancelledError:
        await runtime.internal_jobs.cancel_parent(parent)
        raise
    except asyncio.TimeoutError as exc:
        raise AttentionDiagnosticError(
            "Dependency probe timed out; internal model work was aborted", 504
        ) from exc
    except Exception as exc:
        raise AttentionDiagnosticError("Dependency probe worker failed; no result is available", 503) from exc
    finally:
        await runtime.internal_jobs.finish_parent(parent)
    return {"schema": "qwen-exo-attention-dependency-v1", "method": "span_ablation_fixed_next_token_logprob",
            "probe_token": probe_token if probe_token is not None else tokenizer.decode(probe_ids),
            "base_logprob": base, "blocks": blocks, "warnings": list(dict.fromkeys(warnings)),
            "tokens": tokens, "messages": messages, "rendered_prompt": rendered}
