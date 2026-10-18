# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import replace
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Optional,
    Union,
)

import jinja2
import openai.types.responses as openai_responses_types
import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as ResponseReasoningSummaryAddedPart,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as ResponseReasoningSummaryDonePart,
)
from openai_harmony import Message as OpenAIMessage
from openai_harmony import Role
from qwen_exo_booster.initial_gdn_request import bind_initial_gdn_request
from qwen_exo_booster.memory_hints import (
    MemoryTurnContext,
    is_memory_block,
    memory_block_hints,
)
from qwen_exo_booster.memory_tools import MEMORY_TOOL_NAMES

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    Function,
    MessageProcessingResult,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponseReasoningParam,
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.bounded_store import BoundedStore
from sglang.srt.entrypoints.openai.memory_tool_rounds import (
    MAX_MEMORY_TOOL_ROUNDS,
    MEMORY_ROUND_META_KEY,
    ROUND_LIMIT_OUTPUT,
    ToolResponseTemplate,
    generation_prompt_ids,
    is_memory_tool,
    mcp_call_item,
    mcp_item_to_chat_messages,
    neutralize_control_markers,
    new_memory_call,
    split_memory_calls,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import (
    FileMemoryToolServer,
    MCPToolServer,
    ToolServer,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import QWEN_EXO_REASONING_BUDGET_KEY
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)

_QWEN_EXO_SELF_CHECK_START = "<qwen_exo_self_check>"
_QWEN_EXO_SELF_CHECK_END = "</qwen_exo_self_check>"

_QWEN_EXO_DFLASH_THINK_PHASE = "qwen_exo_dflash_think_phase"
# Qwen chat templates inject a vendor identity system prompt when a
# conversation has no system message; GPT-style clients often send only user
# turns. A neutral system turn keeps the template from adding its own.
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_CONTEXT_LENGTH_ERROR_CODE = "context_length_exceeded"
# Identifies the synthetic tool message that carries retrieved QWEN-EXO memory.


_STORED_RESPONSE_MAX_ENTRIES = 4096
# Matches the idle TTL gateways keep chain baselines for, with headroom.
_STORED_RESPONSE_IDLE_TTL_SECONDS = 2 * 60 * 60
_STORED_MESSAGES_MAX_BYTES = 2 * 1024 * 1024 * 1024


def _chat_message_text(message: dict) -> str:
    """Plain text of a chat message's content (string or text parts)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return ""


def _stored_messages_size(messages: Any) -> int:
    try:
        return len(orjson.dumps(messages, default=str))
    except TypeError:
        return 0


class _ResponsePromptEncoding:
    """Reuse exact encodings within one response, retaining only its latest text."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._prompt: str | None = None
        self._encodings: dict[bool | None, list[int]] = {}

    def encode(
        self, prompt: str, *, add_special_tokens: bool | None = None
    ) -> list[int]:
        if prompt != self._prompt:
            self._prompt = prompt
            self._encodings.clear()
        if add_special_tokens not in self._encodings:
            # Omitted tokenizer options must remain distinct from explicit False.
            self._encodings[add_special_tokens] = (
                self._tokenizer.encode(prompt)
                if add_special_tokens is None
                else self._tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens
                )
            )
        return self._encodings[add_special_tokens]


class _ResponsesInputError(ValueError):
    """Invalid client history, distinct from internal reasoning scoring failures."""


def _log_response_shape(*, request_id, stream, status, items, finish_reason, usage):
    """One structural line per response for diagnosing agent turns that end
    without a tool call: item kinds, malformed calls and token counts only,
    never generated text."""
    kinds = [item.type for item in items]
    calls = [item for item in items if item.type == "function_call"]
    malformed_json = 0
    for call in calls:
        try:
            json.loads(call.arguments or "")
        except ValueError:
            malformed_json += 1
    logger.warning(
        "QWEN_EXO_RESPONSE_SHAPE request_id=%s stream=%s status=%s items=%s "
        "function_calls=%d nameless_calls=%d malformed_json_calls=%d "
        "completion_tokens=%s reasoning_tokens=%s finish_reason=%s",
        request_id,
        stream,
        status,
        kinds,
        len(calls),
        sum(not call.name for call in calls),
        malformed_json,
        usage.completion_tokens if usage else None,
        usage.reasoning_tokens if usage else None,
        finish_reason,
    )


class _QwenExoSelfAskSpillRouter:
    """Keeps phase-two Self-Ask continuations in the reasoning channel."""

    _QUESTION_PREFIX = "Self-question:"
    _THINK_END = "</think>"

    def __init__(self) -> None:
        self._armed = False
        self._in_self_ask = False
        self._buffer = ""

    def arm(self) -> None:
        self._armed = True
        self._in_self_ask = False
        self._buffer = ""

    @staticmethod
    def _partial_token_suffix_length(text: str, token: str) -> int:
        for size in range(min(len(text), len(token) - 1), 0, -1):
            if text.endswith(token[:size]):
                return size
        return 0

    def feed(self, text: str, *, final: bool = False) -> tuple[str, str]:
        if not self._armed:
            return "", str(text or "")
        self._buffer += str(text or "")
        reasoning_parts: list[str] = []
        normal_text = ""

        while self._buffer:
            if self._in_self_ask:
                end = self._buffer.find(self._THINK_END)
                if end >= 0:
                    reasoning_parts.append(self._buffer[:end])
                    self._buffer = self._buffer[end + len(self._THINK_END) :]
                    self._in_self_ask = False
                    continue
                keep = self._partial_token_suffix_length(self._buffer, self._THINK_END)
                emit_end = len(self._buffer) - keep
                if emit_end:
                    reasoning_parts.append(self._buffer[:emit_end])
                    self._buffer = self._buffer[emit_end:]
                break

            stripped = self._buffer.lstrip()
            leading = self._buffer[: len(self._buffer) - len(stripped)]
            if not stripped:
                break
            if self._QUESTION_PREFIX.startswith(stripped):
                break
            if stripped.startswith(self._QUESTION_PREFIX):
                reasoning_parts.append(leading)
                self._buffer = stripped
                self._in_self_ask = True
                continue
            normal_text = self._buffer
            self._buffer = ""
            self._armed = False
            break

        if final and self._buffer:
            stripped = self._buffer.lstrip()
            if self._in_self_ask or (
                stripped and self._QUESTION_PREFIX.startswith(stripped)
            ):
                reasoning_parts.append(self._buffer)
            else:
                normal_text += self._buffer
            self._buffer = ""
            self._armed = False
            self._in_self_ask = False

        return "".join(reasoning_parts), normal_text


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        tool_server: Optional[ToolServer] = None,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

        # Parent OpenAIServingChat.__init__ already populated default_sampling_params.
        if not isinstance(self.default_sampling_params, dict):
            self.default_sampling_params = {}

        self.supports_browsing = (
            tool_server.has_tool("browser") if tool_server else False
        )
        self.supports_code_interpreter = (
            tool_server.has_tool("python") if tool_server else False
        )
        self.tool_server = tool_server
        # Get from model config
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Stored responses and their conversation messages, for
        # previous_response_id chaining and retrieval. Bounded: gateways that
        # chain every turn force store=true, and each msg_store entry holds a
        # conversation's full history.
        self.response_store: BoundedStore[ResponsesResponse] = BoundedStore(
            max_entries=_STORED_RESPONSE_MAX_ENTRIES,
            idle_ttl_seconds=_STORED_RESPONSE_IDLE_TTL_SECONDS,
        )
        self.response_store_lock = asyncio.Lock()
        self.msg_store: BoundedStore[
            Union[list[ChatCompletionMessageParam], list[OpenAIMessage]]
        ] = BoundedStore(
            max_entries=_STORED_RESPONSE_MAX_ENTRIES,
            idle_ttl_seconds=_STORED_RESPONSE_IDLE_TTL_SECONDS,
            max_bytes=_STORED_MESSAGES_MAX_BYTES,
            sizer=_stored_messages_size,
        )

        self.background_tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _has_response_tool(request: ResponsesRequest, *tool_types: str) -> bool:
        return any(tool.type in tool_types for tool in (request.tools or []))

    async def prevalidate_qwen_exo_request(
        self, request: ResponsesRequest
    ) -> ORJSONResponse | None:
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")
        if request.tool_choice == "required" and not any(
            tool.type == "function" for tool in (request.tools or [])
        ):
            return self.create_error_response(
                'tool_choice="required" requires at least one tool with '
                'type="function"; other built-in tool types cannot be forced.'
            )
        if not self.use_harmony and self._has_response_tool(
            request,
            "web_search",
            "web_search_preview",
            "code_interpreter",
        ):
            return self.create_error_response(
                "Built-in web_search and code_interpreter tools are not "
                "supported by the Qwen non-Harmony Responses path; use a "
                "client-executed function tool instead."
            )
        if (
            self.use_harmony
            and self._has_response_tool(request, "web_search", "web_search_preview")
            and not self.supports_browsing
        ):
            return self.create_error_response(
                "web_search requires a browser backend before QWEN-EXO "
                "memory preparation can run."
            )
        if (
            isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and self._has_response_tool(
                request, "web_search", "web_search_preview", "code_interpreter"
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background or streaming mode"
            )
        previous_response_id = request.previous_response_id
        if previous_response_id is not None:
            if not previous_response_id.startswith("resp_"):
                return self._make_invalid_id_error(previous_response_id)
            async with self.response_store_lock:
                previous_response = self.response_store.get(previous_response_id)
                if previous_response is None:
                    return self._make_not_found_error(previous_response_id)
                if (
                    previous_response.status != "completed"
                    or previous_response_id not in self.msg_store
                ):
                    return self.create_error_response(
                        "previous_response_id is not a completed, replayable response",
                        err_type="response_not_replayable",
                        status_code=HTTPStatus.CONFLICT,
                        param="previous_response_id",
                    )
        return None

    # error helpers dedicated for v1/responses
    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: int = 400,
        param: Optional[str] = None,
        code: str | int | None = None,
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code if code is None else code,
        }
        return ORJSONResponse(content={"error": nested_error}, status_code=status_code)

    async def register_cancelled_response(
        self, request: ResponsesRequest
    ) -> ResponsesResponse:
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name=request.model,
            created_time=int(time.time()),
            output=[],
            status="cancelled",
            usage=None,
        )
        async with self.response_store_lock:
            existing = self.response_store.get(request.request_id)
            if existing is not None and existing.status == "cancelled":
                return existing
            self.response_store[request.request_id] = response
        return response

    async def register_in_progress_response(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        *,
        model_name: str,
        created_time: int,
    ) -> ResponsesResponse:
        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        )
        async with self.response_store_lock:
            existing = self.response_store.get(response.id)
            if existing is not None and existing.status == "cancelled":
                return existing
            self.response_store[response.id] = response
        return response

    async def register_pending_cancelled_response(
        self, response_id: str
    ) -> ResponsesResponse:
        response = ResponsesResponse(
            id=str(response_id),
            model=str(self.tokenizer_manager.served_model_name),
            status="cancelled",
            store=True,
            metadata={},
        )
        async with self.response_store_lock:
            existing = self.response_store.get(response.id)
            if existing is not None:
                return existing
            self.response_store[response.id] = response
        return response

    async def register_compaction_response(
        self,
        *,
        response_id: str,
        model_name: str | None,
        user_items: Iterable[dict[str, Any]],
        summary: str,
    ) -> ResponsesResponse:
        summary = str(summary).strip()
        if not summary:
            raise ValueError("Compaction replay summary cannot be empty")
        public_messages: list[ChatCompletionMessageParam] = []
        for item in user_items:
            public_messages.extend(  # type: ignore[arg-type]
                self._normalize_response_items_for_chat(item)
            )
        output_text = ResponseOutputText(
            text=summary,
            annotations=[],
            type="output_text",
            logprobs=None,
        )
        output_message = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
            status="completed",
            type="message",
        )
        response = ResponsesResponse(
            id=str(response_id),
            model=str(model_name or self.tokenizer_manager.served_model_name),
            output=[output_message],
            status="completed",
            store=True,
            metadata={},
        )
        async with self.response_store_lock:
            self.msg_store[response.id] = public_messages
            self.response_store[response.id] = response
        return response

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": status_code,
                }
            }
        )

    def _request_id_prefix(self) -> str:
        return "resp_"

    @staticmethod
    def _requested_reasoning_effort(request: ResponsesRequest | None) -> str | None:
        if (
            request is None
            or request.reasoning is None
            or "effort" not in request.reasoning.model_fields_set
        ):
            return None
        return request.reasoning.effort

    async def _resolve_reasoning_effort(self, request, prev_response, runtime):
        if runtime is None:
            return request
        config = getattr(runtime, "config", None)
        force_auto = getattr(config, "force_auto_reasoning", False)
        effort = self._requested_reasoning_effort(request)
        if not force_auto and (
            getattr(config, "reasoning_mode", "fixed") != "dynamic"
            or effort is not None
        ):
            return request
        try:
            messages = self._construct_input_messages(
                request, prev_response
            )
        except (ValueError, TypeError) as exc:
            raise _ResponsesInputError(str(exc)) from exc
        selected = await runtime.plan_reasoning_effort(
            request.request_id,
            messages,
            request=request,
            previous_response_id=(
                prev_response.id if prev_response is not None else request.previous_response_id
            ),
        )
        reasoning = (
            request.reasoning.model_copy(update={"effort": selected})
            if request.reasoning is not None
            else ResponseReasoningParam(effort=selected)
        )
        resolved = request.model_copy(update={"reasoning": reasoning})
        resolved._qwen_exo_auto_reasoning = True
        return resolved

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ORJSONResponse]:
        # Validate model
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")

        # FIXME: If the engine is dead, raise an error
        # This is required for the streaming case

        # ``tool_choice="required"`` only works with ``function`` tools.
        if request.tool_choice == "required" and not any(
            tool.type == "function" for tool in (request.tools or [])
        ):
            return self.create_error_response(
                'tool_choice="required" requires at least one tool with '
                'type="function"; other built-in tool types cannot be forced.'
            )

        if (
            self.use_harmony
            and self._has_response_tool(request, "web_search", "web_search_preview")
            and not self.supports_browsing
        ):
            return self.create_error_response(
                "web_search requires a browser backend. Set EXA_API_KEY on the "
                "SGLang server to enable native Exa-backed web search, or "
                "configure a browser MCP tool server. Create an Exa API key at "
                "https://dashboard.exa.ai/api-keys."
            )

        # Handle the previous response ID
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None
        qwen_exo_runtime = (
            getattr(raw_request.app.state, "qwen_exo_runtime", None)
            if raw_request is not None
            else None
        )
        try:
            request = await self._resolve_reasoning_effort(
                request, prev_response, qwen_exo_runtime
            )
        except asyncio.CancelledError:
            raise
        except _ResponsesInputError as exc:
            return self.create_error_response(
                str(exc), err_type="invalid_request_error", status_code=400
            )
        except Exception:
            logger.exception("QWEN-EXO dynamic reasoning selection failed")
            return self.create_error_response(
                "Dynamic reasoning selection failed; no generation was started",
                err_type="reasoning_planning_error",
                status_code=503,
            )

        if qwen_exo_runtime is not None and not self.use_harmony:
            request._qwen_exo_memory_block = (
                await qwen_exo_runtime.memory_block_for_turn(
                    request.request_id,
                    blocked_reason=self._memory_tools_blocked_reason(request),
                    turn=self._memory_turn_context(request, prev_response),
                )
            )

        try:
            model_name = request.model
            tokenizer = self.tokenizer_manager.tokenizer
            prompt_encoding = _ResponsePromptEncoding(tokenizer)
            processed_messages: Optional[MessageProcessingResult] = None

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                (
                    messages,
                    request_prompts,
                    engine_prompts,
                    processed_messages,
                ) = await self._make_request(request, prev_response, tokenizer)

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata
        qwen_exo_observe = (
            qwen_exo_runtime is not None and qwen_exo_runtime.observer.mode != "off"
        )
        qwen_exo_score_bias = bool(
            qwen_exo_runtime is not None
            and getattr(qwen_exo_runtime, "score_bias_enabled", False)
        )
        if qwen_exo_runtime is not None:
            context_len = int(self.tokenizer_manager.model_config.context_len)
            output_cap = int(qwen_exo_runtime.max_output_tokens)
            requested_output_tokens = int(request.max_output_tokens or output_cap)
            reserved_output_tokens = max(
                1, min(requested_output_tokens, output_cap)
            ) + int(self.tokenizer_manager.num_reserved_tokens)

            def rendered_length(prompts: list[Any]) -> int:
                lengths = [
                    (
                        len(prompt)
                        if isinstance(prompt, list)
                        else (
                            len(prompt_encoding.encode(prompt))
                            if isinstance(prompt, str)
                            else 0
                        )
                    )
                    for prompt in prompts
                ]
                return max(lengths, default=0)

            async def rebuild_without_private_state(current_request):
                if self.use_harmony:
                    rebuilt = self._make_request_with_harmony(
                        current_request, prev_response
                    )
                    return (*rebuilt, None)
                return await self._make_request(
                    current_request, prev_response, tokenizer
                )

            has_multimodal_input = processed_messages is not None and any(
                getattr(processed_messages, field, None)
                for field in (
                    "image_data",
                    "video_data",
                    "audio_data",
                    "modalities",
                )
            )
            rendered_prompt_tokens = rendered_length(engine_prompts)
            over_context_budget = (
                rendered_prompt_tokens + reserved_output_tokens > context_len
            )
            if qwen_exo_runtime.has_restored_capsule(request.request_id) and (
                has_multimodal_input or over_context_budget
            ):
                request = qwen_exo_runtime.drop_restored_capsule_for_context(
                    request,
                    rendered_prompt_tokens=rendered_prompt_tokens,
                    context_length=context_len,
                    reserved_output_tokens=reserved_output_tokens,
                    reason=(
                        "multimodal_post_expansion_unknown"
                        if has_multimodal_input
                        else "context_capacity"
                    ),
                )
                try:
                    (
                        messages,
                        request_prompts,
                        engine_prompts,
                        processed_messages,
                    ) = await rebuild_without_private_state(request)
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    jinja2.TemplateError,
                ) as exc:
                    logger.exception("Error rebuilding prompt without QWEN-EXO capsule")
                    return self.create_error_response(f"{exc} {exc.__cause__}")
            has_multimodal_input = processed_messages is not None and any(
                getattr(processed_messages, field, None)
                for field in (
                    "image_data",
                    "video_data",
                    "audio_data",
                    "modalities",
                )
            )
            rendered_prompt_tokens = rendered_length(engine_prompts)
            over_context_budget = (
                rendered_prompt_tokens + reserved_output_tokens > context_len
            )
            original_instructions = getattr(
                raw_request.state,
                "qwen_exo_original_instructions",
                request.instructions,
            )
            original_extra_key = getattr(
                raw_request.state,
                "qwen_exo_original_extra_key",
                request.extra_key,
            )
            has_remaining_private_context = (
                request.instructions != original_instructions
                or request.extra_key != original_extra_key
            )
            if has_remaining_private_context and (
                has_multimodal_input or over_context_budget
            ):
                request = request.model_copy(
                    update={
                        "instructions": original_instructions,
                        "extra_key": original_extra_key,
                    }
                )
                qwen_exo_runtime.telemetry.emit(
                    request.request_id,
                    "private_context.dropped_context_budget",
                    {
                        "rendered_prompt_tokens": rendered_prompt_tokens,
                        "context_length": context_len,
                        "reserved_output_tokens": reserved_output_tokens,
                        "reason": (
                            "multimodal_post_expansion_unknown"
                            if has_multimodal_input
                            else "context_capacity"
                        ),
                    },
                )
                try:
                    (
                        messages,
                        request_prompts,
                        engine_prompts,
                        processed_messages,
                    ) = await rebuild_without_private_state(request)
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    jinja2.TemplateError,
                ) as exc:
                    logger.exception(
                        "Error rebuilding prompt without QWEN-EXO private context"
                    )
                    return self.create_error_response(f"{exc} {exc.__cause__}")

        if qwen_exo_runtime is not None:
            public_instructions = (
                getattr(
                    raw_request.state,
                    "qwen_exo_original_instructions",
                    request.instructions,
                )
                if raw_request is not None
                else request.instructions
            )
            request = request.model_copy(update={"instructions": public_instructions})

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ("web_search", "web_search_preview", "code_interpreter")
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and "
                "streaming mode"
            )

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        tool_list = []
        if self.use_harmony:
            if self.supports_browsing:
                tool_list.append("browser")
            if self.supports_code_interpreter:
                tool_list.append("python")
        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.get_tool_session(tool_name)
                        )
                        for tool_name in tool_list
                    }
                    tool_sessions = {}
                    for tool_name in tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(tool_list) == 0
                    tool_sessions = {}
                for i, engine_prompt in enumerate(engine_prompts):
                    # Calculate default max tokens from context length minus prompt length
                    if isinstance(engine_prompt, list):
                        prompt_length = len(engine_prompt)
                    elif isinstance(engine_prompt, str):
                        prompt_length = len(prompt_encoding.encode(engine_prompt))
                    else:
                        prompt_length = 0

                    context_len = (
                        self.tokenizer_manager.model_config.context_len
                        if hasattr(self.tokenizer_manager.model_config, "context_len")
                        else 4096
                    )
                    # Account for reserved tokens (e.g., EAGLE speculative decoding slots)
                    # that the tokenizer_manager adds during validation
                    num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                    input_token_count = prompt_length + num_reserved_tokens
                    if input_token_count >= context_len:
                        message = (
                            f"The input ({input_token_count} tokens) is longer than "
                            f"the model's context length ({context_len} tokens)."
                        )
                        return self.create_error_response(
                            message,
                            err_type="invalid_request_error",
                            status_code=HTTPStatus.BAD_REQUEST,
                            param="input",
                            code=_CONTEXT_LENGTH_ERROR_CODE,
                        )
                    default_max_tokens = max(
                        context_len - prompt_length - num_reserved_tokens, 512
                    )  # Ensure minimum 512 tokens
                    if qwen_exo_runtime is not None:
                        default_max_tokens = min(
                            default_max_tokens,
                            int(qwen_exo_runtime.max_output_tokens),
                        )
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.default_sampling_params,
                        stop=(
                            processed_messages.stop
                            if processed_messages
                            else request.stop
                        ),
                        tool_call_constraint=(
                            processed_messages.tool_call_constraint
                            if processed_messages
                            else None
                        ),
                    )
                    if qwen_exo_runtime is not None:
                        custom_params, bound_extra_key = bind_initial_gdn_request(
                            qwen_exo_runtime,
                            custom_params=dict(
                                sampling_params.get("custom_params") or {}
                            ),
                            extra_key=self._compute_extra_key(request),
                            previous_response_id=request.previous_response_id,
                            response_id=request.request_id,
                        )
                        # Radix matching is exact-token, so shared heads (system
                        # prompt, tools, a forked subagent's inherited context)
                        # are reused across conversations, as in chat/completions.
                        custom_params["qwen_exo_kind"] = "user"
                        score_bias_builder = getattr(
                            qwen_exo_runtime, "score_bias_payload", None
                        )
                        if callable(score_bias_builder):
                            score_bias_prompt_ids = (
                                list(engine_prompt)
                                if isinstance(engine_prompt, list)
                                else prompt_encoding.encode(
                                    engine_prompt, add_special_tokens=False
                                ).copy()
                            )
                            score_bias_blocks = score_bias_builder(
                                request.request_id, score_bias_prompt_ids
                            )
                            if score_bias_blocks:
                                custom_params["qwen_exo_score_bias_blocks"] = list(
                                    score_bias_blocks
                                )
                        capture_builder = getattr(
                            qwen_exo_runtime, "score_bias_capture_payload", None
                        )
                        if callable(capture_builder):
                            trajectory_spans = capture_builder(
                                request.request_id, score_bias_prompt_ids
                            )
                            if trajectory_spans:
                                custom_params["qwen_exo_trajectory_spans"] = list(
                                    trajectory_spans
                                )
                        query_builder = getattr(
                            qwen_exo_runtime, "score_bias_user_query_payload", None
                        )
                        if callable(query_builder):
                            user_query = query_builder(request, score_bias_prompt_ids)
                            if user_query:
                                custom_params["qwen_exo_score_bias_user_query"] = (
                                    user_query
                                )
                        latent_builder = getattr(
                            qwen_exo_runtime, "latent_transplant_payload", None
                        )
                        if callable(latent_builder):
                            latent_transplant = latent_builder(request)
                            if latent_transplant:
                                custom_params["qwen_exo_latent_transplant"] = (
                                    latent_transplant
                                )
                        editor_request = qwen_exo_runtime.activation_editor_request(
                            custom_params.get("qwen_exo_activation_editor")
                        )
                        editor_spec = editor_request["spec"]
                        if editor_spec is not None:
                            custom_params["qwen_exo_activation_editor"] = editor_spec
                        editor_cache_identity = str(editor_request["cache_identity"])
                        sampling_params["custom_params"] = custom_params
                    effective_extra_key = self._compute_extra_key(request)
                    if qwen_exo_runtime is not None:
                        effective_extra_key = bound_extra_key
                        editor_marker = f"qwen-exo-editor={editor_cache_identity}"
                        effective_extra_key = (
                            f"{effective_extra_key}|{editor_marker}"
                            if effective_extra_key
                            else editor_marker
                        )

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    # Create GenerateReqInput for SGLang
                    if isinstance(engine_prompt, str):
                        prompt_kwargs = {"text": engine_prompt}
                    else:
                        prompt_kwargs = {"input_ids": engine_prompt}

                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        image_data=(
                            processed_messages.image_data
                            if processed_messages
                            else None
                        ),
                        video_data=(
                            processed_messages.video_data
                            if processed_messages
                            else None
                        ),
                        audio_data=(
                            processed_messages.audio_data
                            if processed_messages
                            else None
                        ),
                        modalities=(
                            processed_messages.modalities
                            if processed_messages
                            else None
                        ),
                        sampling_params=sampling_params,
                        stream=request.stream,
                        rid=request.request_id,
                        session_id=request.session_id,
                        extra_key=effective_extra_key,
                        background=request.background,
                        return_logprob=qwen_exo_observe,
                        logprob_start_len=0 if qwen_exo_score_bias else None,
                        no_logs=qwen_exo_runtime is not None,
                        # Same contract as chat: a tool-call grammar (strict
                        # tools) must stay inactive until </think>, otherwise it
                        # masks </think> and forces a tool call out of the
                        # reasoning phase.
                        require_reasoning=self._is_thinking_enabled_for_request(
                            request
                        ),
                    )

                    if (
                        isinstance(context, SimpleContext)
                        and (
                            adapted_request.input_ids is not None
                            or isinstance(adapted_request.text, str)
                        )
                        and self._memory_tools_active(request)
                    ):
                        generator = self._generate_with_memory_tools(
                            request.request_id,
                            request_prompts[i],
                            adapted_request,
                            sampling_params,
                            context,
                            raw_request=raw_request,
                            response_request=request,
                            priority=request.priority,
                            prompt_encoding=prompt_encoding,
                        )
                    else:
                        generator = self._generate_with_builtin_tools(
                            request.request_id,
                            request_prompts[i],
                            adapted_request,
                            sampling_params,
                            context,
                            raw_request=raw_request,
                            response_request=request,
                            priority=request.priority,
                            prompt_encoding=prompt_encoding,
                        )
                    if qwen_exo_runtime is not None:
                        generator = qwen_exo_runtime.track_generation(
                            request.request_id, generator
                        )
                    generators.append(generator)
            except ValueError as e:
                return self._error_response_for_exception(e)

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages. The retrieved-memory attachment is
            # deliberately left out: it is evidence for this turn only, and
            # storing it would replay it as permanent history on every later
            # turn of a ``previous_response_id`` chain.
            if request.store:
                public_messages = (
                    self._construct_input_messages_with_harmony(request, prev_response)
                    if self.use_harmony
                    else self._conversation_messages(request, prev_response)
                )
                self.msg_store[request.request_id] = public_messages
            created_time = int(time.time())
            if request.store and not request.background:
                pending_response = await self.register_in_progress_response(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                )
                if pending_response.status == "cancelled":
                    self.tokenizer_manager.abort_request(rid=request.request_id)
                    return pending_response

            if request.background:
                created_time = int(time.time())
                async with self.response_store_lock:
                    dispatch_allowed = (
                        qwen_exo_runtime is None
                        or await qwen_exo_runtime.claim_pending_background_request(
                            request.request_id
                        )
                    )
                    response = ResponsesResponse.from_request(
                        request,
                        sampling_params,
                        model_name=model_name,
                        created_time=created_time,
                        output=[],
                        status="queued" if dispatch_allowed else "cancelled",
                        usage=None,
                    )
                    self.response_store[response.id] = response
                    if not dispatch_allowed:
                        qwen_exo_runtime.acknowledge_request_cancellation(
                            request.request_id
                        )
                        return response

                    # Register the task while holding the store lock so cancel cannot
                    # observe a queued response before its execution is cancellable.
                    task = asyncio.create_task(
                        self._run_background_request(
                            request,
                            sampling_params,
                            result_generator,
                            context,
                            model_name,
                            tokenizer,
                            request_metadata,
                            created_time,
                        ),
                        name=f"create_{response.id}",
                    )
                    self.background_tasks[response.id] = task
                    task.add_done_callback(
                        lambda _: self.background_tasks.pop(response.id, None)
                    )
                return response

            if request.stream:
                if self.use_harmony:
                    return self.responses_stream_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    )
                return self.responses_stream_generator_non_harmony(
                    request,
                    sampling_params,
                    result_generator,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                )
            try:
                result: Union[ORJSONResponse, ResponsesResponse] = (
                    await self.responses_full_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time=created_time,
                    )
                )
                return result
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                response = self.create_error_response(
                    str(detail.get("message", exc.detail)),
                    err_type=str(detail.get("code", "scheduler_abort")),
                    status_code=exc.status_code,
                )
                for key, value in (exc.headers or {}).items():
                    response.headers[key] = value
                return response
            except Exception as e:
                return self._error_response_for_exception(e)
        return self.create_error_response("Unknown error")

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        messages = self._construct_input_messages(request, prev_response)
        if request._qwen_exo_memory_block:
            # The block is this turn's first output item and renders as its
            # own assistant message, exactly as the client's replay of that
            # item renders next turn, so the cached prefix keeps matching.
            messages.append(
                {"role": "assistant", "content": request._qwen_exo_memory_block}
            )

        chat_tools = self._chat_tools_for_request(request)
        # Automatic selection controls the hard token budget, not the leading
        # system tokens. Changing native effort here destroys long-prefix reuse.
        # The shared "high" spelling maps to Qwen3.8's native default "xhigh".
        template_effort = (
            "high" if request._qwen_exo_auto_reasoning
            else (request.reasoning.effort if request.reasoning else None)
        )
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            stream=request.stream,
            tools=chat_tools or None,
            tool_choice=request.tool_choice if chat_tools else "none",
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            stop=request.stop,
            reasoning_effort=template_effort,
            chat_template_kwargs=(
                {"reasoning_effort": template_effort}
                if request._qwen_exo_auto_reasoning else None
            ),
        )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._process_messages(chat_request, is_multimodal)
        request._qwen_exo_template_kwargs = dict(chat_request.chat_template_kwargs or {})

        if is_multimodal:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]

        return messages, request_prompts, engine_prompts, processed_messages

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in " "response API"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = prompt_token_ids
        return messages, [prompt_token_ids], [engine_prompt]

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self._error_response_for_exception(e)

        finish_reason: Mapping[str, Any] | None = None
        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # num_reasoning_tokens isn't wired through HarmonyContext yet; stays 0.
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            output = (
                self._memory_block_output_items(request)
                + self._memory_round_output_items(request, context, tokenizer)
                + self._make_response_output_items(request, final_res["text"], tokenizer)
            )

            # Calculate usage from actual output
            num_reasoning_tokens = 0
            meta_info = None
            if isinstance(final_res, dict) and isinstance(
                final_res.get("meta_info"), dict
            ):
                meta_info = final_res["meta_info"]
            elif hasattr(final_res, "meta_info"):
                meta_info = final_res.meta_info

            if meta_info is not None:
                num_prompt_tokens = meta_info.get("prompt_tokens", 0)
                num_generated_tokens = meta_info.get("completion_tokens", 0)
                num_cached_tokens = meta_info.get("cached_tokens", 0)
                num_reasoning_tokens = meta_info.get("reasoning_tokens", 0)
                raw_finish_reason = meta_info.get("finish_reason")
                if isinstance(raw_finish_reason, Mapping):
                    finish_reason = raw_finish_reason
            elif isinstance(final_res, dict) and (
                final_res.get("prompt_token_ids") is not None
                or final_res.get("output_ids") is not None
            ):
                prompt_token_ids = final_res.get("prompt_token_ids") or []
                output_token_ids = final_res.get("output_ids") or []
                num_prompt_tokens = len(prompt_token_ids)
                num_generated_tokens = len(output_token_ids)
                num_cached_tokens = final_res.get("num_cached_tokens", 0)
            elif hasattr(final_res, "prompt_token_ids") and hasattr(
                final_res, "outputs"
            ):
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res.prompt_token_ids) if final_res.prompt_token_ids else 0
                )
                num_generated_tokens = (
                    len(final_res.outputs[0].token_ids)
                    if final_res.outputs and final_res.outputs[0].token_ids
                    else 0
                )
                num_cached_tokens = getattr(final_res, "num_cached_tokens", 0)
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0
                num_reasoning_tokens = 0

        if self.use_harmony and request.store:
            assert isinstance(context, HarmonyContext)
            self._store_public_harmony_messages(request, context)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=num_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=num_cached_tokens
            )
        request_metadata.final_usage_info = usage

        response_status, incomplete_details = self._response_terminal_status(
            finish_reason
        )
        if not self.use_harmony:
            _log_response_shape(
                request_id=request.request_id,
                stream=False,
                status=response_status,
                items=output,
                finish_reason=finish_reason,
                usage=usage,
            )
        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status=response_status,
            usage=usage,
            incomplete_details=incomplete_details,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        return response

    @staticmethod
    def _wants_reasoning_summary(request: ResponsesRequest) -> bool:
        return request.reasoning is not None and request.reasoning.summary is not None

    def _is_thinking_enabled_for_request(self, request: ResponsesRequest) -> bool:
        """Whether to start the reasoning detector in thinking mode."""
        if not self.reasoning_parser:
            return False
        effort = request.reasoning.effort if request.reasoning is not None else None
        raw_template_kwargs = getattr(self, "default_chat_template_kwargs", None)
        template_kwargs = (
            dict(raw_template_kwargs)
            if isinstance(raw_template_kwargs, Mapping)
            else {}
        )
        if effort is None:
            toggle = (
                self.template_manager.reasoning_config.toggle_param
                if self.template_manager.reasoning_config is not None
                else None
            )
            if toggle is not None and toggle in template_kwargs:
                return bool(template_kwargs[toggle])
            if "enable_thinking" in template_kwargs:
                return bool(template_kwargs["enable_thinking"])
            if "thinking" in template_kwargs:
                return bool(template_kwargs["thinking"])
        if self.reasoning_parser == "hunyuan":
            return effort not in (None, "none", "no_think")
        if self.template_manager.force_reasoning:
            return True
        config = self.template_manager.reasoning_config
        if config is None:
            # Parser-only models (DeepSeek-R1, …) carry the thinking default in
            # the detector itself.
            detector = getattr(self, "_reasoning_detector", None)
            mode = getattr(detector, "reasoning_default", None) if detector else None
            if mode is None or mode == "always":
                return mode == "always"
            if mode == "mistral":
                return effort is not None and effort != "none"
            if mode in ("thinking", "enable_thinking"):
                return effort != "none"
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                return False
            return False
        if config.special_case == "always":
            return True
        if config.special_case == "mistral":
            return effort is not None and effort != "none"
        if config.toggle_param is None or config.default_enabled is None:
            return False
        if effort == "none":
            return False
        return bool(config.default_enabled)

    @staticmethod
    def _partition_qwen_exo_self_ask_spill(text: str) -> tuple[str, str]:
        router = _QwenExoSelfAskSpillRouter()
        router.arm()
        return router.feed(text, final=True)

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
    ):
        if self.reasoning_parser:
            # Templates that prefill ``<think>`` only emit the close tag, so
            # start the detector in thinking mode.
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
            spill_reasoning, content = self._partition_qwen_exo_self_ask_spill(content)
            if spill_reasoning:
                reasoning_content = f"{reasoning_content or ''}{spill_reasoning}"
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            # Mirror the single parsed blob into ``summary`` when the caller opts
            # in via ``reasoning.summary``; full trace stays in ``content``.
            wants_summary = self._wants_reasoning_summary(request)
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=(
                    [
                        ResponseReasoningSummary(
                            type="summary_text", text=reasoning_content
                        )
                    ]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(
                        type="reasoning_text", text=reasoning_content
                    ),
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        chat_tools = self._chat_tools_for_request(request)
        is_required = request.tool_choice == "required"
        tool_call_items: list[ResponseFunctionToolCall] = []
        parsed_via_native = False
        if (
            content
            and chat_tools
            and self.tool_call_parser
            and request.tool_choice != "none"
        ):
            parser = FunctionCallParser(
                chat_tools,
                self.tool_call_parser,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            should_try_native = (
                not is_required or parser.detector.supports_structural_tag()
            )
            if should_try_native and parser.has_tool_call(content):
                try:
                    content, call_info_list = parser.parse_non_stream(content)
                    for call_info in call_info_list:
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=call_info.parameters or "",
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=call_info.name,
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    parsed_via_native = bool(call_info_list)
                except Exception as e:
                    logger.error("Tool call parsing error: %s", e)

        if content and chat_tools and is_required and not parsed_via_native:
            try:
                tool_call_data = orjson.loads(content)
                if isinstance(tool_call_data, dict):
                    tool_call_data = [tool_call_data]
                if isinstance(tool_call_data, list):
                    for tool in tool_call_data:
                        if not isinstance(tool, dict) or "name" not in tool:
                            continue
                        arguments = json.dumps(
                            tool.get("parameters", {}), ensure_ascii=False
                        )
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=arguments,
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=tool["name"],
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    content = ""
            except Exception as e:
                logger.error("Required tool JSON parse error: %s", e)

        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        # Memory tool calls are server-executed; they surface as ``mcp_call``
        # items (see _memory_round_output_items), never as client calls.
        output_items.extend(
            item for item in tool_call_items if not is_memory_tool(item.name)
        )
        return output_items

    @staticmethod
    def _memory_block_output_items(request: ResponsesRequest) -> list:
        if not request._qwen_exo_memory_block:
            return []
        return [
            ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[
                    ResponseOutputText(
                        text=request._qwen_exo_memory_block,
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

    def _memory_round_output_items(
        self,
        request: ResponsesRequest,
        context: ConversationContext,
        tokenizer: Any,
    ) -> list:
        """Items for completed server-side memory rounds, in wire order:
        each round's reasoning/text, then its ``mcp_call`` items."""
        items: list = []
        for round_record in getattr(context, "qwen_exo_memory_rounds", ()):
            items.extend(
                item
                for item in self._make_response_output_items(
                    request, round_record["text"], tokenizer
                )
                # Whitespace the model put before its <tool_call> is not a message.
                if not (
                    isinstance(item, ResponseOutputMessage)
                    and not "".join(part.text for part in item.content).strip()
                )
            )
            items.extend(mcp_call_item(call) for call in round_record["calls"])
        return items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _memory_tools_active(self, request: ResponsesRequest) -> bool:
        """Whether the server-executed qwen-exo memory tools join this turn."""
        return self._memory_tools_blocked_reason(request) is None

    def _memory_tools_blocked_reason(self, request: ResponsesRequest) -> Optional[str]:
        """Why the memory tools stay out of this turn, or ``None`` if they join.

        Only under ``tool_choice="auto"``: a forced call would loop on the
        memory tools, and ``none`` must keep the prompt tool-free.
        """
        if not isinstance(self.tool_server, FileMemoryToolServer):
            return "no_tool_server"
        if self.use_harmony or not self.tool_call_parser:
            return "no_tool_parser"
        if request.tool_choice != "auto":
            return "tool_choice"
        if any(
            tool.type == "function" and tool.name in MEMORY_TOOL_NAMES
            for tool in request.tools
        ):
            return "client_defines_memory_tools"
        # The server loop continues on token ids; media prompts can't be.
        if self._request_has_media(request):
            return "media"
        return None

    def _memory_turn_context(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> MemoryTurnContext:
        """Read memory state from the conversation itself (chained or replayed):
        whether it searched, the server blocks already shown and their hints,
        and the text this turn reacts to (new tool output or user message)."""
        try:
            messages = self._conversation_messages(request, prev_response)
        except (ValueError, TypeError):
            return MemoryTurnContext()
        searched = False
        blocks = 0
        shown: set[tuple[str, int]] = set()
        last_assistant = -1
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if is_memory_block(content):
                blocks += 1
                shown |= memory_block_hints(content)
                continue
            last_assistant = index
            for call in message.get("tool_calls") or ():
                if (call.get("function") or {}).get("name") == "qwen_exo_memory_search":
                    searched = True
        latest = [
            _chat_message_text(message)
            for message in messages[last_assistant + 1 :]
            if isinstance(message, dict) and message.get("role") in ("user", "tool")
        ]
        # The original task keeps short tool output (a file list) on topic.
        task = next(
            (
                _chat_message_text(message)
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        recent = "\n".join(text for text in latest if text)[-3000:]
        return MemoryTurnContext(
            searched=searched,
            blocks_shown=blocks,
            shown_hints=frozenset(shown),
            query_text=recent if task in recent else f"{task[:1000]}\n{recent}".strip(),
        )

    @staticmethod
    def _request_has_media(request: ResponsesRequest) -> bool:
        if isinstance(request.input, str):
            return False
        text_parts = {"input_text", "output_text", "text", "reasoning_text", "summary_text"}
        for item in request.input:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if hasattr(part, "model_dump"):
                    part = part.model_dump(exclude_none=True)
                if isinstance(part, dict) and part.get("type") not in text_parts:
                    return True
        return False

    def _chat_tools_for_request(self, request: ResponsesRequest) -> list[Tool]:
        """Client function tools, then the memory tools. The memory schemas are
        constants appended last, so the rendered tool block is byte-stable
        across turns and the radix prefix keeps matching."""
        chat_tools = self._response_tools_to_chat_tools(request)
        if self._memory_tools_active(request):
            chat_tools.extend(
                Tool.model_validate(schema)
                for schema in self.tool_server.chat_tool_schemas
            )
        return chat_tools

    @staticmethod
    def _response_tools_to_chat_tools(request: ResponsesRequest) -> list[Tool]:
        # Only ``function`` tools flow to chat; built-ins go through harmony.
        chat_tools = []
        for tool in request.tools:
            if tool.type != "function":
                continue
            chat_tools.append(
                Tool(
                    type="function",
                    function=Function(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                        strict=tool.strict,
                    ),
                )
            )
        return chat_tools

    @staticmethod
    def _normalize_response_content_part_for_chat(content_part: Any) -> Any:
        # Default detail=\"auto\" and lift flat min/max_dynamic_patch onto
        # image_url so the image preprocessor sees them.
        if hasattr(content_part, "model_dump"):
            content_part = content_part.model_dump(exclude_none=True)
        if not isinstance(content_part, dict):
            return content_part

        part_type = content_part.get("type")
        if part_type in ("input_text", "output_text"):
            return {"type": "text", "text": content_part.get("text", "")}

        if part_type == "input_image":
            image_url = content_part.get("image_url")
            if isinstance(image_url, dict):
                image_url_obj = image_url.copy()
            else:
                image_url_obj = {"url": image_url}
            if not image_url_obj.get("detail"):
                image_url_obj["detail"] = content_part.get("detail") or "auto"
            for key in ("min_dynamic_patch", "max_dynamic_patch"):
                if key in content_part and key not in image_url_obj:
                    image_url_obj[key] = content_part[key]
            return {"type": "image_url", "image_url": image_url_obj}

        if part_type == "text":
            return content_part

        if part_type == "image_url":
            image_url = content_part.get("image_url")
            if isinstance(image_url, str):
                image_url = {
                    "url": image_url,
                    "detail": content_part.get("detail", "auto"),
                }
            elif isinstance(image_url, dict):
                image_url = image_url.copy()
                if not image_url.get("detail"):
                    image_url["detail"] = content_part.get("detail") or "auto"
            return {**content_part, "image_url": image_url}

        return content_part

    @staticmethod
    def _strip_qwen_exo_self_checks(text: str) -> str:
        remaining = str(text)
        while True:
            start = remaining.find(_QWEN_EXO_SELF_CHECK_START)
            if start < 0:
                break
            end = remaining.find(_QWEN_EXO_SELF_CHECK_END, start)
            if end < 0:
                break
            suffix_start = end + len(_QWEN_EXO_SELF_CHECK_END)
            prefix = remaining[:start].rstrip()
            suffix = remaining[suffix_start:].lstrip()
            remaining = "\n".join(part for part in (prefix, suffix) if part)
        return remaining.strip()

    @classmethod
    def _normalize_response_items_for_chat(cls, message: Any) -> list[Any]:
        """Convert one input item to zero or more chat messages. A replayed
        ``mcp_call`` (server-executed tool) expands to the assistant call plus
        its tool result, matching the prompt the model generated from."""
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if isinstance(message, dict) and message.get("type") == "mcp_call":
            return mcp_item_to_chat_messages(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            # Request validation turns replayed output_text parts into
            # input_text; either way the block must come back verbatim.
            block = _chat_message_text(message)
            if is_memory_block(block):
                return [{"role": "assistant", "content": block}]
        normalized = cls._normalize_response_message_for_chat(message)
        return [] if normalized is None else [normalized]

    @classmethod
    def _normalize_response_message_for_chat(cls, message: Any) -> Any:
        """Convert one Responses-API input item to a chat-completions message."""
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            return message

        # Most chat templates only recognize system/user/assistant/tool;
        # collapse ``developer`` to ``system`` at the boundary.
        if message.get("role") == "developer":
            message = {**message, "role": "system"}

        msg_type = message.get("type")
        if msg_type == "function_call":
            # Coerce ``arguments`` to a valid JSON-object string so the chat
            # template's unconditional ``orjson.loads`` survives truncated or
            # dict-shaped echoes.
            raw = message.get("arguments")
            if isinstance(raw, str):
                try:
                    parsed = orjson.loads(raw) if raw else None
                except orjson.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    raw = "{}"
            elif isinstance(raw, dict):
                raw = orjson.dumps(raw).decode("utf-8")
            else:
                raw = "{}"
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": raw,
                        },
                    }
                ],
            }
        if msg_type == "function_call_output":
            output = message.get("output", "")
            if isinstance(output, list):
                output = [
                    cls._normalize_response_content_part_for_chat(part)
                    for part in output
                ]
            elif output is None or not str(output).strip():
                output = "(tool returned no textual output)"
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id"),
                "content": output,
            }
        # Reasoning items render as {role: assistant, reasoning_content};
        # empty ones drop instead of injecting an empty assistant block.
        if msg_type == "reasoning":
            # Prefer ``summary``; fall back to ``content`` only when summary
            # is empty, since clients often populate both with the same text.
            def _collect(parts):
                out: list[str] = []
                for entry in parts or []:
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if text:
                            out.append(text)
                return out

            text_parts = _collect(message.get("summary"))
            if not text_parts:
                text_parts = _collect(message.get("content"))
            reasoning_content = cls._strip_qwen_exo_self_checks("\n".join(text_parts))
            if not reasoning_content:
                return None
            return {
                "role": "assistant",
                "reasoning_content": reasoning_content,
            }
        if msg_type not in (None, "message"):
            raise ValueError(f"Unsupported Responses API input item type: {msg_type!r}")

        content = message.get("content")
        if isinstance(content, Iterable) and not isinstance(
            content, (str, bytes, dict, list)
        ):
            content = list(content)
        if not isinstance(content, list):
            return {
                k: v
                for k, v in message.items()
                if v is not None and k not in ("id", "status", "type")
            }

        return {
            k: v
            for k, v in {
                **message,
                "content": [
                    cls._normalize_response_content_part_for_chat(part)
                    for part in content
                ],
            }.items()
            if v is not None and k not in ("id", "status", "type")
        }

    @staticmethod
    def _output_message_text(output_item: Any) -> Optional[str]:
        """Return assistant text from a ``message`` output item (joining
        ``output_text`` parts with newlines), or None for non-message items."""
        if isinstance(output_item, ResponseReasoningItem):
            return None
        if hasattr(output_item, "model_dump"):
            output_item = output_item.model_dump(exclude_none=True)
        if not isinstance(output_item, dict):
            return None
        if output_item.get("type") != "message":
            return None

        text_parts = []
        for content in output_item.get("content") or []:
            if isinstance(content, ResponseOutputText):
                text_parts.append(content.text)
                continue
            if hasattr(content, "model_dump"):
                content = content.model_dump(exclude_none=True)
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if text is not None:
                    text_parts.append(text)

        return "\n".join(text_parts) if text_parts else None

    @staticmethod
    def _merge_consecutive_assistant_messages(
        messages: list,
    ) -> list:
        """Collapse runs of consecutive ``assistant`` dicts into one entry,
        joining ``content`` and concatenating ``tool_calls`` and
        ``reasoning_content`` so a logical turn renders as a single block."""
        merged: list = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and merged
                and isinstance(merged[-1], dict)
                and merged[-1].get("role") == "assistant"
                # The server memory block stays its own message.
                and not is_memory_block(msg.get("content"))
                and not is_memory_block(merged[-1].get("content"))
            ):
                prev = merged[-1] = dict(merged[-1])
                # Lift mixed str/list content to list parts so non-text parts
                # (e.g. image_url) survive when the two sides differ in shape.
                new_content = msg.get("content")
                if new_content is not None and new_content != "":
                    prev_content = prev.get("content")
                    if prev_content is None or prev_content == "":
                        prev["content"] = new_content
                    elif isinstance(prev_content, str) and isinstance(new_content, str):
                        sep = "\n\n" if prev_content and new_content else ""
                        prev["content"] = prev_content + sep + new_content
                    else:

                        def _as_parts(c):
                            if isinstance(c, list):
                                return list(c)
                            if isinstance(c, str) and c:
                                return [{"type": "text", "text": c}]
                            return []

                        prev["content"] = _as_parts(prev_content) + _as_parts(
                            new_content
                        )
                new_calls = msg.get("tool_calls")
                if new_calls:
                    prev_calls = prev.get("tool_calls") or []
                    prev["tool_calls"] = prev_calls + list(new_calls)
                new_reasoning = msg.get("reasoning_content")
                if new_reasoning:
                    prev_reasoning = prev.get("reasoning_content")
                    prev["reasoning_content"] = (
                        f"{prev_reasoning}\n{new_reasoning}"
                        if prev_reasoning
                        else new_reasoning
                    )
                continue
            merged.append(msg)
        return merged

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        """Render one Responses request as chat messages."""
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )
        messages.extend(self._conversation_messages(request, prev_response))

        # One Responses-API assistant turn maps to multiple input items
        # (message + function_call(s)); collapse them into one chat message
        # so chat templates render a single assistant block per turn.
        messages = self._merge_consecutive_assistant_messages(messages)

        # Most chat templates expect a single leading ``system`` message;
        # coalesce any ``instructions`` + interleaved ``developer`` entries.
        system_chunks: list[str] = []
        other_msgs: list = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                system_chunks.append(text)
            else:
                other_msgs.append(m)
        system_content = (
            "\n\n".join(system_chunks) if system_chunks else _DEFAULT_SYSTEM_PROMPT
        )
        return [{"role": "system", "content": system_content}] + other_msgs

    def _conversation_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[ChatCompletionMessageParam]:
        """The conversation as chat messages, without ``instructions`` or the
        memory attachment and before assistant merging: the form kept in
        ``msg_store`` for ``previous_response_id`` chaining.

        A chained turn replays the parent's stored messages and *every* parent
        output item (reasoning, tool calls, ``mcp_call``, messages) through the
        same normalization a stateless client's replay takes, so a chained
        prompt renders exactly like the equivalent full-history request.
        """
        messages: list[ChatCompletionMessageParam] = []
        if prev_response is not None:
            for message in self.msg_store[prev_response.id]:
                messages.extend(self._normalize_response_items_for_chat(message))
            for output_item in prev_response.output:
                messages.extend(self._normalize_response_items_for_chat(output_item))
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for input_item in request.input:
                messages.extend(  # type: ignore
                    self._normalize_response_items_for_chat(input_item)
                )
        return messages

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[OpenAIMessage]:
        messages: list[OpenAIMessage] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                any(t in tool_types for t in ("web_search", "web_search_preview"))
                and self.tool_server is not None
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.tool_server is not None
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                browser_description=(
                    self.tool_server.get_tool_description("browser")
                    if self.tool_server and enable_browser
                    else None
                ),
                python_description=(
                    self.tool_server.get_tool_description("python")
                    if self.tool_server and enable_code_interpreter
                    else None
                ),
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions, request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME: Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if (
                len(prev_msgs) > 0
                and hasattr(prev_msgs[-1], "channel")
                and prev_msgs[-1].channel == "final"
            ):  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if (
                        hasattr(prev_msgs[i], "channel")
                        and prev_msgs[i].channel == "final"
                    ):  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                del prev_msgs[prev_final_msg_idx + 1 :]
                for msg in recent_turn_msgs:
                    if hasattr(msg, "channel") and msg.channel != "analysis":  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = list(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    @staticmethod
    def _stringify_response_metadata(
        metadata: dict[str, Any] | None,
    ) -> dict[str, str]:
        return {
            str(key): (
                value
                if isinstance(value, str)
                else json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
            )
            for key, value in (metadata or {}).items()
        }

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return "context length" in message and (
            "input" in message
            or "token count" in message
            or "maximum context" in message
        )

    def _error_response_for_exception(self, exc: Exception) -> ORJSONResponse:
        if self._is_context_length_error(exc):
            return self.create_error_response(
                str(exc),
                err_type="invalid_request_error",
                status_code=HTTPStatus.BAD_REQUEST,
                param="input",
                code=_CONTEXT_LENGTH_ERROR_CODE,
            )
        return self.create_error_response(str(exc))

    @staticmethod
    def _public_response_error(
        exc: Exception,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if isinstance(exc, HTTPException):
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            status_code = int(exc.status_code)
            message = str(detail.get("message", exc.detail))
            internal_code = str(detail.get("code", "scheduler_abort"))
            retry_after = detail.get("retry_after")
        elif OpenAIServingResponses._is_context_length_error(exc):
            status_code = int(HTTPStatus.BAD_REQUEST)
            message = str(exc)
            internal_code = _CONTEXT_LENGTH_ERROR_CODE
            retry_after = None
        else:
            status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)
            message = "Request failed"
            internal_code = type(exc).__name__
            retry_after = None
        if internal_code == _CONTEXT_LENGTH_ERROR_CODE:
            public_code = _CONTEXT_LENGTH_ERROR_CODE
        elif status_code == HTTPStatus.TOO_MANY_REQUESTS:
            public_code = "rate_limit_exceeded"
        elif status_code in {
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            HTTPStatus.UNPROCESSABLE_ENTITY,
        }:
            public_code = "invalid_prompt"
        else:
            public_code = "server_error"
        diagnostics = {
            "qwen_exo_error_code": internal_code,
            "qwen_exo_error_status": str(status_code),
        }
        if retry_after is not None:
            diagnostics["qwen_exo_retry_after"] = str(retry_after)
        return {"message": message, "code": public_code}, diagnostics

    def _store_public_harmony_messages(
        self, request: ResponsesRequest, context: HarmonyContext
    ) -> None:
        public_initial_messages = list(self.msg_store.get(request.request_id, ()))
        context_messages = context.messages
        generated_messages = (
            context_messages[context.num_init_messages :]
            if context_messages is getattr(context, "_messages", None)
            else context_messages
        )
        public_generated_messages = []
        for message in generated_messages:
            author = getattr(message, "author", None)
            role = getattr(author, "role", None)
            if role == Role.TOOL or (
                role == Role.ASSISTANT
                and (
                    getattr(message, "recipient", None) is not None
                    or getattr(message, "channel", None) == "final"
                )
            ):
                public_generated_messages.append(message)
        self.msg_store[request.request_id] = [
            *public_initial_messages,
            *public_generated_messages,
        ]

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        background_error = None
        background_diagnostics: dict[str, str] = {}
        try:
            # Update the status to "in_progress"
            async with self.response_store_lock:
                stored_response = self.response_store.get(request.request_id)
                assert stored_response is not None
                stored_response.status = "in_progress"

            response = await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time,
                *args,
                **kwargs,
            )
        except HTTPException as exc:
            background_error, background_diagnostics = self._public_response_error(exc)
            response = self.create_error_response(
                background_error["message"],
                err_type=background_error["code"],
                status_code=exc.status_code,
            )
        except Exception as exc:
            logger.exception("Background request failed for %s", request.request_id)
            background_error, background_diagnostics = self._public_response_error(exc)
            response = self.create_error_response(
                background_error["message"],
                err_type=background_error["code"],
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in {
                    "completed",
                    "incomplete",
                    "cancelled",
                }:
                    stored_response.status = "failed"
                    stored_response.error = background_error
                    stored_response.metadata = {
                        **self._stringify_response_metadata(stored_response.metadata),
                        **background_diagnostics,
                    }

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status == "cancelled":
                return response
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a terminal response.",
                )

            # Update the status to "cancelled"
            response.status = "cancelled"

        # The response_id is the same as the rid used when submitting the request
        self.tokenizer_manager.abort_request(rid=response_id)

        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_invalid_id_error(self, response_id: str):
        return self.create_error_response(
            message=(
                f"Invalid 'response_id': '{response_id}'. "
                "Expected an ID that begins with 'resp'."
            ),
            err_type="invalid_request_error",
            param="response_id",
        )

    def _make_not_found_error(self, response_id: str):
        return self.create_error_response(
            message=f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        current_content_index = 0
        current_output_index = 0
        current_item_id = f"item_{random_uuid()}"
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        async for ctx in result_generator:
            # Only process context objects that implement the `is_expecting_start()` method,
            # which indicates they support per-turn streaming (e.g., StreamingHarmonyContext).
            # Contexts without this method are skipped, as they do not represent a new turn
            # or are not compatible with per-turn handling in the /v1/responses endpoint.
            if not hasattr(ctx, "is_expecting_start"):
                continue

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            id=f"rs_{random_uuid()}",
                            type="reasoning",
                            summary=[],
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text,
                                    type="reasoning_text",
                                ),
                            ],
                            status="completed",
                        )
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                item_id=current_item_id,
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            )
                        )
                    elif previous_item.channel == "final":
                        text_content = openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            )
                        )

            if ctx.parser.last_content_delta:
                if (
                    ctx.parser.current_channel == "final"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        )
                    )
                elif (
                    ctx.parser.current_channel == "analysis"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        openai_responses_types.ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        )
                    )

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (
                    self.supports_browsing
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("browser.")
                ):
                    function_name = previous_item.recipient[len("browser.") :]
                    action = None
                    parsed_args = orjson.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = openai_responses_types.response_function_web_search.ActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = openai_responses_types.response_function_web_search.ActionOpenPage(
                            type="open_page",
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    elif function_name == "find":
                        action = openai_responses_types.response_function_web_search.ActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.response_function_web_search.ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )

                    # enqueue
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        )
                    )

                if (
                    self.supports_code_interpreter
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("python")
                ):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        )
                    )

        async def empty_async_generator():
            for _ in ():
                yield

        final_response = await self.responses_full_generator(
            request,
            sampling_params,
            empty_async_generator(),
            context,
            model_name,
            tokenizer,
            request_metadata,
            created_time=created_time,
        )
        # Convert final_response to the format expected by ResponseCompletedEvent
        response_dict = final_response.model_dump()
        # OpenAI SDK's Tool union may not know extended types; drop echo.
        response_dict["tools"] = []

        # Convert UsageInfo to ResponseUsage format
        if response_dict.get("usage"):
            usage_info = response_dict["usage"]
            response_dict["usage"] = {
                "input_tokens": usage_info.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": usage_info.get("cached_tokens", 0)
                },
                "output_tokens": usage_info.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": usage_info.get("reasoning_tokens", 0)
                },
                "total_tokens": usage_info.get("total_tokens", 0),
            }

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def responses_stream_generator_non_harmony(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a /v1/responses response as typed OpenAI SSE events for
        non-harmony models. Each engine chunk is run through the reasoning
        and function-call parsers; leftover text becomes
        ``response.output_text.delta``.
        """

        created_time = created_time or int(time.time())
        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        # The streaming Response* event models echo ``tools`` through a
        # narrower OpenAI SDK Tool union; strip it to avoid pydantic
        # validation failures on extended tool types.
        def _sanitize_response_dict(d: dict) -> dict:
            d["tools"] = []
            return d

        initial_response = _sanitize_response_dict(
            ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="in_progress",
                usage=None,
            ).model_dump()
        )
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        chat_tools = self._chat_tools_for_request(request)
        is_required = request.tool_choice == "required"

        # Per model turn; rebuilt after each server-side memory tool round,
        # whose continuation opens a fresh ``<think>`` turn.
        def _new_tool_parser() -> Optional[Union[FunctionCallParser, JsonArrayParser]]:
            if not chat_tools or request.tool_choice == "none":
                return None
            native_supports_structural_tag = False
            if self.tool_call_parser:
                probe = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                native_supports_structural_tag = (
                    probe.detector.supports_structural_tag()
                )
            if is_required and not native_supports_structural_tag:
                return JsonArrayParser()
            if self.tool_call_parser:
                return FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
            return None

        def _new_reasoning_parser() -> Optional[ReasoningParser]:
            if not self.reasoning_parser:
                return None
            return ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )

        tool_parser = _new_tool_parser()
        reasoning_parser_obj = _new_reasoning_parser()

        current_output_index = -1
        reasoning_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        message_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        tool_call_states: dict[int, dict[str, Any]] = {}
        # Items closed during the stream, in wire order. Feeds the final
        # ``response.completed`` snapshot and the stored response.
        emitted_items: list = []
        self_ask_spill_router = _QwenExoSelfAskSpillRouter()
        self_ask_spill_text = ""

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        total_tokens_meta = 0
        reasoning_tokens_meta = 0
        finish_reason: Optional[dict[str, Any]] = None
        stream_offset = 0
        incremental = self.tokenizer_manager.server_args.incremental_streaming_output

        def _open_reasoning_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"rs_{random_uuid()}"
            reasoning_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        wants_summary = self._wants_reasoning_summary(request)

        def _close_reasoning_item():
            if not reasoning_state["open"]:
                return []
            text = reasoning_state["text"]
            completed_item = ResponseReasoningItem(
                id=reasoning_state["item_id"],
                type="reasoning",
                summary=(
                    [ResponseReasoningSummary(type="summary_text", text=text)]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(type="reasoning_text", text=text),
                ],
                status="completed",
            )
            events: list = []
            if wants_summary:
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningSummaryTextDoneEvent(
                            type="response.reasoning_summary_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            text=text,
                        )
                    )
                )
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningSummaryPartDoneEvent(
                            type="response.reasoning_summary_part.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            part=ResponseReasoningSummaryDonePart(
                                type="summary_text", text=text
                            ),
                        )
                    )
                )
            else:
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            content_index=0,
                            text=text,
                        )
                    )
                )
            events += [
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=reasoning_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            reasoning_state["open"] = False
            return events

        def _open_message_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"msg_{random_uuid()}"
            message_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        def _close_message_item():
            if not message_state["open"]:
                return []
            text = message_state["text"]
            text_content = openai_responses_types.ResponseOutputText(
                type="output_text", text=text, annotations=[], logprobs=None
            )
            completed_item = ResponseOutputMessage(
                id=message_state["item_id"],
                type="message",
                role="assistant",
                content=[text_content],
                status="completed",
            )
            events = [
                _send_event(
                    openai_responses_types.ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        content_index=0,
                        text=text,
                        logprobs=[],
                        item_id=message_state["item_id"],
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=message_state["item_id"],
                        output_index=message_state["output_index"],
                        content_index=0,
                        part=text_content,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            message_state["open"] = False
            return events

        def _close_tool_call_state(tool_index: int):
            state = tool_call_states.get(tool_index)
            if state is None or state.get("done"):
                return []
            if state.get("internal"):
                # Server-executed memory call: surfaced as ``mcp_call`` later.
                state["done"] = True
                return []
            arguments = state["arguments"]
            completed_item = ResponseFunctionToolCall(
                arguments=arguments,
                call_id=state["call_id"],
                name=state["name"] or "",
                type="function_call",
                id=state["item_id"],
                status="completed",
            )
            events = [
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=-1,
                        item_id=state["item_id"],
                        output_index=state["output_index"],
                        arguments=arguments,
                        name=state["name"] or "",
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            state["done"] = True
            return events

        def _tool_call_is_open(tool_index: int) -> bool:
            state = tool_call_states.get(tool_index)
            return state is not None and not state["done"]

        def _mcp_call_events(call: dict[str, Any]):
            """Standard hosted-tool events for one server-executed call."""
            nonlocal current_output_index
            current_output_index += 1
            output_index = current_output_index
            completed_item = mcp_call_item(call)
            in_progress_item = completed_item.model_copy(
                update={"output": None, "error": None, "status": "in_progress"}
            )
            terminal_event = (
                openai_responses_types.ResponseMcpCallCompletedEvent(
                    type="response.mcp_call.completed",
                    item_id=call["id"],
                    output_index=output_index,
                    sequence_number=-1,
                )
                if call["ok"]
                else openai_responses_types.ResponseMcpCallFailedEvent(
                    type="response.mcp_call.failed",
                    item_id=call["id"],
                    output_index=output_index,
                    sequence_number=-1,
                )
            )
            events = [
                _send_event(
                    openai_responses_types.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=-1,
                        output_index=output_index,
                        item=in_progress_item,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseMcpCallInProgressEvent(
                        type="response.mcp_call.in_progress",
                        item_id=call["id"],
                        output_index=output_index,
                        sequence_number=-1,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseMcpCallArgumentsDoneEvent(
                        type="response.mcp_call_arguments.done",
                        item_id=call["id"],
                        output_index=output_index,
                        arguments=call["arguments"],
                        sequence_number=-1,
                    )
                ),
                _send_event(terminal_event),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=output_index,
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            return events

        def _tool_call_delta_events(call):
            nonlocal current_output_index
            events = []
            state = tool_call_states.get(call.tool_index)
            if (state is None or state["done"]) and is_memory_tool(call.name):
                tool_call_states[call.tool_index] = {
                    "internal": True,
                    "done": False,
                    "name": call.name,
                }
                return events
            if state is not None and state.get("internal"):
                return events
            if state is None or state["done"]:
                current_output_index += 1
                state = {
                    "item_id": f"fc_{random_uuid()[:8]}",
                    "call_id": f"call_{random_uuid()[:24]}",
                    "output_index": current_output_index,
                    "name": call.name or "",
                    "arguments": "",
                    "added": False,
                    "done": False,
                }
                tool_call_states[call.tool_index] = state
            if not state["added"]:
                state["added"] = True
                # Capture ``call.name`` before the ``added`` event so the
                # name is set on the first emitted item.
                if call.name and not state["name"]:
                    state["name"] = call.name
                events.append(
                    _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=state["output_index"],
                            item=ResponseFunctionToolCall(
                                arguments="",
                                call_id=state["call_id"],
                                name=state["name"],
                                type="function_call",
                                id=state["item_id"],
                                status="in_progress",
                            ),
                        )
                    )
                )
            if call.parameters:
                state["arguments"] += call.parameters
                events.append(
                    _send_event(
                        openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                            type="response.function_call_arguments.delta",
                            sequence_number=-1,
                            item_id=state["item_id"],
                            output_index=state["output_index"],
                            delta=call.parameters,
                        )
                    )
                )
            return events

        for block_item in self._memory_block_output_items(request):
            current_output_index += 1
            yield _send_event(
                openai_responses_types.ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=current_output_index,
                    item=block_item.model_copy(
                        update={"content": [], "status": "in_progress"}
                    ),
                )
            )
            (block_part,) = block_item.content
            yield _send_event(
                openai_responses_types.ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    sequence_number=-1,
                    output_index=current_output_index,
                    item_id=block_item.id,
                    content_index=0,
                    part=block_part.model_copy(update={"text": ""}),
                )
            )
            yield _send_event(
                openai_responses_types.ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=-1,
                    content_index=0,
                    output_index=current_output_index,
                    item_id=block_item.id,
                    delta=block_part.text,
                    logprobs=[],
                )
            )
            yield _send_event(
                openai_responses_types.ResponseTextDoneEvent(
                    type="response.output_text.done",
                    sequence_number=-1,
                    output_index=current_output_index,
                    content_index=0,
                    text=block_part.text,
                    logprobs=[],
                    item_id=block_item.id,
                )
            )
            yield _send_event(
                openai_responses_types.ResponseContentPartDoneEvent(
                    type="response.content_part.done",
                    sequence_number=-1,
                    item_id=block_item.id,
                    output_index=current_output_index,
                    content_index=0,
                    part=block_part,
                )
            )
            yield _send_event(
                openai_responses_types.ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=-1,
                    output_index=current_output_index,
                    item=block_item,
                )
            )
            emitted_items.append(block_item)

        try:
            async for ctx in result_generator:
                if isinstance(ctx, dict):
                    chunk = ctx
                else:
                    chunk = getattr(ctx, "last_output", None)
                if not isinstance(chunk, dict):
                    continue
                meta = chunk.get("meta_info") or {}
                memory_round = meta.get(MEMORY_ROUND_META_KEY)
                if memory_round is not None:
                    # A server-side memory round ended this model turn: close
                    # its items, report the calls, and parse the continuation
                    # turn (which reopens ``<think>``) from scratch.
                    for ev in _close_reasoning_item():
                        yield ev
                    for ev in _close_message_item():
                        yield ev
                    for tool_index in list(tool_call_states):
                        for ev in _close_tool_call_state(tool_index):
                            yield ev
                    for call in memory_round["calls"]:
                        for ev in _mcp_call_events(call):
                            yield ev
                    tool_call_states.clear()
                    tool_parser = _new_tool_parser()
                    reasoning_parser_obj = _new_reasoning_parser()
                    self_ask_spill_router = _QwenExoSelfAskSpillRouter()
                    stream_offset = 0
                    finish_reason = None
                    continue
                prompt_tokens = meta.get("prompt_tokens", prompt_tokens)
                completion_tokens = meta.get("completion_tokens", completion_tokens)
                cached_tokens = meta.get("cached_tokens", cached_tokens)
                total_tokens_meta = meta.get("total_tokens", total_tokens_meta)
                reasoning_tokens_meta = meta.get(
                    "reasoning_tokens", reasoning_tokens_meta
                )
                finish_reason = meta.get("finish_reason") or finish_reason
                if meta.get("qwen_exo_self_ask_boundary"):
                    self_ask_spill_router.arm()

                text = chunk.get("text", "") or ""
                if incremental:
                    delta = text
                else:
                    delta = text[stream_offset:]
                    stream_offset = len(text)
                if not delta and finish_reason is None:
                    continue

                # The reasoning parser owns state transitions. Only its
                # post-reasoning normal delta may reach the tool parser.
                if reasoning_parser_obj is not None:
                    reasoning_chunk, delta = reasoning_parser_obj.parse_stream_chunk(
                        delta
                    )
                else:
                    reasoning_chunk = None
                spill_reasoning, delta = self_ask_spill_router.feed(
                    delta, final=bool(meta.get("finish_reason"))
                )
                if spill_reasoning:
                    self_ask_spill_text += spill_reasoning
                    reasoning_chunk = f"{reasoning_chunk or ''}{spill_reasoning}"

                if reasoning_chunk:
                    if message_state["open"]:
                        for ev in _close_message_item():
                            yield ev
                    if not reasoning_state["open"]:
                        item_id = _open_reasoning_item()
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=reasoning_state["output_index"],
                                item=ResponseReasoningItem(
                                    id=item_id,
                                    type="reasoning",
                                    summary=[],
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        # Clients that opt into ``reasoning.summary`` render
                        # off the ``reasoning_summary_text.*`` event stream,
                        # so mirror the trace into a summary part.
                        if wants_summary:
                            yield _send_event(
                                openai_responses_types.ResponseReasoningSummaryPartAddedEvent(
                                    type="response.reasoning_summary_part.added",
                                    item_id=item_id,
                                    output_index=reasoning_state["output_index"],
                                    summary_index=0,
                                    part=ResponseReasoningSummaryAddedPart(
                                        type="summary_text", text=""
                                    ),
                                    sequence_number=-1,
                                )
                            )
                    reasoning_state["text"] += reasoning_chunk
                    if wants_summary:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningSummaryTextDeltaEvent(
                                type="response.reasoning_summary_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                summary_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )
                    else:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDeltaEvent(
                                type="response.reasoning_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                content_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )

                if not delta:
                    continue

                if isinstance(tool_parser, JsonArrayParser):
                    sp = tool_parser.parse_streaming_increment(delta, chat_tools)
                    normal_text, tool_calls = sp.normal_text or "", sp.calls
                elif tool_parser is not None:
                    normal_text, tool_calls = tool_parser.parse_stream_chunk(delta)
                else:
                    normal_text, tool_calls = delta, []

                # The parser does not order normal text against calls within
                # one chunk. A multi-token chunk (speculative decoding) can end
                # an open call and then carry text after </tool_call>, so the
                # open call's remaining arguments go out before the text closes
                # it; text inside a tool call is never returned as normal text.
                is_open = [_tool_call_is_open(c.tool_index) for c in tool_calls]
                continuing = [c for c, o in zip(tool_calls, is_open) if o]
                tool_calls = [c for c, o in zip(tool_calls, is_open) if not o]
                for call in continuing:
                    for ev in _tool_call_delta_events(call):
                        yield ev

                # Close any open tool-call item before opening a message so
                # ``output_item.done`` lands before the next ``added``.
                if normal_text:
                    if reasoning_state["open"]:
                        for ev in _close_reasoning_item():
                            yield ev
                    for tool_index in list(tool_call_states):
                        for ev in _close_tool_call_state(tool_index):
                            yield ev
                    if not message_state["open"]:
                        item_id = _open_message_item()
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=message_state["output_index"],
                                item=ResponseOutputMessage(
                                    id=item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=message_state["output_index"],
                                item_id=message_state["item_id"],
                                content_index=0,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    message_state["text"] += normal_text
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=0,
                            output_index=message_state["output_index"],
                            item_id=message_state["item_id"],
                            delta=normal_text,
                            logprobs=[],
                        )
                    )

                if not tool_calls:
                    continue

                if reasoning_state["open"]:
                    for ev in _close_reasoning_item():
                        yield ev
                if message_state["open"]:
                    for ev in _close_message_item():
                        yield ev

                for call in tool_calls:
                    for ev in _tool_call_delta_events(call):
                        yield ev
        except Exception as exc:
            if isinstance(exc, HTTPException):
                logger.info(
                    "Streaming /v1/responses terminated with HTTP %s",
                    exc.status_code,
                )
            else:
                logger.exception("Error while streaming /v1/responses")
            failed = _sanitize_response_dict(
                ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="failed",
                    usage=None,
                ).model_dump()
            )
            public_error, error_diagnostics = self._public_response_error(exc)
            failed["error"] = public_error
            failed["metadata"] = {
                **self._stringify_response_metadata(failed.get("metadata")),
                **error_diagnostics,
            }
            yield _send_event(
                openai_responses_types.ResponseFailedEvent(
                    type="response.failed",
                    sequence_number=-1,
                    response=failed,
                )
            )
            return

        for ev in _close_reasoning_item():
            yield ev
        for ev in _close_message_item():
            yield ev
        for tool_index in list(tool_call_states):
            for ev in _close_tool_call_state(tool_index):
                yield ev

        final_output_items = list(emitted_items)
        reclassified_reasoning_tokens = (
            len(
                tokenizer.encode(
                    self_ask_spill_text,
                    add_special_tokens=False,
                )
            )
            if self_ask_spill_text
            else 0
        )
        total_reasoning_tokens = (
            int(reasoning_tokens_meta) + reclassified_reasoning_tokens
        )

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens_meta or (prompt_tokens + completion_tokens),
            reasoning_tokens=total_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=cached_tokens
            )
        request_metadata.final_usage_info = usage

        response_status, incomplete_details = self._response_terminal_status(
            finish_reason
        )
        _log_response_shape(
            request_id=request.request_id,
            stream=True,
            status=response_status,
            items=final_output_items,
            finish_reason=finish_reason,
            usage=usage,
        )
        final_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=final_output_items,
            status=response_status,
            usage=usage,
            incomplete_details=incomplete_details,
        )
        if request.store:
            async with self.response_store_lock:
                stored = self.response_store.get(final_response.id)
                if stored is None or stored.status != "cancelled":
                    self.response_store[final_response.id] = final_response

        response_dict = _sanitize_response_dict(final_response.model_dump())
        if response_dict.get("usage"):
            usage_info = response_dict["usage"]
            response_dict["usage"] = {
                "input_tokens": usage_info.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": cached_tokens,
                },
                "output_tokens": usage_info.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": total_reasoning_tokens,
                },
                "total_tokens": usage_info.get("total_tokens", 0),
            }

        if response_status == "incomplete":
            yield _send_event(
                openai_responses_types.ResponseIncompleteEvent(
                    type="response.incomplete",
                    sequence_number=-1,
                    response=response_dict,
                )
            )
        else:
            yield _send_event(
                openai_responses_types.ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=-1,
                    response=response_dict,
                )
            )

    @staticmethod
    def _copy_sampling_params(sampling_params: Any, **updates: Any) -> Any:
        if isinstance(sampling_params, dict):
            cloned = dict(sampling_params)
            cloned.update(updates)
            return cloned
        cloned = copy.copy(sampling_params)
        for key, value in updates.items():
            setattr(cloned, key, value)
        return cloned

    @classmethod
    def _sampling_params_with_reasoning_stop(
        cls, sampling_params: Any, reasoning_end_token_id: int
    ) -> Any:
        current = (
            sampling_params.get("stop_token_ids")
            if isinstance(sampling_params, dict)
            else getattr(sampling_params, "stop_token_ids", None)
        ) or ()
        stop_token_ids = list(dict.fromkeys([*current, reasoning_end_token_id]))
        return cls._copy_sampling_params(sampling_params, stop_token_ids=stop_token_ids)

    @classmethod
    def _reasoning_phase_sampling_params(
        cls,
        sampling_params: Any,
        reasoning_end_token_id: int,
        max_reasoning_tokens: int,
    ) -> tuple[Any, bool]:
        configured_max = (
            sampling_params.get("max_new_tokens")
            if isinstance(sampling_params, dict)
            else getattr(sampling_params, "max_new_tokens", None)
        )
        cap_applied = configured_max is None or int(configured_max) > int(
            max_reasoning_tokens
        )
        phase_sampling = cls._sampling_params_with_reasoning_stop(
            sampling_params, reasoning_end_token_id
        )
        if cap_applied:
            phase_sampling = cls._copy_sampling_params(
                phase_sampling, max_new_tokens=int(max_reasoning_tokens)
            )
        return phase_sampling, cap_applied

    @classmethod
    def _single_phase_reasoning_sampling_params(
        cls,
        sampling_params: Any,
        reasoning_end_token_id: int,
        max_reasoning_tokens: int,
    ) -> tuple[Any, bool]:
        """Budget the reasoning phase in the scheduler instead of stopping at
        </think>, so a natural reasoning end continues into the answer within
        the same request (no second scheduling round trip)."""
        configured_max = (
            sampling_params.get("max_new_tokens")
            if isinstance(sampling_params, dict)
            else getattr(sampling_params, "max_new_tokens", None)
        )
        cap_applied = configured_max is None or int(configured_max) > int(
            max_reasoning_tokens
        )
        if not cap_applied:
            return sampling_params, False
        custom_params = dict(
            (
                sampling_params.get("custom_params")
                if isinstance(sampling_params, dict)
                else getattr(sampling_params, "custom_params", None)
            )
            or {}
        )
        custom_params[QWEN_EXO_REASONING_BUDGET_KEY] = {
            "end_token_id": int(reasoning_end_token_id),
            "max_tokens": int(max_reasoning_tokens),
        }
        return (
            cls._copy_sampling_params(sampling_params, custom_params=custom_params),
            True,
        )

    @staticmethod
    def _forced_reasoning_boundary(
        finish_reason: dict[str, Any],
        *,
        output_tokens: int,
        max_reasoning_tokens: int,
        cap_applied: bool,
    ) -> bool:
        return (
            cap_applied
            and finish_reason.get("type") == "length"
            and int(output_tokens) >= int(max_reasoning_tokens)
        )

    @staticmethod
    def _matched_reasoning_boundary(
        finish_reason: dict[str, Any], reasoning_end_token_id: int
    ) -> bool:
        if finish_reason.get("type") != "stop":
            return False
        matched = finish_reason.get("matched")
        return (
            matched == reasoning_end_token_id
            or str(matched) == str(reasoning_end_token_id)
            or str(matched).strip() == "</think>"
        )

    def _request_prompt_token_ids(
        self,
        adapted_request: GenerateReqInput,
        prompt_encoding: _ResponsePromptEncoding | None = None,
    ) -> list[int] | None:
        if adapted_request.input_ids is not None:
            return list(adapted_request.input_ids)
        if adapted_request.text is not None:
            if prompt_encoding is None:
                prompt_encoding = _ResponsePromptEncoding(
                    self.tokenizer_manager.tokenizer
                )
            return list(
                prompt_encoding.encode(
                    adapted_request.text, add_special_tokens=False
                )
            )
        return None

    @staticmethod
    def _score_bias_logprob_start_len(
        prompt_token_ids: list[int], trajectory_spans: Any
    ) -> int:
        prompt_length = len(prompt_token_ids)
        starts = []
        for span in trajectory_spans or ():
            try:
                start = int(span["start"])
                end = int(span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= start < end <= prompt_length:
                starts.append(start)
        return max(0, min(starts) - 1) if starts else prompt_length

    @staticmethod
    def _raise_for_generation_abort(finish_reason: dict[str, Any]) -> None:
        if finish_reason.get("type") != "abort":
            return
        detail = {
            "message": finish_reason.get("message", "Scheduler request aborted"),
            "code": finish_reason.get("code", "scheduler_abort"),
        }
        if finish_reason.get("retry_after") is not None:
            detail["retry_after"] = int(finish_reason["retry_after"])
        raw_status_code = finish_reason.get("status_code")
        raise HTTPException(
            status_code=int(
                raw_status_code
                if raw_status_code is not None
                else HTTPStatus.INTERNAL_SERVER_ERROR
            ),
            detail=detail,
        )

    @staticmethod
    def _response_terminal_status(
        finish_reason: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, str] | None]:
        if isinstance(finish_reason, Mapping) and finish_reason.get("type") == "length":
            return "incomplete", {"reason": "max_output_tokens"}
        return "completed", None

    def _reasoning_boundary_tokens(
        self,
        injection: Any,
        reasoning_end_token_id: int,
        *,
        forced: bool = False,
    ) -> tuple[str, tuple[int, ...]]:
        injected_text = injection.text if injection is not None else ""
        if forced:
            # Reasoning budget cutoff: any self-check questions come first, then
            # the stop-and-go instruction, then the </think> boundary.
            injected_text = f"{injected_text}\nlet me do this now stop over thinking\n"
        injected_ids = self.tokenizer_manager.tokenizer.encode(
            injected_text, add_special_tokens=False
        )
        end_text = self.tokenizer_manager.tokenizer.decode(
            [reasoning_end_token_id], skip_special_tokens=False
        )
        if not end_text:
            end_text = "</think>"
        return (
            f"{injected_text}{end_text}",
            tuple([*injected_ids, reasoning_end_token_id]),
        )

    def _phase_two_sampling_params(
        self,
        sampling_params: Any,
        *,
        prompt_length: int,
        consumed_tokens: int,
    ) -> Any:
        configured_max = (
            sampling_params.get("max_new_tokens")
            if isinstance(sampling_params, dict)
            else getattr(sampling_params, "max_new_tokens", None)
        )
        context_length = getattr(
            self.tokenizer_manager.model_config, "context_len", 4096
        )
        context_remaining = max(
            context_length - prompt_length - self.tokenizer_manager.num_reserved_tokens,
            1,
        )
        if configured_max is None:
            remaining = context_remaining
        else:
            remaining = min(
                context_remaining,
                max(int(configured_max) - consumed_tokens, 1),
            )
        return self._copy_sampling_params(sampling_params, max_new_tokens=remaining)

    def _parse_memory_round_calls(
        self, request: ResponsesRequest, text: str
    ) -> tuple[list[Any], list[Any]]:
        """Parse a finished model turn into (memory calls, client calls)."""
        content = text
        if self.reasoning_parser:
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            _reasoning, content = reasoning_parser.parse_non_stream(text)
        if not content:
            return [], []
        tool_parser = FunctionCallParser(
            self._chat_tools_for_request(request),
            self.tool_call_parser,
            tokenizer=self.tokenizer_manager.tokenizer,
        )
        if not tool_parser.has_tool_call(content):
            return [], []
        try:
            _normal_text, calls = tool_parser.parse_non_stream(content)
        except Exception:
            logger.exception("Memory tool round: tool call parsing failed")
            return [], []
        return split_memory_calls(calls)

    async def _run_memory_round(
        self, memory_calls: list[Any], *, limit_reached: bool
    ) -> list[dict[str, Any]]:
        calls = []
        for parsed in memory_calls:
            call = new_memory_call(parsed.name, parsed.parameters)
            if limit_reached:
                call["output"] = ROUND_LIMIT_OUTPUT
            else:
                started = time.perf_counter()
                ok, output = await self.tool_server.executor.run(
                    call["name"], call["arguments"]
                )
                call["duration_ms"] = (time.perf_counter() - started) * 1000.0
                call["ok"] = ok
                call["output"] = neutralize_control_markers(output)
            calls.append(call)
        return calls

    async def _generate_with_memory_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: SimpleContext,
        *,
        raw_request: Optional[Request],
        response_request: ResponsesRequest,
        priority: Optional[int],
        prompt_encoding: _ResponsePromptEncoding | None,
    ) -> AsyncGenerator[Any, None]:
        """Run model turns, executing qwen-exo memory tool calls server-side.

        Each pass is one full model turn (reasoning boundary continuation
        included). When a pass ends with only memory tool calls, the server runs
        them, appends ``output ids + tool-result block + assistant header`` to
        the token prompt (exact KV reuse) and starts the next pass. A synthetic
        chunk carrying ``MEMORY_ROUND_META_KEY`` separates passes so consumers
        can emit ``mcp_call`` items and restart their per-turn parsers.
        """
        tokenizer = self.tokenizer_manager.tokenizer
        template = ToolResponseTemplate(
            tokenizer, response_request._qwen_exo_template_kwargs
        )
        incremental = bool(
            adapted_request.stream
            and self.tokenizer_manager.server_args.incremental_streaming_output
        )
        if adapted_request.input_ids is None:
            # Multimodal-capable models get a text prompt; this request has no
            # media (see _memory_tools_active), so encode it the way the
            # generation loop does for qwen-exo requests.
            adapted_request = replace(
                adapted_request,
                input_ids=list(
                    (prompt_encoding or _ResponsePromptEncoding(tokenizer)).encode(
                        adapted_request.text
                    )
                ),
            )
        prompt_ids = list(adapted_request.input_ids)
        try:
            assistant_header = generation_prompt_ids(
                prompt_ids, tokenizer.convert_tokens_to_ids("<|im_start|>")
            )
            template.continuation_ids(
                output_ids=(), outputs=("",), generation_prompt_ids=()
            )
        except Exception:
            logger.exception("Memory tools disabled: unsupported chat template")
            async for item in self._generate_with_builtin_tools(
                request_id,
                request_prompt,
                adapted_request,
                sampling_params,
                context,
                raw_request=raw_request,
                response_request=response_request,
                priority=priority,
                prompt_encoding=prompt_encoding,
            ):
                yield item
            return
        context.qwen_exo_memory_rounds = []
        pass_request = adapted_request
        pass_sampling_params = sampling_params
        consumed_tokens = 0
        first_usage: dict[str, Any] = {}
        prior_completion = 0
        prior_reasoning = 0
        for round_index in range(MAX_MEMORY_TOOL_ROUNDS + 2):
            pass_ids: list[int] = []
            pass_text = ""
            final_meta: dict[str, Any] = {}
            async for _ in self._generate_with_builtin_tools(
                request_id,
                request_prompt,
                pass_request,
                pass_sampling_params,
                context,
                raw_request=raw_request,
                response_request=response_request,
                priority=priority,
                prompt_encoding=prompt_encoding,
                generation_index_start=round_index,
            ):
                chunk = context.last_output
                if isinstance(chunk, dict):
                    ids = [int(token) for token in (chunk.get("output_ids") or ())]
                    text = str(chunk.get("text") or "")
                    if incremental:
                        pass_ids.extend(ids)
                        pass_text += text
                    else:
                        pass_ids, pass_text = ids, text
                    meta = chunk.get("meta_info") or {}
                    if meta.get("finish_reason"):
                        final_meta = dict(meta)
                        if round_index == 0:
                            first_usage = final_meta
                        else:
                            # Report one turn: the client's prompt, plus every
                            # pass's generated tokens.
                            completion = prior_completion + int(
                                meta.get("completion_tokens") or 0
                            )
                            patched = dict(meta)
                            patched.update(
                                prompt_tokens=first_usage.get("prompt_tokens", 0),
                                cached_tokens=first_usage.get("cached_tokens", 0),
                                completion_tokens=completion,
                                reasoning_tokens=prior_reasoning
                                + int(meta.get("reasoning_tokens") or 0),
                                total_tokens=int(first_usage.get("prompt_tokens") or 0)
                                + completion,
                            )
                            context.append_output({**chunk, "meta_info": patched})
                yield context

            finish = final_meta.get("finish_reason") or {}
            if not isinstance(finish, Mapping) or finish.get("type") != "stop":
                return
            memory_calls, client_calls = self._parse_memory_round_calls(
                response_request, pass_text
            )
            if not memory_calls:
                return
            if client_calls:
                # The turn ends here with the client's calls; results of memory
                # calls made alongside them would never reach the model.
                logger.warning(
                    "QWEN_EXO_MEMORY_TOOL_ROUND request_id=%s dropped=%d "
                    "reason=mixed_with_client_calls",
                    request_id,
                    len(memory_calls),
                )
                return
            if round_index > MAX_MEMORY_TOOL_ROUNDS:
                return
            calls = await self._run_memory_round(
                memory_calls, limit_reached=round_index == MAX_MEMORY_TOOL_ROUNDS
            )
            logger.info(
                "QWEN_EXO_MEMORY_TOOL_ROUND request_id=%s round=%d calls=%s ok=%s "
                "output_chars=%s",
                request_id,
                round_index,
                [call["name"] for call in calls],
                [call["ok"] for call in calls],
                [len(call["output"]) for call in calls],
            )
            runtime = (
                getattr(raw_request.app.state, "qwen_exo_runtime", None)
                if raw_request is not None
                else None
            )
            if runtime is not None:
                await runtime.record_memory_tool_round(request_id, round_index, calls)
            context.qwen_exo_memory_rounds.append({"text": pass_text, "calls": calls})
            context.append_output(
                {
                    "text": "",
                    "output_ids": [],
                    "meta_info": {MEMORY_ROUND_META_KEY: {"calls": calls}},
                }
            )
            yield context

            prompt_ids = [
                *prompt_ids,
                *pass_ids,
                *template.continuation_ids(
                    output_ids=pass_ids,
                    outputs=[call["output"] for call in calls],
                    generation_prompt_ids=assistant_header,
                ),
            ]
            consumed_tokens += len(pass_ids)
            prior_completion += int(final_meta.get("completion_tokens") or 0)
            prior_reasoning += int(final_meta.get("reasoning_tokens") or 0)
            pass_sampling_params = self._phase_two_sampling_params(
                sampling_params,
                prompt_length=len(prompt_ids),
                consumed_tokens=consumed_tokens,
            )
            pass_request = replace(
                adapted_request,
                text=None,
                input_ids=prompt_ids,
                sampling_params=pass_sampling_params,
            )

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        response_request: ResponsesRequest | None = None,
        priority: Optional[int] = None,
        prompt_encoding: _ResponsePromptEncoding | None = None,
        generation_index_start: int = 0,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        if prompt_encoding is None:
            prompt_encoding = _ResponsePromptEncoding(self.tokenizer_manager.tokenizer)
        orig_priority = priority or 0
        generation_index = generation_index_start
        incremental_logprobs = bool(
            adapted_request.stream
            and self.tokenizer_manager.server_args.incremental_streaming_output
        )
        thinking_enabled = (
            self._is_thinking_enabled_for_request(response_request)
            if response_request is not None
            else None
        )

        while True:
            qwen_exo_runtime = (
                getattr(raw_request.app.state, "qwen_exo_runtime", None)
                if raw_request is not None
                else None
            )
            phase_custom_params = dict(sampling_params.get("custom_params") or {})
            phase_custom_params[_QWEN_EXO_DFLASH_THINK_PHASE] = bool(
                thinking_enabled
                and qwen_exo_runtime is not None
                and getattr(qwen_exo_runtime, "think_context_enabled", False)
            )
            sampling_params = self._copy_sampling_params(
                sampling_params, custom_params=phase_custom_params
            )
            adapted_request = replace(adapted_request, sampling_params=sampling_params)
            replay_prompt = (
                adapted_request.input_ids
                if adapted_request.input_ids is not None
                else adapted_request.text
            )
            if qwen_exo_runtime is not None and replay_prompt is not None:
                qwen_exo_runtime.register_generation_prompt(
                    request_id,
                    replay_prompt,
                    generation_index=generation_index,
                )

            prompt_token_ids = self._request_prompt_token_ids(
                adapted_request, prompt_encoding
            )
            trajectory_spans = ()
            if qwen_exo_runtime is not None and prompt_token_ids is not None:
                score_bias_builder = getattr(
                    qwen_exo_runtime, "score_bias_payload", None
                )
                if callable(score_bias_builder):
                    custom_params = dict(sampling_params.get("custom_params") or {})
                    score_bias_blocks = score_bias_builder(request_id, prompt_token_ids)
                    if score_bias_blocks:
                        custom_params["qwen_exo_score_bias_blocks"] = list(
                            score_bias_blocks
                        )
                    else:
                        custom_params.pop("qwen_exo_score_bias_blocks", None)
                    capture_builder = getattr(
                        qwen_exo_runtime, "score_bias_capture_payload", None
                    )
                    if callable(capture_builder):
                        trajectory_spans = capture_builder(request_id, prompt_token_ids)
                        if trajectory_spans:
                            custom_params["qwen_exo_trajectory_spans"] = list(
                                trajectory_spans
                            )
                        else:
                            custom_params.pop("qwen_exo_trajectory_spans", None)
                    sampling_params = dict(sampling_params)
                    sampling_params["custom_params"] = custom_params
            score_bias_logprob_start_len = None
            if (
                qwen_exo_runtime is not None
                and getattr(qwen_exo_runtime, "score_bias_enabled", False)
                and prompt_token_ids is not None
            ):
                score_bias_logprob_start_len = self._score_bias_logprob_start_len(
                    prompt_token_ids, trajectory_spans
                )
                telemetry = getattr(qwen_exo_runtime, "telemetry", None)
                emit = getattr(telemetry, "emit", None)
                if callable(emit) and trajectory_spans:
                    emit(
                        request_id,
                        "score_bias.input_logprob_window",
                        {
                            "generation_index": generation_index,
                            "prompt_tokens": len(prompt_token_ids),
                            "logprob_start_len": score_bias_logprob_start_len,
                            "scored_input_tokens": (
                                len(prompt_token_ids) - score_bias_logprob_start_len
                            ),
                            "capture_count": len(trajectory_spans),
                        },
                    )

            # Budget stopping applies to target-accepted tokens independently
            # of Observer/refresh; only memory injection needs that context lane.
            reasoning_end_token_id = (
                qwen_exo_runtime.reasoning_end_token_id
                if qwen_exo_runtime is not None
                and thinking_enabled is not False
                and prompt_token_ids is not None
                else None
            )
            generation_request = (
                replace(
                    adapted_request,
                    logprob_start_len=score_bias_logprob_start_len,
                )
                if score_bias_logprob_start_len is not None
                else adapted_request
            )
            if (
                qwen_exo_runtime is not None
                and not self.use_harmony
                and isinstance(generation_request.text, str)
                and generation_request.text
                and generation_request.input_ids is None
                and not any(
                    getattr(generation_request, field, None)
                    for field in (
                        "image_data", "video_data", "audio_data", "modalities"
                    )
                )
            ):
                # Keep the original text for recording, and leave internal
                # scoring/THINK IDs in their original no-special-token form.
                # The native tokenizer path uses default special-token options.
                generation_request = replace(
                    generation_request,
                    input_ids=list(prompt_encoding.encode(generation_request.text)),
                )
            max_reasoning_tokens = 0
            reasoning_cap_applied = False
            # Without a think-context injection there is nothing to insert at a
            # natural </think>, so the answer continues in the same request.
            single_phase_reasoning = reasoning_end_token_id is not None and not (
                getattr(qwen_exo_runtime, "think_context_enabled", False)
            )
            if reasoning_end_token_id is not None:
                max_reasoning_tokens = qwen_exo_runtime.reasoning_budget_for_effort(
                    self._requested_reasoning_effort(response_request)
                )
                phase_sampling_params, reasoning_cap_applied = (
                    self._single_phase_reasoning_sampling_params(
                        sampling_params,
                        reasoning_end_token_id,
                        max_reasoning_tokens,
                    )
                    if single_phase_reasoning
                    else self._reasoning_phase_sampling_params(
                        sampling_params,
                        reasoning_end_token_id,
                        max_reasoning_tokens,
                    )
                )
                generation_request = replace(
                    generation_request,
                    sampling_params=phase_sampling_params,
                )
            phase_output_ids: list[int] = []
            phase_output_text = ""
            phase_finish_reason: dict[str, Any] = {}
            phase_prompt_tokens = 0
            phase_cached_tokens = 0
            output_tokens_before = getattr(context, "num_output_tokens", 0)
            reasoning_tokens_before = getattr(context, "num_reasoning_tokens", 0)
            logger.info(
                "QWEN_EXO_GENERATION_START request_id=%s generation_index=%d "
                "prompt_tokens=%d reasoning_end_token_id=%s max_reasoning_tokens=%d",
                request_id,
                generation_index,
                len(prompt_token_ids),
                reasoning_end_token_id,
                max_reasoning_tokens,
            )
            generator = self.tokenizer_manager.generate_request(
                generation_request, raw_request
            )

            async for res in generator:
                meta_info = res.get("meta_info") or {}
                finish_reason = meta_info.get("finish_reason") or {}
                self._raise_for_generation_abort(finish_reason)
                if meta_info.get("prompt_tokens") is not None:
                    phase_prompt_tokens = int(meta_info["prompt_tokens"])
                if meta_info.get("cached_tokens") is not None:
                    phase_cached_tokens = int(meta_info["cached_tokens"])
                raw_output_ids = [int(token) for token in (res.get("output_ids") or ())]
                output_text = str(res.get("text") or "")
                if incremental_logprobs:
                    phase_output_ids.extend(raw_output_ids)
                    phase_output_text += output_text
                else:
                    phase_output_ids = raw_output_ids
                    phase_output_text = output_text
                if finish_reason:
                    phase_finish_reason = finish_reason
                if finish_reason:
                    logger.warning(
                        "QWEN_EXO_GENERATION_RESULT request_id=%s generation_index=%d "
                        "prompt_tokens=%s cached_tokens=%s output_id_count=%d "
                        "output_id_tail=%s output_text_bytes=%d finish_reason=%s "
                        "reasoning_end_token_id=%s",
                        request_id,
                        generation_index,
                        meta_info.get("prompt_tokens"),
                        meta_info.get("cached_tokens"),
                        len(raw_output_ids),
                        raw_output_ids[-16:],
                        len(output_text.encode("utf-8")),
                        finish_reason,
                        reasoning_end_token_id,
                    )
                if qwen_exo_runtime is not None:
                    qwen_exo_runtime.observe_generation_result(
                        request_id,
                        res,
                        incremental_logprobs=incremental_logprobs,
                        generation_index=generation_index,
                        thinking_enabled=thinking_enabled,
                    )
                public_result = res
                if reasoning_end_token_id is not None and (
                    self._matched_reasoning_boundary(
                        finish_reason, reasoning_end_token_id
                    )
                    or (
                        self._forced_reasoning_boundary(
                            finish_reason,
                            output_tokens=len(phase_output_ids),
                            max_reasoning_tokens=max_reasoning_tokens,
                            cap_applied=reasoning_cap_applied,
                        )
                        and reasoning_end_token_id not in phase_output_ids
                    )
                ):
                    public_result = dict(res)
                    public_meta = dict(meta_info)
                    public_meta.pop("finish_reason", None)
                    public_result["meta_info"] = public_meta
                context.append_output(public_result)
                yield context
            matched_reasoning_boundary = (
                reasoning_end_token_id is not None
                and self._matched_reasoning_boundary(
                    phase_finish_reason, reasoning_end_token_id
                )
            )
            forced_reasoning_boundary = (
                reasoning_end_token_id is not None
                and self._forced_reasoning_boundary(
                    phase_finish_reason,
                    output_tokens=len(phase_output_ids),
                    max_reasoning_tokens=max_reasoning_tokens,
                    cap_applied=reasoning_cap_applied,
                )
                # A length finish after a natural </think> is the answer's own
                # max_new_tokens, not the reasoning budget.
                and reasoning_end_token_id not in phase_output_ids
            )
            if (
                single_phase_reasoning
                and not forced_reasoning_boundary
                and reasoning_end_token_id in phase_output_ids
            ):
                reasoning_end = phase_output_ids.index(reasoning_end_token_id) + 1
                context.num_reasoning_tokens = reasoning_tokens_before + reasoning_end
                qwen_exo_runtime.record_reasoning_boundary(
                    request_id,
                    injection=None,
                    committed_text="",
                    token_ids=(),
                    generation_index=generation_index,
                )
            if matched_reasoning_boundary or forced_reasoning_boundary:
                assert reasoning_end_token_id is not None
                assert qwen_exo_runtime is not None
                if forced_reasoning_boundary:
                    injection = await qwen_exo_runtime.build_reasoning_cutoff_injection(
                        request_id,
                        observed_tokens=len(phase_output_ids),
                        generation_index=generation_index,
                        max_reasoning_tokens=max_reasoning_tokens,
                    )
                else:
                    injection = await qwen_exo_runtime.await_think_context(request_id)
                boundary_text, boundary_ids = self._reasoning_boundary_tokens(
                    injection,
                    reasoning_end_token_id,
                    forced=forced_reasoning_boundary,
                )
                combined_prefix_text = f"{phase_output_text}{boundary_text}"
                combined_prefix_ids = [*phase_output_ids, *boundary_ids]
                synthetic_result = {
                    "text": (
                        boundary_text if incremental_logprobs else combined_prefix_text
                    ),
                    "output_ids": (
                        list(boundary_ids)
                        if incremental_logprobs
                        else combined_prefix_ids
                    ),
                    "meta_info": {
                        "qwen_exo_self_ask_boundary": injection is not None,
                    },
                }
                context.append_output(synthetic_result)
                context.num_prompt_tokens = phase_prompt_tokens or len(prompt_token_ids)
                context.num_cached_tokens = phase_cached_tokens
                context.num_output_tokens = output_tokens_before + len(
                    combined_prefix_ids
                )
                context.num_reasoning_tokens = reasoning_tokens_before + len(
                    combined_prefix_ids
                )
                qwen_exo_runtime.record_reasoning_boundary(
                    request_id,
                    injection=injection,
                    committed_text=boundary_text,
                    token_ids=boundary_ids,
                    generation_index=generation_index,
                )
                yield context

                continuation_prompt_ids = [
                    *prompt_token_ids,
                    *combined_prefix_ids,
                ]
                continuation_sampling_params = self._phase_two_sampling_params(
                    sampling_params,
                    prompt_length=len(continuation_prompt_ids),
                    # The forced stop-overthinking instruction is an internal
                    # control prompt, not user-visible model output. Keep it in
                    # the continuation prompt while charging only the generated
                    # reasoning tokens and the synthetic </think> boundary
                    # against max_new_tokens.
                    consumed_tokens=(
                        len(phase_output_ids) + 1
                        if forced_reasoning_boundary
                        else len(combined_prefix_ids)
                    ),
                )
                continuation_custom_params = dict(
                    continuation_sampling_params.get("custom_params") or {}
                )
                continuation_custom_params[_QWEN_EXO_DFLASH_THINK_PHASE] = False
                continuation_sampling_params = self._copy_sampling_params(
                    continuation_sampling_params,
                    custom_params=continuation_custom_params,
                )
                continuation_request = replace(
                    adapted_request,
                    text=None,
                    input_ids=continuation_prompt_ids,
                    sampling_params=continuation_sampling_params,
                    # </think> is already in the prompt: the answer grammar
                    # applies from the first continuation token.
                    require_reasoning=False,
                )
                if score_bias_logprob_start_len is not None:
                    # Keep the phase-one logprob window on phase two. With
                    # return_logprob enabled, the default zero would cap the
                    # radix match at zero and force a full prompt re-prefill.
                    continuation_request = replace(
                        continuation_request,
                        logprob_start_len=score_bias_logprob_start_len,
                    )
                if hasattr(context, "num_processed_tokens"):
                    context.num_processed_tokens = 0
                continuation_output_ids: list[int] = []
                continuation_text_seen = ""
                continuation_runtime_ids = 0
                continuation_runtime_text = ""
                continuation_runtime_signals = 0
                continuation_generator = self.tokenizer_manager.generate_request(
                    continuation_request, raw_request
                )
                async for continuation_result in continuation_generator:
                    continuation_meta = continuation_result.get("meta_info") or {}
                    continuation_finish = continuation_meta.get("finish_reason") or {}
                    self._raise_for_generation_abort(continuation_finish)
                    raw_ids = [
                        int(token)
                        for token in (continuation_result.get("output_ids") or ())
                    ]
                    if incremental_logprobs:
                        continuation_output_ids.extend(raw_ids)
                    else:
                        continuation_output_ids = raw_ids
                    runtime_result = continuation_result
                    if not incremental_logprobs:
                        runtime_result = dict(continuation_result)
                        runtime_result["output_ids"] = raw_ids[
                            continuation_runtime_ids:
                        ]
                        continuation_text = str(continuation_result.get("text") or "")
                        runtime_result["text"] = (
                            continuation_text[len(continuation_runtime_text) :]
                            if continuation_text.startswith(continuation_runtime_text)
                            else continuation_text
                        )
                        runtime_meta = dict(continuation_meta)
                        for signal_key in (
                            "output_token_logprobs",
                            "qwen_exo_q_norm",
                            "qwen_exo_q_drift",
                            "qwen_exo_memory_energy",
                            "qwen_exo_q_sketch",
                        ):
                            values = runtime_meta.get(signal_key)
                            if values is not None:
                                runtime_meta[signal_key] = values[
                                    continuation_runtime_signals:
                                ]
                        runtime_result["meta_info"] = runtime_meta
                        continuation_runtime_ids = len(raw_ids)
                        continuation_runtime_text = continuation_text
                        continuation_runtime_signals = len(
                            continuation_meta.get("output_token_logprobs") or ()
                        )
                    qwen_exo_runtime.observe_generation_result(
                        request_id,
                        runtime_result,
                        incremental_logprobs=True,
                        generation_index=generation_index,
                        thinking_enabled=thinking_enabled,
                    )
                    public_result = continuation_result
                    if isinstance(context, SimpleContext):
                        public_result = dict(continuation_result)
                        continuation_text = str(continuation_result.get("text") or "")
                        if not incremental_logprobs:
                            public_result["text"] = (
                                f"{combined_prefix_text}{continuation_text}"
                            )
                            public_result["output_ids"] = [
                                *combined_prefix_ids,
                                *raw_ids,
                            ]
                        public_meta = dict(continuation_meta)
                        if continuation_finish:
                            completion_tokens = len(combined_prefix_ids) + len(
                                continuation_output_ids
                            )
                            public_meta.update(
                                {
                                    "prompt_tokens": context.num_prompt_tokens,
                                    "cached_tokens": phase_cached_tokens,
                                    "completion_tokens": completion_tokens,
                                    "reasoning_tokens": len(combined_prefix_ids),
                                    "total_tokens": (
                                        context.num_prompt_tokens + completion_tokens
                                    ),
                                }
                            )
                        public_result["meta_info"] = public_meta
                    context.append_output(public_result)
                    chunk_text = str(continuation_result.get("text") or "")
                    continuation_text_seen = (
                        continuation_text_seen + chunk_text
                        if incremental_logprobs
                        else chunk_text
                    )
                    if continuation_finish:
                        # Structure only (marker counts, ids); never the text.
                        logger.warning(
                            "QWEN_EXO_CONTINUATION_RESULT request_id=%s "
                            "generation_index=%d forced=%s output_id_count=%d "
                            "tool_call_open=%d tool_call_close=%d "
                            "output_id_tail=%s finish_reason=%s",
                            request_id,
                            generation_index,
                            forced_reasoning_boundary,
                            len(continuation_output_ids),
                            continuation_text_seen.count("<tool_call>"),
                            continuation_text_seen.count("</tool_call>"),
                            continuation_output_ids[-16:],
                            continuation_finish,
                        )
                    yield context
                context.num_prompt_tokens = phase_prompt_tokens or len(prompt_token_ids)
                context.num_cached_tokens = phase_cached_tokens
                context.num_output_tokens = (
                    output_tokens_before
                    + len(combined_prefix_ids)
                    + len(continuation_output_ids)
                )
                context.num_reasoning_tokens = reasoning_tokens_before + len(
                    combined_prefix_ids
                )

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_request = context.messages[-1] if context.messages else None
            tool_call_payload = None
            if tool_request is not None:
                tool_call_payload = {
                    "recipient": str(getattr(tool_request, "recipient", "") or ""),
                    "arguments": "\n".join(
                        str(text)
                        for part in getattr(tool_request, "content", ()) or ()
                        if (text := getattr(part, "text", None))
                    ),
                }
            tool_output = await context.call_tool()
            context.append_output(tool_output)
            if qwen_exo_runtime is not None:
                observation_parts = []
                for message in tool_output:
                    for part in getattr(message, "content", ()) or ():
                        text = getattr(part, "text", None)
                        if text:
                            observation_parts.append(str(text))
                await qwen_exo_runtime.recall_after_tool(
                    request_id,
                    "\n".join(observation_parts),
                    generation_index=generation_index,
                    tool_call=tool_call_payload,
                )

            # Prepare for the next generation turn
            # Render the updated conversation for the next completion
            prompt_token_ids = list(context.render_for_completion())

            sampling_params = self._phase_two_sampling_params(
                sampling_params,
                prompt_length=len(prompt_token_ids),
                consumed_tokens=0,
            )
            adapted_request = replace(
                adapted_request,
                text=None,
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            if hasattr(context, "num_processed_tokens"):
                context.num_processed_tokens = 0

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1
            generation_index += 1
