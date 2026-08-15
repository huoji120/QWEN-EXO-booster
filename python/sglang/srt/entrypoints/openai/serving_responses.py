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
import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse
from openai_harmony import Message as OpenAIMessage
from openai_harmony import Role
from qwen_exo_booster.memory_span import locate_memory_span
from qwen_exo_booster.pipeline import response_memory_metadata
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
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid

import openai.types.responses as openai_responses_types
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

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)

_QWEN_EXO_SELF_CHECK_START = "<qwen_exo_self_check>"
_QWEN_EXO_SELF_CHECK_END = "</qwen_exo_self_check>"


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

        # Response storage for background and retrieval operations
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # Message storage for conversation continuity
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.msg_store: dict[
            str, Union[list[ChatCompletionMessageParam], list[OpenAIMessage]]
        ] = {}

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
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code,
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
            normalized = self._normalize_response_message_for_chat(item)
            if normalized is not None:
                public_messages.append(normalized)  # type: ignore[arg-type]
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

        try:
            model_name = request.model
            tokenizer = self.tokenizer_manager.tokenizer
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
        qwen_exo_runtime = (
            getattr(raw_request.app.state, "qwen_exo_runtime", None)
            if raw_request is not None
            else None
        )
        qwen_exo_observe = (
            qwen_exo_runtime is not None and qwen_exo_runtime.observer.mode != "off"
        )
        qwen_exo_score_bias = bool(
            qwen_exo_runtime is not None
            and getattr(qwen_exo_runtime, "score_bias_enabled", False)
        )
        memory_state = (
            getattr(raw_request.state, "qwen_exo_memory_state", None)
            if raw_request is not None
            else None
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
                            len(tokenizer.encode(prompt))
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
                for field in ("image_data", "video_data", "audio_data", "modalities")
            )
            rendered_prompt_tokens = rendered_length(engine_prompts) + len(
                getattr(memory_state, "radix_prefix_token_ids", ()) or ()
            )
            over_context_budget = (
                rendered_prompt_tokens + reserved_output_tokens > context_len
            )
            if (
                memory_state is not None
                and (
                    memory_state.private_attachment is not None
                    or memory_state.policy_attachment is not None
                )
                and (has_multimodal_input or over_context_budget)
            ):
                (
                    request,
                    memory_state,
                ) = await qwen_exo_runtime.drop_memory_attachment_for_context(
                    request,
                    rendered_prompt_tokens=rendered_prompt_tokens,
                    context_length=context_len,
                    reserved_output_tokens=reserved_output_tokens,
                    reason=(
                        "multimodal_post_expansion_unknown"
                        if has_multimodal_input
                        else "context_capacity"
                    ),
                    include_policy=(
                        has_multimodal_input or memory_state.private_attachment is None
                    ),
                )
                raw_request.state.qwen_exo_memory_state = memory_state
                raw_request.state.qwen_exo_memory = (
                    memory_state.public_dict() if memory_state is not None else None
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
                    logger.exception("Error rebuilding prompt without QWEN-EXO memory")
                    return self.create_error_response(f"{exc} {exc.__cause__}")

            has_multimodal_input = processed_messages is not None and any(
                getattr(processed_messages, field, None)
                for field in ("image_data", "video_data", "audio_data", "modalities")
            )
            rendered_prompt_tokens = rendered_length(engine_prompts) + len(
                getattr(memory_state, "radix_prefix_token_ids", ()) or ()
            )
            over_context_budget = (
                rendered_prompt_tokens + reserved_output_tokens > context_len
            )
            if (
                memory_state is not None
                and memory_state.policy_attachment is not None
                and (has_multimodal_input or over_context_budget)
            ):
                (
                    request,
                    memory_state,
                ) = await qwen_exo_runtime.drop_memory_attachment_for_context(
                    request,
                    rendered_prompt_tokens=rendered_prompt_tokens,
                    context_length=context_len,
                    reserved_output_tokens=reserved_output_tokens,
                    reason=(
                        "multimodal_post_expansion_unknown"
                        if has_multimodal_input
                        else "context_capacity"
                    ),
                    include_policy=True,
                )
                raw_request.state.qwen_exo_memory_state = memory_state
                raw_request.state.qwen_exo_memory = (
                    memory_state.public_dict() if memory_state is not None else None
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
                        "Error rebuilding prompt without QWEN-EXO PolicyData"
                    )
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
            memory_diagnostics = response_memory_metadata(
                memory_state,
                fallback=(
                    getattr(raw_request.state, "qwen_exo_memory", None)
                    if raw_request is not None
                    else None
                ),
                observer_mode=qwen_exo_runtime.observer.mode,
            )
            response_metadata = self._stringify_response_metadata(request.metadata)
            encoded_memory_diagnostics = json.dumps(
                memory_diagnostics,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            response_metadata["memory"] = encoded_memory_diagnostics
            response_metadata["qwen_exo"] = encoded_memory_diagnostics
            request = request.model_copy(update={"metadata": response_metadata})
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
                    memory_state = (
                        getattr(raw_request.state, "qwen_exo_memory_state", None)
                        if raw_request is not None
                        else None
                    )
                    native_prefix_ids = tuple(
                        getattr(memory_state, "radix_prefix_token_ids", ()) or ()
                    )
                    if native_prefix_ids:
                        rendered_ids = (
                            tokenizer.encode(engine_prompt)
                            if isinstance(engine_prompt, str)
                            else list(engine_prompt)
                        )
                        engine_prompt = [*native_prefix_ids, *rendered_ids]
                    # Calculate default max tokens from context length minus prompt length
                    if isinstance(engine_prompt, list):
                        prompt_length = len(engine_prompt)
                    elif isinstance(engine_prompt, str):
                        prompt_length = len(tokenizer.encode(engine_prompt))
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
                        custom_params = dict(sampling_params.get("custom_params") or {})
                        custom_params["qwen_exo_kind"] = "user"
                        if memory_state is not None:
                            if native_prefix_ids:
                                custom_params.update(
                                    {
                                        "qwen_exo_radix_prefix_page_id": (
                                            memory_state.radix_prefix_page_id
                                        ),
                                        "qwen_exo_radix_prefix_identity": (
                                            memory_state.radix_prefix_identity
                                        ),
                                        "qwen_exo_radix_prefix_tokens": len(
                                            native_prefix_ids
                                        ),
                                    }
                                )
                                if (
                                    memory_state.radix_prefix_source_digest
                                    and memory_state.radix_prefix_local_positions
                                ):
                                    custom_params["qwen_exo_native_bank_selection"] = {
                                        "source_digest": (
                                            memory_state.radix_prefix_source_digest
                                        ),
                                        "page_id": memory_state.radix_prefix_page_id,
                                        "local_positions": list(
                                            memory_state.radix_prefix_local_positions
                                        ),
                                        "prefix_identity": (
                                            memory_state.radix_prefix_identity
                                        ),
                                    }
                            memory_span = locate_memory_span(
                                engine_prompt,
                                tokenizer,
                                memory_state.private_attachment,
                            )
                            if memory_span is not None:
                                custom_params.update(
                                    {
                                        "qwen_exo_memory_start": memory_span[0],
                                        "qwen_exo_memory_length": memory_span[1],
                                        "qwen_exo_memory_key": (
                                            f"{memory_state.attachment_digest}:"
                                            f"{memory_span[0]}:{memory_span[1]}"
                                        ),
                                    }
                                )
                            elif memory_state.private_attachment:
                                qwen_exo_runtime.telemetry.emit(
                                    request.request_id,
                                    "observer.memory_span_unresolved",
                                    {
                                        "attachment_digest": memory_state.attachment_digest,
                                        "attached_tokens": memory_state.attached_tokens,
                                    },
                                )
                            if (
                                native_prefix_ids
                                and memory_state.radix_prefix_source_digest
                                and memory_state.radix_prefix_local_positions
                            ):
                                custom_params.update(
                                    {
                                        "qwen_exo_memory_start": 0,
                                        "qwen_exo_memory_length": len(
                                            native_prefix_ids
                                        ),
                                        "qwen_exo_memory_key": (
                                            "qwen-exo-native:"
                                            f"{memory_state.radix_prefix_identity}"
                                        ),
                                    }
                                )
                        score_bias_builder = getattr(
                            qwen_exo_runtime, "score_bias_payload", None
                        )
                        if callable(score_bias_builder):
                            score_bias_prompt_ids = (
                                list(engine_prompt)
                                if isinstance(engine_prompt, list)
                                else tokenizer.encode(
                                    engine_prompt, add_special_tokens=False
                                )
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
                    )

                    generator = self._generate_with_builtin_tools(
                        request.request_id,
                        request_prompts[i],
                        adapted_request,
                        sampling_params,
                        context,
                        raw_request=raw_request,
                        priority=request.priority,
                        native_prefix_ids=native_prefix_ids,
                    )
                    if qwen_exo_runtime is not None:
                        generator = qwen_exo_runtime.track_generation(
                            request.request_id, generator
                        )
                    generators.append(generator)
            except ValueError as e:
                return self.create_error_response(str(e))

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages
            if request.store:
                public_messages = (
                    self._construct_input_messages_with_harmony(request, prev_response)
                    if self.use_harmony
                    else self._construct_input_messages(request, prev_response)
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
                result: Union[
                    ORJSONResponse, ResponsesResponse
                ] = await self.responses_full_generator(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time=created_time,
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
                return self.create_error_response(str(e))
        return self.create_error_response("Unknown error")

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        messages = self._construct_input_messages(request, prev_response)

        chat_tools = self._response_tools_to_chat_tools(request)
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
            reasoning_effort=(request.reasoning.effort if request.reasoning else None),
        )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._process_messages(chat_request, is_multimodal)

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
            return self.create_error_response(str(e))

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

            output = self._make_response_output_items(
                request, final_res["text"], tokenizer
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

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
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

        chat_tools = self._response_tools_to_chat_tools(request)
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
        output_items.extend(tool_call_items)
        return output_items

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
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )

        # Prepend the conversation history
        if prev_response is not None:
            prev_msg = self.msg_store[prev_response.id]
            for message in prev_msg:
                normalized = self._normalize_response_message_for_chat(message)
                if normalized is not None:
                    messages.append(normalized)

            for output_item in prev_response.output:
                assistant_text = self._output_message_text(output_item)
                if assistant_text is None:
                    continue
                messages.append({"role": "assistant", "content": assistant_text})

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for input_item in request.input:
                normalized = self._normalize_response_message_for_chat(input_item)
                if normalized is not None:
                    messages.append(normalized)  # type: ignore

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
        if system_chunks:
            return [
                {"role": "system", "content": "\n\n".join(system_chunks)}
            ] + other_msgs
        return other_msgs

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
    def _public_response_error(
        exc: Exception,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if isinstance(exc, HTTPException):
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            status_code = int(exc.status_code)
            message = str(detail.get("message", exc.detail))
            internal_code = str(detail.get("code", "scheduler_abort"))
            retry_after = detail.get("retry_after")
        else:
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            message = "Request failed"
            internal_code = type(exc).__name__
            retry_after = None
        public_code = (
            "rate_limit_exceeded"
            if status_code == HTTPStatus.TOO_MANY_REQUESTS
            else (
                "invalid_prompt"
                if status_code
                in {
                    HTTPStatus.BAD_REQUEST,
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                }
                else "server_error"
            )
        )
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
                if stored_response.status not in ("completed", "cancelled"):
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

        chat_tools = self._response_tools_to_chat_tools(request)
        is_required = request.tool_choice == "required"
        tool_parser: Optional[Union[FunctionCallParser, JsonArrayParser]] = None
        if chat_tools and request.tool_choice != "none":
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
                tool_parser = JsonArrayParser()
            elif self.tool_call_parser:
                tool_parser = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
        reasoning_parser_obj: Optional[ReasoningParser] = None
        if self.reasoning_parser:
            reasoning_parser_obj = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )

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

        try:
            async for ctx in result_generator:
                if isinstance(ctx, dict):
                    chunk = ctx
                else:
                    chunk = getattr(ctx, "last_output", None)
                if not isinstance(chunk, dict):
                    continue
                meta = chunk.get("meta_info") or {}
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
                    tool_index = call.tool_index
                    state = tool_call_states.get(tool_index)
                    if state is None or state.get("done"):
                        current_output_index += 1
                        item_id = f"fc_{random_uuid()[:8]}"
                        call_id = f"call_{random_uuid()[:24]}"
                        state = {
                            "item_id": item_id,
                            "call_id": call_id,
                            "output_index": current_output_index,
                            "name": call.name or "",
                            "arguments": "",
                            "added": False,
                            "done": False,
                        }
                        tool_call_states[tool_index] = state
                    if not state["added"]:
                        state["added"] = True
                        # Capture ``call.name`` before the ``added`` event so
                        # the name is set on the first emitted item.
                        if call.name and not state["name"]:
                            state["name"] = call.name
                        yield _send_event(
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
                    if call.parameters:
                        state["arguments"] += call.parameters
                        yield _send_event(
                            openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                                type="response.function_call_arguments.delta",
                                sequence_number=-1,
                                item_id=state["item_id"],
                                output_index=state["output_index"],
                                delta=call.parameters,
                            )
                        )
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

        final_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=final_output_items,
            status="completed",
            usage=usage,
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
        self, adapted_request: GenerateReqInput
    ) -> list[int] | None:
        if adapted_request.input_ids is not None:
            return list(adapted_request.input_ids)
        if adapted_request.text is not None:
            return list(
                self.tokenizer_manager.tokenizer.encode(
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

    def _reasoning_boundary_tokens(
        self,
        injection: Any,
        reasoning_end_token_id: int,
        *,
        forced: bool = False,
    ) -> tuple[str, tuple[int, ...]]:
        injected_text = injection.text if injection is not None else ""
        if forced:
            injected_text = "\nlet me do this now stop over thinking\n"
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

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        priority: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        orig_priority = priority or 0
        generation_index = 0
        incremental_logprobs = bool(
            adapted_request.stream
            and self.tokenizer_manager.server_args.incremental_streaming_output
        )
        native_prefix_ids = tuple(kwargs.get("native_prefix_ids") or ())

        while True:
            qwen_exo_runtime = (
                getattr(raw_request.app.state, "qwen_exo_runtime", None)
                if raw_request is not None
                else None
            )
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

            prompt_token_ids = self._request_prompt_token_ids(adapted_request)
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

            reasoning_end_token_id = (
                qwen_exo_runtime.reasoning_end_token_id
                if qwen_exo_runtime is not None
                and qwen_exo_runtime.think_context_enabled
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
            max_reasoning_tokens = 0
            reasoning_cap_applied = False
            if reasoning_end_token_id is not None:
                max_reasoning_tokens = int(
                    getattr(qwen_exo_runtime, "max_reasoning_tokens", 3072)
                )
                phase_sampling_params, reasoning_cap_applied = (
                    self._reasoning_phase_sampling_params(
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
                "prompt_tokens=%d native_prefix_tokens=%d reasoning_end_token_id=%s "
                "max_reasoning_tokens=%d",
                request_id,
                generation_index,
                len(prompt_token_ids),
                len(native_prefix_ids),
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
                    )
                public_result = res
                if reasoning_end_token_id is not None and (
                    self._matched_reasoning_boundary(
                        finish_reason, reasoning_end_token_id
                    )
                    or self._forced_reasoning_boundary(
                        finish_reason,
                        output_tokens=len(phase_output_ids),
                        max_reasoning_tokens=max_reasoning_tokens,
                        cap_applied=reasoning_cap_applied,
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
            )
            if matched_reasoning_boundary or forced_reasoning_boundary:
                assert reasoning_end_token_id is not None
                assert qwen_exo_runtime is not None
                if forced_reasoning_boundary:
                    await qwen_exo_runtime.discard_think_context_for_reasoning_budget(
                        request_id,
                        observed_tokens=len(phase_output_ids),
                        generation_index=generation_index,
                    )
                    injection = None
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
                continuation_request = replace(
                    adapted_request,
                    text=None,
                    input_ids=continuation_prompt_ids,
                    sampling_params=continuation_sampling_params,
                )
                if hasattr(context, "num_processed_tokens"):
                    context.num_processed_tokens = 0
                continuation_output_ids: list[int] = []
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
            prompt_token_ids = [
                *native_prefix_ids,
                *context.render_for_completion(),
            ]

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
