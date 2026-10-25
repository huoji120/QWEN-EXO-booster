# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.types import ListToolsResult
except ImportError as e:
    ClientSession = sse_client = ListToolsResult = e

from openai_harmony import ToolDescription, ToolNamespaceConfig

logger = logging.getLogger(__name__)


async def list_server_and_tools(server_url: str):

    async with (
        sse_client(url=server_url) as streams,
        ClientSession(*streams) as session,
    ):
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response


def trim_schema(schema: dict) -> dict:
    # Turn JSON Schema from MCP generated into Harmony's variant.
    if "title" in schema:
        del schema["title"]
    if "default" in schema and schema["default"] is None:
        del schema["default"]
    if "anyOf" in schema:
        # Turn "anyOf": [{"type": "type-1"}, {"type": "type-2"}]
        # into "type": ["type-1", "type-2"]
        # if there's more than 1 types, also remove "null" type as Harmony will
        # just ignore it
        types = [
            type_dict["type"]
            for type_dict in schema["anyOf"]
            if type_dict["type"] != "null"
        ]
        schema["type"] = types
        del schema["anyOf"]
    if "properties" in schema:
        schema["properties"] = {
            k: trim_schema(v) for k, v in schema["properties"].items()
        }
    return schema


def post_process_tools_description(
    list_tools_result: "ListToolsResult",
) -> "ListToolsResult":
    # Adapt the MCP tool result for Harmony
    for tool in list_tools_result.tools:
        tool.inputSchema = trim_schema(tool.inputSchema)

    # Some tools schema don't need to be part of the prompt (e.g. simple text
    # in text out for Python)
    list_tools_result.tools = [
        tool
        for tool in list_tools_result.tools
        if getattr(tool.annotations, "include_in_prompt", True)
    ]

    return list_tools_result


class ToolServer(ABC):

    @abstractmethod
    def has_tool(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_description(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_session(self, tool_name: str) -> AbstractAsyncContextManager[Any]: ...


class MCPToolServer(ToolServer):

    def __init__(self):
        self.harmony_tool_descriptions = {}

    async def add_tool_server(self, server_url: str):
        tool_urls = server_url.split(",")
        self.harmony_tool_descriptions = {}
        self.urls: dict[str, str] = {}
        for url in tool_urls:
            url = f"http://{url}/sse"
            initialize_response, list_tools_response = await list_server_and_tools(url)

            list_tools_response = post_process_tools_description(list_tools_response)

            tool_from_mcp = ToolNamespaceConfig(
                name=initialize_response.serverInfo.name,
                description=initialize_response.instructions,
                tools=[
                    ToolDescription.new(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema,
                    )
                    for tool in list_tools_response.tools
                ],
            )
            self.harmony_tool_descriptions[tool_from_mcp.name] = tool_from_mcp
            if tool_from_mcp.name not in self.urls:
                self.urls[tool_from_mcp.name] = url
            else:
                logger.warning(
                    "Tool %s already exists. Ignoring duplicate tool server %s",
                    tool_from_mcp.name,
                    url,
                )

    def has_tool(self, tool_name: str):
        return tool_name in self.harmony_tool_descriptions

    def get_tool_description(self, tool_name: str):
        return self.harmony_tool_descriptions.get(tool_name)

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        url = self.urls.get(tool_name)
        if url:
            async with (
                sse_client(url=url) as streams,
                ClientSession(*streams) as session,
            ):
                await session.initialize()
                yield session
        else:
            logger.warning("Tool %s not found", tool_name)


class DemoToolServer(ToolServer):

    def __init__(self, *, enable_python: bool = True):
        from sglang.srt.entrypoints.tool import (
            HarmonyBrowserTool,
            HarmonyPythonTool,
            Tool,
        )

        self.tools: dict[str, Tool] = {}
        browser_tool = HarmonyBrowserTool()
        if browser_tool.enabled:
            self.tools["browser"] = browser_tool
        if enable_python:
            python_tool = HarmonyPythonTool()
            if python_tool.enabled:
                self.tools["python"] = python_tool

    def has_tool(self, tool_name: str):
        return tool_name in self.tools

    def get_tool_description(self, tool_name: str):
        if tool_name not in self.tools:
            return None
        if tool_name == "browser":
            return ToolNamespaceConfig.browser()
        elif tool_name == "python":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown tool {tool_name}")

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        yield self.tools[tool_name]

    async def aclose(self):
        browser = self.tools.get("browser")
        exa_client = getattr(browser, "exa_client", None) if browser else None
        if exa_client is not None:
            await exa_client.close()


class NativeToolServer(DemoToolServer):
    """Built-in SGLang hosted tools that do not require an external MCP server."""

    def __init__(self):
        super().__init__(enable_python=False)

class FileMemoryToolServer(ToolServer):
    """Server-executed qwen-exo memory tools over the ordinary Knowledge files.

    Unlike the harmony builtins these are plain function tools rendered into
    the Qwen prompt; ``serving_responses`` runs them through ``executor``.
    """

    def __init__(self, root, state_directory):
        import threading

        from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
        from qwen_exo_booster.memory_tools import (
            MEMORY_TOOL_SCHEMAS,
            MemoryToolExecutor,
        )

        self.executor = MemoryToolExecutor(FileKnowledgeIndex(root, state_directory))
        self._memory_executors = {(None, "shared"): self.executor}
        self._memory_write_windows = {None: self.executor._write_window}
        self._memory_executor_lock = threading.Lock()
        scope_descriptions = {
            "shared": " This request uses shared memory only: reads, searches and work-memory writes target shared storage. No private memory is accessible.",
            "private": " This request uses private memory: reads and searches include shared and your own private files only. Unprefixed read paths are shared; private/ read paths are your own. No other key's private memory is accessible.",
            "private_only": " This request uses private_only memory: reads, searches and counts include ONLY your own private files, never shared or other owners. Reads require private/ paths; unprefixed shared paths are denied.",
            "super": " This request uses super memory: reads and searches include shared files, your own private/ files, and all existing other owners' files at private-keys/<key_id>/<relative-path>, including removed or revoked owners. Unprefixed read paths are shared. These read rights grant no key administration or conversation access.",
        }
        own_writes = (
            " All writes, edits and deletes target ONLY your own private work-memory notes, "
            "with or without the private/ prefix. Shared and other owners' files cannot be "
            "changed; private-keys/ write paths are rejected. Use private/work-memory/... "
            "to read back a note you wrote."
        )
        self._scoped_chat_tool_schemas = {
            mode: tuple(
                {**schema, "function": {
                    **schema["function"],
                    "description": schema["function"]["description"] + description
                    + (own_writes if mode != "shared" else ""),
                }}
                for schema in MEMORY_TOOL_SCHEMAS
            )
            for mode, description in scope_descriptions.items()
        }
        self.chat_tool_schemas = self._scoped_chat_tool_schemas["shared"]
        self._descriptions = {
            schema["function"]["name"]: ToolDescription.new(
                name=schema["function"]["name"],
                description=schema["function"]["description"],
                parameters=schema["function"]["parameters"],
            )
            for schema in MEMORY_TOOL_SCHEMAS
        }

    def executor_for_memory_scope(self, key_id: str | None, memory_scope: str):
        """Select from authenticated request identity, never model tool arguments."""
        from qwen_exo_booster.memory_tools import MemoryToolExecutor, MemoryWriteWindow
        from qwen_exo_booster.scoped_memory import get_private_memory_index

        self.chat_tool_schemas_for_memory_scope(key_id, memory_scope)
        with self._memory_executor_lock:
            cache_key = (key_id, memory_scope)
            executor = self._memory_executors.get(cache_key)
            if executor is None:
                shared = self.executor.index
                window = self._memory_write_windows.get(key_id)
                if window is None:
                    window = MemoryWriteWindow()
                    self._memory_write_windows[key_id] = window
                if memory_scope == "shared":
                    executor = MemoryToolExecutor(shared, write_window=window)
                else:
                    private = get_private_memory_index(shared.root, shared.state_directory, key_id)
                    executor = MemoryToolExecutor(
                        private, shared_index=shared, key_id=key_id,
                        memory_scope=memory_scope, write_window=window,
                    )
                self._memory_executors[cache_key] = executor
            return executor

    def chat_tool_schemas_for_memory_scope(self, key_id: str | None, memory_scope: str):
        from qwen_exo_booster.api_keys import ApiKeyStoreError, validate_memory_scope
        from qwen_exo_booster.file_knowledge import FileKnowledgeError
        from qwen_exo_booster.scoped_memory import validate_memory_key_id

        try:
            memory_scope = validate_memory_scope(memory_scope)
        except ApiKeyStoreError as exc:
            raise FileKnowledgeError(str(exc)) from exc
        if key_id is not None:
            validate_memory_key_id(key_id)
        elif memory_scope != "shared":
            raise FileKnowledgeError("private memory requires an authenticated owner")
        return self._scoped_chat_tool_schemas[memory_scope]

    def has_tool(self, tool_name: str):
        return tool_name in self._descriptions

    def get_tool_description(self, tool_name: str):
        return self._descriptions.get(tool_name)

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        executor = self.executor

        class Session:
            async def call_tool(self, name, arguments):
                _ok, text = await executor.run(name, arguments)
                return type(
                    "ToolResult", (), {"content": [type("Text", (), {"text": text})()]}
                )()

        yield Session()
