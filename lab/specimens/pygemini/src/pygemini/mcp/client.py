"""MCP (Model Context Protocol) client integration.

Connects to configured MCP servers via stdio transport, discovers their
tools, and wraps each one as a ``DiscoveredMCPTool`` (a ``BaseTool`` subclass)
so the AgentLoop can dispatch them like any built-in tool.

Configuration example (in settings.toml):

    [mcp_servers.my_server]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env = {}
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pygemini.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from pygemini.core.config import Config, MCPServerConfig

# ---------------------------------------------------------------------------
# Conditional MCP SDK import
# ---------------------------------------------------------------------------

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DiscoveredMCPTool
# ---------------------------------------------------------------------------


class DiscoveredMCPTool(BaseTool):
    """A tool discovered from an MCP server, wrapped as a BaseTool.

    The tool name is prefixed with ``mcp_{server_alias}_`` to avoid
    collisions with built-in tools or tools from other MCP servers.
    """

    def __init__(
        self,
        server_alias: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict,
        client_session: Any,  # mcp.ClientSession
    ) -> None:
        self._server_alias = server_alias
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._tool_schema = tool_schema
        self._session = client_session

    # -- BaseTool interface -------------------------------------------------

    @property
    def name(self) -> str:
        """Prefixed tool name: ``mcp_{alias}_{tool}``."""
        return f"mcp_{self._server_alias}_{self._tool_name}"

    @property
    def description(self) -> str:
        return self._tool_description

    @property
    def parameter_schema(self) -> dict:
        return self._tool_schema

    async def execute(
        self,
        params: dict,
        abort_signal: asyncio.Event | None = None,
    ) -> ToolResult:
        """Call the remote MCP tool and return its result."""
        try:
            result = await self._session.call_tool(
                self._tool_name, arguments=params
            )
        except Exception as exc:
            logger.error(
                "MCP tool %s/%s failed: %s",
                self._server_alias,
                self._tool_name,
                exc,
            )
            return ToolResult(
                llm_content=f"Error calling MCP tool: {exc}",
                display_content=f"[red]MCP tool error:[/red] {exc}",
                is_error=True,
            )

        # Parse result — MCP tool results have a `content` list of
        # content blocks. Concatenate text blocks for the LLM.
        text_parts: list[str] = []
        for block in getattr(result, "content", []):
            if hasattr(block, "text"):
                text_parts.append(block.text)
            else:
                # Non-text content (images, etc.) — include a placeholder.
                text_parts.append(f"[{getattr(block, 'type', 'unknown')} content]")

        content = "\n".join(text_parts) if text_parts else str(result)
        is_error = getattr(result, "isError", False)

        return ToolResult(
            llm_content=content,
            display_content=content,
            is_error=is_error,
        )


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """Connects to MCP servers and discovers their tools.

    Each configured server is launched as a subprocess (stdio transport).
    The client discovers available tools via ``list_tools()`` and wraps
    them as ``DiscoveredMCPTool`` instances.
    """

    def __init__(self, config: Config) -> None:
        self._servers = config.mcp_servers
        self._sessions: dict[str, Any] = {}
        # Track context managers so we can clean up on disconnect.
        self._cleanup_callbacks: list[Any] = []

    async def connect_all(self) -> list[DiscoveredMCPTool]:
        """Connect to all configured MCP servers and discover tools.

        For each server in ``config.mcp_servers``:
        1. Launch the server process using its command + args.
        2. Create an MCP ``ClientSession``.
        3. Call ``list_tools()`` to discover available tools.
        4. Wrap each tool as ``DiscoveredMCPTool``.

        Returns:
            All discovered tools across all servers.
        """
        if not MCP_AVAILABLE:
            if self._servers:
                logger.warning(
                    "MCP servers are configured but the 'mcp' package is not "
                    "installed. Install it with: pip install mcp"
                )
            return []

        if not self._servers:
            return []

        all_tools: list[DiscoveredMCPTool] = []
        for alias, server_config in self._servers.items():
            try:
                tools = await self.connect_server(alias, server_config)
                all_tools.extend(tools)
                logger.info(
                    "MCP server '%s': discovered %d tool(s)",
                    alias,
                    len(tools),
                )
            except Exception as exc:
                logger.error(
                    "Failed to connect to MCP server '%s': %s", alias, exc
                )

        return all_tools

    async def connect_server(
        self, alias: str, server_config: MCPServerConfig
    ) -> list[DiscoveredMCPTool]:
        """Connect to a single MCP server and return its discovered tools.

        Args:
            alias: Server alias used as the tool name prefix.
            server_config: Command, args, and env for the server.

        Returns:
            List of ``DiscoveredMCPTool`` instances from this server.
        """
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "MCP SDK is not installed. Install with: pip install mcp"
            )

        server_params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env if server_config.env else None,
        )

        # Launch the server process via stdio transport.
        # stdio_client is an async context manager that yields
        # (read_stream, write_stream). We need to keep it alive
        # for the lifetime of the session.
        transport_cm = stdio_client(server_params)
        transport = await transport_cm.__aenter__()
        self._cleanup_callbacks.append(transport_cm)

        read_stream, write_stream = transport

        # Create and initialise the client session.
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        self._cleanup_callbacks.append(session_cm)

        await session.initialize()

        self._sessions[alias] = session

        # Discover tools.
        tools_response = await session.list_tools()
        discovered: list[DiscoveredMCPTool] = []

        for tool_info in tools_response.tools:
            schema = (
                tool_info.inputSchema
                if hasattr(tool_info, "inputSchema")
                else {}
            )
            discovered.append(
                DiscoveredMCPTool(
                    server_alias=alias,
                    tool_name=tool_info.name,
                    tool_description=getattr(
                        tool_info, "description", ""
                    )
                    or "",
                    tool_schema=schema,
                    client_session=session,
                )
            )

        return discovered

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and clean up resources."""
        errors: list[str] = []

        # Clean up in reverse order (sessions first, then transports).
        for cm in reversed(self._cleanup_callbacks):
            try:
                await cm.__aexit__(None, None, None)
            except Exception as exc:
                errors.append(str(exc))

        self._cleanup_callbacks.clear()
        self._sessions.clear()

        if errors:
            logger.warning(
                "Errors during MCP disconnect: %s", "; ".join(errors)
            )
        else:
            logger.debug("All MCP servers disconnected")

    async def _call_tool(
        self, server_alias: str, tool_name: str, arguments: dict
    ) -> str:
        """Call a tool on an MCP server directly.

        Args:
            server_alias: The server alias to target.
            tool_name: The tool name (without prefix).
            arguments: Tool arguments.

        Returns:
            The tool result as a string.

        Raises:
            KeyError: If the server alias is not connected.
        """
        session = self._sessions.get(server_alias)
        if session is None:
            raise KeyError(
                f"MCP server '{server_alias}' is not connected"
            )

        result = await session.call_tool(tool_name, arguments=arguments)

        # Concatenate text content blocks.
        text_parts: list[str] = []
        for block in getattr(result, "content", []):
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "\n".join(text_parts) if text_parts else str(result)
