"""Tests for pygemini.mcp.client (MCPClient, DiscoveredMCPTool)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pygemini.core.config import Config, MCPServerConfig
from pygemini.mcp.client import DiscoveredMCPTool, MCPClient
from pygemini.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(servers: dict[str, MCPServerConfig] | None = None) -> MCPClient:
    """Build an MCPClient from a minimal Config."""
    config = Config(mcp_servers=servers or {})
    return MCPClient(config)


def _fake_tool(
    server_alias: str = "test_server",
    tool_name: str = "my_tool",
    description: str = "Does something",
    schema: dict | None = None,
) -> DiscoveredMCPTool:
    """Create a DiscoveredMCPTool with a mock session."""
    session = MagicMock()
    session.call_tool = AsyncMock()
    return DiscoveredMCPTool(
        server_alias=server_alias,
        tool_name=tool_name,
        tool_description=description,
        tool_schema=schema or {"type": "object", "properties": {}},
        client_session=session,
    )


# ---------------------------------------------------------------------------
# TestDiscoveredMCPTool
# ---------------------------------------------------------------------------


class TestDiscoveredMCPTool:
    """DiscoveredMCPTool should satisfy the BaseTool interface."""

    def test_name_prefixed_with_mcp_and_alias(self) -> None:
        tool = _fake_tool(server_alias="myserver", tool_name="list_files")
        assert tool.name == "mcp_myserver_list_files"

    def test_name_prefix_format(self) -> None:
        tool = _fake_tool(server_alias="fs", tool_name="read")
        assert tool.name.startswith("mcp_fs_")

    def test_description_returned(self) -> None:
        tool = _fake_tool(description="List all files in a directory")
        assert tool.description == "List all files in a directory"

    def test_empty_description_returned(self) -> None:
        tool = _fake_tool(description="")
        assert tool.description == ""

    def test_schema_returned(self) -> None:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        tool = _fake_tool(schema=schema)
        assert tool.parameter_schema == schema

    def test_empty_schema_returned(self) -> None:
        """Schema passed directly is stored and returned unchanged."""
        session = MagicMock()
        tool = DiscoveredMCPTool(
            server_alias="s",
            tool_name="t",
            tool_description="",
            tool_schema={},
            client_session=session,
        )
        assert tool.parameter_schema == {}

    async def test_execute_calls_session_call_tool(self) -> None:
        tool = _fake_tool(tool_name="greet")
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False
        tool._session.call_tool = AsyncMock(return_value=mock_result)

        result = await tool.execute({"name": "world"})

        tool._session.call_tool.assert_called_once_with(
            "greet", arguments={"name": "world"}
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False

    async def test_execute_concatenates_text_content(self) -> None:
        tool = _fake_tool()
        block1 = MagicMock()
        block1.text = "Hello"
        block2 = MagicMock()
        block2.text = " World"
        mock_result = MagicMock()
        mock_result.content = [block1, block2]
        mock_result.isError = False
        tool._session.call_tool = AsyncMock(return_value=mock_result)

        result = await tool.execute({})
        assert "Hello" in result.llm_content
        assert "World" in result.llm_content

    async def test_execute_handles_non_text_block(self) -> None:
        """Non-text content blocks should produce a placeholder, not crash."""
        tool = _fake_tool()
        block = MagicMock(spec=[])  # no .text attribute
        block.type = "image"
        mock_result = MagicMock()
        mock_result.content = [block]
        mock_result.isError = False
        tool._session.call_tool = AsyncMock(return_value=mock_result)

        result = await tool.execute({})
        assert isinstance(result, ToolResult)

    async def test_execute_returns_error_on_session_exception(self) -> None:
        tool = _fake_tool()
        tool._session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))

        result = await tool.execute({})
        assert result.is_error is True
        assert "connection lost" in result.llm_content

    async def test_execute_propagates_iserror_flag(self) -> None:
        tool = _fake_tool()
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = True
        tool._session.call_tool = AsyncMock(return_value=mock_result)

        result = await tool.execute({})
        assert result.is_error is True


# ---------------------------------------------------------------------------
# TestMCPClient — no servers / MCP unavailable
# ---------------------------------------------------------------------------


class TestMCPClient:
    """MCPClient.connect_all() handles edge cases gracefully."""

    async def test_no_servers_returns_empty_list(self) -> None:
        client = _make_client(servers={})
        tools = await client.connect_all()
        assert tools == []

    async def test_mcp_unavailable_with_no_servers_returns_empty(self) -> None:
        """If MCP is not installed and no servers configured → empty list, no warning."""
        client = _make_client(servers={})
        with patch("pygemini.mcp.client.MCP_AVAILABLE", False):
            tools = await client.connect_all()
        assert tools == []

    async def test_mcp_unavailable_with_servers_returns_empty_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If MCP is not installed but servers are configured → warn + empty list."""
        servers = {"myserver": MCPServerConfig(command="npx", args=["-y", "some-server"])}
        client = _make_client(servers=servers)
        with patch("pygemini.mcp.client.MCP_AVAILABLE", False):
            import logging
            with caplog.at_level(logging.WARNING, logger="pygemini.mcp.client"):
                tools = await client.connect_all()
        assert tools == []
        assert any("mcp" in r.message.lower() or "install" in r.message.lower() for r in caplog.records)

    async def test_connect_all_aggregates_tools_from_multiple_servers(self) -> None:
        """connect_all() should call connect_server for each configured server."""
        servers = {
            "server_a": MCPServerConfig(command="cmd_a"),
            "server_b": MCPServerConfig(command="cmd_b"),
        }
        client = _make_client(servers=servers)

        tool_a = _fake_tool(server_alias="server_a", tool_name="tool_a")
        tool_b = _fake_tool(server_alias="server_b", tool_name="tool_b")

        async def fake_connect_server(alias: str, server_config):
            if alias == "server_a":
                return [tool_a]
            return [tool_b]

        with patch("pygemini.mcp.client.MCP_AVAILABLE", True):
            client.connect_server = fake_connect_server  # type: ignore[method-assign]
            tools = await client.connect_all()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "mcp_server_a_tool_a" in names
        assert "mcp_server_b_tool_b" in names

    async def test_connect_all_skips_failed_server(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """connect_all() should log an error and continue if one server fails."""
        servers = {
            "good": MCPServerConfig(command="good_cmd"),
            "bad": MCPServerConfig(command="bad_cmd"),
        }
        client = _make_client(servers=servers)

        good_tool = _fake_tool(server_alias="good", tool_name="ok_tool")

        async def fake_connect_server(alias: str, server_config):
            if alias == "bad":
                raise RuntimeError("Server failed to start")
            return [good_tool]

        with patch("pygemini.mcp.client.MCP_AVAILABLE", True):
            client.connect_server = fake_connect_server  # type: ignore[method-assign]
            import logging
            with caplog.at_level(logging.ERROR, logger="pygemini.mcp.client"):
                tools = await client.connect_all()

        assert len(tools) == 1
        assert tools[0].name == "mcp_good_ok_tool"

    async def test_disconnect_all_cleans_up_callbacks(self) -> None:
        """disconnect_all() should call __aexit__ on all cleanup callbacks."""
        client = _make_client()

        cm1 = AsyncMock()
        cm2 = AsyncMock()
        client._cleanup_callbacks = [cm1, cm2]
        client._sessions = {"s1": MagicMock()}

        await client.disconnect_all()

        cm1.__aexit__.assert_called_once()
        cm2.__aexit__.assert_called_once()
        assert client._cleanup_callbacks == []
        assert client._sessions == {}

    async def test_disconnect_all_handles_errors_gracefully(self) -> None:
        """Errors during disconnect should be logged, not raised."""
        client = _make_client()

        cm = AsyncMock()
        cm.__aexit__ = AsyncMock(side_effect=RuntimeError("cleanup error"))
        client._cleanup_callbacks = [cm]

        # Should not raise
        await client.disconnect_all()
        assert client._cleanup_callbacks == []
