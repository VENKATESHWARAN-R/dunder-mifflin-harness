"""Tests for pygemini.tools.memory (SaveMemoryTool)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.memory import SaveMemoryTool
from pygemini.context.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock(spec=MemoryStore)
    store.save = MagicMock()
    return store


@pytest.fixture
def tool(mock_store: MagicMock) -> SaveMemoryTool:
    return SaveMemoryTool(memory_store=mock_store)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: SaveMemoryTool) -> None:
        assert tool.name == "save_memory"

    def test_description_mentions_memory(self, tool: SaveMemoryTool) -> None:
        desc = tool.description
        assert "memory" in desc.lower() or "save" in desc.lower()

    def test_parameter_schema_type(self, tool: SaveMemoryTool) -> None:
        assert tool.parameter_schema["type"] == "object"

    def test_parameter_schema_has_content(self, tool: SaveMemoryTool) -> None:
        assert "content" in tool.parameter_schema["properties"]

    def test_parameter_schema_requires_content(self, tool: SaveMemoryTool) -> None:
        assert "content" in tool.parameter_schema["required"]

    def test_to_function_declaration_structure(self, tool: SaveMemoryTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "save_memory"
        assert "description" in decl
        assert "parameters" in decl


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject missing or empty content."""

    def test_valid_content(self, tool: SaveMemoryTool) -> None:
        assert tool.validate_params({"content": "User prefers dark mode."}) is None

    def test_missing_content(self, tool: SaveMemoryTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "content" in result.lower()

    def test_empty_content_whitespace(self, tool: SaveMemoryTool) -> None:
        result = tool.validate_params({"content": "   "})
        assert result is not None

    def test_empty_content_string(self, tool: SaveMemoryTool) -> None:
        result = tool.validate_params({"content": ""})
        assert result is not None

    def test_non_string_content(self, tool: SaveMemoryTool) -> None:
        result = tool.validate_params({"content": 123})
        assert result is not None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show the content, truncated at 80 - prefix chars."""

    def test_shows_content(self, tool: SaveMemoryTool) -> None:
        desc = tool.get_description({"content": "User likes Python."})
        assert "User likes Python." in desc

    def test_truncates_long_content(self, tool: SaveMemoryTool) -> None:
        long_content = "X" * 200
        desc = tool.get_description({"content": long_content})
        assert len(desc) <= 85  # "Save memory: " prefix (13) + up to 80 - 13 content

    def test_short_content_not_truncated(self, tool: SaveMemoryTool) -> None:
        short = "short fact"
        desc = tool.get_description({"content": short})
        assert short in desc

    def test_missing_content_graceful(self, tool: SaveMemoryTool) -> None:
        desc = tool.get_description({})
        assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    def test_returns_none(self, tool: SaveMemoryTool) -> None:
        assert tool.should_confirm({"content": "some fact"}) is None


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    """execute should call memory_store.save and return confirmation."""

    async def test_calls_memory_store_save(
        self, tool: SaveMemoryTool, mock_store: MagicMock
    ) -> None:
        await tool.execute({"content": "User's name is Alice."})
        mock_store.save.assert_called_once_with("User's name is Alice.")

    async def test_returns_tool_result(
        self, tool: SaveMemoryTool
    ) -> None:
        result = await tool.execute({"content": "Fact to remember."})
        assert isinstance(result, ToolResult)

    async def test_result_not_error(
        self, tool: SaveMemoryTool
    ) -> None:
        result = await tool.execute({"content": "Fact to remember."})
        assert not result.is_error

    async def test_llm_content_mentions_content(
        self, tool: SaveMemoryTool
    ) -> None:
        result = await tool.execute({"content": "User prefers dark mode."})
        assert "User prefers dark mode." in result.llm_content

    async def test_display_content_indicates_saved(
        self, tool: SaveMemoryTool
    ) -> None:
        result = await tool.execute({"content": "Test fact."})
        assert "saved" in result.display_content.lower() or "memory" in result.display_content.lower()

    async def test_save_called_with_exact_content(
        self, tool: SaveMemoryTool, mock_store: MagicMock
    ) -> None:
        content = "The project uses Python 3.12."
        await tool.execute({"content": content})
        args, kwargs = mock_store.save.call_args
        assert args[0] == content

    async def test_save_called_once_per_execute(
        self, tool: SaveMemoryTool, mock_store: MagicMock
    ) -> None:
        await tool.execute({"content": "First fact."})
        await tool.execute({"content": "Second fact."})
        assert mock_store.save.call_count == 2
