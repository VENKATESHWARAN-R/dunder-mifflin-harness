"""Unit tests for pygemini.tools.base.

Covers:
- ToolResult creation and repr
- ToolConfirmation creation and repr
- BaseTool.to_function_declaration returns correct dict
- BaseTool.validate_params default returns None
- BaseTool.should_confirm default returns None
- BaseTool.get_description default returns self.description
"""

from __future__ import annotations

import asyncio

import pytest

from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------


class MockTool(BaseTool):
    """Minimal concrete implementation used across all tests."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing."

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "count": {"type": "integer", "description": "Number of items"},
            },
            "required": ["path"],
        }

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        return ToolResult(llm_content=f"executed with path={params.get('path')}")


class ShouldConfirmTool(MockTool):
    """Variant that overrides should_confirm to return a confirmation."""

    @property
    def name(self) -> str:
        return "confirm_tool"

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        return ToolConfirmation(
            description="Are you sure?",
            details={"path": params.get("path", "")},
        )


class ValidatingTool(MockTool):
    """Variant that overrides validate_params."""

    @property
    def name(self) -> str:
        return "validating_tool"

    def validate_params(self, params: dict) -> str | None:
        if "path" not in params:
            return "Missing required parameter: path"
        return None


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_basic_creation(self) -> None:
        result = ToolResult(llm_content="some content")
        assert result.llm_content == "some content"
        assert result.display_content == ""
        assert result.is_error is False

    def test_explicit_display_content(self) -> None:
        result = ToolResult(llm_content="llm", display_content="display")
        assert result.display_content == "display"

    def test_is_error_true(self) -> None:
        result = ToolResult(llm_content="error text", is_error=True)
        assert result.is_error is True

    def test_repr_short_content(self) -> None:
        result = ToolResult(llm_content="short")
        r = repr(result)
        assert "short" in r
        assert "ToolResult" in r
        assert "ERROR" not in r

    def test_repr_error_tag(self) -> None:
        result = ToolResult(llm_content="bad", is_error=True)
        assert "ERROR" in repr(result)

    def test_repr_truncates_long_content(self) -> None:
        long_text = "x" * 100
        result = ToolResult(llm_content=long_text)
        r = repr(result)
        # Repr should contain ellipsis for content longer than 60 chars
        assert "..." in r

    def test_repr_does_not_truncate_exactly_60(self) -> None:
        text = "y" * 60
        result = ToolResult(llm_content=text)
        r = repr(result)
        assert "..." not in r

    def test_repr_truncates_at_61(self) -> None:
        text = "z" * 61
        result = ToolResult(llm_content=text)
        r = repr(result)
        assert "..." in r

    def test_full_construction(self) -> None:
        result = ToolResult(
            llm_content="content",
            display_content="[bold]display[/bold]",
            is_error=False,
        )
        assert result.llm_content == "content"
        assert result.display_content == "[bold]display[/bold]"


# ---------------------------------------------------------------------------
# ToolConfirmation
# ---------------------------------------------------------------------------


class TestToolConfirmation:
    def test_basic_creation(self) -> None:
        conf = ToolConfirmation(description="delete /tmp/x")
        assert conf.description == "delete /tmp/x"
        assert conf.details == {}

    def test_with_details(self) -> None:
        conf = ToolConfirmation(
            description="run shell", details={"command": "rm -rf /"}
        )
        assert conf.details["command"] == "rm -rf /"

    def test_repr_no_details(self) -> None:
        conf = ToolConfirmation(description="do something")
        r = repr(conf)
        assert "do something" in r
        assert "ToolConfirmation" in r
        # No detail keys should appear
        assert "details=" not in r

    def test_repr_with_details(self) -> None:
        conf = ToolConfirmation(
            description="write file",
            details={"path": "/tmp/f", "content": "hello"},
        )
        r = repr(conf)
        assert "write file" in r
        assert "path" in r
        assert "content" in r

    def test_repr_empty_details_dict_treated_as_no_details(self) -> None:
        conf = ToolConfirmation(description="action", details={})
        r = repr(conf)
        # When details is empty, no bracket section should appear
        assert "[" not in r

    def test_details_default_factory(self) -> None:
        c1 = ToolConfirmation(description="a")
        c2 = ToolConfirmation(description="b")
        c1.details["key"] = "val"
        assert "key" not in c2.details


# ---------------------------------------------------------------------------
# BaseTool — to_function_declaration
# ---------------------------------------------------------------------------


class TestToFunctionDeclaration:
    def test_returns_dict(self) -> None:
        tool = MockTool()
        decl = tool.to_function_declaration()
        assert isinstance(decl, dict)

    def test_contains_name(self) -> None:
        tool = MockTool()
        assert tool.to_function_declaration()["name"] == "mock_tool"

    def test_contains_description(self) -> None:
        tool = MockTool()
        assert tool.to_function_declaration()["description"] == "A mock tool for testing."

    def test_contains_parameters(self) -> None:
        tool = MockTool()
        decl = tool.to_function_declaration()
        assert "parameters" in decl
        assert decl["parameters"] == tool.parameter_schema

    def test_exact_keys(self) -> None:
        tool = MockTool()
        decl = tool.to_function_declaration()
        assert set(decl.keys()) == {"name", "description", "parameters"}

    def test_parameters_schema_structure(self) -> None:
        tool = MockTool()
        schema = tool.to_function_declaration()["parameters"]
        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "required" in schema


# ---------------------------------------------------------------------------
# BaseTool — validate_params default
# ---------------------------------------------------------------------------


class TestValidateParams:
    def test_default_returns_none(self) -> None:
        tool = MockTool()
        assert tool.validate_params({"path": "/tmp/x"}) is None

    def test_default_with_empty_params(self) -> None:
        tool = MockTool()
        assert tool.validate_params({}) is None

    def test_override_returns_error_message(self) -> None:
        tool = ValidatingTool()
        result = tool.validate_params({})
        assert isinstance(result, str)
        assert "path" in result.lower()

    def test_override_returns_none_when_valid(self) -> None:
        tool = ValidatingTool()
        assert tool.validate_params({"path": "/tmp/x"}) is None


# ---------------------------------------------------------------------------
# BaseTool — should_confirm default
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    def test_default_returns_none(self) -> None:
        tool = MockTool()
        assert tool.should_confirm({"path": "/tmp/x"}) is None

    def test_default_with_empty_params(self) -> None:
        tool = MockTool()
        assert tool.should_confirm({}) is None

    def test_override_returns_confirmation(self) -> None:
        tool = ShouldConfirmTool()
        conf = tool.should_confirm({"path": "/tmp/danger"})
        assert isinstance(conf, ToolConfirmation)
        assert conf.description == "Are you sure?"
        assert conf.details["path"] == "/tmp/danger"

    def test_override_returns_confirmation_type(self) -> None:
        tool = ShouldConfirmTool()
        result = tool.should_confirm({})
        assert result is not None
        assert isinstance(result, ToolConfirmation)


# ---------------------------------------------------------------------------
# BaseTool — get_description default
# ---------------------------------------------------------------------------


class TestGetDescription:
    def test_default_returns_description(self) -> None:
        tool = MockTool()
        assert tool.get_description({}) == tool.description

    def test_default_ignores_params(self) -> None:
        tool = MockTool()
        assert tool.get_description({"path": "/x"}) == tool.description


# ---------------------------------------------------------------------------
# BaseTool — execute (async)
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_execute_returns_tool_result(self) -> None:
        tool = MockTool()
        result = await tool.execute({"path": "/tmp/test"})
        assert isinstance(result, ToolResult)

    async def test_execute_result_content(self) -> None:
        tool = MockTool()
        result = await tool.execute({"path": "/tmp/foo"})
        assert "path=/tmp/foo" in result.llm_content

    async def test_execute_with_abort_signal(self) -> None:
        tool = MockTool()
        signal = asyncio.Event()
        result = await tool.execute({"path": "/x"}, abort_signal=signal)
        assert isinstance(result, ToolResult)

    async def test_execute_no_abort_signal(self) -> None:
        tool = MockTool()
        result = await tool.execute({"path": "/x"}, abort_signal=None)
        assert isinstance(result, ToolResult)


# ---------------------------------------------------------------------------
# BaseTool is abstract — cannot instantiate directly
# ---------------------------------------------------------------------------


class TestAbstractness:
    def test_cannot_instantiate_base_tool(self) -> None:
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_partial_subclass_missing_execute_is_abstract(self) -> None:
        class Incomplete(BaseTool):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def description(self) -> str:
                return "desc"

            @property
            def parameter_schema(self) -> dict:
                return {}

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
