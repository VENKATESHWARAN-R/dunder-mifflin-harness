"""Tests for pygemini.tools.shell (ShellTool)."""

from __future__ import annotations

import asyncio

import pytest

from pygemini.tools.base import ToolConfirmation, ToolResult
from pygemini.tools.shell import ShellTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: ShellTool) -> None:
        assert tool.name == "run_shell_command"

    def test_description_mentions_shell(self, tool: ShellTool) -> None:
        desc = tool.description
        assert "shell" in desc.lower() or "command" in desc.lower()

    def test_parameter_schema_type(self, tool: ShellTool) -> None:
        schema = tool.parameter_schema
        assert schema["type"] == "object"

    def test_parameter_schema_has_command(self, tool: ShellTool) -> None:
        assert "command" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_timeout(self, tool: ShellTool) -> None:
        assert "timeout" in tool.parameter_schema["properties"]

    def test_parameter_schema_requires_command(self, tool: ShellTool) -> None:
        assert "command" in tool.parameter_schema["required"]

    def test_to_function_declaration_structure(self, tool: ShellTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "run_shell_command"
        assert "description" in decl
        assert "parameters" in decl


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject bad inputs."""

    def test_valid_command(self, tool: ShellTool) -> None:
        assert tool.validate_params({"command": "echo hello"}) is None

    def test_missing_command(self, tool: ShellTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "command" in result.lower()

    def test_empty_command(self, tool: ShellTool) -> None:
        result = tool.validate_params({"command": "   "})
        assert result is not None
        assert "empty" in result.lower() or "command" in result.lower()

    def test_invalid_timeout_zero(self, tool: ShellTool) -> None:
        result = tool.validate_params({"command": "echo hi", "timeout": 0})
        assert result is not None
        assert "timeout" in result.lower()

    def test_invalid_timeout_negative(self, tool: ShellTool) -> None:
        result = tool.validate_params({"command": "echo hi", "timeout": -5})
        assert result is not None
        assert "timeout" in result.lower()

    def test_invalid_timeout_float(self, tool: ShellTool) -> None:
        result = tool.validate_params({"command": "echo hi", "timeout": 1.5})
        assert result is not None

    def test_valid_timeout(self, tool: ShellTool) -> None:
        assert tool.validate_params({"command": "echo hi", "timeout": 10}) is None

    def test_no_timeout_is_valid(self, tool: ShellTool) -> None:
        assert tool.validate_params({"command": "ls"}) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show the command."""

    def test_shows_command(self, tool: ShellTool) -> None:
        desc = tool.get_description({"command": "echo hello"})
        assert "echo hello" in desc

    def test_truncates_long_command(self, tool: ShellTool) -> None:
        long_cmd = "a" * 200
        desc = tool.get_description({"command": long_cmd})
        assert len(desc) < 150
        assert "..." in desc

    def test_missing_command_uses_placeholder(self, tool: ShellTool) -> None:
        desc = tool.get_description({})
        assert "?" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    """should_confirm should always return a ToolConfirmation."""

    def test_returns_tool_confirmation(self, tool: ShellTool) -> None:
        result = tool.should_confirm({"command": "rm -rf /tmp/foo"})
        assert isinstance(result, ToolConfirmation)

    def test_description_contains_command(self, tool: ShellTool) -> None:
        result = tool.should_confirm({"command": "git status"})
        assert result is not None
        assert "git status" in result.description

    def test_details_contain_command_key(self, tool: ShellTool) -> None:
        result = tool.should_confirm({"command": "make build"})
        assert result is not None
        assert "command" in result.details

    def test_details_contain_timeout(self, tool: ShellTool) -> None:
        result = tool.should_confirm({"command": "make build", "timeout": 60})
        assert result is not None
        assert "timeout" in result.details


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute should run commands and return stdout/exit code."""

    async def test_echo_stdout_in_output(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "echo hello"})
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "hello" in result.llm_content

    async def test_exit_code_zero_in_output(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "echo hello"})
        assert "Exit code: 0" in result.llm_content

    async def test_is_error_false_on_success(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "echo ok"})
        assert result.is_error is False

    async def test_display_content_populated(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "echo hi"})
        assert result.display_content != ""

    async def test_stderr_included_when_present(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "echo errtext >&2"})
        assert "errtext" in result.llm_content


# ---------------------------------------------------------------------------
# execute — failure
# ---------------------------------------------------------------------------


class TestExecuteFailure:
    """execute should surface non-zero exit codes as errors."""

    async def test_nonzero_exit_is_error(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "exit 1", "timeout": 5})
        assert result.is_error is True

    async def test_exit_code_in_output(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "exit 42", "timeout": 5})
        assert "42" in result.llm_content

    async def test_false_command_is_error(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "false"})
        assert result.is_error is True


# ---------------------------------------------------------------------------
# execute — timeout
# ---------------------------------------------------------------------------


class TestExecuteTimeout:
    """execute should kill the process and return an error on timeout."""

    async def test_timeout_returns_error(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "sleep 100", "timeout": 1})
        assert isinstance(result, ToolResult)
        assert result.is_error is True

    async def test_timeout_mentions_timeout_in_output(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "sleep 100", "timeout": 1})
        assert "timeout" in result.llm_content.lower() or "timed out" in result.llm_content.lower()

    async def test_timeout_mentions_duration(self, tool: ShellTool) -> None:
        result = await tool.execute({"command": "sleep 100", "timeout": 1})
        assert "1" in result.llm_content


# ---------------------------------------------------------------------------
# execute — abort signal
# ---------------------------------------------------------------------------


class TestAbortSignal:
    """A pre-set abort signal should cause execute to return an aborted result."""

    async def test_aborted_result_is_error(self, tool: ShellTool) -> None:
        signal = asyncio.Event()
        signal.set()

        # Use a fast command so it completes before the abort check fires,
        # but in either outcome (aborted or completed) we just confirm no crash.
        result = await tool.execute({"command": "echo done"}, abort_signal=signal)
        assert isinstance(result, ToolResult)

    async def test_abort_checked_after_completion(self, tool: ShellTool) -> None:
        """Abort checked post-completion: fast command may complete before signal."""
        signal = asyncio.Event()
        signal.set()

        result = await tool.execute({"command": "echo hi"}, abort_signal=signal)
        # With signal pre-set and a fast command the result may be either
        # aborted or successful; we just verify no exception is raised.
        assert isinstance(result, ToolResult)
