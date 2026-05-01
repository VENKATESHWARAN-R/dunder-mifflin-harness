"""Tests for pygemini.tools.ask_user (AskUserTool)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.ask_user import AskUserTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_emitter(confirmation_response: bool = True) -> MagicMock:
    """Return a mock EventEmitter whose request_confirmation returns the given bool."""
    emitter = MagicMock()
    emitter.request_confirmation = AsyncMock(return_value=confirmation_response)
    return emitter


@pytest.fixture
def tool_approved() -> AskUserTool:
    """AskUserTool with an emitter that always confirms."""
    return AskUserTool(event_emitter=_make_event_emitter(True))


@pytest.fixture
def tool_declined() -> AskUserTool:
    """AskUserTool with an emitter that always declines."""
    return AskUserTool(event_emitter=_make_event_emitter(False))


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool_approved: AskUserTool) -> None:
        assert tool_approved.name == "ask_user"

    def test_description_mentions_question(self, tool_approved: AskUserTool) -> None:
        desc = tool_approved.description
        assert "question" in desc.lower() or "ask" in desc.lower()

    def test_parameter_schema_type(self, tool_approved: AskUserTool) -> None:
        assert tool_approved.parameter_schema["type"] == "object"

    def test_parameter_schema_has_question(self, tool_approved: AskUserTool) -> None:
        assert "question" in tool_approved.parameter_schema["properties"]

    def test_parameter_schema_requires_question(self, tool_approved: AskUserTool) -> None:
        assert "question" in tool_approved.parameter_schema["required"]

    def test_to_function_declaration_structure(self, tool_approved: AskUserTool) -> None:
        decl = tool_approved.to_function_declaration()
        assert decl["name"] == "ask_user"
        assert "description" in decl
        assert "parameters" in decl


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject missing or empty questions."""

    def test_valid_question(self, tool_approved: AskUserTool) -> None:
        assert tool_approved.validate_params({"question": "What is your name?"}) is None

    def test_missing_question(self, tool_approved: AskUserTool) -> None:
        result = tool_approved.validate_params({})
        assert result is not None
        assert "question" in result.lower()

    def test_empty_question_whitespace(self, tool_approved: AskUserTool) -> None:
        result = tool_approved.validate_params({"question": "   "})
        assert result is not None

    def test_empty_question_string(self, tool_approved: AskUserTool) -> None:
        result = tool_approved.validate_params({"question": ""})
        assert result is not None

    def test_non_string_question(self, tool_approved: AskUserTool) -> None:
        result = tool_approved.validate_params({"question": 42})
        assert result is not None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show the question, truncated at 100 chars."""

    def test_shows_question(self, tool_approved: AskUserTool) -> None:
        desc = tool_approved.get_description({"question": "What is your name?"})
        assert "What is your name?" in desc

    def test_truncates_long_question(self, tool_approved: AskUserTool) -> None:
        long_q = "A" * 200
        desc = tool_approved.get_description({"question": long_q})
        # The whole description should be shorter than 200 + prefix chars
        assert "..." in desc
        assert len(desc) < 220

    def test_short_question_not_truncated(self, tool_approved: AskUserTool) -> None:
        short_q = "Short?"
        desc = tool_approved.get_description({"question": short_q})
        assert "..." not in desc
        assert short_q in desc

    def test_missing_question_uses_placeholder(self, tool_approved: AskUserTool) -> None:
        desc = tool_approved.get_description({})
        assert "?" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    def test_returns_none(self, tool_approved: AskUserTool) -> None:
        assert tool_approved.should_confirm({"question": "anything"}) is None


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    """execute should call event_emitter.request_confirmation and return result."""

    async def test_user_approved_returns_confirmed_response(
        self, tool_approved: AskUserTool
    ) -> None:
        result = await tool_approved.execute({"question": "Proceed?"})
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "confirmed" in result.llm_content.lower()

    async def test_user_declined_returns_declined_response(
        self, tool_declined: AskUserTool
    ) -> None:
        result = await tool_declined.execute({"question": "Proceed?"})
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "declined" in result.llm_content.lower()

    async def test_request_confirmation_called_with_question(
        self, tool_approved: AskUserTool
    ) -> None:
        question = "What is the meaning of life?"
        await tool_approved.execute({"question": question})
        tool_approved._event_emitter.request_confirmation.assert_called_once()
        call_kwargs = tool_approved._event_emitter.request_confirmation.call_args
        # description should contain the question
        assert question in (call_kwargs.kwargs.get("description") or call_kwargs.args[0])

    async def test_display_content_shows_question(
        self, tool_approved: AskUserTool
    ) -> None:
        result = await tool_approved.execute({"question": "Are you sure?"})
        assert "Are you sure?" in result.display_content

    async def test_display_content_shows_response(
        self, tool_approved: AskUserTool
    ) -> None:
        result = await tool_approved.execute({"question": "Continue?"})
        assert "confirmed" in result.display_content.lower()

    async def test_declined_display_content_shows_declined(
        self, tool_declined: AskUserTool
    ) -> None:
        result = await tool_declined.execute({"question": "Delete everything?"})
        assert "declined" in result.display_content.lower()
