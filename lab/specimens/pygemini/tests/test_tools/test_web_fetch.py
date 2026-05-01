"""Tests for pygemini.tools.web_fetch (WebFetchTool)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pygemini.tools.base import ToolConfirmation, ToolResult
from pygemini.tools.web_fetch import WebFetchTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> WebFetchTool:
    return WebFetchTool()


def _make_mock_response(
    text: str = "<html><body><h1>Hello</h1><p>World</p></body></html>",
    content_type: str = "text/html; charset=utf-8",
    status_code: int = 200,
) -> MagicMock:
    """Build a mock httpx.Response."""
    response = MagicMock()
    response.text = text
    response.headers = {"content-type": content_type}
    response.status_code = status_code
    response.raise_for_status = MagicMock()  # no-op for 2xx
    return response


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: WebFetchTool) -> None:
        assert tool.name == "web_fetch"

    def test_description_mentions_url(self, tool: WebFetchTool) -> None:
        desc = tool.description
        assert "url" in desc.lower() or "fetch" in desc.lower()

    def test_parameter_schema_type(self, tool: WebFetchTool) -> None:
        assert tool.parameter_schema["type"] == "object"

    def test_parameter_schema_has_url(self, tool: WebFetchTool) -> None:
        assert "url" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_max_length(self, tool: WebFetchTool) -> None:
        assert "max_length" in tool.parameter_schema["properties"]

    def test_parameter_schema_requires_url(self, tool: WebFetchTool) -> None:
        assert "url" in tool.parameter_schema["required"]

    def test_to_function_declaration_structure(self, tool: WebFetchTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "web_fetch"
        assert "description" in decl
        assert "parameters" in decl


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject bad inputs."""

    def test_valid_https_url(self, tool: WebFetchTool) -> None:
        assert tool.validate_params({"url": "https://example.com"}) is None

    def test_valid_http_url(self, tool: WebFetchTool) -> None:
        assert tool.validate_params({"url": "http://example.com"}) is None

    def test_missing_url(self, tool: WebFetchTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "url" in result.lower()

    def test_url_without_scheme(self, tool: WebFetchTool) -> None:
        result = tool.validate_params({"url": "example.com"})
        assert result is not None
        assert "http" in result.lower()

    def test_url_with_ftp_scheme(self, tool: WebFetchTool) -> None:
        result = tool.validate_params({"url": "ftp://example.com"})
        assert result is not None

    def test_invalid_max_length_zero(self, tool: WebFetchTool) -> None:
        result = tool.validate_params({"url": "https://x.com", "max_length": 0})
        assert result is not None
        assert "max_length" in result.lower()

    def test_invalid_max_length_negative(self, tool: WebFetchTool) -> None:
        result = tool.validate_params({"url": "https://x.com", "max_length": -1})
        assert result is not None

    def test_valid_max_length(self, tool: WebFetchTool) -> None:
        assert tool.validate_params({"url": "https://x.com", "max_length": 1000}) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show the url."""

    def test_shows_url(self, tool: WebFetchTool) -> None:
        desc = tool.get_description({"url": "https://example.com"})
        assert "https://example.com" in desc

    def test_missing_url_uses_placeholder(self, tool: WebFetchTool) -> None:
        desc = tool.get_description({})
        assert "?" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    """should_confirm should return a ToolConfirmation with the url."""

    def test_returns_tool_confirmation(self, tool: WebFetchTool) -> None:
        result = tool.should_confirm({"url": "https://example.com"})
        assert isinstance(result, ToolConfirmation)

    def test_description_contains_url(self, tool: WebFetchTool) -> None:
        result = tool.should_confirm({"url": "https://example.com"})
        assert result is not None
        assert "https://example.com" in result.description

    def test_details_contain_url_key(self, tool: WebFetchTool) -> None:
        result = tool.should_confirm({"url": "https://example.com"})
        assert result is not None
        assert "url" in result.details


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute should fetch content and convert HTML to text."""

    async def test_html_response_converts_to_text(self, tool: WebFetchTool) -> None:
        mock_response = _make_mock_response(
            text="<html><body><h1>Hello World</h1></body></html>",
            content_type="text/html",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "Hello World" in result.llm_content

    async def test_llm_content_contains_url(self, tool: WebFetchTool) -> None:
        mock_response = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert "https://example.com" in result.llm_content

    async def test_plain_text_response_returned_as_is(self, tool: WebFetchTool) -> None:
        plain_text = "Just plain text content here."
        mock_response = _make_mock_response(
            text=plain_text,
            content_type="text/plain",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com/file.txt"})

        assert not result.is_error
        assert plain_text in result.llm_content

    async def test_max_length_truncates_content(self, tool: WebFetchTool) -> None:
        long_text = "x" * 10000
        mock_response = _make_mock_response(
            text=long_text,
            content_type="text/plain",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com", "max_length": 100})

        assert not result.is_error
        # The returned text (minus the url prefix) should be at most 100 chars
        content_after_header = result.llm_content.split("\n\n", 1)[-1]
        assert len(content_after_header) <= 100

    async def test_display_content_populated(self, tool: WebFetchTool) -> None:
        mock_response = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert result.display_content != ""


# ---------------------------------------------------------------------------
# execute — errors
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    """execute should return error ToolResults for various failure modes."""

    async def test_timeout_error(self, tool: WebFetchTool) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert result.is_error
        assert "timeout" in result.llm_content.lower() or "timed out" in result.llm_content.lower()

    async def test_connection_error(self, tool: WebFetchTool) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert result.is_error
        assert "connect" in result.llm_content.lower()

    async def test_http_404_error(self, tool: WebFetchTool) -> None:
        import httpx

        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        http_error = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_response = _make_mock_response()
        mock_response.raise_for_status = MagicMock(side_effect=http_error)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com/missing"})

        assert result.is_error
        assert "404" in result.llm_content

    async def test_http_500_error(self, tool: WebFetchTool) -> None:
        import httpx

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        http_error = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_response = _make_mock_response()
        mock_response.raise_for_status = MagicMock(side_effect=http_error)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"url": "https://example.com"})

        assert result.is_error
        assert "500" in result.llm_content


# ---------------------------------------------------------------------------
# execute — abort signal
# ---------------------------------------------------------------------------


class TestAbortSignal:
    """abort_signal is accepted without crashing (not currently checked mid-fetch)."""

    async def test_execute_with_abort_signal_set_does_not_crash(
        self, tool: WebFetchTool
    ) -> None:
        signal = asyncio.Event()
        signal.set()

        mock_response = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(
                {"url": "https://example.com"}, abort_signal=signal
            )

        assert isinstance(result, ToolResult)
