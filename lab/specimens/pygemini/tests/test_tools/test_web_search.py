"""Tests for pygemini.tools.web_search (WebSearchTool)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.web_search import WebSearchTool, _format_results_llm, _parse_grounding_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_no_key() -> WebSearchTool:
    """WebSearchTool with no API key and GEMINI_API_KEY unset."""
    return WebSearchTool(api_key=None)


@pytest.fixture
def tool_with_key() -> WebSearchTool:
    return WebSearchTool(api_key="test-api-key-123")


def _make_grounding_response(chunks: list[dict]) -> MagicMock:
    """Build a mock Gemini response with grounding metadata.

    Each chunk dict should have keys: uri, title, snippet (optional).
    """
    response = MagicMock()

    candidate = MagicMock()

    grounding_meta = MagicMock()
    mock_chunks = []
    mock_supports = []

    for i, chunk_data in enumerate(chunks):
        web = MagicMock()
        web.uri = chunk_data.get("uri", "")
        web.title = chunk_data.get("title", "")

        chunk = MagicMock()
        chunk.web = web
        mock_chunks.append(chunk)

        snippet = chunk_data.get("snippet", "")
        if snippet:
            support = MagicMock()
            support.grounding_chunk_indices = [i]
            segment = MagicMock()
            segment.text = snippet
            support.segment = segment
            mock_supports.append(support)

    grounding_meta.grounding_chunks = mock_chunks
    grounding_meta.grounding_supports = mock_supports

    candidate.grounding_metadata = grounding_meta
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool_with_key: WebSearchTool) -> None:
        assert tool_with_key.name == "google_web_search"

    def test_description_mentions_search(self, tool_with_key: WebSearchTool) -> None:
        desc = tool_with_key.description
        assert "search" in desc.lower()

    def test_parameter_schema_type(self, tool_with_key: WebSearchTool) -> None:
        assert tool_with_key.parameter_schema["type"] == "object"

    def test_parameter_schema_has_query(self, tool_with_key: WebSearchTool) -> None:
        assert "query" in tool_with_key.parameter_schema["properties"]

    def test_parameter_schema_requires_query(self, tool_with_key: WebSearchTool) -> None:
        assert "query" in tool_with_key.parameter_schema["required"]

    def test_to_function_declaration_structure(self, tool_with_key: WebSearchTool) -> None:
        decl = tool_with_key.to_function_declaration()
        assert decl["name"] == "google_web_search"
        assert "description" in decl
        assert "parameters" in decl


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject missing or empty queries."""

    def test_valid_query(self, tool_with_key: WebSearchTool) -> None:
        assert tool_with_key.validate_params({"query": "python asyncio"}) is None

    def test_missing_query(self, tool_with_key: WebSearchTool) -> None:
        result = tool_with_key.validate_params({})
        assert result is not None
        assert "query" in result.lower()

    def test_empty_query(self, tool_with_key: WebSearchTool) -> None:
        result = tool_with_key.validate_params({"query": "   "})
        assert result is not None

    def test_none_query(self, tool_with_key: WebSearchTool) -> None:
        result = tool_with_key.validate_params({"query": None})
        assert result is not None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show the query."""

    def test_shows_query(self, tool_with_key: WebSearchTool) -> None:
        desc = tool_with_key.get_description({"query": "best python libraries"})
        assert "best python libraries" in desc

    def test_missing_query_uses_placeholder(self, tool_with_key: WebSearchTool) -> None:
        desc = tool_with_key.get_description({})
        assert "?" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    """should_confirm should return None (no confirmation needed)."""

    def test_returns_none(self, tool_with_key: WebSearchTool) -> None:
        assert tool_with_key.should_confirm({"query": "anything"}) is None


# ---------------------------------------------------------------------------
# execute — no API key
# ---------------------------------------------------------------------------


class TestExecuteNoApiKey:
    """execute should return an error when no API key is available."""

    async def test_no_key_returns_error(
        self, tool_no_key: WebSearchTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = await tool_no_key.execute({"query": "test"})
        assert isinstance(result, ToolResult)
        assert result.is_error

    async def test_no_key_error_message_mentions_api_key(
        self, tool_no_key: WebSearchTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = await tool_no_key.execute({"query": "test"})
        assert "api" in result.llm_content.lower() or "key" in result.llm_content.lower()

    async def test_env_var_key_is_used(
        self, tool_no_key: WebSearchTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GEMINI_API_KEY is set, the tool should attempt the search."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-xyz")

        with patch.object(
            tool_no_key, "_search_sync", return_value=[{"title": "Example", "snippet": "A snippet", "url": "https://example.com"}]
        ):
            result = await tool_no_key.execute({"query": "python"})

        assert not result.is_error


# ---------------------------------------------------------------------------
# execute — success (mocked _search_sync)
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute should parse results and return formatted content."""

    async def test_results_included_in_llm_content(
        self, tool_with_key: WebSearchTool
    ) -> None:
        fake_results = [
            {"title": "Python Docs", "snippet": "Official docs.", "url": "https://docs.python.org"},
            {"title": "Real Python", "snippet": "Tutorials and more.", "url": "https://realpython.com"},
        ]
        with patch.object(tool_with_key, "_search_sync", return_value=fake_results):
            result = await tool_with_key.execute({"query": "python tutorial"})

        assert not result.is_error
        assert "Python Docs" in result.llm_content
        assert "https://docs.python.org" in result.llm_content

    async def test_query_present_in_llm_content(
        self, tool_with_key: WebSearchTool
    ) -> None:
        fake_results = [
            {"title": "Result", "snippet": "Snippet", "url": "https://example.com"}
        ]
        with patch.object(tool_with_key, "_search_sync", return_value=fake_results):
            result = await tool_with_key.execute({"query": "asyncio tutorial"})

        assert "asyncio tutorial" in result.llm_content

    async def test_no_results_returns_non_error(
        self, tool_with_key: WebSearchTool
    ) -> None:
        with patch.object(tool_with_key, "_search_sync", return_value=[]):
            result = await tool_with_key.execute({"query": "xyzzy nothing found"})

        assert not result.is_error
        assert "no" in result.llm_content.lower() or "found" in result.llm_content.lower()

    async def test_search_exception_returns_error(
        self, tool_with_key: WebSearchTool
    ) -> None:
        with patch.object(
            tool_with_key, "_search_sync", side_effect=RuntimeError("network error")
        ):
            result = await tool_with_key.execute({"query": "test"})

        assert result.is_error
        assert "failed" in result.llm_content.lower() or "error" in result.llm_content.lower()

    async def test_display_content_shows_result_count(
        self, tool_with_key: WebSearchTool
    ) -> None:
        fake_results = [
            {"title": "A", "snippet": "s1", "url": "https://a.com"},
            {"title": "B", "snippet": "s2", "url": "https://b.com"},
        ]
        with patch.object(tool_with_key, "_search_sync", return_value=fake_results):
            result = await tool_with_key.execute({"query": "test"})

        assert "2" in result.display_content


# ---------------------------------------------------------------------------
# _parse_grounding_metadata (unit tests for the helper)
# ---------------------------------------------------------------------------


class TestParseGroundingMetadata:
    """_parse_grounding_metadata should extract results gracefully."""

    def test_parses_chunks_with_snippets(self) -> None:
        response = _make_grounding_response(
            [
                {"uri": "https://a.com", "title": "A", "snippet": "Snippet A"},
                {"uri": "https://b.com", "title": "B", "snippet": "Snippet B"},
            ]
        )
        results = _parse_grounding_metadata(response)
        assert len(results) == 2
        assert results[0]["url"] == "https://a.com"
        assert results[0]["title"] == "A"
        assert results[0]["snippet"] == "Snippet A"

    def test_deduplicates_by_url(self) -> None:
        response = _make_grounding_response(
            [
                {"uri": "https://a.com", "title": "A"},
                {"uri": "https://a.com", "title": "A duplicate"},
            ]
        )
        results = _parse_grounding_metadata(response)
        assert len(results) == 1

    def test_no_candidates_returns_empty(self) -> None:
        response = MagicMock()
        response.candidates = []
        assert _parse_grounding_metadata(response) == []

    def test_no_grounding_metadata_returns_empty(self) -> None:
        response = MagicMock()
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response.candidates = [candidate]
        assert _parse_grounding_metadata(response) == []

    def test_chunk_without_web_skipped(self) -> None:
        response = MagicMock()
        candidate = MagicMock()
        grounding_meta = MagicMock()

        chunk = MagicMock()
        chunk.web = None
        grounding_meta.grounding_chunks = [chunk]
        grounding_meta.grounding_supports = []

        candidate.grounding_metadata = grounding_meta
        response.candidates = [candidate]

        results = _parse_grounding_metadata(response)
        assert results == []

    def test_exception_returns_empty(self) -> None:
        """Parsing a broken response should not raise; return []."""
        response = MagicMock()
        response.candidates = MagicMock(side_effect=AttributeError("boom"))
        # Should not raise
        results = _parse_grounding_metadata(response)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _format_results_llm (unit tests)
# ---------------------------------------------------------------------------


class TestFormatResultsLlm:
    def test_includes_title_and_url(self) -> None:
        results = [{"title": "My Title", "snippet": "", "url": "https://x.com"}]
        text = _format_results_llm("test query", results)
        assert "My Title" in text
        assert "https://x.com" in text

    def test_includes_snippet_when_present(self) -> None:
        results = [{"title": "T", "snippet": "Some relevant snippet", "url": "https://x.com"}]
        text = _format_results_llm("q", results)
        assert "Some relevant snippet" in text

    def test_includes_query_header(self) -> None:
        text = _format_results_llm("python tips", [])
        assert "python tips" in text

    def test_numbered_entries(self) -> None:
        results = [
            {"title": "First", "snippet": "", "url": "https://a.com"},
            {"title": "Second", "snippet": "", "url": "https://b.com"},
        ]
        text = _format_results_llm("q", results)
        assert "1." in text
        assert "2." in text
