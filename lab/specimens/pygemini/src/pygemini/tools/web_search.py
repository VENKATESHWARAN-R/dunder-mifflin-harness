"""web_search tool — searches the web via Google using Gemini grounding."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pygemini.tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Search the web using Google via Gemini's google_search_retrieval grounding.

    Unlike a grounding *config* attached to every model call, this tool is
    invoked explicitly by the model.  It issues a dedicated ``generate_content``
    call with grounding enabled and parses the resulting grounding metadata to
    surface titles, snippets, and URLs back to the model.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "google_web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Google. Returns search results with titles, "
            "snippets, and URLs."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
            },
            "required": ["query"],
        }

    def validate_params(self, params: dict) -> str | None:
        query = params.get("query")
        if not query:
            return "Missing required parameter: query"
        if not isinstance(query, str) or not query.strip():
            return "Parameter 'query' must be a non-empty string"
        return None

    def get_description(self, params: dict) -> str:
        query = params.get("query", "?")
        return f"Search: {query}"

    def should_confirm(self, params: dict) -> None:
        return None

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        query: str = params["query"].strip()

        api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return ToolResult(
                llm_content="Error: No Gemini API key available for web search.",
                display_content=(
                    "[red]Web search failed:[/red] GEMINI_API_KEY is not set. "
                    "Set the environment variable or pass api_key to WebSearchTool."
                ),
                is_error=True,
            )

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._search_sync, query, api_key
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                llm_content=(
                    f"Error: Web search failed: {exc}\n\n"
                    "Tip: use the web_fetch tool to retrieve a specific URL directly."
                ),
                display_content=(
                    f"[red]Web search failed:[/red] {exc}\n"
                    "[dim]Tip: use [bold]web_fetch[/bold] to retrieve a specific URL.[/dim]"
                ),
                is_error=True,
            )

        if not results:
            return ToolResult(
                llm_content=f"No search results found for '{query}'.",
                display_content=(
                    f"[yellow]No results found[/yellow] for [bold]{query!r}[/bold]."
                ),
            )

        llm_content = _format_results_llm(query, results)
        display_content = (
            f"[green]Found {len(results)} result{'s' if len(results) != 1 else ''}[/green]"
            f" for [bold]{query!r}[/bold]"
        )
        return ToolResult(llm_content=llm_content, display_content=display_content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_sync(self, query: str, api_key: str) -> list[dict[str, str]]:
        """Synchronous call to Gemini with google_search_retrieval grounding.

        Returns a list of dicts, each with keys ``title``, ``snippet``, ``url``.
        Raises on hard errors; returns an empty list when grounding metadata is
        absent (e.g. the model simply had no results).
        """
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]

        client = genai.Client(api_key=api_key)

        grounding_tool = types.Tool(
            google_search_retrieval=types.GoogleSearchRetrieval()
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=types.GenerateContentConfig(tools=[grounding_tool]),
        )

        return _parse_grounding_metadata(response)


# ---------------------------------------------------------------------------
# Grounding metadata parser
# ---------------------------------------------------------------------------

def _parse_grounding_metadata(response: Any) -> list[dict[str, str]]:
    """Extract search results from a grounded Gemini response.

    The google-genai SDK exposes grounding metadata on
    ``response.candidates[0].grounding_metadata``.  The structure contains a
    ``grounding_chunks`` list (each with a ``web`` sub-object holding ``uri``
    and ``title``) and a ``search_entry_point`` with rendered snippets.

    We do a best-effort parse that degrades gracefully when fields are absent.
    """
    results: list[dict[str, str]] = []

    try:
        candidates = response.candidates or []
        if not candidates:
            return results

        grounding_meta = getattr(candidates[0], "grounding_metadata", None)
        if grounding_meta is None:
            return results

        # grounding_chunks: list of chunks, each may have a .web attribute
        chunks = getattr(grounding_meta, "grounding_chunks", None) or []

        # grounding_supports: provides snippet text aligned to chunks
        supports = getattr(grounding_meta, "grounding_supports", None) or []

        # Build a mapping from chunk index → snippet text (first support wins)
        snippet_by_index: dict[int, str] = {}
        for support in supports:
            indices = getattr(support, "grounding_chunk_indices", None) or []
            segment = getattr(support, "segment", None)
            text = getattr(segment, "text", "") if segment else ""
            for idx in indices:
                if idx not in snippet_by_index and text:
                    snippet_by_index[idx] = text

        for i, chunk in enumerate(chunks):
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            uri = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or ""
            snippet = snippet_by_index.get(i, "")
            if uri:
                results.append({"title": title, "snippet": snippet, "url": uri})

        # Deduplicate by URL while preserving order
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)
        return unique

    except Exception:  # noqa: BLE001
        # Parsing errors should not crash the tool; return whatever we have
        return results


def _format_results_llm(query: str, results: list[dict[str, str]]) -> str:
    """Format search results as plain text for the LLM."""
    lines: list[str] = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet']}")
        lines.append(f"   URL: {r['url']}")
        lines.append("")
    return "\n".join(lines).rstrip()
