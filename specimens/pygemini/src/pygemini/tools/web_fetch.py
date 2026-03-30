"""web_fetch tool — fetch a URL and convert HTML to readable text."""

from __future__ import annotations

import asyncio

import html2text
import httpx

from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult

_USER_AGENT = (
    "Mozilla/5.0 (compatible; PyGeminiCLI/1.0; +https://github.com/pygemini)"
)


class WebFetchTool(BaseTool):
    """Fetch content from a URL and convert HTML to readable text."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL and convert HTML to readable text. "
            "Returns the page content as plain text, truncated to max_length characters."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must begin with http:// or https://).",
                },
                "max_length": {
                    "type": "integer",
                    "description": (
                        "Maximum number of characters to return. Defaults to 50000."
                    ),
                },
            },
            "required": ["url"],
        }

    def validate_params(self, params: dict) -> str | None:
        url = params.get("url")
        if not url:
            return "Missing required parameter: url"
        if not isinstance(url, str) or not (
            url.startswith("http://") or url.startswith("https://")
        ):
            return "Parameter 'url' must start with http:// or https://"
        max_length = params.get("max_length")
        if max_length is not None:
            if not isinstance(max_length, int) or max_length <= 0:
                return "Parameter 'max_length' must be a positive integer"
        return None

    def get_description(self, params: dict) -> str:
        url = params.get("url", "?")
        return f"Fetch {url}"

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        url = params.get("url", "?")
        return ToolConfirmation(
            description=f"Fetch {url}",
            details={"url": url},
        )

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        url: str = params["url"]
        max_length: int = params.get("max_length", 50000)

        timeout = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=60.0)
        headers = {"User-Agent": _USER_AGENT}

        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=headers
            ) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if "text/html" in content_type:
                converter = html2text.HTML2Text()
                converter.ignore_links = False
                converter.ignore_images = True
                converter.body_width = 0
                text = converter.handle(response.text)
            else:
                # Plain text or other text/* types — use raw body
                text = response.text

            if len(text) > max_length:
                text = text[:max_length]

            llm_content = f"Content from {url}:\n\n{text}"
            display_content = (
                f"[green]Fetched[/green] {url} ({len(text):,} chars)"
            )
            return ToolResult(
                llm_content=llm_content,
                display_content=display_content,
            )

        except httpx.TimeoutException:
            return ToolResult(
                llm_content=f"Error: Request timed out fetching {url}",
                display_content=f"[red]Timeout:[/red] {url}",
                is_error=True,
            )
        except httpx.ConnectError as exc:
            return ToolResult(
                llm_content=f"Error: Could not connect to {url}: {exc}",
                display_content=f"[red]Connection error:[/red] {url}",
                is_error=True,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return ToolResult(
                llm_content=f"Error: HTTP {status} fetching {url}",
                display_content=f"[red]HTTP {status}:[/red] {url}",
                is_error=True,
            )
