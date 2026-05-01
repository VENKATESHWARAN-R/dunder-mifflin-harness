"""read_file tool — reads file contents with optional offset and limit."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pygemini.tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    """Read a file's contents, optionally with line offset and limit."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Returns the file content with line numbers. "
            "Use offset and limit to read specific portions of large files."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to CWD).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Default: 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Default: all lines.",
                },
            },
            "required": ["path"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "path" not in params:
            return "Missing required parameter: path"
        offset = params.get("offset", 1)
        if isinstance(offset, int) and offset < 1:
            return "offset must be >= 1"
        limit = params.get("limit")
        if limit is not None and isinstance(limit, int) and limit < 1:
            return "limit must be >= 1"
        return None

    def get_description(self, params: dict) -> str:
        path = params.get("path", "?")
        offset = params.get("offset")
        limit = params.get("limit")
        desc = f"Read {path}"
        if offset or limit:
            parts = []
            if offset:
                parts.append(f"from line {offset}")
            if limit:
                parts.append(f"limit {limit} lines")
            desc += f" ({', '.join(parts)})"
        return desc

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        path = Path(params["path"]).expanduser()
        offset = params.get("offset", 1)
        limit = params.get("limit")

        if not path.exists():
            return ToolResult(
                llm_content=f"Error: File not found: {path}",
                display_content=f"[red]File not found:[/red] {path}",
                is_error=True,
            )

        if not path.is_file():
            return ToolResult(
                llm_content=f"Error: Not a file: {path}",
                display_content=f"[red]Not a file:[/red] {path}",
                is_error=True,
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                llm_content=f"Error: Binary file cannot be read as text: {path}",
                display_content=f"[red]Binary file:[/red] {path}",
                is_error=True,
            )
        except PermissionError:
            return ToolResult(
                llm_content=f"Error: Permission denied: {path}",
                display_content=f"[red]Permission denied:[/red] {path}",
                is_error=True,
            )

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Apply offset (1-based)
        start = max(0, offset - 1)
        end = start + limit if limit else total_lines
        selected = lines[start:end]

        # Format with line numbers
        numbered_lines = []
        for i, line in enumerate(selected, start=start + 1):
            numbered_lines.append(f"{i:>6}\t{line.rstrip()}")

        result_text = "\n".join(numbered_lines)

        header = f"File: {path} ({total_lines} lines)"
        if start > 0 or end < total_lines:
            header += f" [showing lines {start + 1}-{min(end, total_lines)}]"

        llm_content = f"{header}\n\n{result_text}"

        # Display: truncated preview
        display_lines = numbered_lines[:50]
        display_text = "\n".join(display_lines)
        if len(numbered_lines) > 50:
            display_text += f"\n... ({len(numbered_lines) - 50} more lines)"

        return ToolResult(
            llm_content=llm_content,
            display_content=f"[dim]{header}[/dim]\n{display_text}",
        )
