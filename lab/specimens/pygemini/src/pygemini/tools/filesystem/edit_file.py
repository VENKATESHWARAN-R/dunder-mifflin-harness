"""edit_file tool — search-and-replace editing with diff display."""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text, appending '...' if it exceeds *max_len*."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_diff_display(diff_lines: list[str]) -> str:
    """Format unified diff lines with Rich markup for terminal display."""
    parts: list[str] = []
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            parts.append(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            parts.append(f"[cyan]{line}[/cyan]")
        elif line.startswith("-"):
            parts.append(f"[red]{line}[/red]")
        elif line.startswith("+"):
            parts.append(f"[green]{line}[/green]")
        else:
            parts.append(line)
    return "\n".join(parts)


class EditFileTool(BaseTool):
    """Edit a file by replacing an exact string match with new content."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing an exact string match with new content. "
            "The old_string must match exactly, including all whitespace, "
            "indentation, and newlines. The match must be unique within the file."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (absolute or relative to CWD).",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The exact string to find in the file. Must match exactly "
                        "including whitespace and indentation. Must appear exactly once."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string to substitute for old_string.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "path" not in params:
            return "Missing required parameter: path"
        if "old_string" not in params:
            return "Missing required parameter: old_string"
        if "new_string" not in params:
            return "Missing required parameter: new_string"
        if not params["old_string"]:
            return "old_string must not be empty"
        if params["old_string"] == params["new_string"]:
            return "old_string and new_string must be different"
        return None

    def get_description(self, params: dict) -> str:
        path = params.get("path", "?")
        old_len = len(params.get("old_string", ""))
        new_len = len(params.get("new_string", ""))
        return f"Edit {path}: replace {old_len} chars with {new_len} chars"

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        path = params.get("path", "?")
        old_string = params.get("old_string", "")
        new_string = params.get("new_string", "")

        return ToolConfirmation(
            description=f"Edit {path}",
            details={
                "path": path,
                "old_string": _truncate(old_string),
                "new_string": _truncate(new_string),
            },
        )

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        path = Path(params["path"]).expanduser()
        old_string: str = params["old_string"]
        new_string: str = params["new_string"]

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ToolResult(
                llm_content=f"Error: File not found: {path}",
                display_content=f"[red]File not found:[/red] {path}",
                is_error=True,
            )
        except PermissionError:
            return ToolResult(
                llm_content=f"Error: Permission denied reading {path}",
                display_content=f"[red]Permission denied:[/red] {path}",
                is_error=True,
            )
        except UnicodeDecodeError:
            return ToolResult(
                llm_content=f"Error: {path} appears to be a binary file",
                display_content=f"[red]Binary file:[/red] {path}",
                is_error=True,
            )

        # Count occurrences
        count = content.count(old_string)

        if count == 0:
            return ToolResult(
                llm_content=f"Error: old_string not found in {path}",
                display_content=f"[red]Not found:[/red] old_string not found in {path}",
                is_error=True,
            )

        if count > 1:
            return ToolResult(
                llm_content=(
                    f"Error: old_string found {count} times in {path} — "
                    "must be unique. Provide more context to make it unique."
                ),
                display_content=(
                    f"[red]Ambiguous match:[/red] old_string found {count} times in {path}"
                ),
                is_error=True,
            )

        # Exactly one match — perform the replacement
        new_content = content.replace(old_string, new_string, 1)

        try:
            path.write_text(new_content, encoding="utf-8")
        except PermissionError:
            return ToolResult(
                llm_content=f"Error: Permission denied writing to {path}",
                display_content=f"[red]Permission denied:[/red] {path}",
                is_error=True,
            )
        except OSError as e:
            return ToolResult(
                llm_content=f"Error writing to {path}: {e}",
                display_content=f"[red]Write error:[/red] {e}",
                is_error=True,
            )

        # Generate unified diff
        old_lines = content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=str(path),
                tofile=str(path),
            )
        )
        diff_text = "".join(diff_lines)

        old_len = len(old_string)
        new_len = len(new_string)
        llm_content = (
            f"Edited {path}: replaced {old_len} chars with {new_len} chars\n{diff_text}"
        )
        display_content = _format_diff_display(
            [line.rstrip("\n") for line in diff_lines]
        )

        return ToolResult(
            llm_content=llm_content,
            display_content=display_content,
        )
