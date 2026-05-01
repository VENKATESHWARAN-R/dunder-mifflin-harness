"""write_file tool — writes content to a file with confirmation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult


class WriteFileTool(BaseTool):
    """Write content to a file, creating parent directories if needed."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Creates parent directories as needed."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (absolute or relative to CWD).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
                "create_directories": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist. Default: true.",
                },
            },
            "required": ["path", "content"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "path" not in params:
            return "Missing required parameter: path"
        if "content" not in params:
            return "Missing required parameter: content"
        return None

    def get_description(self, params: dict) -> str:
        path = params.get("path", "?")
        content = params.get("content", "")
        line_count = content.count("\n") + (1 if content else 0)
        return f"Write to {path} ({line_count} lines)"

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        path = params.get("path", "?")
        content = params.get("content", "")
        line_count = content.count("\n") + (1 if content else 0)

        # Show a preview (first 10 lines)
        preview_lines = content.split("\n")[:10]
        preview = "\n".join(preview_lines)
        if len(content.split("\n")) > 10:
            preview += "\n..."

        return ToolConfirmation(
            description=f"Write {line_count} lines to {path}",
            details={
                "path": path,
                "lines": line_count,
                "bytes": len(content.encode("utf-8")),
                "preview": preview,
            },
        )

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        path = Path(params["path"]).expanduser()
        content = params["content"]
        create_dirs = params.get("create_directories", True)

        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            existed = path.exists()
            path.write_text(content, encoding="utf-8")

            byte_count = len(content.encode("utf-8"))
            line_count = content.count("\n") + (1 if content else 0)
            action = "Updated" if existed else "Created"

            llm_content = f"{action} {path} ({line_count} lines, {byte_count} bytes)"

            return ToolResult(
                llm_content=llm_content,
                display_content=f"[green]{action}:[/green] {path} ({line_count} lines)",
            )
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
