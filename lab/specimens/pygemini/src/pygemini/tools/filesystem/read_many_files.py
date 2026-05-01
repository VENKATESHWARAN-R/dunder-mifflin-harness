"""read_many_files tool — reads multiple files at once and returns all contents."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pygemini.tools.base import BaseTool, ToolResult


class ReadManyFilesTool(BaseTool):
    """Read multiple files at once, returns contents of all files."""

    @property
    def name(self) -> str:
        return "read_many_files"

    @property
    def description(self) -> str:
        return (
            "Read multiple files at once. Returns the contents of all requested files "
            "with line numbers and clear file headers. If a file cannot be read (not "
            "found, binary, permission denied), its error is included in the output "
            "and reading continues for the remaining files."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read (absolute or relative to CWD).",
                },
            },
            "required": ["paths"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "paths" not in params:
            return "Missing required parameter: paths"
        paths = params["paths"]
        if not isinstance(paths, list) or len(paths) == 0:
            return "paths must be a non-empty list of file path strings"
        return None

    def get_description(self, params: dict) -> str:
        paths = params.get("paths", [])
        count = len(paths)
        if count == 0:
            return "Read 0 files"
        preview = [Path(p).name for p in paths[:3]]
        label = ", ".join(preview)
        if count > 3:
            label += f", ... (+{count - 3} more)"
        return f"Read {count} file{'s' if count != 1 else ''}: {label}"

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        paths = params["paths"]

        llm_sections: list[str] = []
        display_previews: list[str] = []
        total_lines = 0
        files_read = 0
        files_failed = 0

        for raw_path in paths:
            path = Path(raw_path).expanduser()

            # --- attempt to read the file ---
            if not path.exists():
                error_msg = f"Error: File not found: {path}"
                llm_sections.append(
                    f"--- File: {path} ---\n{error_msg}"
                )
                display_previews.append(
                    f"[red]✗[/red] [bold]{path}[/bold] — file not found"
                )
                files_failed += 1
                continue

            if not path.is_file():
                error_msg = f"Error: Not a file: {path}"
                llm_sections.append(
                    f"--- File: {path} ---\n{error_msg}"
                )
                display_previews.append(
                    f"[red]✗[/red] [bold]{path}[/bold] — not a file"
                )
                files_failed += 1
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                error_msg = f"Error: Binary file cannot be read as text: {path}"
                llm_sections.append(
                    f"--- File: {path} ---\n{error_msg}"
                )
                display_previews.append(
                    f"[red]✗[/red] [bold]{path}[/bold] — binary file"
                )
                files_failed += 1
                continue
            except PermissionError:
                error_msg = f"Error: Permission denied: {path}"
                llm_sections.append(
                    f"--- File: {path} ---\n{error_msg}"
                )
                display_previews.append(
                    f"[red]✗[/red] [bold]{path}[/bold] — permission denied"
                )
                files_failed += 1
                continue

            lines = content.splitlines(keepends=True)
            file_line_count = len(lines)
            total_lines += file_line_count

            # Format with line numbers (same style as ReadFileTool)
            numbered_lines = [
                f"{i:>6}\t{line.rstrip()}"
                for i, line in enumerate(lines, start=1)
            ]
            file_text = "\n".join(numbered_lines)

            header = f"--- File: {path} ({file_line_count} lines) ---"
            llm_sections.append(f"{header}\n{file_text}")

            # Display: show up to 10 lines per file as a truncated preview
            preview_lines = numbered_lines[:10]
            preview_text = "\n".join(preview_lines)
            if file_line_count > 10:
                preview_text += f"\n  ... ({file_line_count - 10} more lines)"
            display_previews.append(
                f"[green]✓[/green] [bold]{path}[/bold] ({file_line_count} lines)\n"
                f"[dim]{preview_text}[/dim]"
            )
            files_read += 1

        llm_content = "\n\n".join(llm_sections)

        # Build display summary line
        summary_parts: list[str] = []
        if files_read:
            summary_parts.append(
                f"[green]Read {files_read} file{'s' if files_read != 1 else ''} "
                f"({total_lines} lines total)[/green]"
            )
        if files_failed:
            summary_parts.append(
                f"[red]{files_failed} file{'s' if files_failed != 1 else ''} failed[/red]"
            )
        summary = ", ".join(summary_parts) if summary_parts else "No files processed"

        display_content = summary + "\n" + "\n\n".join(display_previews)

        return ToolResult(
            llm_content=llm_content,
            display_content=display_content,
            is_error=(files_read == 0 and files_failed > 0),
        )
