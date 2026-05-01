"""list_directory tool — lists directory contents with optional recursion."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pygemini.tools.base import BaseTool, ToolResult


class ListDirectoryTool(BaseTool):
    """List the contents of a directory, optionally recursively."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Returns entries sorted with directories "
            "first, then files, each with type and size information. Supports recursive "
            "listing and toggling visibility of hidden files (dotfiles)."
        )

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list (absolute or relative to CWD).",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list contents recursively. Default: false.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "If true, include hidden files and directories (dotfiles). Default: false.",
                },
            },
            "required": ["path"],
        }

    def validate_params(self, params: dict) -> str | None:
        if "path" not in params:
            return "Missing required parameter: path"
        return None

    def get_description(self, params: dict) -> str:
        path = params.get("path", "?")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)
        desc = f"List {path}"
        flags = []
        if recursive:
            flags.append("recursive")
        if include_hidden:
            flags.append("including hidden")
        if flags:
            desc += f" ({', '.join(flags)})"
        return desc

    def should_confirm(self, params: dict) -> None:
        return None

    async def execute(
        self, params: dict, abort_signal: asyncio.Event | None = None
    ) -> ToolResult:
        path = Path(params["path"]).expanduser()
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)

        if not path.exists():
            return ToolResult(
                llm_content=f"Error: Directory not found: {path}",
                display_content=f"[red]Directory not found:[/red] {path}",
                is_error=True,
            )

        if not path.is_dir():
            return ToolResult(
                llm_content=f"Error: Not a directory: {path}",
                display_content=f"[red]Not a directory:[/red] {path}",
                is_error=True,
            )

        try:
            entries = _collect_entries(path, recursive=recursive, include_hidden=include_hidden)
        except PermissionError:
            return ToolResult(
                llm_content=f"Error: Permission denied: {path}",
                display_content=f"[red]Permission denied:[/red] {path}",
                is_error=True,
            )

        if not entries:
            summary = f"Directory: {path} (empty)"
            return ToolResult(
                llm_content=summary,
                display_content=f"[dim]{summary}[/dim]",
            )

        if recursive:
            llm_lines, display_lines = _format_recursive(path, entries)
        else:
            llm_lines, display_lines = _format_flat(entries)

        dir_count = sum(1 for e in entries if e.is_dir())
        file_count = sum(1 for e in entries if e.is_file())
        summary = (
            f"Directory: {path} "
            f"({dir_count} director{'ies' if dir_count != 1 else 'y'}, "
            f"{file_count} file{'s' if file_count != 1 else ''})"
        )

        llm_content = f"{summary}\n\n" + "\n".join(llm_lines)
        display_content = f"[dim]{summary}[/dim]\n" + "\n".join(display_lines)

        return ToolResult(
            llm_content=llm_content,
            display_content=display_content,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_entries(
    root: Path,
    *,
    recursive: bool,
    include_hidden: bool,
) -> list[Path]:
    """Return sorted entries under *root*.

    Sort order: directories first (alphabetically), then files (alphabetically).
    Hidden entries are skipped unless *include_hidden* is True.
    """
    if recursive:
        raw = list(root.rglob("*"))
    else:
        raw = list(root.iterdir())

    if not include_hidden:
        raw = [e for e in raw if not any(part.startswith(".") for part in e.relative_to(root).parts)]

    dirs = sorted([e for e in raw if e.is_dir()], key=lambda p: p.name.lower())
    files = sorted([e for e in raw if e.is_file()], key=lambda p: p.name.lower())
    return dirs + files


def _format_flat(entries: list[Path]) -> tuple[list[str], list[str]]:
    """Format a flat (non-recursive) listing."""
    llm_lines: list[str] = []
    display_lines: list[str] = []

    for entry in entries:
        if entry.is_dir():
            llm_lines.append(f"[DIR]  {entry.name}/")
            display_lines.append(f"[green]\U0001F4C1  {entry.name}/[/green]")
        else:
            size = _human_size(entry.stat().st_size)
            llm_lines.append(f"[FILE] {entry.name}  ({size})")
            display_lines.append(f"{entry.name}  [dim]({size})[/dim]")

    return llm_lines, display_lines


def _format_recursive(root: Path, entries: list[Path]) -> tuple[list[str], list[str]]:
    """Format a recursive listing with indentation showing directory depth."""
    llm_lines: list[str] = []
    display_lines: list[str] = []

    for entry in entries:
        rel = entry.relative_to(root)
        depth = len(rel.parts) - 1
        indent = "    " * depth

        if entry.is_dir():
            llm_lines.append(f"{indent}[DIR]  {entry.name}/")
            display_lines.append(f"{indent}[green]\U0001F4C1  {entry.name}/[/green]")
        else:
            size = _human_size(entry.stat().st_size)
            llm_lines.append(f"{indent}[FILE] {entry.name}  ({size})")
            display_lines.append(f"{indent}{entry.name}  [dim]({size})[/dim]")

    return llm_lines, display_lines


def _human_size(n: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"
