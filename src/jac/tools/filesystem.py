"""Filesystem agent tools — `read_file`, `write_file`, `edit_file`,
`list_directory`, `search_files`, `grep_files`.

Every tool returns a typed `ToolResult` subtype and never raises. Each
function carries a `.approval: ToolApprovalMeta` attribute the factory's
approval wrapper reads at call time.

`compute_edit` / `apply_edit` / `preview_write` are split out so the
wrapper can compute and surface a unified diff before the gate fires; the
wrapper then calls `apply_edit` on approval without re-reading the file.
"""

from __future__ import annotations

import difflib as _difflib
import re as _re
from dataclasses import dataclass
from pathlib import Path

from jac.tools.types import (
    DirectoryListResult,
    DirEntry,
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    GrepMatch,
    GrepResult,
    RiskLevel,
    SearchResult,
    ToolApprovalMeta,
    ToolStatus,
)


# Directories that add noise without value for agent exploration.
_DEFAULT_IGNORE: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ty_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        ".build",
        ".DS_Store",
    }
)


def _preview(s: str, max_len: int = 40) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def _collect_dir_entries(
    path: Path,
    current_depth: int,
    max_depth: int,
    ignore: frozenset[str],
    show_hidden: bool,
) -> list[DirEntry]:
    entries: list[DirEntry] = []
    try:
        children = sorted(path.iterdir())
    except PermissionError:
        return entries

    for child in children:
        if not show_hidden and child.name.startswith("."):
            continue
        if child.name in ignore:
            continue
        is_dir = child.is_dir()
        try:
            size = child.stat().st_size if not is_dir else 0
        except OSError:
            size = 0
        entries.append(
            DirEntry(name=child.name, path=str(child), is_dir=is_dir, size=size)
        )
        if is_dir and current_depth < max_depth - 1:
            entries.extend(
                _collect_dir_entries(
                    child, current_depth + 1, max_depth, ignore, show_hidden
                )
            )
    return entries


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def read_file(
    path: str, start_line: int = 1, end_line: int | None = None
) -> FileReadResult:
    """Read a text file. start_line and end_line are 1-indexed and inclusive."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FileReadResult(
            status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path
        )
    except PermissionError:
        return FileReadResult(
            status=ToolStatus.PERMISSION_DENIED,
            error=f"permission denied: {path}",
            path=path,
        )
    except OSError as exc:
        return FileReadResult(status=ToolStatus.ERROR, error=str(exc), path=path)

    lines = raw.splitlines(keepends=True)
    total = len(lines)
    start_idx = max(0, start_line - 1)
    sliced = lines[start_idx:end_line]
    truncated = len(sliced) < total
    return FileReadResult(
        path=path,
        content="".join(sliced),
        lines_total=total,
        lines_returned=len(sliced),
        truncated=truncated,
        warnings=["reading partial file; use start_line/end_line to navigate"]
        if truncated
        else [],
    )


setattr(
    read_file,
    "approval",
    ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda path, **_: f"Read `{path}`",
        timeout_seconds=30.0,
    ),
)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def preview_write(path: str, content: str) -> str:
    """Build a unified diff for a prospective write."""
    p = Path(path)
    try:
        before = p.read_text(encoding="utf-8") if p.exists() else ""
    except OSError:
        before = ""
    return "".join(
        _difflib.unified_diff(
            before.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{p.name}" if before else "/dev/null",
            tofile=f"b/{p.name}",
        )
    )


async def write_file(path: str, content: str) -> FileWriteResult:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    created = not p.exists()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        return FileWriteResult(
            status=ToolStatus.PERMISSION_DENIED,
            error=f"permission denied: {path}",
            path=path,
        )
    except OSError as exc:
        return FileWriteResult(status=ToolStatus.ERROR, error=str(exc), path=path)
    return FileWriteResult(
        path=path, bytes_written=len(content.encode("utf-8")), created=created
    )


setattr(
    write_file,
    "approval",
    ToolApprovalMeta(
        category="file_write",
        risk_level=RiskLevel.LOW,
        reversible=True,
        description_fn=lambda path, content="", **_: (
            f"Write {len(content):,} bytes → `{path}`"
        ),
    ),
)


# ---------------------------------------------------------------------------
# edit_file (compute_edit + apply_edit split out for the approval wrapper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedEdit:
    """An edit computed but not yet applied."""

    path: str
    new_content: str
    diff: str
    replacements_made: int


def compute_edit(
    path: str, old_string: str, new_string: str, replace_all: bool = False
) -> PreparedEdit | FileEditResult:
    """Compute the prospective edit without writing.

    Returns a `PreparedEdit` on success, or a `FileEditResult` carrying the
    error if the file is missing / unreadable / the match is ambiguous.
    """
    p = Path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FileEditResult(
            status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path
        )
    except PermissionError:
        return FileEditResult(
            status=ToolStatus.PERMISSION_DENIED,
            error=f"permission denied: {path}",
            path=path,
        )
    except OSError as exc:
        return FileEditResult(status=ToolStatus.ERROR, error=str(exc), path=path)

    count = content.count(old_string)
    if count == 0:
        return FileEditResult(
            status=ToolStatus.ERROR,
            error=f"old_string not found in `{path}`",
            path=path,
        )
    if count > 1 and not replace_all:
        return FileEditResult(
            status=ToolStatus.ERROR,
            error=(
                f"old_string matches {count} times in `{path}`; "
                "set replace_all=True to replace all"
            ),
            path=path,
        )

    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    made = count if replace_all else 1
    diff = "".join(
        _difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{p.name}",
            tofile=f"b/{p.name}",
        )
    )
    return PreparedEdit(
        path=path, new_content=new_content, diff=diff, replacements_made=made
    )


def apply_edit(prepared: PreparedEdit) -> FileEditResult:
    """Apply a previously-computed edit, writing the new content to disk."""
    p = Path(prepared.path)
    try:
        p.write_text(prepared.new_content, encoding="utf-8")
    except OSError as exc:
        return FileEditResult(
            status=ToolStatus.ERROR, error=str(exc), path=prepared.path
        )
    return FileEditResult(
        path=prepared.path,
        replacements_made=prepared.replacements_made,
        diff=prepared.diff,
    )


async def edit_file(
    path: str, old_string: str, new_string: str, replace_all: bool = False
) -> FileEditResult:
    """Replace an exact string in a file.

    Fails if `old_string` matches more than once and `replace_all` is False.
    """
    prepared = compute_edit(path, old_string, new_string, replace_all)
    if isinstance(prepared, FileEditResult):
        return prepared
    return apply_edit(prepared)


setattr(
    edit_file,
    "approval",
    ToolApprovalMeta(
        category="file_write",
        risk_level=RiskLevel.LOW,
        reversible=True,
        description_fn=lambda path, old_string="", **_: (
            f"Edit `{path}` — replace `{_preview(old_string)}`"
        ),
    ),
)


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


async def list_directory(
    path: str,
    ignore: list[str] | None = None,
    max_depth: int = 1,
    show_hidden: bool = False,
) -> DirectoryListResult:
    """List directory entries.

    `max_depth=1` is flat; increase for project-level overviews. Common
    noisy directories (.git, __pycache__, node_modules, etc.) are ignored
    by default.
    """
    p = Path(path)
    if not p.exists():
        return DirectoryListResult(
            status=ToolStatus.NOT_FOUND, error=f"not found: {path}", path=path
        )
    if not p.is_dir():
        return DirectoryListResult(
            status=ToolStatus.ERROR, error=f"not a directory: {path}", path=path
        )

    effective_ignore = _DEFAULT_IGNORE | frozenset(ignore or [])
    entries = _collect_dir_entries(p, 0, max_depth, effective_ignore, show_hidden)
    return DirectoryListResult(path=path, entries=entries)


setattr(
    list_directory,
    "approval",
    ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda path, max_depth=1, **_: (
            f"List `{path}`" + (f" (depth {max_depth})" if max_depth > 1 else "")
        ),
    ),
)


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


async def search_files(root: str, pattern: str) -> SearchResult:
    """Find files under root matching a glob pattern (recursive)."""
    p = Path(root)
    if not p.exists():
        return SearchResult(
            status=ToolStatus.NOT_FOUND,
            error=f"root not found: {root}",
            root=root,
            pattern=pattern,
        )
    matches = [str(m) for m in sorted(p.rglob(pattern))]
    return SearchResult(root=root, pattern=pattern, matches=matches)


setattr(
    search_files,
    "approval",
    ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda root, pattern="", **_: (
            f"Search `{root}` for files matching `{pattern}`"
        ),
    ),
)


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------


async def grep_files(
    root: str,
    pattern: str,
    context_lines: int = 0,
    max_matches: int = 100,
    recursive: bool = True,
    include: str | None = None,
) -> GrepResult:
    """Search file contents under root using a regex."""
    try:
        rx = _re.compile(pattern)
    except _re.error as exc:
        return GrepResult(status=ToolStatus.ERROR, error=f"invalid regex: {exc}")

    p = Path(root)
    if not p.exists():
        return GrepResult(status=ToolStatus.NOT_FOUND, error=f"root not found: {root}")

    glob_fn = p.rglob if recursive else p.glob
    file_glob = include or "*"
    matches: list[GrepMatch] = []
    searched = 0
    total_matches = 0

    for filepath in sorted(glob_fn(file_glob)):
        if not filepath.is_file():
            continue
        searched += 1
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_lines = text.splitlines()
        for lineno, line in enumerate(file_lines, 1):
            if not rx.search(line):
                continue
            total_matches += 1
            if len(matches) < max_matches:
                before = file_lines[max(0, lineno - 1 - context_lines) : lineno - 1]
                after = file_lines[
                    lineno : min(len(file_lines), lineno + context_lines)
                ]
                matches.append(
                    GrepMatch(
                        file=str(filepath),
                        line_number=lineno,
                        line=line,
                        context_before=before,
                        context_after=after,
                    )
                )

    warnings: list[str] = []
    if total_matches > max_matches:
        warnings.append(
            f"results capped at {max_matches}; {total_matches} total matches — "
            "refine your pattern"
        )

    return GrepResult(
        matches=matches,
        searched_files=searched,
        total_matches=total_matches,
        warnings=warnings,
    )


setattr(
    grep_files,
    "approval",
    ToolApprovalMeta(
        category="file_read",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda root, pattern="", **_: (
            f"Search file contents in `{root}` for `{pattern}`"
        ),
    ),
)
