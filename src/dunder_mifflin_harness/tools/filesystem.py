"""Filesystem helpers for attachments, edit previews, and agent tools."""

from __future__ import annotations

import mimetypes
import os
import re as _re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path

from dunder_mifflin_harness.tools.types import (
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


# ---------------------------------------------------------------------------
# CLI attachment helpers — used by cli/parser.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileAttachment:
    """Structured file content attached to a user message."""

    path: Path
    display_path: str
    content: str
    size: int
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class AttachmentWarning:
    """A recoverable problem while loading a user-requested attachment."""

    reference: str
    message: str


def resolve_user_path(reference: str, cwd: Path) -> Path:
    """Resolve a user-provided path against the session cwd."""
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def load_file_attachment(
    reference: str,
    cwd: Path,
    max_bytes: int,
) -> FileAttachment | AttachmentWarning:
    """Load a text file attachment or return a visible warning."""
    path = resolve_user_path(reference, cwd)
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        stat = os.fstat(fd)
    except FileNotFoundError:
        return AttachmentWarning(reference, f"file not found: {reference}")
    except PermissionError:
        return AttachmentWarning(reference, f"permission denied: {reference}")
    except OSError as exc:
        return AttachmentWarning(reference, f"could not open {reference}: {exc}")

    try:
        if stat_module.S_ISDIR(stat.st_mode):
            return AttachmentWarning(reference, f"directories are not attachable yet: {reference}")

        if not stat_module.S_ISREG(stat.st_mode):
            return AttachmentWarning(reference, f"not a regular file: {reference}")

        if stat.st_size > max_bytes:
            return AttachmentWarning(
                reference,
                f"file is too large to attach ({stat.st_size} bytes): {reference}",
            )

        try:
            data = b""
            while len(data) <= max_bytes:
                chunk = os.read(fd, max_bytes + 1 - len(data))
                if not chunk:
                    break
                data += chunk
        except PermissionError:
            return AttachmentWarning(reference, f"permission denied: {reference}")
        except OSError as exc:
            return AttachmentWarning(reference, f"could not read {reference}: {exc}")
    finally:
        if fd is not None:
            os.close(fd)

    if len(data) > max_bytes:
        return AttachmentWarning(
            reference,
            f"file is too large to attach (over {max_bytes} bytes): {reference}",
        )

    if b"\x00" in data:
        return AttachmentWarning(reference, f"binary file cannot be attached: {reference}")

    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return AttachmentWarning(
            reference,
            f"file is not valid UTF-8 text: {reference}",
        )

    mime_type, _encoding = mimetypes.guess_type(path.name)
    try:
        display_path = str(path.relative_to(cwd))
    except ValueError:
        display_path = str(path)

    return FileAttachment(
        path=path,
        display_path=display_path,
        content=content,
        size=stat.st_size,
        mime_type=mime_type,
    )


def format_attachments_for_prompt(attachments: list[FileAttachment]) -> str:
    """Render structured attachments for the current plain LLM backend."""
    if not attachments:
        return ""

    parts: list[str] = ["\n\nAttached files:"]
    for attachment in attachments:
        parts.append(
            "\n"
            f"--- {attachment.display_path} ({attachment.size} bytes) ---\n"
            f"{attachment.content}\n"
            f"--- End {attachment.display_path} ---"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent tools — callable with `.approval: ToolApprovalMeta` set via setattr
# ---------------------------------------------------------------------------

# Directories that add noise without value for agent exploration.
_DEFAULT_IGNORE: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ty_cache",
    "node_modules", ".venv", "venv", "env",
    "dist", "build", ".build",
    ".DS_Store",
})


def _preview(s: str, max_len: int = 40) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def _collect_dir_entries(
    path: Path,
    current_depth: int,
    max_depth: int,
    ignore: frozenset[str],
    show_hidden: bool,
) -> list[DirEntry]:
    """Recursively collect directory entries up to max_depth levels deep."""
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
        entries.append(DirEntry(name=child.name, path=str(child), is_dir=is_dir, size=size))
        if is_dir and current_depth < max_depth - 1:
            entries.extend(_collect_dir_entries(child, current_depth + 1, max_depth, ignore, show_hidden))
    return entries


async def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> FileReadResult:
    """Read a text file. start_line and end_line are 1-indexed and inclusive."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FileReadResult(status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path)
    except PermissionError:
        return FileReadResult(status=ToolStatus.PERMISSION_DENIED, error=f"permission denied: {path}", path=path)
    except OSError as exc:
        return FileReadResult(status=ToolStatus.ERROR, error=str(exc), path=path)

    lines = raw.splitlines(keepends=True)
    total = len(lines)
    start_idx = max(0, start_line - 1)
    # end_line is 1-indexed inclusive; Python slice end_line excludes index end_line-1 → use end_line directly
    sliced = lines[start_idx:end_line]
    truncated = len(sliced) < total
    return FileReadResult(
        path=path,
        content="".join(sliced),
        lines_total=total,
        lines_returned=len(sliced),
        truncated=truncated,
        warnings=["reading partial file; use start_line/end_line to navigate"] if truncated else [],
    )


setattr(read_file, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda path, **_: f"Read `{path}`",
))


async def write_file(path: str, content: str) -> FileWriteResult:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    created = not p.exists()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        return FileWriteResult(status=ToolStatus.PERMISSION_DENIED, error=f"permission denied: {path}", path=path)
    except OSError as exc:
        return FileWriteResult(status=ToolStatus.ERROR, error=str(exc), path=path)
    return FileWriteResult(path=path, bytes_written=len(content.encode("utf-8")), created=created)


setattr(write_file, "approval", ToolApprovalMeta(
    category="file_write",
    risk_level=RiskLevel.LOW,
    reversible=True,
    description_fn=lambda path, content="", **_: f"Write {len(content):,} bytes → `{path}`",
))


async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> FileEditResult:
    """Replace an exact string in a file. Fails if old_string matches more than once and replace_all is False."""
    if old_string == "":
        return FileEditResult(status=ToolStatus.ERROR, error="old_string must not be empty", path=path)

    p = Path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FileEditResult(status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path)
    except PermissionError:
        return FileEditResult(status=ToolStatus.PERMISSION_DENIED, error=f"permission denied: {path}", path=path)
    except OSError as exc:
        return FileEditResult(status=ToolStatus.ERROR, error=str(exc), path=path)

    count = content.count(old_string)
    if count == 0:
        return FileEditResult(status=ToolStatus.ERROR, error=f"old_string not found in `{path}`", path=path)
    if count > 1 and not replace_all:
        return FileEditResult(
            status=ToolStatus.ERROR,
            error=f"old_string matches {count} times in `{path}`; set replace_all=True to replace all",
            path=path,
        )

    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
    made = count if replace_all else 1
    try:
        p.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        return FileEditResult(status=ToolStatus.ERROR, error=str(exc), path=path)
    return FileEditResult(path=path, replacements_made=made)


setattr(edit_file, "approval", ToolApprovalMeta(
    category="file_write",
    risk_level=RiskLevel.LOW,
    reversible=True,
    description_fn=lambda path, old_string="", **_: f"Edit `{path}` — replace `{_preview(old_string)}`",
))


async def list_directory(
    path: str,
    ignore: list[str] | None = None,
    max_depth: int = 1,
    show_hidden: bool = False,
) -> DirectoryListResult:
    """List directory entries. max_depth=1 is flat; increase for project-level overviews.
    Common noisy directories (.git, __pycache__, node_modules, etc.) are ignored by default.
    """
    p = Path(path)
    if not p.exists():
        return DirectoryListResult(status=ToolStatus.NOT_FOUND, error=f"not found: {path}", path=path)
    if not p.is_dir():
        return DirectoryListResult(status=ToolStatus.ERROR, error=f"not a directory: {path}", path=path)

    effective_ignore = _DEFAULT_IGNORE | frozenset(ignore or [])
    entries = _collect_dir_entries(p, 0, max_depth, effective_ignore, show_hidden)
    return DirectoryListResult(path=path, entries=entries)


setattr(list_directory, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda path, max_depth=1, **_: f"List `{path}`" + (f" (depth {max_depth})" if max_depth > 1 else ""),
))


async def search_files(root: str, pattern: str) -> SearchResult:
    """Find files under root matching a glob pattern (recursive)."""
    p = Path(root)
    if not p.exists():
        return SearchResult(status=ToolStatus.NOT_FOUND, error=f"root not found: {root}", root=root, pattern=pattern)
    matches = [str(m) for m in sorted(p.rglob(pattern))]
    return SearchResult(root=root, pattern=pattern, matches=matches)


setattr(search_files, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda root, pattern="", **_: f"Search `{root}` for files matching `{pattern}`",
))


async def grep_files(
    root: str,
    pattern: str,
    context_lines: int = 0,
    max_matches: int = 100,
    recursive: bool = True,
    include: str | None = None,
) -> GrepResult:
    """Search file contents under root using a regex. Returns matches with optional surrounding context."""
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
                after = file_lines[lineno : min(len(file_lines), lineno + context_lines)]
                matches.append(GrepMatch(
                    file=str(filepath),
                    line_number=lineno,
                    line=line,
                    context_before=before,
                    context_after=after,
                ))

    warnings: list[str] = []
    if total_matches > max_matches:
        warnings.append(
            f"results capped at {max_matches}; {total_matches} total matches found — refine your pattern"
        )

    return GrepResult(
        matches=matches,
        searched_files=searched,
        total_matches=total_matches,
        warnings=warnings,
    )


setattr(grep_files, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda root, pattern="", **_: f"Search file contents in `{root}` for `{pattern}`",
))
