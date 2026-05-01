"""Parsing for interactive CLI input."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from jac.tools.filesystem import (
    AttachmentWarning,
    FileAttachment,
    load_file_attachment,
)

_FILE_REF_PATTERN = re.compile(r"(?<![A-Za-z0-9])@(?:\"([^\"]+)\"|(\S+))")


class ParsedInputKind(StrEnum):
    """Kinds of user input understood by the CLI."""

    EMPTY = "empty"
    PLAIN = "plain"
    SLASH = "slash"
    SHELL = "shell"


@dataclass(frozen=True, slots=True)
class SlashInput:
    """A slash command plus raw argument text."""

    command: str
    args: str = ""


@dataclass(frozen=True, slots=True)
class ParsedInput:
    """Structured representation of one prompt_toolkit input."""

    kind: ParsedInputKind
    text: str = ""
    slash: SlashInput | None = None
    shell_command: str | None = None
    attachments: list[FileAttachment] = field(default_factory=list)
    warnings: list[AttachmentWarning] = field(default_factory=list)


def extract_file_references(text: str) -> list[str]:
    """Extract supported @ file references from text."""
    references: list[str] = []
    for match in _FILE_REF_PATTERN.finditer(text):
        references.append(match.group(1) or match.group(2))
    return references


def parse_input(
    text: str,
    *,
    cwd: Path,
    max_attachment_bytes: int,
) -> ParsedInput:
    """Parse a raw interactive input string."""
    stripped = text.strip()
    if not stripped:
        return ParsedInput(kind=ParsedInputKind.EMPTY)

    if stripped.startswith("/"):
        parts = stripped[1:].split(maxsplit=1)
        command = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return ParsedInput(
            kind=ParsedInputKind.SLASH,
            text=stripped,
            slash=SlashInput(command=command, args=args),
        )

    if stripped.startswith("!"):
        return ParsedInput(
            kind=ParsedInputKind.SHELL,
            text=stripped,
            shell_command=stripped[1:].strip(),
        )

    attachments: list[FileAttachment] = []
    warnings: list[AttachmentWarning] = []
    for reference in extract_file_references(text):
        loaded = load_file_attachment(reference, cwd, max_attachment_bytes)
        if isinstance(loaded, AttachmentWarning):
            warnings.append(loaded)
        else:
            attachments.append(loaded)

    return ParsedInput(
        kind=ParsedInputKind.PLAIN,
        text=text,
        attachments=attachments,
        warnings=warnings,
    )
