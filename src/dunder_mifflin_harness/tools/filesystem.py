"""Filesystem helpers for attachments and edit previews."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path


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
    try:
        stat = path.stat()
    except FileNotFoundError:
        return AttachmentWarning(reference, f"file not found: {reference}")
    except PermissionError:
        return AttachmentWarning(reference, f"permission denied: {reference}")
    except OSError as exc:
        return AttachmentWarning(reference, f"could not inspect {reference}: {exc}")

    if path.is_dir():
        return AttachmentWarning(reference, f"directories are not attachable yet: {reference}")

    if stat.st_size > max_bytes:
        return AttachmentWarning(
            reference,
            f"file is too large to attach ({stat.st_size} bytes): {reference}",
        )

    try:
        data = path.read_bytes()
    except PermissionError:
        return AttachmentWarning(reference, f"permission denied: {reference}")
    except OSError as exc:
        return AttachmentWarning(reference, f"could not read {reference}: {exc}")

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
