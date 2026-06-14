import asyncio
import os
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
    read_file,
    write_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("abc", encoding="utf-8")

    result = asyncio.run(edit_file(str(path), "", "X", replace_all=True))

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert path.read_text(encoding="utf-8") == "abc"


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = load_file_attachment("pipe", tmp_path, max_bytes=100)

    assert isinstance(result, AttachmentWarning)
    assert "not a regular text file" in result.message


def test_read_file_rejects_binary_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00")

    result = asyncio.run(read_file(str(path)))

    assert result.status == ToolStatus.ERROR
    assert "not valid UTF-8" in (result.error or "")


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.txt"
    path.write_bytes(b"x" * 1_000_001)

    result = asyncio.run(read_file(str(path)))

    assert result.status == ToolStatus.ERROR
    assert "too large" in (result.error or "")


def test_write_file_rejects_non_regular_target(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = asyncio.run(write_file(str(fifo), "content"))

    assert result.status == ToolStatus.ERROR
    assert "not a regular file" in (result.error or "")
