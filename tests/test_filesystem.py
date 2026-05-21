import asyncio
import os
from pathlib import Path

import pytest

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
)
from dunder_mifflin_harness.tools.types import ToolStatus


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    loaded = load_file_attachment("pipe", tmp_path, max_bytes=100)

    assert isinstance(loaded, AttachmentWarning)
    assert "not a readable file" in loaded.message


def test_load_file_attachment_enforces_read_cap(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    loaded = load_file_attachment("note.txt", tmp_path, max_bytes=4)

    assert isinstance(loaded, AttachmentWarning)
    assert "too large" in loaded.message


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hello", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert "must not be empty" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "hello"
