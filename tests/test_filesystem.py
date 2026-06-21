import asyncio
import os
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "events.pipe"
    os.mkfifo(fifo)

    loaded = load_file_attachment("events.pipe", tmp_path, max_bytes=1024)

    assert isinstance(loaded, AttachmentWarning)
    assert "only regular files" in loaded.message


def test_load_file_attachment_reads_regular_file(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    loaded = load_file_attachment("note.txt", tmp_path, max_bytes=1024)

    assert not isinstance(loaded, AttachmentWarning)
    assert loaded.content == "hello"
    assert loaded.size == 5


def test_edit_file_rejects_empty_old_string_without_changing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(file_path), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert file_path.read_text(encoding="utf-8") == "abc"
