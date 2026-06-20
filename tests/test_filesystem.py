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
    target = tmp_path / "note.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="x", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"


def test_file_tools_reject_non_regular_paths(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    read_result = asyncio.run(read_file(str(directory)))
    write_result = asyncio.run(write_file(str(directory), "content"))
    edit_result = asyncio.run(edit_file(str(directory), "old", "new"))

    assert read_result.status == ToolStatus.ERROR
    assert write_result.status == ToolStatus.ERROR
    assert edit_result.status == ToolStatus.ERROR


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = load_file_attachment("pipe", tmp_path, max_bytes=100)

    assert isinstance(result, AttachmentWarning)
    assert "not a regular file" in result.message
