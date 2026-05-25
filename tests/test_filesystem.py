import asyncio
import os
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
    read_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(
            path=str(target),
            old_string="",
            new_string="X",
            replace_all=True,
        )
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = load_file_attachment("pipe", cwd=tmp_path, max_bytes=1000)

    assert isinstance(result, AttachmentWarning)
    assert "not a regular file" in result.message


def test_read_file_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = asyncio.run(read_file(str(fifo)))

    assert result.status == ToolStatus.ERROR
    assert "not a regular file" in (result.error or "")
