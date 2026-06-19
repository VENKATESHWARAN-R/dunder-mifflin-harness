import asyncio
import os
from pathlib import Path

import pytest

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
    read_file,
    write_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX fifo support")
def test_file_tools_reject_fifo_paths(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    attachment = load_file_attachment("pipe", tmp_path, max_bytes=100)
    read_result = asyncio.run(read_file(str(fifo)))
    write_result = asyncio.run(write_file(str(fifo), "content"))

    assert isinstance(attachment, AttachmentWarning)
    assert "not a regular file" in attachment.message
    assert read_result.status == ToolStatus.ERROR
    assert write_result.status == ToolStatus.ERROR
