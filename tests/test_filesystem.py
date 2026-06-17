import asyncio
import os
from pathlib import Path

import pytest

from dunder_mifflin_harness.tools.filesystem import (
    edit_file,
    load_file_attachment,
    read_file,
    write_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_attachment_loader_rejects_fifo(tmp_path: Path) -> None:
    fifo_path = tmp_path / "stream"
    os.mkfifo(fifo_path)

    result = load_file_attachment("stream", cwd=tmp_path, max_bytes=1000)

    assert result.message == "not a regular file: stream"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_agent_file_tools_reject_fifo(tmp_path: Path) -> None:
    fifo_path = tmp_path / "stream"
    os.mkfifo(fifo_path)

    read_result = asyncio.run(read_file(str(fifo_path)))
    write_result = asyncio.run(write_file(str(fifo_path), "data"))
    edit_result = asyncio.run(edit_file(str(fifo_path), "old", "new"))

    assert read_result.status == ToolStatus.ERROR
    assert write_result.status == ToolStatus.ERROR
    assert edit_result.status == ToolStatus.ERROR
    assert "not a regular file" in (read_result.error or "")
    assert "not a regular file" in (write_result.error or "")
    assert "not a regular file" in (edit_result.error or "")
