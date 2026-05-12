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


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.error == "old_string must not be empty"
    assert result.replacements_made == 0
    assert path.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
def test_load_file_attachment_rejects_fifo(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    result = load_file_attachment("pipe", cwd=tmp_path, max_bytes=100)

    assert isinstance(result, AttachmentWarning)
    assert "not a regular file" in result.message
