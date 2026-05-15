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
    target = tmp_path / "notes.txt"
    target.write_text("important data", encoding="utf-8")

    result = asyncio.run(
        edit_file(
            str(target),
            old_string="",
            new_string="corruption",
            replace_all=True,
        )
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "important data"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX fifo support")
def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    loaded = load_file_attachment("pipe", tmp_path, max_bytes=1000)

    assert isinstance(loaded, AttachmentWarning)
    assert "not a regular file" in loaded.message
