import asyncio
import os
import threading
import time
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_mutating(tmp_path: Path) -> None:
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


def test_edit_file_can_replace_all_non_empty_matches(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("red blue red", encoding="utf-8")

    result = asyncio.run(
        edit_file(
            path=str(target),
            old_string="red",
            new_string="green",
            replace_all=True,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 2
    assert target.read_text(encoding="utf-8") == "green blue green"


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    loaded: list[object] = []

    def load() -> None:
        loaded.append(load_file_attachment("pipe", tmp_path, max_bytes=1000))

    thread = threading.Thread(target=load, daemon=True)
    started_at = time.monotonic()
    thread.start()
    thread.join(timeout=1)

    if thread.is_alive():
        write_fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(write_fd)
        thread.join(timeout=1)

    assert time.monotonic() - started_at < 1
    assert not thread.is_alive()
    assert isinstance(loaded[0], AttachmentWarning)
    assert "not a regular file" in loaded[0].message
