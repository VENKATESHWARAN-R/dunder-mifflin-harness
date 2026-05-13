import asyncio
import os
from pathlib import Path

import pytest

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
    read_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_load_file_attachment_rejects_non_regular_files(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not available on this platform")

    os.mkfifo(tmp_path / "pipe")

    loaded = load_file_attachment("pipe", tmp_path, max_bytes=1000)

    assert isinstance(loaded, AttachmentWarning)
    assert "only regular files" in loaded.message


def test_load_file_attachment_detects_growth_beyond_limit(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    loaded = load_file_attachment("note.txt", tmp_path, max_bytes=4)

    assert isinstance(loaded, AttachmentWarning)
    assert "too large" in loaded.message


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert "old_string must not be empty" in (result.error or "")
    assert path.read_text(encoding="utf-8") == "abc"


def test_text_tools_return_errors_for_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"\xff")

    read_result = asyncio.run(read_file(str(path)))
    edit_result = asyncio.run(edit_file(str(path), "a", "b"))

    assert read_result.status == ToolStatus.ERROR
    assert edit_result.status == ToolStatus.ERROR
    assert "UTF-8" in (read_result.error or "")
    assert "UTF-8" in (edit_result.error or "")
