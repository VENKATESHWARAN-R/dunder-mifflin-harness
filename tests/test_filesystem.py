import asyncio
import os
import time
from pathlib import Path

import pytest

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    grep_files,
    load_file_attachment,
    read_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not available on this platform")

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    started_at = time.monotonic()
    result = load_file_attachment("pipe", tmp_path, max_bytes=1024)
    elapsed = time.monotonic() - started_at

    assert isinstance(result, AttachmentWarning)
    assert "not a regular file" in result.message
    assert elapsed < 0.5


def test_read_file_returns_error_for_non_utf8_file(tmp_path: Path) -> None:
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"\xff\xfe")

    result = asyncio.run(read_file(str(binary)))

    assert result.status == ToolStatus.ERROR
    assert "not valid UTF-8" in (result.error or "")


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * 1_000_001)

    result = asyncio.run(read_file(str(large)))

    assert result.status == ToolStatus.ERROR
    assert "too large" in (result.error or "")


def test_edit_file_rejects_empty_old_string_without_corrupting_file(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert target.read_text(encoding="utf-8") == "abc"


def test_grep_files_skips_non_regular_files_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not available on this platform")

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    text_file = tmp_path / "notes.txt"
    text_file.write_text("needle\n", encoding="utf-8")

    started_at = time.monotonic()
    result = asyncio.run(grep_files(str(tmp_path), "needle"))
    elapsed = time.monotonic() - started_at

    assert result.status == ToolStatus.OK
    assert result.total_matches == 1
    assert elapsed < 0.5
