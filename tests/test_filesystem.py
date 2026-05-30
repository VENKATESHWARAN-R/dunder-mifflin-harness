import asyncio
import os
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    edit_file,
    grep_files,
    load_file_attachment,
    read_file,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_replaces_exact_match(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")

    result = asyncio.run(edit_file(str(path), "world", "there"))

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert path.read_text(encoding="utf-8") == "hello there"


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    original = "abc"
    path.write_text(original, encoding="utf-8")

    result = asyncio.run(edit_file(str(path), "", "X", replace_all=True))

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert path.read_text(encoding="utf-8") == original


def test_load_file_attachment_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = load_file_attachment("pipe", tmp_path, max_bytes=100)

    assert "only regular files" in result.message


def test_read_file_rejects_large_full_file_read(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 1_000_001, encoding="utf-8")

    result = asyncio.run(read_file(str(path)))

    assert result.status == ToolStatus.ERROR
    assert "too large" in result.error


def test_read_file_allows_bounded_line_read_from_large_file(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("first\n" + ("x" * 1_000_001), encoding="utf-8")

    result = asyncio.run(read_file(str(path), start_line=1, end_line=1))

    assert result.status == ToolStatus.OK
    assert result.content == "first\n"
    assert result.truncated


def test_grep_files_skips_large_files(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    small.write_text("needle\n", encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_text("needle\n" + ("x" * 1_000_001), encoding="utf-8")

    result = asyncio.run(grep_files(str(tmp_path), "needle", include="*.txt"))

    assert result.status == ToolStatus.OK
    assert [match.file for match in result.matches] == [str(small)]
    assert any("skipped large file" in warning for warning in result.warnings)
