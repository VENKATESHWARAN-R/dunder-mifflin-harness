import asyncio
import os
from pathlib import Path

from dunder_mifflin_harness.tools import filesystem as fs
from dunder_mifflin_harness.tools.filesystem import AttachmentWarning
from dunder_mifflin_harness.tools.types import ToolStatus


def test_load_file_attachment_rejects_non_regular_file(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    result = fs.load_file_attachment("pipe", tmp_path, max_bytes=1000)

    assert isinstance(result, AttachmentWarning)
    assert "not a regular file" in result.message


def test_read_file_binary_returns_structured_error(tmp_path: Path) -> None:
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"\xff\xfe")

    result = asyncio.run(fs.read_file(str(binary)))

    assert result.status == ToolStatus.ERROR
    assert "UTF-8" in (result.error or "")


def test_read_file_caps_large_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fs, "_DEFAULT_MAX_FILE_READ_BYTES", 5)
    note = tmp_path / "note.txt"
    note.write_text("abcdef", encoding="utf-8")

    result = asyncio.run(fs.read_file(str(note)))

    assert result.status == ToolStatus.OK
    assert result.content == "abcde"
    assert result.truncated
    assert "truncated" in result.warnings[0]


def test_edit_file_rejects_empty_old_string_without_corrupting_file(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        fs.edit_file(str(note), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert "old_string" in (result.error or "")
    assert note.read_text(encoding="utf-8") == "abc"


def test_edit_file_binary_returns_structured_error(tmp_path: Path) -> None:
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"\xff\xfe")

    result = asyncio.run(fs.edit_file(str(binary), old_string="x", new_string="y"))

    assert result.status == ToolStatus.ERROR
    assert "UTF-8" in (result.error or "")
