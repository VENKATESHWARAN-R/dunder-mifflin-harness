import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file, grep_files, read_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_read_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    target = tmp_path / "binary.dat"
    target.write_bytes(b"\xff\xfe")

    result = asyncio.run(read_file(str(target)))

    assert result.status == ToolStatus.ERROR
    assert "UTF-8" in (result.error or "")


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"x" * 1_000_001)

    result = asyncio.run(read_file(str(target)))

    assert result.status == ToolStatus.ERROR
    assert "too large" in (result.error or "")


def test_edit_file_rejects_empty_old_string_without_modifying(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("safe content", encoding="utf-8")

    result = asyncio.run(edit_file(str(target), "", "unsafe", replace_all=True))

    assert result.status == ToolStatus.ERROR
    assert "must not be empty" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "safe content"


def test_grep_files_skips_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")
    (tmp_path / "text.txt").write_text("hello\n", encoding="utf-8")

    result = asyncio.run(grep_files(str(tmp_path), "hello"))

    assert result.status == ToolStatus.OK
    assert result.total_matches == 1
    assert result.matches[0].file.endswith("text.txt")
