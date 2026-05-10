import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_changing_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"


def test_edit_file_rejects_empty_old_string_for_empty_files(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X")
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == ""
