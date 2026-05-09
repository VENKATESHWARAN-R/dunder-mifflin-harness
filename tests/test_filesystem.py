import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_mutating(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert "must not be empty" in (result.error or "")
    assert path.read_text(encoding="utf-8") == "abc"


def test_edit_file_rejects_empty_old_string_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="", new_string="X")
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert path.read_text(encoding="utf-8") == ""


def test_edit_file_replaces_matching_string(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("dunder paper", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="dunder", new_string="mifflin")
    )

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert path.read_text(encoding="utf-8") == "mifflin paper"
