import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert "old_string must not be empty" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "abc"


def test_edit_file_replaces_all_non_empty_matches(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("alpha alpha", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="alpha", new_string="beta", replace_all=True)
    )

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 2
    assert target.read_text(encoding="utf-8") == "beta beta"
