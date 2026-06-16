import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("abc", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert path.read_text(encoding="utf-8") == "abc"


def test_edit_file_replaces_nonempty_string(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(path), old_string="world", new_string="there")
    )

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert path.read_text(encoding="utf-8") == "hello there"
