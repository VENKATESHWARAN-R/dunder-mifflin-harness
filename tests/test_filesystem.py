import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_replaces_single_match(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello world\n", encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="world", new_string="paper")
    )

    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert target.read_text(encoding="utf-8") == "hello paper\n"


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    original = "abc\n"
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        edit_file(str(target), old_string="", new_string="X", replace_all=True)
    )

    assert result.status == ToolStatus.ERROR
    assert result.error == "old_string must not be empty"
    assert target.read_text(encoding="utf-8") == original
