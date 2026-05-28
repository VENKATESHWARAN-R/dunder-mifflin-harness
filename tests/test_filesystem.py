import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
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
