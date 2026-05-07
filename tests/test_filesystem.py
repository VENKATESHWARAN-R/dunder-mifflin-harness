import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("abc", encoding="utf-8")

    result = asyncio.run(edit_file(str(path), old_string="", new_string="X"))

    assert result.status == ToolStatus.ERROR
    assert result.error == "old_string must not be empty"
    assert path.read_text(encoding="utf-8") == "abc"
