import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    edit_file,
    load_file_attachment,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_load_file_attachment_rejects_special_files() -> None:
    loaded = load_file_attachment("/dev/null", cwd=Path.cwd(), max_bytes=1000)

    assert isinstance(loaded, AttachmentWarning)
    assert "regular files" in loaded.message


def test_edit_file_rejects_empty_old_string_without_changing_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("abc", encoding="utf-8")

    result = asyncio.run(edit_file(str(target), "", "X", replace_all=True))

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "abc"
