import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import (
    AttachmentWarning,
    FileAttachment,
    edit_file,
    load_file_attachment,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = asyncio.run(
        edit_file(
            path=str(target),
            old_string="",
            new_string="CORRUPT",
            replace_all=True,
        )
    )

    assert result.status == ToolStatus.ERROR
    assert "must not be empty" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_load_file_attachment_rejects_non_regular_files() -> None:
    result = load_file_attachment("/dev/null", cwd=Path("/"), max_bytes=1000)

    assert isinstance(result, AttachmentWarning)
    assert "only regular files" in result.message


def test_load_file_attachment_accepts_regular_text_file(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    result = load_file_attachment("note.txt", cwd=tmp_path, max_bytes=1000)

    assert isinstance(result, FileAttachment)
    assert result.content == "hello"
    assert result.display_path == "note.txt"
