import asyncio
from pathlib import Path

from dunder_mifflin_harness.tools.filesystem import edit_file, grep_files, read_file, search_files
from dunder_mifflin_harness.tools.types import ToolStatus


def test_edit_file_rejects_empty_old_string_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")

    result = asyncio.run(
        edit_file(
            path=str(target),
            old_string="",
            new_string="X",
            replace_all=True,
        )
    )

    assert result.status == ToolStatus.ERROR
    assert result.replacements_made == 0
    assert target.read_text(encoding="utf-8") == "hello"


def test_read_file_returns_error_for_non_utf8_file(tmp_path: Path) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"\xff\xfe\x00")

    result = asyncio.run(read_file(path=str(target)))

    assert result.status == ToolStatus.ERROR
    assert "UTF-8" in result.error


def test_read_file_rejects_large_unsliced_file(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_text("x" * 1_000_001, encoding="utf-8")

    result = asyncio.run(read_file(path=str(target)))

    assert result.status == ToolStatus.ERROR
    assert "too large" in result.error


def test_search_files_caps_results_and_ignores_noise_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.txt").write_text("ignored", encoding="utf-8")
    for index in range(3):
        (tmp_path / f"match-{index}.txt").write_text("match", encoding="utf-8")

    result = asyncio.run(search_files(root=str(tmp_path), pattern="*.txt", max_results=2))

    assert result.status == ToolStatus.OK
    assert len(result.matches) == 2
    assert result.warnings
    assert all(".git" not in match for match in result.matches)


def test_grep_files_ignores_noise_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.txt").write_text("needle", encoding="utf-8")
    visible = tmp_path / "visible.txt"
    visible.write_text("needle\n", encoding="utf-8")

    result = asyncio.run(grep_files(root=str(tmp_path), pattern="needle"))

    assert result.status == ToolStatus.OK
    assert [match.file for match in result.matches] == [str(visible)]
