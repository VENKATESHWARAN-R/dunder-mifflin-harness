"""Filesystem tool tests — happy + failure paths against tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from jac.tools.filesystem import (
    PreparedEdit,
    compute_edit,
    edit_file,
    grep_files,
    list_directory,
    preview_write,
    read_file,
    search_files,
    write_file,
)
from jac.tools.types import ToolStatus


# ---- read_file -------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_returns_content(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("line1\nline2\n", encoding="utf-8")
    result = await read_file(str(target))
    assert result.status == ToolStatus.OK
    assert result.content == "line1\nline2\n"
    assert result.lines_total == 2
    assert result.truncated is False


@pytest.mark.asyncio
async def test_read_file_missing_returns_not_found(tmp_path: Path) -> None:
    result = await read_file(str(tmp_path / "nope.txt"))
    assert result.status == ToolStatus.NOT_FOUND
    assert result.error is not None and "not found" in result.error


@pytest.mark.asyncio
async def test_read_file_partial_slice(tmp_path: Path) -> None:
    target = tmp_path / "many.txt"
    target.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    result = await read_file(str(target), start_line=2, end_line=4)
    assert result.status == ToolStatus.OK
    assert result.content == "b\nc\nd\n"
    assert result.truncated is True
    assert result.warnings


# ---- write_file ------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.txt"
    result = await write_file(str(target), "payload")
    assert result.status == ToolStatus.OK
    assert result.created is True
    assert target.read_text(encoding="utf-8") == "payload"


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    result = await write_file(str(target), "new")
    assert result.status == ToolStatus.OK
    assert result.created is False
    assert target.read_text(encoding="utf-8") == "new"


def test_preview_write_returns_diff_for_new_file(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    diff = preview_write(str(target), "hello\n")
    assert "/dev/null" in diff
    assert "+hello" in diff


# ---- edit_file -------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_file_replaces_single_match(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = await edit_file(str(target), "x = 1", "x = 42")
    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert target.read_text(encoding="utf-8") == "x = 42\ny = 2\n"


@pytest.mark.asyncio
async def test_edit_file_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("a\na\na\n", encoding="utf-8")
    result = await edit_file(str(target), "a", "b", replace_all=True)
    assert result.status == ToolStatus.OK
    assert result.replacements_made == 3
    assert target.read_text(encoding="utf-8") == "b\nb\nb\n"


@pytest.mark.asyncio
async def test_edit_file_ambiguous_match_errors(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("a\na\n", encoding="utf-8")
    result = await edit_file(str(target), "a", "b")
    assert result.status == ToolStatus.ERROR
    assert result.error is not None and "matches 2 times" in result.error


@pytest.mark.asyncio
async def test_edit_file_no_match_errors(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("hello\n", encoding="utf-8")
    result = await edit_file(str(target), "missing", "x")
    assert result.status == ToolStatus.ERROR


def test_compute_edit_returns_prepared_edit_for_valid_match(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("alpha\n", encoding="utf-8")
    prepared = compute_edit(str(target), "alpha", "beta")
    assert isinstance(prepared, PreparedEdit)
    assert prepared.new_content == "beta\n"
    assert "alpha" in prepared.diff


# ---- list_directory --------------------------------------------------------


@pytest.mark.asyncio
async def test_list_directory_lists_immediate_children(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("", encoding="utf-8")
    result = await list_directory(str(tmp_path))
    assert result.status == ToolStatus.OK
    names = {e.name for e in result.entries}
    assert "a.txt" in names
    assert "sub" in names
    assert "b.txt" not in names


@pytest.mark.asyncio
async def test_list_directory_respects_max_depth(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("", encoding="utf-8")
    result = await list_directory(str(tmp_path), max_depth=2)
    names = {e.name for e in result.entries}
    assert "b.txt" in names


@pytest.mark.asyncio
async def test_list_directory_skips_hidden_by_default(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("", encoding="utf-8")
    result = await list_directory(str(tmp_path))
    assert all(e.name != ".hidden" for e in result.entries)
    result_with_hidden = await list_directory(str(tmp_path), show_hidden=True)
    assert any(e.name == ".hidden" for e in result_with_hidden.entries)


@pytest.mark.asyncio
async def test_list_directory_missing(tmp_path: Path) -> None:
    result = await list_directory(str(tmp_path / "nope"))
    assert result.status == ToolStatus.NOT_FOUND


# ---- search_files ----------------------------------------------------------


@pytest.mark.asyncio
async def test_search_files_globs_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("", encoding="utf-8")
    result = await search_files(str(tmp_path), "*.py")
    assert result.status == ToolStatus.OK
    assert any(m.endswith("a.py") for m in result.matches)
    assert any(m.endswith("b.py") for m in result.matches)


# ---- grep_files ------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text(
        "def hello():\n    return 'world'\n", encoding="utf-8"
    )
    result = await grep_files(str(tmp_path), r"def \w+")
    assert result.status == ToolStatus.OK
    assert result.total_matches == 1
    assert result.matches[0].line == "def hello():"


@pytest.mark.asyncio
async def test_grep_files_context_lines(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("a\nb\nMATCH\nc\nd\n", encoding="utf-8")
    result = await grep_files(str(tmp_path), "MATCH", context_lines=1)
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.context_before == ["b"]
    assert m.context_after == ["c"]


@pytest.mark.asyncio
async def test_grep_files_invalid_regex(tmp_path: Path) -> None:
    result = await grep_files(str(tmp_path), "(")
    assert result.status == ToolStatus.ERROR
    assert result.error is not None and "invalid regex" in result.error
