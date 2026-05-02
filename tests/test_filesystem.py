"""Tests for filesystem agent tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jac.tools.filesystem import (
    edit_file,
    grep_files,
    list_directory,
    read_file,
    search_files,
    write_file,
)
from jac.tools.types import RiskLevel, ToolStatus


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_content(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")
    result = asyncio.run(read_file(str(f)))
    assert result.status == ToolStatus.OK
    assert result.content == "line1\nline2\nline3\n"
    assert result.lines_total == 3
    assert not result.truncated


def test_read_file_not_found(tmp_path: Path) -> None:
    result = asyncio.run(read_file(str(tmp_path / "missing.txt")))
    assert result.status == ToolStatus.NOT_FOUND


def test_read_file_line_range(tmp_path: Path) -> None:
    f = tmp_path / "multi.txt"
    f.write_text("a\nb\nc\nd\n")
    result = asyncio.run(read_file(str(f), start_line=2, end_line=3))
    assert result.content == "b\nc\n"
    assert result.lines_returned == 2
    assert result.truncated


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_new_file(tmp_path: Path) -> None:
    p = tmp_path / "new.txt"
    result = asyncio.run(write_file(str(p), "hello"))
    assert result.status == ToolStatus.OK
    assert result.created is True
    assert p.read_text() == "hello"
    assert result.bytes_written == 5


def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    p = tmp_path / "existing.txt"
    p.write_text("old content")
    result = asyncio.run(write_file(str(p), "new content"))
    assert result.status == ToolStatus.OK
    assert result.created is False
    assert p.read_text() == "new content"


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "deep.txt"
    result = asyncio.run(write_file(str(p), "deep"))
    assert result.status == ToolStatus.OK
    assert p.read_text() == "deep"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_replaces_string(tmp_path: Path) -> None:
    p = tmp_path / "edit.txt"
    p.write_text("hello world\n")
    result = asyncio.run(edit_file(str(p), "world", "python"))
    assert result.status == ToolStatus.OK
    assert result.replacements_made == 1
    assert p.read_text() == "hello python\n"


def test_edit_file_populates_diff(tmp_path: Path) -> None:
    p = tmp_path / "diff.txt"
    p.write_text("hello world\n")
    result = asyncio.run(edit_file(str(p), "world", "python"))
    assert result.diff
    assert "-hello world" in result.diff
    assert "+hello python" in result.diff


def test_edit_file_not_found(tmp_path: Path) -> None:
    result = asyncio.run(edit_file(str(tmp_path / "ghost.txt"), "x", "y"))
    assert result.status == ToolStatus.NOT_FOUND


def test_edit_file_string_not_found(tmp_path: Path) -> None:
    p = tmp_path / "nochange.txt"
    p.write_text("hello world\n")
    result = asyncio.run(edit_file(str(p), "missing", "replacement"))
    assert result.status == ToolStatus.ERROR
    assert result.error is not None


def test_edit_file_ambiguous_match_errors(tmp_path: Path) -> None:
    p = tmp_path / "dup.txt"
    p.write_text("x x x")
    result = asyncio.run(edit_file(str(p), "x", "y"))
    assert result.status == ToolStatus.ERROR
    assert "3" in (result.error or "")


def test_edit_file_replace_all(tmp_path: Path) -> None:
    p = tmp_path / "multi.txt"
    p.write_text("x x x")
    result = asyncio.run(edit_file(str(p), "x", "y", replace_all=True))
    assert result.status == ToolStatus.OK
    assert result.replacements_made == 3
    assert p.read_text() == "y y y"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_returns_entries(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    result = asyncio.run(list_directory(str(tmp_path)))
    assert result.status == ToolStatus.OK
    names = {e.name for e in result.entries}
    assert {"a.txt", "b.txt"} <= names


def test_list_directory_not_found(tmp_path: Path) -> None:
    result = asyncio.run(list_directory(str(tmp_path / "missing")))
    assert result.status == ToolStatus.NOT_FOUND


def test_list_directory_ignores_pycache(tmp_path: Path) -> None:
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "real.py").write_text("")
    result = asyncio.run(list_directory(str(tmp_path)))
    names = {e.name for e in result.entries}
    assert "__pycache__" not in names
    assert "real.py" in names


def test_list_directory_max_depth(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("")
    result = asyncio.run(list_directory(str(tmp_path), max_depth=2))
    names = {e.name for e in result.entries}
    assert "nested.txt" in names


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


def test_search_files_finds_glob_pattern(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("pass")
    (tmp_path / "bar.txt").write_text("text")
    result = asyncio.run(search_files(str(tmp_path), "*.py"))
    assert result.status == ToolStatus.OK
    assert any("foo.py" in m for m in result.matches)
    assert not any("bar.txt" in m for m in result.matches)


def test_search_files_root_not_found(tmp_path: Path) -> None:
    result = asyncio.run(search_files(str(tmp_path / "ghost"), "*.py"))
    assert result.status == ToolStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------


def test_grep_files_finds_pattern(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("def hello():\n    pass\n")
    result = asyncio.run(grep_files(str(tmp_path), r"def \w+"))
    assert result.status == ToolStatus.OK
    assert result.total_matches >= 1
    assert any("hello" in m.line for m in result.matches)


def test_grep_files_invalid_regex(tmp_path: Path) -> None:
    result = asyncio.run(grep_files(str(tmp_path), "[invalid"))
    assert result.status == ToolStatus.ERROR


def test_grep_files_context_lines(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("before\ntarget\nafter\n")
    result = asyncio.run(grep_files(str(tmp_path), "target", context_lines=1))
    assert result.matches
    assert result.matches[0].context_before == ["before"]
    assert result.matches[0].context_after == ["after"]


def test_grep_files_max_matches_cap(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("\n".join(["hit"] * 20))
    result = asyncio.run(grep_files(str(tmp_path), "hit", max_matches=5))
    assert len(result.matches) == 5
    assert result.total_matches == 20
    assert result.warnings


# ---------------------------------------------------------------------------
# Approval metadata
# ---------------------------------------------------------------------------


def test_read_file_approval_is_read_only() -> None:
    assert read_file.approval.risk_level == RiskLevel.READ_ONLY
    assert read_file.approval.category == "file_read"
    assert read_file.approval.reversible


def test_write_file_approval_is_low_risk() -> None:
    assert write_file.approval.risk_level == RiskLevel.LOW
    assert write_file.approval.category == "file_write"


def test_edit_file_approval_is_low_risk() -> None:
    assert edit_file.approval.risk_level == RiskLevel.LOW
    assert edit_file.approval.category == "file_write"


def test_list_directory_approval_is_read_only() -> None:
    assert list_directory.approval.risk_level == RiskLevel.READ_ONLY


def test_search_files_approval_is_read_only() -> None:
    assert search_files.approval.risk_level == RiskLevel.READ_ONLY


def test_grep_files_approval_is_read_only() -> None:
    assert grep_files.approval.risk_level == RiskLevel.READ_ONLY
