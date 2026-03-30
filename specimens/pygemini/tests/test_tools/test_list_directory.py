"""Tests for pygemini.tools.filesystem.list_directory."""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.filesystem.list_directory import ListDirectoryTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> ListDirectoryTool:
    return ListDirectoryTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: ListDirectoryTool) -> None:
        assert tool.name == "list_directory"

    def test_description_mentions_directory(self, tool: ListDirectoryTool) -> None:
        assert "directory" in tool.description.lower()

    def test_description_mentions_recursive(self, tool: ListDirectoryTool) -> None:
        assert "recursive" in tool.description.lower()

    def test_parameter_schema_has_path(self, tool: ListDirectoryTool) -> None:
        assert "path" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_recursive(self, tool: ListDirectoryTool) -> None:
        assert "recursive" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_include_hidden(self, tool: ListDirectoryTool) -> None:
        assert "include_hidden" in tool.parameter_schema["properties"]

    def test_parameter_schema_required_is_path(self, tool: ListDirectoryTool) -> None:
        assert tool.parameter_schema["required"] == ["path"]

    def test_to_function_declaration_structure(self, tool: ListDirectoryTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "list_directory"
        assert "description" in decl
        assert decl["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should catch bad inputs before execution."""

    def test_missing_path_returns_error(self, tool: ListDirectoryTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "path" in result.lower()

    def test_valid_path_returns_none(self, tool: ListDirectoryTool) -> None:
        assert tool.validate_params({"path": "/some/dir"}) is None

    def test_valid_path_with_recursive_flag(self, tool: ListDirectoryTool) -> None:
        assert tool.validate_params({"path": "/some/dir", "recursive": True}) is None

    def test_valid_path_with_include_hidden_flag(self, tool: ListDirectoryTool) -> None:
        assert tool.validate_params({"path": "/some/dir", "include_hidden": True}) is None

    def test_valid_path_with_all_params(self, tool: ListDirectoryTool) -> None:
        assert tool.validate_params(
            {"path": "/some/dir", "recursive": True, "include_hidden": True}
        ) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should produce human-readable summaries."""

    def test_path_only(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo/bar"})
        assert "/foo/bar" in desc

    def test_missing_path_uses_placeholder(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({})
        assert "?" in desc

    def test_recursive_flag_mentioned(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo", "recursive": True})
        assert "recursive" in desc.lower()

    def test_include_hidden_flag_mentioned(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo", "include_hidden": True})
        assert "hidden" in desc.lower()

    def test_both_flags_mentioned(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo", "recursive": True, "include_hidden": True})
        assert "recursive" in desc.lower()
        assert "hidden" in desc.lower()

    def test_no_flags_no_parenthetical(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo"})
        assert "(" not in desc

    def test_recursive_false_not_mentioned(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo", "recursive": False})
        assert "recursive" not in desc.lower()

    def test_include_hidden_false_not_mentioned(self, tool: ListDirectoryTool) -> None:
        desc = tool.get_description({"path": "/foo", "include_hidden": False})
        assert "hidden" not in desc.lower()


# ---------------------------------------------------------------------------
# execute — flat listing
# ---------------------------------------------------------------------------


class TestExecuteFlat:
    """execute should list directory contents with dirs first, then files."""

    async def test_lists_files_and_dirs(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "alpha.txt").write_text("hello")
        (tmp_path / "beta.py").write_text("code")

        result = await tool.execute({"path": str(tmp_path)})

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "subdir" in result.llm_content
        assert "alpha.txt" in result.llm_content
        assert "beta.py" in result.llm_content

    async def test_dirs_appear_before_files(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        """[DIR] entries should precede [FILE] entries in flat listing output."""
        (tmp_path / "z_dir").mkdir()
        (tmp_path / "a_file.txt").write_text("x")

        result = await tool.execute({"path": str(tmp_path)})

        dir_pos = result.llm_content.index("[DIR]")
        file_pos = result.llm_content.index("[FILE]")
        assert dir_pos < file_pos

    async def test_file_sizes_present(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "sized.txt").write_text("12345678")  # 8 bytes

        result = await tool.execute({"path": str(tmp_path)})

        assert not result.is_error
        # Size unit should appear (B / KB / MB etc.)
        assert "B" in result.llm_content

    async def test_dir_entries_have_dir_marker(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "mysubdir").mkdir()

        result = await tool.execute({"path": str(tmp_path)})

        assert "[DIR]" in result.llm_content

    async def test_file_entries_have_file_marker(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "file.txt").write_text("data")

        result = await tool.execute({"path": str(tmp_path)})

        assert "[FILE]" in result.llm_content

    async def test_summary_shows_dir_count(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "d1").mkdir()
        (tmp_path / "d2").mkdir()

        result = await tool.execute({"path": str(tmp_path)})

        assert "2 director" in result.llm_content

    async def test_summary_shows_file_count(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "f1.txt").write_text("a")
        (tmp_path / "f2.txt").write_text("b")
        (tmp_path / "f3.txt").write_text("c")

        result = await tool.execute({"path": str(tmp_path)})

        assert "3 file" in result.llm_content

    async def test_empty_directory(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        result = await tool.execute({"path": str(tmp_path)})

        assert not result.is_error
        assert "empty" in result.llm_content.lower()

    async def test_display_content_populated(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "f.txt").write_text("x")

        result = await tool.execute({"path": str(tmp_path)})

        assert result.display_content != ""

    async def test_entries_sorted_alphabetically_within_type(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "m.txt").write_text("m")

        result = await tool.execute({"path": str(tmp_path)})

        a_pos = result.llm_content.index("a.txt")
        m_pos = result.llm_content.index("m.txt")
        z_pos = result.llm_content.index("z.txt")
        assert a_pos < m_pos < z_pos


# ---------------------------------------------------------------------------
# execute — recursive listing
# ---------------------------------------------------------------------------


class TestExecuteRecursive:
    """execute with recursive=True should descend into subdirectories."""

    async def test_recursive_finds_nested_file(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "leaf.txt").write_text("leaf content")

        result = await tool.execute({"path": str(tmp_path), "recursive": True})

        assert not result.is_error
        assert "leaf.txt" in result.llm_content

    async def test_recursive_finds_nested_dir(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "sub" / "deep").mkdir(parents=True)

        result = await tool.execute({"path": str(tmp_path), "recursive": True})

        assert "deep" in result.llm_content

    async def test_recursive_total_count_includes_nested(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file1.txt").write_text("a")
        (sub / "file2.txt").write_text("b")
        (tmp_path / "root.txt").write_text("c")

        result = await tool.execute({"path": str(tmp_path), "recursive": True})

        assert "3 file" in result.llm_content

    async def test_recursive_flat_does_not_show_nested(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("hidden in flat")

        result = await tool.execute({"path": str(tmp_path), "recursive": False})

        # flat listing shows "sub" dir but not its contents
        assert "sub" in result.llm_content
        assert "nested.txt" not in result.llm_content


# ---------------------------------------------------------------------------
# execute — hidden files
# ---------------------------------------------------------------------------


class TestExecuteHidden:
    """execute should show/hide dotfiles based on include_hidden flag."""

    async def test_hidden_files_excluded_by_default(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("ok")

        result = await tool.execute({"path": str(tmp_path)})

        assert ".hidden" not in result.llm_content
        assert "visible.txt" in result.llm_content

    async def test_include_hidden_shows_dotfiles(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / ".hidden").write_text("secret")

        result = await tool.execute(
            {"path": str(tmp_path), "include_hidden": True}
        )

        assert ".hidden" in result.llm_content

    async def test_hidden_dirs_excluded_by_default(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "src").mkdir()

        result = await tool.execute({"path": str(tmp_path)})

        assert ".git" not in result.llm_content
        assert "src" in result.llm_content

    async def test_include_hidden_shows_hidden_dirs(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / ".git").mkdir()

        result = await tool.execute(
            {"path": str(tmp_path), "include_hidden": True}
        )

        assert ".git" in result.llm_content


# ---------------------------------------------------------------------------
# execute — error cases
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    """execute should return error ToolResults for invalid inputs."""

    async def test_not_found(self, tool: ListDirectoryTool, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"

        result = await tool.execute({"path": str(missing)})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "not found" in result.llm_content.lower()

    async def test_not_found_mentions_path(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        missing = tmp_path / "ghost_dir"

        result = await tool.execute({"path": str(missing)})

        assert str(missing) in result.llm_content

    async def test_not_a_directory(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("I am a file, not a dir")

        result = await tool.execute({"path": str(f)})

        assert result.is_error
        assert "not a directory" in result.llm_content.lower()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod-based permission denial not reliable on Windows",
    )
    async def test_permission_denied(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        protected = tmp_path / "locked"
        protected.mkdir()
        protected.chmod(0o000)

        try:
            result = await tool.execute({"path": str(protected)})
            assert result.is_error
            assert "permission" in result.llm_content.lower()
        finally:
            protected.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# execute — abort_signal (no-op, must not crash)
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_execute_with_abort_signal_set(
        self, tool: ListDirectoryTool, tmp_path: Path
    ) -> None:
        (tmp_path / "file.txt").write_text("data")
        signal = asyncio.Event()
        signal.set()

        result = await tool.execute({"path": str(tmp_path)}, abort_signal=signal)

        assert not result.is_error
