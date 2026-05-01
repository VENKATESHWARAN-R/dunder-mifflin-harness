"""Tests for pygemini.tools.filesystem.read_file."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.filesystem.read_file import ReadFileTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> ReadFileTool:
    return ReadFileTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestReadFileMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: ReadFileTool) -> None:
        assert tool.name == "read_file"

    def test_description_contains_key_phrases(self, tool: ReadFileTool) -> None:
        desc = tool.description
        assert "Read" in desc
        assert "line numbers" in desc

    def test_to_function_declaration_structure(self, tool: ReadFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "read_file"
        assert "description" in decl
        assert decl["parameters"]["type"] == "object"
        assert "path" in decl["parameters"]["properties"]
        assert "offset" in decl["parameters"]["properties"]
        assert "limit" in decl["parameters"]["properties"]
        assert decl["parameters"]["required"] == ["path"]

    def test_to_function_declaration_offset_is_integer(self, tool: ReadFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["parameters"]["properties"]["offset"]["type"] == "integer"

    def test_to_function_declaration_limit_is_integer(self, tool: ReadFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["parameters"]["properties"]["limit"]["type"] == "integer"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should catch bad inputs before execution."""

    def test_valid_path_only(self, tool: ReadFileTool) -> None:
        assert tool.validate_params({"path": "/some/file.py"}) is None

    def test_valid_with_offset_and_limit(self, tool: ReadFileTool) -> None:
        assert tool.validate_params({"path": "/f", "offset": 5, "limit": 10}) is None

    def test_missing_path_returns_error(self, tool: ReadFileTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "path" in result.lower()

    def test_offset_zero_returns_error(self, tool: ReadFileTool) -> None:
        result = tool.validate_params({"path": "/f", "offset": 0})
        assert result is not None
        assert "offset" in result

    def test_negative_offset_returns_error(self, tool: ReadFileTool) -> None:
        result = tool.validate_params({"path": "/f", "offset": -5})
        assert result is not None
        assert "offset" in result

    def test_positive_offset_valid(self, tool: ReadFileTool) -> None:
        assert tool.validate_params({"path": "/f", "offset": 1}) is None

    def test_limit_zero_returns_error(self, tool: ReadFileTool) -> None:
        result = tool.validate_params({"path": "/f", "limit": 0})
        assert result is not None
        assert "limit" in result

    def test_negative_limit_returns_error(self, tool: ReadFileTool) -> None:
        result = tool.validate_params({"path": "/f", "limit": -1})
        assert result is not None
        assert "limit" in result

    def test_positive_limit_valid(self, tool: ReadFileTool) -> None:
        assert tool.validate_params({"path": "/f", "limit": 1}) is None

    def test_none_limit_is_valid(self, tool: ReadFileTool) -> None:
        # limit defaults to None (read all) — must not be an error
        assert tool.validate_params({"path": "/f", "limit": None}) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should produce human-readable summaries."""

    def test_path_only(self, tool: ReadFileTool) -> None:
        desc = tool.get_description({"path": "/foo/bar.py"})
        assert "/foo/bar.py" in desc

    def test_with_offset(self, tool: ReadFileTool) -> None:
        desc = tool.get_description({"path": "app.py", "offset": 10})
        assert "app.py" in desc
        assert "10" in desc

    def test_with_limit(self, tool: ReadFileTool) -> None:
        desc = tool.get_description({"path": "app.py", "limit": 20})
        assert "20" in desc

    def test_with_offset_and_limit(self, tool: ReadFileTool) -> None:
        desc = tool.get_description({"path": "app.py", "offset": 5, "limit": 15})
        assert "5" in desc
        assert "15" in desc

    def test_missing_path_uses_placeholder(self, tool: ReadFileTool) -> None:
        desc = tool.get_description({})
        assert "?" in desc


# ---------------------------------------------------------------------------
# execute — happy path
# ---------------------------------------------------------------------------


class TestExecuteReadValidFile:
    """execute should return file contents with line numbers."""

    async def test_reads_entire_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\nline three\n")

        result = await tool.execute({"path": str(f)})

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "line one" in result.llm_content
        assert "line two" in result.llm_content
        assert "line three" in result.llm_content

    async def test_line_numbers_present(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "numbered.txt"
        f.write_text("alpha\nbeta\ngamma\n")

        result = await tool.execute({"path": str(f)})

        # Lines are formatted as "     1\talpha" etc.
        assert "1" in result.llm_content
        assert "2" in result.llm_content
        assert "3" in result.llm_content

    async def test_result_header_contains_filename(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "myfile.py"
        f.write_text("x = 1\n")

        result = await tool.execute({"path": str(f)})

        assert "myfile.py" in result.llm_content

    async def test_result_header_contains_line_count(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "counted.txt"
        f.write_text("a\nb\nc\n")

        result = await tool.execute({"path": str(f)})

        assert "3 lines" in result.llm_content

    async def test_display_content_populated(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "disp.txt"
        f.write_text("hello\n")

        result = await tool.execute({"path": str(f)})

        assert result.display_content != ""


# ---------------------------------------------------------------------------
# execute — offset and limit
# ---------------------------------------------------------------------------


class TestExecuteOffsetLimit:
    """execute should slice lines correctly using offset and limit."""

    async def test_offset_skips_leading_lines(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "offset.txt"
        f.write_text("line1\nline2\nline3\nline4\n")

        result = await tool.execute({"path": str(f), "offset": 3})

        assert "line3" in result.llm_content
        assert "line4" in result.llm_content
        assert "line1" not in result.llm_content
        assert "line2" not in result.llm_content

    async def test_limit_caps_output(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "limit.txt"
        f.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")

        result = await tool.execute({"path": str(f), "limit": 2})

        assert "alpha" in result.llm_content
        assert "beta" in result.llm_content
        assert "gamma" not in result.llm_content

    async def test_offset_and_limit_combined(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "slice.txt"
        lines = [f"line{i}" for i in range(1, 11)]
        f.write_text("\n".join(lines) + "\n")

        # Read lines 3-5 (offset=3, limit=3)
        result = await tool.execute({"path": str(f), "offset": 3, "limit": 3})

        assert "line3" in result.llm_content
        assert "line5" in result.llm_content
        assert "line1" not in result.llm_content
        assert "line6" not in result.llm_content

    async def test_header_shows_range_when_sliced(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "range.txt"
        f.write_text("a\nb\nc\nd\ne\n")

        result = await tool.execute({"path": str(f), "offset": 2, "limit": 2})

        # Header should indicate partial view
        assert "showing lines" in result.llm_content

    async def test_no_range_annotation_when_reading_all(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "all.txt"
        f.write_text("only\nthis\n")

        result = await tool.execute({"path": str(f)})

        assert "showing lines" not in result.llm_content

    async def test_offset_beyond_file_returns_empty_content(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "short.txt"
        f.write_text("just one line\n")

        result = await tool.execute({"path": str(f), "offset": 100})

        assert not result.is_error
        # llm_content exists but the body after the header is empty
        assert "File:" in result.llm_content


# ---------------------------------------------------------------------------
# execute — error cases
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    """execute should return error ToolResults for unreadable inputs."""

    async def test_file_not_found(self, tool: ReadFileTool, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.txt"

        result = await tool.execute({"path": str(missing)})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "not found" in result.llm_content.lower() or "Error" in result.llm_content

    async def test_file_not_found_llm_content_mentions_path(self, tool: ReadFileTool, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.txt"

        result = await tool.execute({"path": str(missing)})

        assert str(missing) in result.llm_content

    async def test_directory_instead_of_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        result = await tool.execute({"path": str(tmp_path)})

        assert result.is_error
        assert "not a file" in result.llm_content.lower() or "Not a file" in result.llm_content

    async def test_binary_file_detection(self, tool: ReadFileTool, tmp_path: Path) -> None:
        binary_file = tmp_path / "data.bin"
        # Write bytes that are invalid UTF-8
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

        result = await tool.execute({"path": str(binary_file)})

        assert result.is_error
        assert "binary" in result.llm_content.lower()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod-based permission denial not reliable on Windows",
    )
    async def test_permission_denied(self, tool: ReadFileTool, tmp_path: Path) -> None:
        protected = tmp_path / "secret.txt"
        protected.write_text("top secret\n")
        # Remove all read permissions
        protected.chmod(0o000)

        try:
            result = await tool.execute({"path": str(protected)})
            assert result.is_error
            assert "permission" in result.llm_content.lower()
        finally:
            # Restore so tmp_path cleanup can delete the file
            protected.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# execute — abort_signal ignored (no-op, but should not crash)
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_execute_with_abort_signal_set(self, tool: ReadFileTool, tmp_path: Path) -> None:
        import asyncio

        f = tmp_path / "abort.txt"
        f.write_text("content\n")
        signal = asyncio.Event()
        signal.set()  # Already triggered

        # ReadFileTool does not check the signal, but it must not crash
        result = await tool.execute({"path": str(f)}, abort_signal=signal)
        assert not result.is_error
