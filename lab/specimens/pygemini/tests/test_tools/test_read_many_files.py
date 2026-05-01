"""Tests for pygemini.tools.filesystem.read_many_files."""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from pygemini.tools.base import ToolResult
from pygemini.tools.filesystem.read_many_files import ReadManyFilesTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> ReadManyFilesTool:
    return ReadManyFilesTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: ReadManyFilesTool) -> None:
        assert tool.name == "read_many_files"

    def test_description_mentions_multiple(self, tool: ReadManyFilesTool) -> None:
        assert "multiple" in tool.description.lower() or "many" in tool.description.lower()

    def test_description_mentions_line_numbers(self, tool: ReadManyFilesTool) -> None:
        assert "line numbers" in tool.description.lower()

    def test_parameter_schema_has_paths(self, tool: ReadManyFilesTool) -> None:
        assert "paths" in tool.parameter_schema["properties"]

    def test_parameter_schema_paths_is_array(self, tool: ReadManyFilesTool) -> None:
        assert tool.parameter_schema["properties"]["paths"]["type"] == "array"

    def test_parameter_schema_required_is_paths(self, tool: ReadManyFilesTool) -> None:
        assert tool.parameter_schema["required"] == ["paths"]

    def test_to_function_declaration_structure(self, tool: ReadManyFilesTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "read_many_files"
        assert "description" in decl
        assert decl["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should reject missing or empty paths."""

    def test_missing_paths_key_returns_error(self, tool: ReadManyFilesTool) -> None:
        result = tool.validate_params({})
        assert result is not None
        assert "paths" in result.lower()

    def test_empty_list_returns_error(self, tool: ReadManyFilesTool) -> None:
        result = tool.validate_params({"paths": []})
        assert result is not None

    def test_non_list_returns_error(self, tool: ReadManyFilesTool) -> None:
        result = tool.validate_params({"paths": "/single/path"})
        assert result is not None

    def test_valid_single_path(self, tool: ReadManyFilesTool) -> None:
        assert tool.validate_params({"paths": ["/some/file.py"]}) is None

    def test_valid_multiple_paths(self, tool: ReadManyFilesTool) -> None:
        assert tool.validate_params({"paths": ["/a.py", "/b.py", "/c.py"]}) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show file count and filenames."""

    def test_single_file_shows_filename(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({"paths": ["/foo/bar.py"]})
        assert "bar.py" in desc

    def test_single_file_count(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({"paths": ["/foo/bar.py"]})
        assert "1" in desc

    def test_multiple_files_shows_count(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({"paths": ["/a.py", "/b.py", "/c.py"]})
        assert "3" in desc

    def test_multiple_files_shows_first_filenames(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({"paths": ["/a.py", "/b.py", "/c.py"]})
        assert "a.py" in desc
        assert "b.py" in desc

    def test_more_than_three_shows_overflow(self, tool: ReadManyFilesTool) -> None:
        paths = [f"/file{i}.py" for i in range(5)]
        desc = tool.get_description({"paths": paths})
        assert "+" in desc or "more" in desc.lower()

    def test_empty_paths_shows_zero(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({"paths": []})
        assert "0" in desc

    def test_missing_paths_key(self, tool: ReadManyFilesTool) -> None:
        desc = tool.get_description({})
        assert "0" in desc


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute should return combined contents of all requested files."""

    async def test_reads_single_file(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\n")

        result = await tool.execute({"paths": [str(f)]})

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "line one" in result.llm_content
        assert "line two" in result.llm_content

    async def test_reads_multiple_files(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "alpha.txt"
        f1.write_text("content alpha\n")
        f2 = tmp_path / "beta.txt"
        f2.write_text("content beta\n")

        result = await tool.execute({"paths": [str(f1), str(f2)]})

        assert not result.is_error
        assert "content alpha" in result.llm_content
        assert "content beta" in result.llm_content

    async def test_file_headers_present(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "myfile.py"
        f.write_text("x = 1\n")

        result = await tool.execute({"paths": [str(f)]})

        assert "myfile.py" in result.llm_content

    async def test_file_header_shows_line_count(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "counted.txt"
        f.write_text("a\nb\nc\n")

        result = await tool.execute({"paths": [str(f)]})

        assert "3 lines" in result.llm_content

    async def test_line_numbers_present(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "numbered.txt"
        f.write_text("alpha\nbeta\ngamma\n")

        result = await tool.execute({"paths": [str(f)]})

        # Lines formatted as "     1\talpha" etc.
        assert "1" in result.llm_content
        assert "2" in result.llm_content
        assert "3" in result.llm_content

    async def test_each_file_has_separator_header(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "first.txt"
        f1.write_text("first content\n")
        f2 = tmp_path / "second.txt"
        f2.write_text("second content\n")

        result = await tool.execute({"paths": [str(f1), str(f2)]})

        # Each file should have a "--- File: ... ---" header
        assert "--- File:" in result.llm_content

    async def test_display_content_populated(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "disp.txt"
        f.write_text("some text\n")

        result = await tool.execute({"paths": [str(f)]})

        assert result.display_content != ""


# ---------------------------------------------------------------------------
# execute — partial failure
# ---------------------------------------------------------------------------


class TestExecutePartialFailure:
    """When some files fail, the others should still be read."""

    async def test_missing_file_does_not_stop_reading(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("good content\n")
        missing = tmp_path / "missing.txt"  # intentionally not created

        result = await tool.execute({"paths": [str(missing), str(good)]})

        assert not result.is_error
        assert "good content" in result.llm_content

    async def test_missing_file_error_in_output(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("ok\n")
        missing = tmp_path / "ghost.txt"

        result = await tool.execute({"paths": [str(missing), str(good)]})

        assert "not found" in result.llm_content.lower() or "Error" in result.llm_content

    async def test_binary_file_error_in_output(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("readable\n")
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

        result = await tool.execute({"paths": [str(binary), str(good)]})

        assert not result.is_error
        assert "readable" in result.llm_content

    async def test_partial_failure_is_not_error(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("data\n")
        bad = tmp_path / "nonexistent.txt"

        result = await tool.execute({"paths": [str(bad), str(good)]})

        assert not result.is_error

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod-based permission denial not reliable on Windows",
    )
    async def test_permission_denied_file_continues(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("accessible\n")
        protected = tmp_path / "secret.txt"
        protected.write_text("top secret\n")
        protected.chmod(0o000)

        try:
            result = await tool.execute({"paths": [str(protected), str(good)]})
            assert not result.is_error
            assert "accessible" in result.llm_content
            assert "permission" in result.llm_content.lower()
        finally:
            protected.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# execute — all failures
# ---------------------------------------------------------------------------


class TestExecuteAllFailure:
    """When all files fail, is_error should be True."""

    async def test_all_missing_is_error(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"

        result = await tool.execute({"paths": [str(a), str(b)]})

        assert result.is_error

    async def test_all_missing_error_mentions_files(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        missing = tmp_path / "ghost.txt"

        result = await tool.execute({"paths": [str(missing)]})

        assert result.is_error
        assert str(missing) in result.llm_content

    async def test_all_binary_is_error(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

        result = await tool.execute({"paths": [str(binary)]})

        assert result.is_error


# ---------------------------------------------------------------------------
# execute — abort_signal (no-op, must not crash)
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_execute_with_abort_signal_set(
        self, tool: ReadManyFilesTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "abort.txt"
        f.write_text("content\n")
        signal = asyncio.Event()
        signal.set()

        result = await tool.execute({"paths": [str(f)]}, abort_signal=signal)

        assert not result.is_error
