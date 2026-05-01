"""Tests for pygemini.tools.filesystem.edit_file."""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from pygemini.tools.base import ToolConfirmation, ToolResult
from pygemini.tools.filesystem.edit_file import EditFileTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> EditFileTool:
    return EditFileTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: EditFileTool) -> None:
        assert tool.name == "edit_file"

    def test_description_mentions_replace(self, tool: EditFileTool) -> None:
        assert "replac" in tool.description.lower()

    def test_description_mentions_exact(self, tool: EditFileTool) -> None:
        assert "exact" in tool.description.lower()

    def test_parameter_schema_has_path(self, tool: EditFileTool) -> None:
        assert "path" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_old_string(self, tool: EditFileTool) -> None:
        assert "old_string" in tool.parameter_schema["properties"]

    def test_parameter_schema_has_new_string(self, tool: EditFileTool) -> None:
        assert "new_string" in tool.parameter_schema["properties"]

    def test_parameter_schema_required_fields(self, tool: EditFileTool) -> None:
        required = tool.parameter_schema["required"]
        assert "path" in required
        assert "old_string" in required
        assert "new_string" in required

    def test_to_function_declaration_structure(self, tool: EditFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "edit_file"
        assert "description" in decl
        assert decl["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should catch bad inputs before execution."""

    def test_missing_path_returns_error(self, tool: EditFileTool) -> None:
        result = tool.validate_params({"old_string": "a", "new_string": "b"})
        assert result is not None
        assert "path" in result.lower()

    def test_missing_old_string_returns_error(self, tool: EditFileTool) -> None:
        result = tool.validate_params({"path": "/f", "new_string": "b"})
        assert result is not None
        assert "old_string" in result

    def test_missing_new_string_returns_error(self, tool: EditFileTool) -> None:
        result = tool.validate_params({"path": "/f", "old_string": "a"})
        assert result is not None
        assert "new_string" in result

    def test_empty_old_string_returns_error(self, tool: EditFileTool) -> None:
        result = tool.validate_params({"path": "/f", "old_string": "", "new_string": "b"})
        assert result is not None
        assert "old_string" in result

    def test_same_old_and_new_returns_error(self, tool: EditFileTool) -> None:
        result = tool.validate_params(
            {"path": "/f", "old_string": "same", "new_string": "same"}
        )
        assert result is not None

    def test_valid_params_returns_none(self, tool: EditFileTool) -> None:
        assert (
            tool.validate_params(
                {"path": "/f", "old_string": "old content", "new_string": "new content"}
            )
            is None
        )

    def test_empty_new_string_is_valid(self, tool: EditFileTool) -> None:
        # Deletion (replacing with empty string) is allowed
        assert (
            tool.validate_params({"path": "/f", "old_string": "something", "new_string": ""})
            is None
        )


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should show char counts for old and new strings."""

    def test_shows_path(self, tool: EditFileTool) -> None:
        desc = tool.get_description(
            {"path": "/foo/bar.py", "old_string": "abc", "new_string": "xyz"}
        )
        assert "/foo/bar.py" in desc

    def test_shows_old_char_count(self, tool: EditFileTool) -> None:
        desc = tool.get_description(
            {"path": "/f", "old_string": "abcde", "new_string": "x"}
        )
        assert "5" in desc

    def test_shows_new_char_count(self, tool: EditFileTool) -> None:
        desc = tool.get_description(
            {"path": "/f", "old_string": "a", "new_string": "xyz"}
        )
        assert "3" in desc

    def test_missing_path_uses_placeholder(self, tool: EditFileTool) -> None:
        desc = tool.get_description({"old_string": "a", "new_string": "b"})
        assert "?" in desc

    def test_missing_strings_default_to_zero_chars(self, tool: EditFileTool) -> None:
        desc = tool.get_description({"path": "/f"})
        assert "0" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    """should_confirm should always return a ToolConfirmation with path and strings."""

    def test_returns_tool_confirmation(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/f", "old_string": "old", "new_string": "new"}
        )
        assert isinstance(result, ToolConfirmation)

    def test_description_contains_path(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/my/file.py", "old_string": "old", "new_string": "new"}
        )
        assert result is not None
        assert "/my/file.py" in result.description

    def test_details_has_path_key(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/f.py", "old_string": "old", "new_string": "new"}
        )
        assert result is not None
        assert "path" in result.details

    def test_details_has_old_string(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/f", "old_string": "old content", "new_string": "new"}
        )
        assert result is not None
        assert "old_string" in result.details

    def test_details_has_new_string(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/f", "old_string": "old", "new_string": "new content"}
        )
        assert result is not None
        assert "new_string" in result.details

    def test_long_old_string_is_truncated(self, tool: EditFileTool) -> None:
        long_string = "x" * 300
        result = tool.should_confirm(
            {"path": "/f", "old_string": long_string, "new_string": "new"}
        )
        assert result is not None
        # The stored old_string should be shorter than the original (truncated)
        assert len(result.details["old_string"]) < len(long_string)
        assert result.details["old_string"].endswith("...")

    def test_long_new_string_is_truncated(self, tool: EditFileTool) -> None:
        long_string = "y" * 300
        result = tool.should_confirm(
            {"path": "/f", "old_string": "old", "new_string": long_string}
        )
        assert result is not None
        assert len(result.details["new_string"]) < len(long_string)
        assert result.details["new_string"].endswith("...")

    def test_short_strings_not_truncated(self, tool: EditFileTool) -> None:
        result = tool.should_confirm(
            {"path": "/f", "old_string": "short old", "new_string": "short new"}
        )
        assert result is not None
        assert result.details["old_string"] == "short old"
        assert result.details["new_string"] == "short new"


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute should replace the text and write the file."""

    async def test_basic_replacement(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 2\nz = 3\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "y = 2", "new_string": "y = 42"}
        )

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert f.read_text() == "x = 1\ny = 42\nz = 3\n"

    async def test_file_is_written_after_replace(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "target.txt"
        f.write_text("hello world\n")

        await tool.execute(
            {"path": str(f), "old_string": "hello", "new_string": "goodbye"}
        )

        assert "goodbye" in f.read_text()
        assert "hello" not in f.read_text()

    async def test_result_mentions_file_path(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "myfile.txt"
        f.write_text("foo bar baz\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "foo", "new_string": "qux"}
        )

        assert str(f) in result.llm_content

    async def test_result_mentions_char_counts(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "chars.txt"
        f.write_text("abc def\n")
        old_s = "abc"
        new_s = "xyz"

        result = await tool.execute(
            {"path": str(f), "old_string": old_s, "new_string": new_s}
        )

        assert str(len(old_s)) in result.llm_content
        assert str(len(new_s)) in result.llm_content

    async def test_diff_in_output(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "diff_test.txt"
        f.write_text("line one\nline two\nline three\n")

        result = await tool.execute(
            {
                "path": str(f),
                "old_string": "line two",
                "new_string": "line TWO",
            }
        )

        # Unified diff markers should appear in the LLM content
        assert "-" in result.llm_content
        assert "+" in result.llm_content

    async def test_display_content_populated(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "disp.txt"
        f.write_text("alpha\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "alpha", "new_string": "beta"}
        )

        assert result.display_content != ""

    async def test_multiline_replacement(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "multiline.py"
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")

        result = await tool.execute(
            {
                "path": str(f),
                "old_string": "def foo():\n    return 1",
                "new_string": "def foo():\n    return 99",
            }
        )

        assert not result.is_error
        assert "return 99" in f.read_text()


# ---------------------------------------------------------------------------
# execute — old_string not found
# ---------------------------------------------------------------------------


class TestExecuteNotFound:
    """execute should error when old_string is absent from the file."""

    async def test_not_found_is_error(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("some content\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "DOES NOT EXIST", "new_string": "replacement"}
        )

        assert result.is_error

    async def test_not_found_message(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("some content\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "missing text", "new_string": "x"}
        )

        assert "not found" in result.llm_content.lower() or "old_string" in result.llm_content

    async def test_file_unchanged_when_not_found(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        original = "original content\n"
        f = tmp_path / "unchanged.txt"
        f.write_text(original)

        await tool.execute(
            {"path": str(f), "old_string": "ghost", "new_string": "replacement"}
        )

        assert f.read_text() == original


# ---------------------------------------------------------------------------
# execute — multiple matches
# ---------------------------------------------------------------------------


class TestExecuteMultipleMatches:
    """execute should error when old_string matches more than once."""

    async def test_multiple_matches_is_error(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "dupes.txt"
        f.write_text("repeat\nrepeat\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "repeat", "new_string": "unique"}
        )

        assert result.is_error

    async def test_multiple_matches_message_contains_count(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "dupes.txt"
        f.write_text("dup dup dup\n")

        result = await tool.execute(
            {"path": str(f), "old_string": "dup", "new_string": "x"}
        )

        assert "3" in result.llm_content

    async def test_file_unchanged_when_multiple_matches(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        original = "word word\n"
        f = tmp_path / "unchanged.txt"
        f.write_text(original)

        await tool.execute(
            {"path": str(f), "old_string": "word", "new_string": "other"}
        )

        assert f.read_text() == original


# ---------------------------------------------------------------------------
# execute — file error cases
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    """execute should return error ToolResults for unreadable/unwritable files."""

    async def test_file_not_found(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        missing = tmp_path / "ghost.txt"

        result = await tool.execute(
            {"path": str(missing), "old_string": "old", "new_string": "new"}
        )

        assert result.is_error
        assert "not found" in result.llm_content.lower() or "Error" in result.llm_content

    async def test_file_not_found_mentions_path(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        missing = tmp_path / "ghost.txt"

        result = await tool.execute(
            {"path": str(missing), "old_string": "old", "new_string": "new"}
        )

        assert str(missing) in result.llm_content

    async def test_binary_file_is_error(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

        result = await tool.execute(
            {"path": str(binary), "old_string": "PNG", "new_string": "GIF"}
        )

        assert result.is_error
        assert "binary" in result.llm_content.lower()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod-based permission denial not reliable on Windows",
    )
    async def test_permission_denied_read(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        protected = tmp_path / "secret.txt"
        protected.write_text("content\n")
        protected.chmod(0o000)

        try:
            result = await tool.execute(
                {"path": str(protected), "old_string": "content", "new_string": "replaced"}
            )
            assert result.is_error
            assert "permission" in result.llm_content.lower()
        finally:
            protected.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# execute — abort_signal (no-op, must not crash)
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_execute_with_abort_signal_set(
        self, tool: EditFileTool, tmp_path: Path
    ) -> None:
        f = tmp_path / "abort.txt"
        f.write_text("old content\n")
        signal = asyncio.Event()
        signal.set()

        result = await tool.execute(
            {"path": str(f), "old_string": "old content", "new_string": "new content"},
            abort_signal=signal,
        )

        assert not result.is_error
        assert "new content" in f.read_text()
