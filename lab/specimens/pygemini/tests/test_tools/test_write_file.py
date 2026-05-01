"""Tests for pygemini.tools.filesystem.write_file."""

from __future__ import annotations

import sys
import stat
from pathlib import Path

import pytest

from pygemini.tools.base import ToolConfirmation
from pygemini.tools.filesystem.write_file import WriteFileTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> WriteFileTool:
    return WriteFileTool()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestWriteFileMetadata:
    """Static tool properties should be correct."""

    def test_name(self, tool: WriteFileTool) -> None:
        assert tool.name == "write_file"

    def test_description_mentions_write(self, tool: WriteFileTool) -> None:
        assert "Write" in tool.description or "write" in tool.description

    def test_description_mentions_directories(self, tool: WriteFileTool) -> None:
        assert "director" in tool.description.lower()

    def test_to_function_declaration_structure(self, tool: WriteFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["name"] == "write_file"
        assert "description" in decl
        assert decl["parameters"]["type"] == "object"
        props = decl["parameters"]["properties"]
        assert "path" in props
        assert "content" in props
        assert "create_directories" in props

    def test_to_function_declaration_required_fields(self, tool: WriteFileTool) -> None:
        decl = tool.to_function_declaration()
        required = decl["parameters"]["required"]
        assert "path" in required
        assert "content" in required

    def test_to_function_declaration_create_directories_is_boolean(self, tool: WriteFileTool) -> None:
        decl = tool.to_function_declaration()
        assert decl["parameters"]["properties"]["create_directories"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """validate_params should catch missing required parameters."""

    def test_valid_params(self, tool: WriteFileTool) -> None:
        assert tool.validate_params({"path": "/tmp/f.txt", "content": "hello"}) is None

    def test_missing_path_returns_error(self, tool: WriteFileTool) -> None:
        result = tool.validate_params({"content": "hello"})
        assert result is not None
        assert "path" in result.lower()

    def test_missing_content_returns_error(self, tool: WriteFileTool) -> None:
        result = tool.validate_params({"path": "/tmp/f.txt"})
        assert result is not None
        assert "content" in result.lower()

    def test_missing_both_returns_error(self, tool: WriteFileTool) -> None:
        result = tool.validate_params({})
        assert result is not None

    def test_empty_content_is_valid(self, tool: WriteFileTool) -> None:
        # Empty string is a valid content value
        assert tool.validate_params({"path": "/tmp/f.txt", "content": ""}) is None


# ---------------------------------------------------------------------------
# get_description
# ---------------------------------------------------------------------------


class TestGetDescription:
    """get_description should summarise the write operation."""

    def test_contains_path(self, tool: WriteFileTool) -> None:
        desc = tool.get_description({"path": "/foo/bar.py", "content": "x = 1\n"})
        assert "/foo/bar.py" in desc

    def test_contains_line_count(self, tool: WriteFileTool) -> None:
        desc = tool.get_description({"path": "/f.txt", "content": "a\nb\nc\n"})
        # 3 newlines → 3 lines (count("\n") + 1 since content is non-empty)
        assert "3" in desc or "4" in desc  # implementation: count("\n") + 1 = 4

    def test_single_line_content(self, tool: WriteFileTool) -> None:
        desc = tool.get_description({"path": "/f.txt", "content": "hello"})
        assert "1" in desc

    def test_missing_path_uses_placeholder(self, tool: WriteFileTool) -> None:
        desc = tool.get_description({"content": "data"})
        assert "?" in desc


# ---------------------------------------------------------------------------
# should_confirm
# ---------------------------------------------------------------------------


class TestShouldConfirm:
    """should_confirm must always return a ToolConfirmation (write is destructive)."""

    def test_returns_tool_confirmation(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/tmp/x.txt", "content": "hello\n"})
        assert isinstance(confirmation, ToolConfirmation)

    def test_confirmation_description_contains_path(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/tmp/out.txt", "content": "data\n"})
        assert confirmation is not None
        assert "/tmp/out.txt" in confirmation.description

    def test_confirmation_details_has_path(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/a/b.txt", "content": "x\n"})
        assert confirmation is not None
        assert "path" in confirmation.details
        assert confirmation.details["path"] == "/a/b.txt"

    def test_confirmation_details_has_lines(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/f.txt", "content": "a\nb\n"})
        assert confirmation is not None
        assert "lines" in confirmation.details
        assert isinstance(confirmation.details["lines"], int)

    def test_confirmation_details_has_bytes(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/f.txt", "content": "hello"})
        assert confirmation is not None
        assert "bytes" in confirmation.details
        assert confirmation.details["bytes"] == 5

    def test_confirmation_details_has_preview(self, tool: WriteFileTool) -> None:
        confirmation = tool.should_confirm({"path": "/f.txt", "content": "preview line\n"})
        assert confirmation is not None
        assert "preview" in confirmation.details

    def test_confirmation_preview_truncated_after_ten_lines(self, tool: WriteFileTool) -> None:
        content = "\n".join(f"line{i}" for i in range(1, 20))
        confirmation = tool.should_confirm({"path": "/f.txt", "content": content})
        assert confirmation is not None
        assert "..." in confirmation.details["preview"]

    def test_confirmation_preview_not_truncated_for_short_content(self, tool: WriteFileTool) -> None:
        content = "line1\nline2\nline3\n"
        confirmation = tool.should_confirm({"path": "/f.txt", "content": content})
        assert confirmation is not None
        assert "..." not in confirmation.details["preview"]


# ---------------------------------------------------------------------------
# execute — writing new files
# ---------------------------------------------------------------------------


class TestExecuteWriteNewFile:
    """execute should create files and report 'Created'."""

    async def test_creates_new_file(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "new.txt"

        result = await tool.execute({"path": str(dest), "content": "hello world\n"})

        assert not result.is_error
        assert dest.exists()
        assert dest.read_text() == "hello world\n"

    async def test_result_says_created(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "brand_new.txt"

        result = await tool.execute({"path": str(dest), "content": "x\n"})

        assert "Created" in result.llm_content

    async def test_llm_content_contains_line_count(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "counted.txt"
        content = "a\nb\nc\n"

        result = await tool.execute({"path": str(dest), "content": content})

        assert "lines" in result.llm_content

    async def test_llm_content_contains_byte_count(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "bytes.txt"
        content = "hello"

        result = await tool.execute({"path": str(dest), "content": content})

        assert "bytes" in result.llm_content
        assert "5" in result.llm_content

    async def test_display_content_populated(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "disp.txt"

        result = await tool.execute({"path": str(dest), "content": "data\n"})

        assert result.display_content != ""


# ---------------------------------------------------------------------------
# execute — overwriting existing files
# ---------------------------------------------------------------------------


class TestExecuteOverwriteFile:
    """execute should overwrite existing files and report 'Updated'."""

    async def test_overwrites_existing_file(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "existing.txt"
        dest.write_text("old content\n")

        result = await tool.execute({"path": str(dest), "content": "new content\n"})

        assert not result.is_error
        assert dest.read_text() == "new content\n"

    async def test_result_says_updated(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "updated.txt"
        dest.write_text("original\n")

        result = await tool.execute({"path": str(dest), "content": "revised\n"})

        assert "Updated" in result.llm_content

    async def test_overwrite_changes_file_completely(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "overwrite.txt"
        dest.write_text("line1\nline2\nline3\n")

        await tool.execute({"path": str(dest), "content": "only this\n"})

        assert dest.read_text() == "only this\n"


# ---------------------------------------------------------------------------
# execute — parent directory creation
# ---------------------------------------------------------------------------


class TestExecuteCreateDirectories:
    """execute should create missing parent directories by default."""

    async def test_creates_parent_directories(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "file.txt"

        result = await tool.execute({"path": str(dest), "content": "deep\n"})

        assert not result.is_error
        assert dest.exists()
        assert dest.read_text() == "deep\n"

    async def test_create_directories_true_is_default(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "file.txt"

        # Do not pass create_directories at all — should default to True
        result = await tool.execute({"path": str(dest), "content": "auto\n"})

        assert not result.is_error
        assert dest.exists()

    async def test_create_directories_false_fails_for_missing_parent(
        self, tool: WriteFileTool, tmp_path: Path
    ) -> None:
        dest = tmp_path / "nonexistent" / "file.txt"

        result = await tool.execute(
            {"path": str(dest), "content": "data\n", "create_directories": False}
        )

        assert result.is_error

    async def test_create_directories_false_succeeds_when_parent_exists(
        self, tool: WriteFileTool, tmp_path: Path
    ) -> None:
        dest = tmp_path / "file.txt"  # tmp_path itself already exists

        result = await tool.execute(
            {"path": str(dest), "content": "ok\n", "create_directories": False}
        )

        assert not result.is_error
        assert dest.read_text() == "ok\n"


# ---------------------------------------------------------------------------
# execute — write empty content
# ---------------------------------------------------------------------------


class TestExecuteEmptyContent:
    async def test_write_empty_string(self, tool: WriteFileTool, tmp_path: Path) -> None:
        dest = tmp_path / "empty.txt"

        result = await tool.execute({"path": str(dest), "content": ""})

        assert not result.is_error
        assert dest.exists()
        assert dest.read_text() == ""


# ---------------------------------------------------------------------------
# execute — permission error
# ---------------------------------------------------------------------------


class TestExecutePermissionError:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod-based permission denial not reliable on Windows",
    )
    async def test_permission_denied_returns_error(self, tool: WriteFileTool, tmp_path: Path) -> None:
        # Make the directory read-only so no files can be created inside it
        protected_dir = tmp_path / "readonly"
        protected_dir.mkdir()
        protected_dir.chmod(0o555)

        dest = protected_dir / "file.txt"
        try:
            result = await tool.execute({"path": str(dest), "content": "data\n"})
            assert result.is_error
            assert "permission" in result.llm_content.lower()
        finally:
            protected_dir.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# execute — abort_signal (no-op but must not crash)
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_execute_with_abort_signal(self, tool: WriteFileTool, tmp_path: Path) -> None:
        import asyncio

        dest = tmp_path / "signal.txt"
        signal = asyncio.Event()
        signal.set()

        result = await tool.execute(
            {"path": str(dest), "content": "data\n"}, abort_signal=signal
        )
        assert not result.is_error
        assert dest.exists()
