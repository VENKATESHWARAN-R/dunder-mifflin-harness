"""Tests for pygemini.cli.input — @file and !command expansion."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch


from pygemini.cli.input import expand_file_references, expand_shell_commands, process_input


# ---------------------------------------------------------------------------
# TestExpandFileReferences
# ---------------------------------------------------------------------------


class TestExpandFileReferences:
    """Tests for expand_file_references()."""

    def test_single_file(self, tmp_path: Path) -> None:
        """@path/to/file is replaced with a fenced block containing the file contents."""
        f = tmp_path / "hello.txt"
        f.write_text("hello world\n")

        result = expand_file_references(f"Please review @{f}")

        assert "hello world" in result
        assert f"--- Contents of {f} ---" in result
        assert f"--- End of {f} ---" in result
        # The original @ref should be gone
        assert f"@{f}" not in result

    def test_multiple_files(self, tmp_path: Path) -> None:
        """Multiple @file refs in a single input are all expanded."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content-a")
        f2.write_text("content-b")

        result = expand_file_references(f"@{f1} and @{f2}")

        assert "content-a" in result
        assert "content-b" in result
        assert f"--- Contents of {f1} ---" in result
        assert f"--- Contents of {f2} ---" in result

    def test_file_not_found(self, tmp_path: Path) -> None:
        """@nonexistent.txt is replaced with a [Error: File not found: ...] message."""
        missing = tmp_path / "nonexistent.txt"

        result = expand_file_references(f"@{missing}")

        assert "Error: File not found" in result
        assert str(missing) in result

    def test_binary_file(self, tmp_path: Path) -> None:
        """@binary.bin is replaced with a [Error: Binary file cannot be read: ...] message."""
        binary = tmp_path / "binary.bin"
        binary.write_bytes(bytes(range(256)))

        result = expand_file_references(f"@{binary}")

        assert "Error: Binary file cannot be read" in result
        assert str(binary) in result

    def test_no_email_false_positive(self) -> None:
        """user@example.com must NOT be expanded (the @ is preceded by alphanumeric)."""
        text = "Contact user@example.com for help."

        result = expand_file_references(text)

        assert result == text

    def test_no_refs(self) -> None:
        """Plain text with no @ refs is returned unchanged."""
        text = "Just a plain message with no special syntax."

        result = expand_file_references(text)

        assert result == text


# ---------------------------------------------------------------------------
# TestExpandShellCommands
# ---------------------------------------------------------------------------


class TestExpandShellCommands:
    """Tests for expand_shell_commands()."""

    def test_simple_command(self) -> None:
        """!echo hello is replaced with the command's output."""
        result = expand_shell_commands("!echo hello")

        assert "hello" in result
        # The raw !echo should be gone
        assert "!echo hello" not in result

    def test_command_output_wrapped_in_markers(self) -> None:
        """Command output is wrapped in --- Output of ... --- markers."""
        result = expand_shell_commands("!echo hi")

        assert "--- Output of `echo hi` ---" in result
        assert "--- End of command output ---" in result

    def test_failed_command(self) -> None:
        """A command that exits non-zero still returns its stderr in the output block."""
        # `false` exits with code 1; `ls /no_such_dir_xyz` writes to stderr
        result = expand_shell_commands("!ls /no_such_dir_xyz_pygemini")

        # We get a wrapped output block (not an exception), and stderr is present
        assert "--- Output of" in result
        assert "--- End of command output ---" in result

    def test_timeout(self) -> None:
        """Commands that exceed the timeout are replaced with a timeout error message."""
        timeout_exc = subprocess.TimeoutExpired(cmd="sleep 100", timeout=10)

        with patch("pygemini.cli.input.subprocess.run", side_effect=timeout_exc):
            result = expand_shell_commands("!sleep 100")

        assert "Error: Command timed out after 10s" in result
        assert "sleep 100" in result

    def test_not_at_start(self) -> None:
        """!command embedded mid-line (not at start) is NOT expanded."""
        text = "some text !echo embedded"

        result = expand_shell_commands(text)

        assert result == text

    def test_no_commands(self) -> None:
        """Plain text with no ! lines is returned unchanged."""
        text = "No commands here at all."

        result = expand_shell_commands(text)

        assert result == text


# ---------------------------------------------------------------------------
# TestProcessInput
# ---------------------------------------------------------------------------


class TestProcessInput:
    """Tests for process_input(), which applies both expansion passes."""

    def test_both_expansions(self, tmp_path: Path) -> None:
        """Input containing both @file and !command is fully expanded."""
        f = tmp_path / "note.txt"
        f.write_text("file contents here")

        text = f"@{f}\n!echo done"
        result = process_input(text)

        assert "file contents here" in result
        assert "done" in result
        assert "--- Contents of" in result
        assert "--- Output of" in result

    def test_plain_text(self) -> None:
        """Plain text with no special syntax passes through unchanged."""
        text = "Nothing special here."

        result = process_input(text)

        assert result == text
