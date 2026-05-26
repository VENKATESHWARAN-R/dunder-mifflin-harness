import asyncio
import sys
import time
from pathlib import Path

from dunder_mifflin_harness.tools.shell import run_shell, truncate_output
from dunder_mifflin_harness.tools.types import ToolStatus


def test_run_shell_captures_output(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(command="printf hello", cwd=str(tmp_path))
    )

    assert result.status == ToolStatus.OK
    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.succeeded


def test_run_shell_nonzero_exit_sets_error_status(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(command="exit 1", cwd=str(tmp_path))
    )

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == 1
    assert not result.succeeded


def test_run_shell_timeout(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(command="sleep 10", cwd=str(tmp_path), timeout_seconds=0.1)
    )

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_timeout_kills_child_processes_holding_pipes(tmp_path: Path) -> None:
    command = (
        f"{sys.executable} -c "
        "\"import subprocess, time; subprocess.Popen(['sleep', '2']); time.sleep(2)\""
    )

    started = time.monotonic()
    result = asyncio.run(run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1))
    elapsed = time.monotonic() - started

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out
    assert elapsed < 1.0


def test_run_shell_spawn_error_returns_tool_error(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(run_shell(command="printf hello", cwd=str(missing_cwd)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert not result.succeeded
    assert result.error


def test_run_shell_truncates_large_output_without_unbounded_capture(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(
            command=f"{sys.executable} -c \"print('x' * 20000)\"",
            cwd=str(tmp_path),
            max_output_chars=1000,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert result.warnings == ["output was truncated"]
    assert len(result.stdout) <= 1000


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
