import asyncio
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


def test_run_shell_timeout_kills_process_group_with_open_pipes(tmp_path: Path) -> None:
    started_at = time.monotonic()

    result = asyncio.run(
        run_shell(
            command="sh -c 'sleep 60 & wait'",
            cwd=str(tmp_path),
            timeout_seconds=0.1,
        )
    )

    assert time.monotonic() - started_at < 3
    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_spawn_error_returns_structured_result(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(run_shell(command="echo hello", cwd=str(missing_cwd)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert result.cwd == str(missing_cwd)
    assert result.error


def test_run_shell_caps_large_output(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(
            command="python3 -c 'import sys; sys.stdout.write(\"x\" * 1000000)'",
            cwd=str(tmp_path),
            max_output_chars=1000,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) <= 1000


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
