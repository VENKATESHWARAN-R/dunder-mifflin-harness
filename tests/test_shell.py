import asyncio
import shlex
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


def test_run_shell_timeout_kills_child_process_group(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)' & wait"

    started_at = time.monotonic()
    result = asyncio.run(
        asyncio.wait_for(
            run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1),
            timeout=2.0,
        )
    )

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out
    assert time.monotonic() - started_at < 2.0


def test_run_shell_output_capture_is_bounded(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 10000)'"

    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), max_output_chars=500)
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert "truncated" in result.stdout
    assert len(result.stdout) < 600


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
