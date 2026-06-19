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


def test_run_shell_timeout_kills_child_holding_stdout(tmp_path: Path) -> None:
    script = "import time; print('ready', flush=True); time.sleep(5)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    started = time.monotonic()
    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1)
    )

    assert time.monotonic() - started < 1.0
    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_bad_cwd_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = asyncio.run(run_shell(command="pwd", cwd=str(missing)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert "failed to start command" in (result.error or "")


def test_run_shell_caps_large_output(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(
            command=f"{shlex.quote(sys.executable)} -c \"print('x' * 1000)\"",
            cwd=str(tmp_path),
            max_output_chars=100,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) <= 100


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
