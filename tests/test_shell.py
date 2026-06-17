import asyncio
import shlex
import sys
import time
from pathlib import Path

from dunder_mifflin_harness.tools.shell import (
    read_process_output,
    run_shell,
    run_shell_background,
    truncate_output,
)
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


def test_run_shell_timeout_kills_child_process_holding_pipe(tmp_path: Path) -> None:
    command = (
        f"{shlex.quote(sys.executable)} -c "
        "'import subprocess,time; subprocess.Popen([\"sleep\",\"5\"]); time.sleep(5)'"
    )

    started_at = time.monotonic()
    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1)
    )
    elapsed = time.monotonic() - started_at

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out
    assert elapsed < 2


def test_run_shell_missing_cwd_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = asyncio.run(run_shell(command="pwd", cwd=str(missing)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert "failed to start command" in (result.error or "")


def test_run_shell_truncates_large_output(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 20000)'"

    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), max_output_chars=1000)
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) < 1200


def test_background_shell_output_is_capped(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 2000000)'"

    async def run() -> tuple[int, str, list[str]]:
        started = await run_shell_background(command=command, cwd=str(tmp_path))
        assert started.status == ToolStatus.OK
        for _ in range(50):
            output = await read_process_output(started.process_id, max_output_chars=2000)
            if not output.running:
                return len(output.stdout), output.stdout, output.warnings
            await asyncio.sleep(0.05)
        raise AssertionError("background process did not finish")

    output_len, stdout, warnings = asyncio.run(run())

    assert output_len < 3000
    assert "output truncated" in stdout
    assert warnings == ["output was truncated"]


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
