import asyncio
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


def test_run_shell_timeout_kills_child_processes(tmp_path: Path) -> None:
    start = time.monotonic()

    result = asyncio.run(
        run_shell(command="sh -c 'sleep 5'", cwd=str(tmp_path), timeout_seconds=0.1)
    )

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out
    assert time.monotonic() - start < 2


def test_run_shell_spawn_failure_returns_error(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(run_shell(command="printf hello", cwd=str(missing_cwd)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert result.error


def test_run_shell_captures_bounded_output(tmp_path: Path) -> None:
    result = asyncio.run(
        run_shell(
            command="yes x | head -c 20000",
            cwd=str(tmp_path),
            max_output_chars=1000,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) <= 1000


def test_run_shell_background_starts_and_captures_output(tmp_path: Path) -> None:
    async def run() -> tuple[ToolStatus, str]:
        started = await run_shell_background(command="printf hello", cwd=str(tmp_path))
        output = await read_process_output(started.process_id)
        for _ in range(10):
            if output.stdout == "hello":
                break
            await asyncio.sleep(0.05)
            output = await read_process_output(started.process_id)
        return started.status, output.stdout

    status, stdout = asyncio.run(run())

    assert status == ToolStatus.OK
    assert stdout == "hello"


def test_run_shell_background_spawn_failure_returns_error(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(
        run_shell_background(command="printf hello", cwd=str(missing_cwd))
    )

    assert result.status == ToolStatus.ERROR
    assert not result.process_id
    assert result.error


def test_read_process_output_is_bounded(tmp_path: Path) -> None:
    async def run() -> tuple[ToolStatus, bool, str]:
        started = await run_shell_background(
            command="python3 -c 'import sys; sys.stdout.write(\"x\" * 20000)'",
            cwd=str(tmp_path),
        )
        output = await read_process_output(started.process_id, max_output_chars=1000)
        for _ in range(20):
            if output.warnings or len(output.stdout) >= 1000:
                break
            await asyncio.sleep(0.05)
            output = await read_process_output(started.process_id, max_output_chars=1000)
        return started.status, bool(output.warnings), output.stdout

    status, warned, stdout = asyncio.run(run())

    assert status == ToolStatus.OK
    assert warned
    assert len(stdout) <= 1000


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
