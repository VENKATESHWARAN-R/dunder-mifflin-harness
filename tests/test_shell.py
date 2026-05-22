import asyncio
import shlex
import sys
import time
from pathlib import Path

from dunder_mifflin_harness.tools.shell import (
    PROCESS_REGISTRY,
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


def test_run_shell_timeout_kills_child_process_group(tmp_path: Path) -> None:
    started = time.monotonic()

    result = asyncio.run(
        run_shell(command="sh -c 'sleep 5'", cwd=str(tmp_path), timeout_seconds=0.1)
    )

    assert result.status == ToolStatus.TIMEOUT
    assert time.monotonic() - started < 2


def test_run_shell_spawn_failure_returns_structured_error(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(run_shell(command="pwd", cwd=str(missing_cwd)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert result.error


def test_read_process_output_truncates_without_loading_full_file(tmp_path: Path) -> None:
    async def run_background_command() -> None:
        command = (
            f"{shlex.quote(sys.executable)} -c "
            f"{shlex.quote('import sys; sys.stdout.write(\"x\" * 20000)')}"
        )
        started = await run_shell_background(command=command, cwd=str(tmp_path))
        assert started.status == ToolStatus.OK
        info = PROCESS_REGISTRY[started.process_id]
        try:
            await asyncio.wait_for(info.process.wait(), timeout=5)
            output = await read_process_output(started.process_id, max_output_chars=1000)
        finally:
            PROCESS_REGISTRY.pop(started.process_id, None)
            Path(info.stdout_path).unlink(missing_ok=True)
            Path(info.stderr_path).unlink(missing_ok=True)

        assert output.status == ToolStatus.OK
        assert output.warnings == ["output was truncated"]
        assert len(output.stdout) <= 1000

    asyncio.run(run_background_command())


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
