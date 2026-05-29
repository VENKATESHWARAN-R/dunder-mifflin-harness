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


def test_run_shell_timeout_kills_children_holding_output_pipes(tmp_path: Path) -> None:
    script = "import subprocess, time; subprocess.Popen(['sleep', '60']); time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    started_at = time.monotonic()
    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1)
    )

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out
    assert time.monotonic() - started_at < 2.0


def test_run_shell_spawn_error_returns_structured_error(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    result = asyncio.run(run_shell(command="printf hi", cwd=str(missing_cwd)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert "failed to start command" in (result.error or "")


def test_run_shell_large_output_is_capped(tmp_path: Path) -> None:
    script = "import sys; sys.stdout.write('x' * 20000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), max_output_chars=1000)
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert result.warnings == ["output was truncated"]
    assert len(result.stdout) <= 1000


def test_background_process_output_is_capped(tmp_path: Path) -> None:
    script = "import sys; sys.stdout.write('x' * 20000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def run_and_read():
        background = await run_shell_background(command=command, cwd=str(tmp_path))
        for _ in range(50):
            output = await read_process_output(
                background.process_id,
                max_output_chars=1000,
            )
            if output.exit_code is not None:
                return background, output
            await asyncio.sleep(0.02)
        return background, output

    background, output = asyncio.run(run_and_read())

    assert background.status == ToolStatus.OK
    assert output.warnings == ["output was truncated"]
    assert len(output.stdout) <= 1000


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
