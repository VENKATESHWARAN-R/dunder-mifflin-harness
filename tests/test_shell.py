import asyncio
import time
from pathlib import Path

from dunder_mifflin_harness.tools.shell import run_shell, run_shell_background, truncate_output
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


def test_run_shell_timeout_kills_children_holding_pipes(tmp_path: Path) -> None:
    start = time.monotonic()

    result = asyncio.run(
        run_shell(
            command="sh -c 'sleep 10 & wait'",
            cwd=str(tmp_path),
            timeout_seconds=0.1,
        )
    )

    assert time.monotonic() - start < 2
    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_timeout_returns_when_background_child_keeps_pipe(tmp_path: Path) -> None:
    start = time.monotonic()

    result = asyncio.run(
        run_shell(
            command=(
                "python3 -c 'import os, time; pid = os.fork(); "
                "print(\"done\") if pid else (time.sleep(10), os._exit(0))'"
            ),
            cwd=str(tmp_path),
            timeout_seconds=0.1,
        )
    )

    assert time.monotonic() - start < 2
    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_spawn_error_returns_tool_result(tmp_path: Path) -> None:
    result = asyncio.run(run_shell(command="pwd", cwd=str(tmp_path / "missing")))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert "failed to start" in (result.error or "")


def test_run_shell_background_spawn_error_returns_tool_result(tmp_path: Path) -> None:
    result = asyncio.run(run_shell_background(command="pwd", cwd=str(tmp_path / "missing")))

    assert result.status == ToolStatus.ERROR
    assert not result.process_id
    assert "failed to start" in (result.error or "")


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
