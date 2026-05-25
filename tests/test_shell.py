import asyncio
import sys
from pathlib import Path

from dunder_mifflin_harness.tools.shell import (
    run_shell,
    run_shell_background,
    truncate_output,
)
from dunder_mifflin_harness.tools.types import ToolStatus


def test_run_shell_captures_output(tmp_path: Path) -> None:
    result = asyncio.run(run_shell(command="printf hello", cwd=str(tmp_path)))

    assert result.status == ToolStatus.OK
    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.succeeded


def test_run_shell_nonzero_exit_sets_error_status(tmp_path: Path) -> None:
    result = asyncio.run(run_shell(command="exit 1", cwd=str(tmp_path)))

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
        f'{sys.executable} -c "import subprocess, time; '
        "subprocess.Popen(['sh', '-c', 'sleep 5; echo late']); "
        'time.sleep(10)"'
    )

    result = asyncio.run(
        asyncio.wait_for(
            run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1),
            timeout=2.0,
        )
    )

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_invalid_cwd_returns_structured_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = asyncio.run(run_shell(command="printf hello", cwd=str(missing)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert not result.succeeded


def test_run_shell_truncates_without_buffering_unbounded_output(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"import sys; sys.stdout.write('a' * 200000)\""

    result = asyncio.run(
        run_shell(
            command=command,
            cwd=str(tmp_path),
            timeout_seconds=2,
            max_output_chars=1_000,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) < 1_200
    assert "truncated" in result.warnings[0]


def test_run_shell_background_invalid_cwd_returns_structured_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    result = asyncio.run(run_shell_background(command="sleep 1", cwd=str(missing)))

    assert result.status == ToolStatus.ERROR
    assert result.process_id == ""


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
