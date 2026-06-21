import asyncio
import shlex
import sys
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


def test_run_shell_timeout_kills_process_group_with_inherited_pipes(tmp_path: Path) -> None:
    child_script = "import time; time.sleep(10)"
    parent_script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        "time.sleep(10)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_script)}"

    async def invoke() -> object:
        return await asyncio.wait_for(
            run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.1),
            timeout=2.0,
        )

    result = asyncio.run(invoke())

    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out


def test_run_shell_returns_error_for_missing_cwd(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = asyncio.run(run_shell(command="pwd", cwd=str(missing)))

    assert result.status == ToolStatus.ERROR
    assert result.exit_code == -1
    assert "No such file or directory" in (result.error or "")


def test_run_shell_caps_high_volume_output(tmp_path: Path) -> None:
    script = "import sys; sys.stdout.write('x' * 200000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), max_output_chars=1000)
    )

    assert result.status == ToolStatus.OK
    assert result.truncated
    assert len(result.stdout) <= 1000


def test_background_shell_caps_output_file(tmp_path: Path) -> None:
    script = "import sys; sys.stdout.write('x' * 1200000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def invoke() -> object:
        result = await run_shell_background(command=command, cwd=str(tmp_path))
        await asyncio.sleep(0.5)
        return result

    result = asyncio.run(invoke())

    assert result.status == ToolStatus.OK
    # Regression guard: this used to write without bound; the helper now caps
    # background streams near 1MB plus a small truncation marker.
    from dunder_mifflin_harness.tools.shell import PROCESS_REGISTRY

    stdout_path = Path(PROCESS_REGISTRY[result.process_id].stdout_path)
    assert stdout_path.stat().st_size < 1_100_000


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
