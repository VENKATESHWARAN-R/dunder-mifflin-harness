"""Shell tool tests — run_shell, background process lifecycle, cleanup."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from jac.tools.shell import (
    PROCESS_REGISTRY,
    cleanup_processes,
    read_process_output,
    run_shell,
    run_shell_background,
)
from jac.tools.types import ToolStatus


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with no leaked background processes."""
    cleanup_processes()
    yield
    cleanup_processes()


@pytest.mark.asyncio
async def test_run_shell_happy_path() -> None:
    result = await run_shell("echo hello")
    assert result.status == ToolStatus.OK
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_run_shell_nonzero_exit() -> None:
    result = await run_shell("exit 7")
    assert result.status == ToolStatus.ERROR
    assert result.exit_code == 7


@pytest.mark.asyncio
async def test_run_shell_timeout() -> None:
    result = await run_shell("sleep 5", timeout_seconds=0.2)
    assert result.status == ToolStatus.TIMEOUT
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_run_shell_respects_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("hi", encoding="utf-8")
    result = await run_shell("ls", cwd=str(tmp_path))
    assert "marker.txt" in result.stdout


@pytest.mark.asyncio
async def test_background_process_lifecycle() -> None:
    started = await run_shell_background(
        f"{sys.executable} -c \"import time; print('hello'); time.sleep(0.05)\""
    )
    assert started.status == ToolStatus.OK
    assert started.process_id in PROCESS_REGISTRY

    # Poll a few times until the process exits.
    for _ in range(50):
        output = await read_process_output(started.process_id)
        if not output.running:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("background process did not exit in time")

    assert output.status == ToolStatus.OK
    assert "hello" in output.stdout
    assert output.exit_code == 0


@pytest.mark.asyncio
async def test_read_process_output_unknown_id() -> None:
    result = await read_process_output("does-not-exist")
    assert result.status == ToolStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_cleanup_processes_clears_registry_and_files() -> None:
    started = await run_shell_background(
        f'{sys.executable} -c "import time; time.sleep(2)"'
    )
    info = PROCESS_REGISTRY[started.process_id]
    stdout_path = Path(info.stdout_path)
    assert stdout_path.exists()

    cleanup_processes()

    assert PROCESS_REGISTRY == {}
    assert not stdout_path.exists()
