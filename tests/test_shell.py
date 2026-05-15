import asyncio
import os
from pathlib import Path
import shlex
import sys
import time

import pytest

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


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
def test_run_shell_timeout_kills_descendant_processes(tmp_path: Path) -> None:
    sentinel = tmp_path / "leaked.txt"
    child_code = (
        "import pathlib, time; "
        "time.sleep(0.5); "
        f"pathlib.Path({str(sentinel)!r}).write_text('leaked', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "time.sleep(10)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)} >/dev/null 2>&1"

    result = asyncio.run(
        run_shell(command=command, cwd=str(tmp_path), timeout_seconds=0.2)
    )
    time.sleep(0.7)

    assert result.status == ToolStatus.TIMEOUT
    assert not sentinel.exists()


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")
