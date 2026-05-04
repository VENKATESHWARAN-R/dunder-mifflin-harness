import asyncio
import time
from pathlib import Path

from jac.tools.shell import (
    PROCESS_REGISTRY,
    list_processes,
    read_process_output,
    run_shell,
    run_shell_background,
    truncate_output,
)
from jac.tools.types import RiskLevel, ToolStatus


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


def test_truncate_output_preserves_head_and_tail() -> None:
    text = "a" * 150 + "b" * 150

    truncated = truncate_output(text, 220)

    assert truncated.startswith("a")
    assert "truncated" in truncated
    assert truncated.endswith("b")


# ---------------------------------------------------------------------------
# run_shell_background / list_processes / read_process_output
# ---------------------------------------------------------------------------


def test_run_shell_background_returns_process_id(tmp_path: Path) -> None:
    result = asyncio.run(run_shell_background("sleep 10", cwd=str(tmp_path)))
    try:
        assert result.status == ToolStatus.OK
        assert len(result.process_id) == 8
        assert result.command == "sleep 10"
    finally:
        info = PROCESS_REGISTRY.pop(result.process_id, None)
        if info:
            info.process.kill()


def test_list_processes_shows_registered_process(tmp_path: Path) -> None:
    bg = asyncio.run(run_shell_background("sleep 10", cwd=str(tmp_path)))
    try:
        result = asyncio.run(list_processes())
        assert result.status == ToolStatus.OK
        ids = {p.process_id for p in result.processes}
        assert bg.process_id in ids
    finally:
        info = PROCESS_REGISTRY.pop(bg.process_id, None)
        if info:
            info.process.kill()


def test_read_process_output_captures_stdout(tmp_path: Path) -> None:
    bg = asyncio.run(run_shell_background("printf hello", cwd=str(tmp_path)))
    time.sleep(0.2)
    try:
        result = asyncio.run(read_process_output(bg.process_id))
        assert result.status == ToolStatus.OK
        assert "hello" in result.stdout
    finally:
        PROCESS_REGISTRY.pop(bg.process_id, None)


def test_read_process_output_not_found() -> None:
    result = asyncio.run(read_process_output("nonexistent"))
    assert result.status == ToolStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Approval metadata
# ---------------------------------------------------------------------------


def test_run_shell_approval_is_high_risk() -> None:
    assert run_shell.approval.risk_level == RiskLevel.HIGH
    assert run_shell.approval.category == "shell"
    assert not run_shell.approval.reversible


def test_run_shell_background_approval_is_high_risk() -> None:
    assert run_shell_background.approval.risk_level == RiskLevel.HIGH
    assert not run_shell_background.approval.reversible


def test_list_processes_approval_is_read_only() -> None:
    assert list_processes.approval.risk_level == RiskLevel.READ_ONLY
    assert list_processes.approval.reversible


def test_read_process_output_approval_is_read_only() -> None:
    assert read_process_output.approval.risk_level == RiskLevel.READ_ONLY
    assert read_process_output.approval.reversible
