"""Shell execution helpers for CLI and agent tools."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dunder_mifflin_harness.tools.types import (
    BackgroundProcessResult,
    ProcessEntry,
    ProcessListResult,
    ProcessOutputResult,
    RiskLevel,
    ShellToolResult,
    ToolApprovalMeta,
    ToolStatus,
)


def truncate_output(text: str, max_chars: int) -> str:
    """Truncate long output while preserving the head and tail."""
    if len(text) <= max_chars:
        return text
    if max_chars < 200:
        return text[:max_chars]
    half = (max_chars - 80) // 2
    omitted = len(text) - (half * 2)
    return (
        f"{text[:half]}\n"
        f"... truncated {omitted} characters ...\n"
        f"{text[-half:]}"
    )


@dataclass
class _ProcessInfo:
    process: asyncio.subprocess.Process
    command: str
    stdout_path: str
    stderr_path: str
    started_at: str
    stdout_task: asyncio.Task[bool] | None = None
    stderr_task: asyncio.Task[bool] | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    watcher_task: asyncio.Task[int] | None = None


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}
_BACKGROUND_OUTPUT_BYTES = 1_000_000


def _subprocess_kwargs() -> dict[str, bool]:
    """Create a new process group so timeouts can stop grandchildren too."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _signal_process_tree(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        return


async def _wait_after_signal(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
    timeout_seconds: float,
) -> None:
    _signal_process_tree(process, sig)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return


def _format_captured_bytes(data: bytes, total_bytes: int, max_bytes: int) -> str:
    if total_bytes <= max_bytes:
        return data.decode("utf-8", errors="replace")
    if max_bytes <= 0:
        return ""
    if max_bytes < 200:
        return data[:max_bytes].decode("utf-8", errors="replace")

    half = (max_bytes - 80) // 2
    omitted = max(0, total_bytes - (half * 2))
    marker = f"\n... truncated {omitted} bytes ...\n".encode()
    return (data[:half] + marker + data[-half:]).decode("utf-8", errors="replace")


async def _read_stream_capped(
    stream: asyncio.StreamReader | None,
    max_output_chars: int,
) -> tuple[str, bool]:
    if stream is None:
        return "", False

    max_bytes = max(0, max_output_chars)
    captured_limit = max_bytes if max_bytes < 200 else max_bytes + 1
    captured = bytearray()
    tail = bytearray()
    total_bytes = 0

    while chunk := await stream.read(8192):
        total_bytes += len(chunk)
        if len(captured) < captured_limit:
            remaining = captured_limit - len(captured)
            captured.extend(chunk[:remaining])
        if max_bytes >= 200:
            tail.extend(chunk)
            half = (max_bytes - 80) // 2
            if len(tail) > half:
                del tail[: len(tail) - half]

    if total_bytes > max_bytes and max_bytes >= 200:
        half = (max_bytes - 80) // 2
        data = bytes(captured[:half] + tail[-half:])
    else:
        data = bytes(captured)
    return _format_captured_bytes(data, total_bytes, max_bytes), total_bytes > max_bytes


async def _drain_stream_to_file(
    stream: asyncio.StreamReader | None,
    path: str,
    max_bytes: int,
) -> bool:
    if stream is None:
        return False

    captured = 0
    truncated = False
    with open(path, "wb") as output:
        while chunk := await stream.read(8192):
            remaining = max_bytes - captured
            if remaining > 0:
                output.write(chunk[:remaining])
                captured += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated = True
    return truncated


async def _remember_exit_code(process: asyncio.subprocess.Process) -> int:
    return await process.wait()


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout_seconds: float = 30.0,
    max_output_chars: int = 10_000,
) -> ShellToolResult:
    """Execute a shell command and return captured stdout/stderr."""
    resolved_cwd = Path(cwd) if cwd else Path.cwd()
    started_at = time.monotonic()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(resolved_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_subprocess_kwargs(),
        )
    except OSError as exc:
        return ShellToolResult(
            status=ToolStatus.ERROR,
            error=f"failed to start command: {exc}",
            exit_code=-1,
            command=command,
            cwd=str(resolved_cwd),
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )

    timed_out = False
    stdout_task = asyncio.create_task(_read_stream_capped(process.stdout, max_output_chars))
    stderr_task = asyncio.create_task(_read_stream_capped(process.stderr, max_output_chars))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        await _wait_after_signal(process, signal.SIGTERM, timeout_seconds=1.0)
        if process.returncode is None:
            await _wait_after_signal(process, signal.SIGKILL, timeout_seconds=1.0)

    stdout_raw, stdout_truncated = await stdout_task
    stderr_raw, stderr_truncated = await stderr_task

    duration_ms = int((time.monotonic() - started_at) * 1000)
    truncated = stdout_truncated or stderr_truncated

    if timed_out:
        status = ToolStatus.TIMEOUT
    elif process.returncode != 0:
        status = ToolStatus.ERROR
    else:
        status = ToolStatus.OK

    return ShellToolResult(
        status=status,
        warnings=["output was truncated"] if truncated else [],
        error=f"exited with code {process.returncode}" if status == ToolStatus.ERROR else None,
        stdout=stdout_raw,
        stderr=stderr_raw,
        exit_code=process.returncode if process.returncode is not None else -1,
        command=command,
        cwd=str(resolved_cwd),
        timed_out=timed_out,
        truncated=truncated,
        duration_ms=duration_ms,
    )


setattr(run_shell, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.HIGH,
    reversible=False,
    description_fn=lambda command, cwd=None, **_: (
        f"Run `{command}`" + (f" in `{cwd}`" if cwd else "")
    ),
))


async def run_shell_background(
    command: str,
    cwd: str | None = None,
) -> BackgroundProcessResult:
    """Start a long-running command in the background. Returns a process_id for polling via read_process_output."""
    resolved_cwd = Path(cwd) if cwd else Path.cwd()
    process_id = uuid.uuid4().hex[:8]

    stdout_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stdout",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stderr",
    )
    stdout_path = stdout_file.name
    stderr_path = stderr_file.name
    stdout_file.close()
    stderr_file.close()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(resolved_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_subprocess_kwargs(),
        )
    except OSError as exc:
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=f"failed to start command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        stdout_task=asyncio.create_task(
            _drain_stream_to_file(process.stdout, stdout_path, _BACKGROUND_OUTPUT_BYTES)
        ),
        stderr_task=asyncio.create_task(
            _drain_stream_to_file(process.stderr, stderr_path, _BACKGROUND_OUTPUT_BYTES)
        ),
        watcher_task=asyncio.create_task(_remember_exit_code(process)),
    )

    return BackgroundProcessResult(
        process_id=process_id,
        command=command,
        cwd=str(resolved_cwd),
    )


setattr(run_shell_background, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.HIGH,
    reversible=False,
    description_fn=lambda command, cwd=None, **_: (
        f"Start background: `{command}`" + (f" in `{cwd}`" if cwd else "")
    ),
))


async def list_processes() -> ProcessListResult:
    """List all background processes started in this session."""
    entries = [
        ProcessEntry(
            process_id=pid,
            command=info.command,
            started_at=info.started_at,
            running=info.process.returncode is None,
            exit_code=info.process.returncode,
        )
        for pid, info in PROCESS_REGISTRY.items()
    ]
    return ProcessListResult(processes=entries)


setattr(list_processes, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda **_: "List background processes",
))


async def read_process_output(
    process_id: str,
    max_output_chars: int = 10_000,
) -> ProcessOutputResult:
    """Read accumulated stdout/stderr from a background process."""
    info = PROCESS_REGISTRY.get(process_id)
    if not info:
        return ProcessOutputResult(
            status=ToolStatus.NOT_FOUND,
            error=f"no process with id: {process_id}",
            running=False,
        )

    if info.stdout_task and info.stdout_task.done():
        info.stdout_truncated = info.stdout_truncated or info.stdout_task.result()
    if info.stderr_task and info.stderr_task.done():
        info.stderr_truncated = info.stderr_truncated or info.stderr_task.result()

    running = info.process.returncode is None
    try:
        stdout_raw = Path(info.stdout_path).read_text(encoding="utf-8", errors="replace")
        stderr_raw = Path(info.stderr_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = (
        len(stdout_raw) > max_output_chars
        or len(stderr_raw) > max_output_chars
        or info.stdout_truncated
        or info.stderr_truncated
    )
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=truncate_output(stdout_raw, max_output_chars),
        stderr=truncate_output(stderr_raw, max_output_chars),
        running=running,
        exit_code=info.process.returncode,
    )


setattr(read_process_output, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda process_id="", **_: f"Read output of process `{process_id}`",
))
