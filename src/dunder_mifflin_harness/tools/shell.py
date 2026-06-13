"""Shell execution helpers for CLI and agent tools."""

from __future__ import annotations

import asyncio
import contextlib
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
    stdout_task: asyncio.Task[bool]
    stderr_task: asyncio.Task[bool]


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}

_CAPTURE_CHUNK_SIZE = 8192
_PROCESS_EXIT_GRACE_SECONDS = 2.0
_BACKGROUND_OUTPUT_LIMIT_BYTES = 1_000_000


async def _capture_stream(
    stream: asyncio.StreamReader | None,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Drain a stream while keeping only a bounded prefix in memory."""
    if stream is None:
        return b"", False

    chunks: list[bytes] = []
    captured = 0
    truncated = False
    while True:
        chunk = await stream.read(_CAPTURE_CHUNK_SIZE)
        if not chunk:
            break

        remaining = max_bytes - captured
        if remaining > 0:
            kept = chunk[:remaining]
            chunks.append(kept)
            captured += len(kept)
            truncated = truncated or len(kept) < len(chunk)
        else:
            truncated = True

    return b"".join(chunks), truncated


async def _drain_stream_to_file(
    stream: asyncio.StreamReader | None,
    path: str,
    max_bytes: int,
) -> bool:
    """Drain process output to a capped file so verbose daemons cannot fill disk."""
    if stream is None:
        return False

    captured = 0
    truncated = False
    with Path(path).open("wb") as output_file:
        while True:
            chunk = await stream.read(_CAPTURE_CHUNK_SIZE)
            if not chunk:
                break

            remaining = max_bytes - captured
            if remaining > 0:
                kept = chunk[:remaining]
                output_file.write(kept)
                captured += len(kept)
                truncated = truncated or len(kept) < len(chunk)
            else:
                truncated = True

    return truncated


async def _finish_capture_task(
    task: asyncio.Task[tuple[bytes, bool]],
) -> tuple[bytes, bool]:
    try:
        return await asyncio.wait_for(task, timeout=_PROCESS_EXIT_GRACE_SECONDS)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return b"", True


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def _new_process_id() -> str:
    process_id = uuid.uuid4().hex
    while process_id in PROCESS_REGISTRY:
        process_id = uuid.uuid4().hex
    return process_id


def _read_output_file(path: str, max_chars: int) -> tuple[str, bool]:
    max_bytes = max(1, max_chars * 4)
    file_path = Path(path)
    with file_path.open("rb") as output_file:
        data = output_file.read(max_bytes + 1)
    truncated = len(data) > max_bytes or file_path.stat().st_size > max_bytes
    return data[:max_bytes].decode("utf-8", errors="replace"), truncated


def _task_reported_truncation(task: asyncio.Task[bool]) -> bool:
    if not task.done():
        return False
    try:
        return task.result()
    except (Exception, asyncio.CancelledError):
        return True


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
            start_new_session=True,
        )
    except OSError as exc:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        return ShellToolResult(
            status=ToolStatus.ERROR,
            error=str(exc),
            exit_code=-1,
            command=command,
            cwd=str(resolved_cwd),
            duration_ms=duration_ms,
        )

    timed_out = False
    capture_limit = max(1, max_output_chars * 4)
    stdout_task = asyncio.create_task(_capture_stream(process.stdout, capture_limit))
    stderr_task = asyncio.create_task(_capture_stream(process.stderr, capture_limit))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        _kill_process_group(process)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_EXIT_GRACE_SECONDS)

    stdout_bytes, stdout_capped = await _finish_capture_task(stdout_task)
    stderr_bytes, stderr_capped = await _finish_capture_task(stderr_task)

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
    stderr_raw = stderr_bytes.decode("utf-8", errors="replace")
    truncated = (
        stdout_capped
        or stderr_capped
        or len(stdout_raw) > max_output_chars
        or len(stderr_raw) > max_output_chars
    )

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
        stdout=truncate_output(stdout_raw, max_output_chars),
        stderr=truncate_output(stderr_raw, max_output_chars),
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
    process_id = _new_process_id()

    stdout_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stdout", mode="wb",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stderr", mode="wb",
    )
    stdout_file.close()
    stderr_file.close()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(resolved_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        Path(stdout_file.name).unlink(missing_ok=True)
        Path(stderr_file.name).unlink(missing_ok=True)
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=str(exc),
            process_id=process_id,
            command=command,
            cwd=str(resolved_cwd),
        )

    stdout_task = asyncio.create_task(
        _drain_stream_to_file(process.stdout, stdout_file.name, _BACKGROUND_OUTPUT_LIMIT_BYTES)
    )
    stderr_task = asyncio.create_task(
        _drain_stream_to_file(process.stderr, stderr_file.name, _BACKGROUND_OUTPUT_LIMIT_BYTES)
    )

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_file.name,
        stderr_path=stderr_file.name,
        started_at=datetime.now(timezone.utc).isoformat(),
        stdout_task=stdout_task,
        stderr_task=stderr_task,
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

    running = info.process.returncode is None
    try:
        stdout_raw, stdout_truncated = _read_output_file(info.stdout_path, max_output_chars)
        stderr_raw, stderr_truncated = _read_output_file(info.stderr_path, max_output_chars)
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = (
        stdout_truncated
        or stderr_truncated
        or _task_reported_truncation(info.stdout_task)
        or _task_reported_truncation(info.stderr_task)
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
