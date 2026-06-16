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

_STREAM_READ_CHUNK_BYTES = 8192
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 2.0
_BACKGROUND_OUTPUT_MAX_BYTES = 1_000_000


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


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}


async def _read_bounded_stream(
    stream: asyncio.StreamReader | None,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Drain a subprocess stream while keeping only a bounded prefix in memory."""
    if stream is None:
        return b"", False

    captured = bytearray()
    truncated = False
    limit = max(0, max_bytes)
    while True:
        chunk = await stream.read(_STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        if len(captured) < limit:
            remaining = limit - len(captured)
            captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True
    return bytes(captured), truncated


async def _finish_stream_task(
    task: asyncio.Task[tuple[bytes, bool]],
) -> tuple[bytes, bool]:
    try:
        return await asyncio.wait_for(task, timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return b"", True


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()


async def _drain_stream_to_file(
    stream: asyncio.StreamReader | None,
    path: str,
    max_bytes: int,
) -> None:
    """Drain background process output to disk without allowing unbounded growth."""
    if stream is None:
        return

    written = 0
    truncated = False
    with Path(path).open("wb") as output:
        while True:
            chunk = await stream.read(_STREAM_READ_CHUNK_BYTES)
            if not chunk:
                break
            if written < max_bytes:
                remaining = max_bytes - written
                output.write(chunk[:remaining])
                written += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    truncated = True
            else:
                truncated = True
        if truncated:
            output.write(
                f"\n... output truncated after {max_bytes} bytes ...\n".encode("utf-8")
            )


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
            error=f"failed to start command: {exc}",
            exit_code=-1,
            command=command,
            cwd=str(resolved_cwd),
            duration_ms=duration_ms,
        )

    timed_out = False
    stdout_task = asyncio.create_task(
        _read_bounded_stream(process.stdout, max_output_chars)
    )
    stderr_task = asyncio.create_task(
        _read_bounded_stream(process.stderr, max_output_chars)
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        _kill_process_group(process)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                process.wait(),
                timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS,
            )

    stdout_bytes, stdout_stream_truncated = await _finish_stream_task(stdout_task)
    stderr_bytes, stderr_stream_truncated = await _finish_stream_task(stderr_task)

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
    stderr_raw = stderr_bytes.decode("utf-8", errors="replace")
    truncated = (
        stdout_stream_truncated
        or stderr_stream_truncated
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
    process_id = uuid.uuid4().hex[:8]

    stdout_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{process_id}.stdout")
    stderr_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{process_id}.stderr")
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
            error=f"failed to start command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )

    asyncio.create_task(
        _drain_stream_to_file(
            process.stdout,
            stdout_file.name,
            _BACKGROUND_OUTPUT_MAX_BYTES,
        )
    )
    asyncio.create_task(
        _drain_stream_to_file(
            process.stderr,
            stderr_file.name,
            _BACKGROUND_OUTPUT_MAX_BYTES,
        )
    )

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_file.name,
        stderr_path=stderr_file.name,
        started_at=datetime.now(timezone.utc).isoformat(),
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
        stdout_raw = Path(info.stdout_path).read_text(encoding="utf-8", errors="replace")
        stderr_raw = Path(info.stderr_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = len(stdout_raw) > max_output_chars or len(stderr_raw) > max_output_chars
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
