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
    stdout_capture: _FileStreamCapture | None = None
    stderr_capture: _FileStreamCapture | None = None


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}

_STREAM_READ_CHUNK_BYTES = 8192
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 1.0
_FOREGROUND_PIPE_CLEANUP_TIMEOUT_SECONDS = 0.2
_BACKGROUND_STREAM_MAX_BYTES = 1_000_000


class _MemoryStreamCapture:
    """Drain a subprocess stream while retaining only bounded output."""

    def __init__(self, max_output_chars: int) -> None:
        self.max_bytes = max(0, max_output_chars * 4)
        self.chunks: list[bytes] = []
        self.truncated = False

    async def drain(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return

        captured = 0
        while chunk := await stream.read(_STREAM_READ_CHUNK_BYTES):
            remaining = self.max_bytes - captured
            if remaining > 0:
                self.chunks.append(chunk[:remaining])
                captured += min(len(chunk), remaining)
            if len(chunk) > remaining:
                self.truncated = True

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


class _FileStreamCapture:
    """Drain a subprocess stream to a capped file so background jobs cannot fill disk."""

    def __init__(self, path: str, max_bytes: int = _BACKGROUND_STREAM_MAX_BYTES) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.truncated = False

    async def drain(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return

        written = 0
        with Path(self.path).open("wb") as output:
            while chunk := await stream.read(_STREAM_READ_CHUNK_BYTES):
                remaining = self.max_bytes - written
                if remaining > 0:
                    data = chunk[:remaining]
                    output.write(data)
                    output.flush()
                    written += len(data)
                if len(chunk) > remaining:
                    self.truncated = True


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


async def _wait_for_process_exit(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)


async def _finish_drain_tasks(
    tasks: list[asyncio.Task[None]],
    timeout_seconds: float,
    *,
    cancel_on_timeout: bool = True,
) -> bool:
    """Return True when all stream drain tasks finish before the timeout."""
    if not tasks:
        return True

    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    if pending:
        if cancel_on_timeout:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return False

    await asyncio.gather(*done, return_exceptions=True)
    return True


def _read_capped_output_file(path: str, max_output_chars: int) -> tuple[str, bool]:
    max_bytes = max(0, max_output_chars * 4)
    file_path = Path(path)
    size = file_path.stat().st_size
    if max_bytes == 0:
        return "", size > 0

    if size <= max_bytes or max_bytes < 200:
        data = file_path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace"), size > max_bytes

    head_bytes = max_bytes // 2
    tail_bytes = max_bytes - head_bytes
    with file_path.open("rb") as output:
        head = output.read(head_bytes)
        output.seek(max(0, size - tail_bytes))
        tail = output.read(tail_bytes)

    omitted = size - len(head) - len(tail)
    marker = f"\n... truncated {omitted} bytes ...\n".encode()
    return (head + marker + tail).decode("utf-8", errors="replace"), True


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
            error=f"failed to start shell command: {exc}",
            exit_code=-1,
            command=command,
            cwd=str(resolved_cwd),
            duration_ms=duration_ms,
        )

    timed_out = False
    stdout_capture = _MemoryStreamCapture(max_output_chars)
    stderr_capture = _MemoryStreamCapture(max_output_chars)
    drain_tasks = [
        asyncio.create_task(stdout_capture.drain(process.stdout)),
        asyncio.create_task(stderr_capture.drain(process.stderr)),
    ]

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        _kill_process_group(process)
        await _wait_for_process_exit(process)

    pipes_closed = await _finish_drain_tasks(
        drain_tasks,
        timeout_seconds=(
            _PROCESS_CLEANUP_TIMEOUT_SECONDS
            if timed_out
            else _FOREGROUND_PIPE_CLEANUP_TIMEOUT_SECONDS
        ),
        cancel_on_timeout=timed_out,
    )
    if not pipes_closed and not timed_out:
        _kill_process_group(process)
        await _wait_for_process_exit(process)
        await _finish_drain_tasks(
            drain_tasks,
            _PROCESS_CLEANUP_TIMEOUT_SECONDS,
            cancel_on_timeout=True,
        )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_capture.text()
    stderr_raw = stderr_capture.text()
    truncated = (
        stdout_capture.truncated
        or stderr_capture.truncated
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

    stdout_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stdout", mode="w",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stderr", mode="w",
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
            start_new_session=True,
        )
    except OSError as exc:
        with contextlib.suppress(OSError):
            Path(stdout_path).unlink()
        with contextlib.suppress(OSError):
            Path(stderr_path).unlink()
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=f"failed to start shell command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )

    stdout_capture = _FileStreamCapture(stdout_path)
    stderr_capture = _FileStreamCapture(stderr_path)
    asyncio.create_task(stdout_capture.drain(process.stdout))
    asyncio.create_task(stderr_capture.drain(process.stderr))

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
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
        stdout_raw, stdout_truncated = _read_capped_output_file(
            info.stdout_path,
            max_output_chars,
        )
        stderr_raw, stderr_truncated = _read_capped_output_file(
            info.stderr_path,
            max_output_chars,
        )
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = (
        stdout_truncated
        or stderr_truncated
        or (info.stdout_capture.truncated if info.stdout_capture else False)
        or (info.stderr_capture.truncated if info.stderr_capture else False)
        or len(stdout_raw) > max_output_chars
        or len(stderr_raw) > max_output_chars
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
