"""Shell execution helpers for CLI and agent tools."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
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
    drain_tasks: tuple[asyncio.Task[None], ...] = ()


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}

_READ_CHUNK_BYTES = 8192
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 1.0
_BACKGROUND_OUTPUT_LIMIT_BYTES = 1_000_000
_BACKGROUND_TRUNCATION_MARKER = b"\n... output truncated; refine or restart the command ...\n"


@dataclass
class _BoundedCapture:
    """Capture subprocess output without allowing unbounded memory growth."""

    max_bytes: int
    buffer: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0
    truncated: bool = False

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        if self.max_bytes <= 0:
            self.truncated = True
            return

        if not self.truncated:
            self.buffer.extend(chunk)
            if len(self.buffer) <= self.max_bytes:
                return

            self.truncated = True
            if self.max_bytes < 200:
                del self.buffer[self.max_bytes:]
                return

            half = (self.max_bytes - 80) // 2
            overflow = self.buffer[half:]
            self.tail.extend(overflow)
            if len(self.tail) > half:
                del self.tail[:-half]
            del self.buffer[half:]
            return

        if self.max_bytes < 200:
            return

        half = (self.max_bytes - 80) // 2
        self.tail.extend(chunk)
        if len(self.tail) > half:
            del self.tail[:-half]

    def as_text(self) -> str:
        if not self.truncated:
            return bytes(self.buffer).decode("utf-8", errors="replace")
        if self.max_bytes < 200:
            return bytes(self.buffer).decode("utf-8", errors="replace")

        head = bytes(self.buffer).decode("utf-8", errors="replace")
        tail = bytes(self.tail).decode("utf-8", errors="replace")
        omitted = max(0, self.total_bytes - len(self.buffer) - len(self.tail))
        return f"{head}\n... truncated at least {omitted} bytes ...\n{tail}"


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    capture: _BoundedCapture,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            return
        capture.append(chunk)


async def _finish_reader_tasks(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        try:
            await asyncio.wait_for(task, timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


async def _drain_stream_to_file(
    stream: asyncio.StreamReader | None,
    output_path: str,
    max_bytes: int = _BACKGROUND_OUTPUT_LIMIT_BYTES,
) -> None:
    if stream is None:
        return

    written = 0
    marker_written = False
    try:
        with Path(output_path).open("ab") as output:
            while True:
                chunk = await stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    return

                remaining = max_bytes - written
                if remaining > 0:
                    to_write = chunk[:remaining]
                    output.write(to_write)
                    written += len(to_write)
                    if len(to_write) == len(chunk):
                        output.flush()
                        continue

                if not marker_written:
                    output.write(_BACKGROUND_TRUNCATION_MARKER)
                    marker_written = True
                output.flush()
    except OSError:
        return


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
    stdout_capture = _BoundedCapture(max(0, max_output_chars))
    stderr_capture = _BoundedCapture(max(0, max_output_chars))
    reader_tasks = [
        asyncio.create_task(_drain_stream(process.stdout, stdout_capture)),
        asyncio.create_task(_drain_stream(process.stderr, stderr_capture)),
    ]
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
    finally:
        await _finish_reader_tasks(reader_tasks)

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_capture.as_text()
    stderr_raw = stderr_capture.as_text()
    truncated = stdout_capture.truncated or stderr_capture.truncated

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
        delete=False, suffix=f".{process_id}.stdout", mode="wb",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stderr", mode="wb",
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
            error=f"failed to start command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )

    drain_tasks = (
        asyncio.create_task(_drain_stream_to_file(process.stdout, stdout_path)),
        asyncio.create_task(_drain_stream_to_file(process.stderr, stderr_path)),
    )

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        drain_tasks=drain_tasks,
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
