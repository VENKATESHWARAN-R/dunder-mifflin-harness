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


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}


class _OutputCapture:
    """Bound process output in memory while still draining pipes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, max_bytes)
        self.total_bytes = 0
        self._head = bytearray()
        self._tail = bytearray()

    @property
    def truncated(self) -> bool:
        return self.total_bytes > len(self._head) + len(self._tail)

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        if self.max_bytes == 0:
            return
        if self.max_bytes < 200:
            remaining = self.max_bytes - len(self._head)
            if remaining > 0:
                self._head.extend(chunk[:remaining])
            return

        head_limit = (self.max_bytes - 80) // 2
        tail_limit = head_limit
        if len(self._head) < head_limit:
            head_remaining = head_limit - len(self._head)
            self._head.extend(chunk[:head_remaining])
            chunk = chunk[head_remaining:]
        if not chunk:
            return

        self._tail.extend(chunk)
        if len(self._tail) > tail_limit:
            del self._tail[: len(self._tail) - tail_limit]

    def text(self) -> str:
        if not self.truncated:
            return bytes(self._head).decode("utf-8", errors="replace")
        if self.max_bytes < 200:
            return bytes(self._head).decode("utf-8", errors="replace")
        omitted = self.total_bytes - len(self._head) - len(self._tail)
        head = bytes(self._head).decode("utf-8", errors="replace")
        tail = bytes(self._tail).decode("utf-8", errors="replace")
        return f"{head}\n... truncated {omitted} bytes ...\n{tail}"


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    capture: _OutputCapture,
) -> None:
    if stream is None:
        return
    while chunk := await stream.read(8192):
        capture.add(chunk)


def _subprocess_kwargs() -> dict[str, bool]:
    """Return platform-specific subprocess isolation options."""
    return {"start_new_session": True} if os.name != "nt" else {}


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
            return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


async def _finish_reader_tasks(
    *tasks: asyncio.Task[None],
    timeout_seconds: float = 5.0,
) -> None:
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _read_log_file(path: str, max_bytes: int) -> _OutputCapture:
    capture = _OutputCapture(max_bytes)
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
        if size <= capture.max_bytes:
            capture.add(file_path.read_bytes())
            return capture

        capture.total_bytes = size
        if capture.max_bytes == 0:
            return capture
        if capture.max_bytes < 200:
            with file_path.open("rb") as handle:
                capture._head.extend(handle.read(capture.max_bytes))
            return capture

        head_limit = (capture.max_bytes - 80) // 2
        tail_limit = head_limit
        with file_path.open("rb") as handle:
            capture._head.extend(handle.read(head_limit))
            handle.seek(max(size - tail_limit, 0))
            capture._tail.extend(handle.read(tail_limit))
    except OSError:
        raise
    return capture


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
    stdout_capture = _OutputCapture(max_output_chars)
    stderr_capture = _OutputCapture(max_output_chars)
    stdout_task = asyncio.create_task(_drain_stream(process.stdout, stdout_capture))
    stderr_task = asyncio.create_task(_drain_stream(process.stderr, stderr_capture))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        _kill_process_tree(process)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5.0)

    await _finish_reader_tasks(stdout_task, stderr_task)

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_capture.text()
    stderr_raw = stderr_capture.text()
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
        delete=False, suffix=f".{process_id}.stdout", mode="w",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{process_id}.stderr", mode="w",
    )

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(resolved_cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            **_subprocess_kwargs(),
        )
    except OSError as exc:
        stdout_file.close()
        stderr_file.close()
        with contextlib.suppress(OSError):
            Path(stdout_file.name).unlink()
        with contextlib.suppress(OSError):
            Path(stderr_file.name).unlink()
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=str(exc),
            command=command,
            cwd=str(resolved_cwd),
        )
    stdout_file.close()
    stderr_file.close()

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
        stdout_capture = _read_log_file(info.stdout_path, max_output_chars)
        stderr_capture = _read_log_file(info.stderr_path, max_output_chars)
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = stdout_capture.truncated or stderr_capture.truncated
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=stdout_capture.text(),
        stderr=stderr_capture.text(),
        running=running,
        exit_code=info.process.returncode,
    )


setattr(read_process_output, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda process_id="", **_: f"Read output of process `{process_id}`",
))
