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


_CAPTURE_CHUNK_SIZE = 8192
_PROCESS_TERMINATION_GRACE_SECONDS = 1.0


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


@dataclass
class _CapturedStream:
    chunks: list[bytes] = field(default_factory=list)
    bytes_seen: int = 0
    bytes_stored: int = 0
    truncated: bool = False

    def append(self, chunk: bytes, byte_limit: int) -> None:
        self.bytes_seen += len(chunk)
        remaining = byte_limit - self.bytes_stored
        if remaining > 0:
            kept = chunk[:remaining]
            self.chunks.append(kept)
            self.bytes_stored += len(kept)
        if self.bytes_seen > byte_limit:
            self.truncated = True

    def data(self) -> bytes:
        return b"".join(self.chunks)


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}


def _capture_byte_limit(max_output_chars: int) -> int:
    # UTF-8 can use up to 4 bytes per character; this bounds memory while keeping
    # the existing character-oriented output setting.
    return max(1, max_output_chars * 4)


async def _read_stream(
    stream: asyncio.StreamReader | None,
    capture: _CapturedStream,
    byte_limit: int,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(_CAPTURE_CHUNK_SIZE)
        if not chunk:
            return
        capture.append(chunk, byte_limit)


async def _finish_stream_tasks(
    tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
    timeout: float | None,
) -> None:
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
            process.wait(),
            timeout=_PROCESS_TERMINATION_GRACE_SECONDS,
        )


def _decode_capture(capture: _CapturedStream, max_output_chars: int) -> tuple[str, bool]:
    text = capture.data().decode("utf-8", errors="replace")
    truncated = capture.truncated or len(text) > max_output_chars
    return truncate_output(text, max_output_chars), truncated


def _read_text_file_limited(path: str, max_output_chars: int) -> tuple[str, bool]:
    byte_limit = _capture_byte_limit(max_output_chars)
    with Path(path).open("rb") as file:
        data = file.read(byte_limit + 1)
    truncated = len(data) > byte_limit
    if truncated:
        data = data[:byte_limit]
    text = data.decode("utf-8", errors="replace")
    truncated = truncated or len(text) > max_output_chars
    return truncate_output(text, max_output_chars), truncated


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
    byte_limit = _capture_byte_limit(max_output_chars)
    stdout_capture = _CapturedStream()
    stderr_capture = _CapturedStream()
    stdout_task = asyncio.create_task(
        _read_stream(process.stdout, stdout_capture, byte_limit)
    )
    stderr_task = asyncio.create_task(
        _read_stream(process.stderr, stderr_capture, byte_limit)
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        await _terminate_process(process)
    await _finish_stream_tasks(
        (stdout_task, stderr_task),
        _PROCESS_TERMINATION_GRACE_SECONDS if timed_out else None,
    )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout, stdout_truncated = _decode_capture(stdout_capture, max_output_chars)
    stderr, stderr_truncated = _decode_capture(stderr_capture, max_output_chars)
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
        stdout=stdout,
        stderr=stderr,
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
            start_new_session=True,
        )
    except OSError as exc:
        stdout_path = stdout_file.name
        stderr_path = stderr_file.name
        stdout_file.close()
        stderr_file.close()
        with contextlib.suppress(OSError):
            os.unlink(stdout_path)
        with contextlib.suppress(OSError):
            os.unlink(stderr_path)
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=f"failed to start command: {exc}",
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
        stdout, stdout_truncated = _read_text_file_limited(
            info.stdout_path,
            max_output_chars,
        )
        stderr, stderr_truncated = _read_text_file_limited(
            info.stderr_path,
            max_output_chars,
        )
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = stdout_truncated or stderr_truncated
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=stdout,
        stderr=stderr,
        running=running,
        exit_code=info.process.returncode,
    )


setattr(read_process_output, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda process_id="", **_: f"Read output of process `{process_id}`",
))
