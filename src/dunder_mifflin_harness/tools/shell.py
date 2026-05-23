"""Shell execution helpers for CLI and agent tools."""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
import tempfile
import time
import uuid
from contextlib import suppress
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
_READ_CHUNK_BYTES = 8192
_KILL_WAIT_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


async def _read_stream_limited(
    stream: asyncio.StreamReader | None,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Drain a subprocess stream while retaining at most max_bytes in memory."""
    if stream is None:
        return b"", False

    captured = bytearray()
    truncated = False
    limit = max(0, max_bytes)
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > max(0, remaining):
            truncated = True
    return bytes(captured), truncated


def _signal_process_group(process: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the whole process group, falling back to the shell process."""
    with suppress(ProcessLookupError):
        try:
            os.killpg(process.pid, sig)
        except OSError:
            with suppress(ProcessLookupError):
                process.send_signal(sig)


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    _signal_process_group(process, signal.SIGKILL)
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_KILL_WAIT_SECONDS)


async def _collect_stream_tasks(
    stdout_task: asyncio.Task[tuple[bytes, bool]],
    stderr_task: asyncio.Task[tuple[bytes, bool]],
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    try:
        return await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task),
            timeout=_KILL_WAIT_SECONDS,
        )
    except asyncio.TimeoutError:
        results: list[tuple[bytes, bool]] = []
        for task in (stdout_task, stderr_task):
            if task.done() and not task.cancelled():
                results.append(task.result())
            else:
                task.cancel()
                results.append((b"", True))
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return results[0], results[1]


def _read_text_file_limited(path: str, max_chars: int) -> tuple[str, bool]:
    """Read a text preview from a growing process output file without loading it all."""
    max_bytes = max(0, max_chars)
    size = os.path.getsize(path)
    if max_bytes == 0:
        return "", size > 0
    if size <= max_bytes:
        return Path(path).read_text(encoding="utf-8", errors="replace"), False

    if max_bytes < 200:
        with open(path, "rb") as file:
            return file.read(max_bytes).decode("utf-8", errors="replace"), True

    half = (max_bytes - 80) // 2
    with open(path, "rb") as file:
        head = file.read(half)
        file.seek(-half, os.SEEK_END)
        tail = file.read(half)
    omitted = size - (len(head) + len(tail))
    text = (
        f"{head.decode('utf-8', errors='replace')}\n"
        f"... truncated {omitted} bytes ...\n"
        f"{tail.decode('utf-8', errors='replace')}"
    )
    return text, True


def _cleanup_background_processes() -> None:
    for info in list(PROCESS_REGISTRY.values()):
        if info.process.returncode is None:
            _signal_process_group(info.process, signal.SIGTERM)
        for path in (info.stdout_path, info.stderr_path):
            with suppress(OSError):
                os.unlink(path)
    PROCESS_REGISTRY.clear()


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
            error=f"could not start command: {exc}",
            exit_code=-1,
            command=command,
            cwd=str(resolved_cwd),
            duration_ms=duration_ms,
        )

    timed_out = False
    stdout_task = asyncio.create_task(_read_stream_limited(process.stdout, max_output_chars))
    stderr_task = asyncio.create_task(_read_stream_limited(process.stderr, max_output_chars))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        await _kill_process_group(process)

    (stdout_bytes, stdout_truncated), (stderr_bytes, stderr_truncated) = (
        await _collect_stream_tasks(
            stdout_task,
            stderr_task,
        )
    )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
    stderr_raw = stderr_bytes.decode("utf-8", errors="replace")
    truncated = (
        stdout_truncated
        or stderr_truncated
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
    """Start a long-running command in the background.

    Returns a process_id for polling via read_process_output.
    """
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
        stdout_name = stdout_file.name
        stderr_name = stderr_file.name
        stdout_file.close()
        stderr_file.close()
        with suppress(OSError):
            os.unlink(stdout_name)
        with suppress(OSError):
            os.unlink(stderr_name)
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=f"could not start command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )
    finally:
        with suppress(OSError):
            stdout_file.close()
        with suppress(OSError):
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
        stdout_raw, stdout_truncated = _read_text_file_limited(
            info.stdout_path,
            max_output_chars,
        )
        stderr_raw, stderr_truncated = _read_text_file_limited(
            info.stderr_path,
            max_output_chars,
        )
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = stdout_truncated or stderr_truncated
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=stdout_raw,
        stderr=stderr_raw,
        running=running,
        exit_code=info.process.returncode,
    )


setattr(read_process_output, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda process_id="", **_: f"Read output of process `{process_id}`",
))


atexit.register(_cleanup_background_processes)
