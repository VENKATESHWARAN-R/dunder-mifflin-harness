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


@dataclass(frozen=True)
class _CapturedOutput:
    text: str
    truncated: bool


_STREAM_CHUNK_BYTES = 8192


def _output_byte_budget(max_output_chars: int) -> int:
    """Translate the public character cap into a safe byte cap for UTF-8 output."""
    return max(0, max_output_chars) * 4


def _split_capture_budget(max_output_chars: int) -> tuple[int, int]:
    byte_budget = _output_byte_budget(max_output_chars)
    if byte_budget == 0:
        return 0, 0
    if max_output_chars < 200:
        return byte_budget, 0

    head_budget = max(1, (byte_budget - 320) // 2)
    tail_budget = max(1, byte_budget - head_budget)
    return head_budget, tail_budget


def _format_captured_bytes(
    head: bytes,
    tail: bytes,
    total_bytes: int,
    max_output_chars: int,
) -> _CapturedOutput:
    captured_bytes = len(head) + len(tail)
    byte_truncated = total_bytes > captured_bytes

    if byte_truncated and tail:
        omitted = total_bytes - captured_bytes
        text = (
            f"{head.decode('utf-8', errors='replace')}\n"
            f"... truncated at least {omitted} bytes ...\n"
            f"{tail.decode('utf-8', errors='replace')}"
        )
    else:
        text = (head + tail).decode("utf-8", errors="replace")

    char_truncated = len(text) > max_output_chars
    return _CapturedOutput(
        text=truncate_output(text, max_output_chars),
        truncated=byte_truncated or char_truncated,
    )


async def _read_stream_bounded(
    stream: asyncio.StreamReader | None,
    max_output_chars: int,
) -> _CapturedOutput:
    if stream is None:
        return _CapturedOutput(text="", truncated=False)

    head_budget, tail_budget = _split_capture_budget(max_output_chars)
    head = bytearray()
    tail = bytearray()
    total_bytes = 0

    while chunk := await stream.read(_STREAM_CHUNK_BYTES):
        total_bytes += len(chunk)
        remaining_head = max(0, head_budget - len(head))
        if remaining_head:
            head.extend(chunk[:remaining_head])
            chunk = chunk[remaining_head:]
        if tail_budget and chunk:
            tail.extend(chunk)
            if len(tail) > tail_budget:
                del tail[: len(tail) - tail_budget]

    return _format_captured_bytes(bytes(head), bytes(tail), total_bytes, max_output_chars)


def _read_file_bounded(path: str, max_output_chars: int) -> _CapturedOutput:
    head_budget, tail_budget = _split_capture_budget(max_output_chars)
    file_size = Path(path).stat().st_size

    with Path(path).open("rb") as handle:
        head = handle.read(head_budget)
        tail = b""
        if tail_budget and file_size > len(head):
            handle.seek(max(len(head), file_size - tail_budget))
            tail = handle.read(tail_budget)

    return _format_captured_bytes(head, tail, file_size, max_output_chars)


def _signal_process_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, sig)


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    _signal_process_group(process, signal.SIGKILL)
    await process.wait()


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
    stdout_task = asyncio.create_task(_read_stream_bounded(process.stdout, max_output_chars))
    stderr_task = asyncio.create_task(_read_stream_bounded(process.stderr, max_output_chars))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        await _terminate_process_group(process)

    stdout_capture, stderr_capture = await asyncio.gather(stdout_task, stderr_task)

    duration_ms = int((time.monotonic() - started_at) * 1000)
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
        stdout=stdout_capture.text,
        stderr=stderr_capture.text,
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
        stdout_file.close()
        stderr_file.close()
        Path(stdout_file.name).unlink(missing_ok=True)
        Path(stderr_file.name).unlink(missing_ok=True)
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
        stdout_capture = _read_file_bounded(info.stdout_path, max_output_chars)
        stderr_capture = _read_file_bounded(info.stderr_path, max_output_chars)
    except OSError as exc:
        return ProcessOutputResult(status=ToolStatus.ERROR, error=str(exc), running=running)

    truncated = stdout_capture.truncated or stderr_capture.truncated
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=stdout_capture.text,
        stderr=stderr_capture.text,
        running=running,
        exit_code=info.process.returncode,
    )


setattr(read_process_output, "approval", ToolApprovalMeta(
    category="shell",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda process_id="", **_: f"Read output of process `{process_id}`",
))
