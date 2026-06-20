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

_STREAM_CHUNK_SIZE = 8192
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 1.0
_BACKGROUND_OUTPUT_CAP_BYTES = 1_000_000


@dataclass
class _CapturedOutput:
    text: str
    truncated: bool = False


class _BoundedCapture:
    """Capture a stream without allowing unbounded memory growth."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, max_bytes)
        self.total = 0
        if self.max_bytes < 200:
            self.head_limit = self.max_bytes
            self.tail_limit = 0
        else:
            self.head_limit = (self.max_bytes - 80) // 2
            self.tail_limit = self.head_limit
        self.head = bytearray()
        self.tail = bytearray()

    def append(self, chunk: bytes) -> None:
        self.total += len(chunk)
        remaining = chunk

        if len(self.head) < self.head_limit:
            head_room = self.head_limit - len(self.head)
            self.head.extend(remaining[:head_room])
            remaining = remaining[head_room:]

        if self.tail_limit > 0 and remaining:
            self.tail.extend(remaining)
            if len(self.tail) > self.tail_limit:
                del self.tail[: len(self.tail) - self.tail_limit]

    def render(self) -> _CapturedOutput:
        if self.total <= self.max_bytes:
            data = bytes(self.head + self.tail)
            return _CapturedOutput(data.decode("utf-8", errors="replace"))

        if self.max_bytes == 0:
            return _CapturedOutput("", truncated=True)

        head = bytes(self.head).decode("utf-8", errors="replace")
        if self.tail_limit == 0:
            return _CapturedOutput(head, truncated=True)

        tail = bytes(self.tail).decode("utf-8", errors="replace")
        omitted = self.total - len(self.head) - len(self.tail)
        return _CapturedOutput(
            f"{head}\n... truncated {omitted} bytes ...\n{tail}",
            truncated=True,
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
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}


def _subprocess_kwargs() -> dict[str, bool]:
    """Return platform-specific subprocess options."""
    if os.name == "nt":
        return {}
    return {"start_new_session": True}


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if os.name == "nt":
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


async def _wait_for_tasks(tasks: list[asyncio.Task[object]], timeout: float) -> None:
    if not tasks:
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)
    for task in tasks:
        if not task.done():
            task.cancel()


async def _read_stream(
    stream: asyncio.StreamReader | None,
    max_output_chars: int,
) -> _CapturedOutput:
    if stream is None:
        return _CapturedOutput("")

    capture = _BoundedCapture(max_output_chars)
    while True:
        chunk = await stream.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            break
        capture.append(chunk)
    return capture.render()


def _task_output(task: asyncio.Task[_CapturedOutput]) -> _CapturedOutput:
    if task.cancelled() or not task.done():
        return _CapturedOutput("")
    with contextlib.suppress(Exception):
        return task.result()
    return _CapturedOutput("")


async def _drain_stream_to_file(
    stream: asyncio.StreamReader | None,
    path: str,
    max_bytes: int,
) -> None:
    written = 0
    truncated = False
    with Path(path).open("wb") as output:
        if stream is not None:
            while True:
                chunk = await stream.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                remaining = max_bytes - written
                if remaining > 0:
                    output.write(chunk[:remaining])
                    written += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    truncated = True
        if truncated:
            output.write(f"\n... output truncated at {max_bytes} bytes ...\n".encode())


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
    except (OSError, ValueError) as exc:
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
    process_task = asyncio.create_task(process.wait())
    stdout_task = asyncio.create_task(_read_stream(process.stdout, max_output_chars))
    stderr_task = asyncio.create_task(_read_stream(process.stderr, max_output_chars))
    tasks = [process_task, stdout_task, stderr_task]

    _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    if pending:
        timed_out = True
        _kill_process_group(process)
        await _wait_for_tasks(list(pending), _PROCESS_CLEANUP_TIMEOUT_SECONDS)

    duration_ms = int((time.monotonic() - started_at) * 1000)

    stdout = _task_output(stdout_task)
    stderr = _task_output(stderr_task)
    truncated = stdout.truncated or stderr.truncated

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
        stdout=stdout.text,
        stderr=stderr.text,
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{process_id}.stdout") as stdout_file:
        stdout_path = stdout_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{process_id}.stderr") as stderr_file:
        stderr_path = stderr_file.name

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(resolved_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_subprocess_kwargs(),
        )
    except (OSError, ValueError) as exc:
        return BackgroundProcessResult(
            status=ToolStatus.ERROR,
            error=f"failed to start shell command: {exc}",
            command=command,
            cwd=str(resolved_cwd),
        )

    stdout_task = asyncio.create_task(
        _drain_stream_to_file(process.stdout, stdout_path, _BACKGROUND_OUTPUT_CAP_BYTES)
    )
    stderr_task = asyncio.create_task(
        _drain_stream_to_file(process.stderr, stderr_path, _BACKGROUND_OUTPUT_CAP_BYTES)
    )

    PROCESS_REGISTRY[process_id] = _ProcessInfo(
        process=process,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
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
