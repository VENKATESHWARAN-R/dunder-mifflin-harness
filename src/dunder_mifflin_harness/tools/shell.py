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
    return f"{text[:half]}\n... truncated {omitted} characters ...\n{text[-half:]}"


@dataclass
class _ProcessInfo:
    process: asyncio.subprocess.Process
    command: str
    stdout_path: str
    stderr_path: str
    started_at: str


# In-memory registry of background processes — lives for the duration of the harness session.
PROCESS_REGISTRY: dict[str, _ProcessInfo] = {}


class _BoundedOutput:
    """Capture command output without allowing unbounded memory growth."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, max_bytes)
        self.total_bytes = 0
        self._data = bytearray()
        self._head = bytearray()
        self._tail = bytearray()

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self.max_bytes

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return

        self.total_bytes += len(chunk)
        if not self.truncated:
            self._data.extend(chunk)
            return

        if not self._head and not self._tail:
            combined = bytes(self._data) + chunk
            self._data.clear()
            if self.max_bytes < 200:
                self._head.extend(combined[: self.max_bytes])
                return

            half = (self.max_bytes - 80) // 2
            self._head.extend(combined[:half])
            self._tail.extend(combined[-half:])
            return

        if self.max_bytes < 200:
            return

        half = (self.max_bytes - 80) // 2
        self._tail.extend(chunk)
        if len(self._tail) > half:
            del self._tail[: len(self._tail) - half]

    def text(self) -> str:
        if not self.truncated:
            return bytes(self._data).decode("utf-8", errors="replace")

        head = bytes(self._head).decode("utf-8", errors="replace")
        if self.max_bytes < 200:
            return head

        tail = bytes(self._tail).decode("utf-8", errors="replace")
        omitted = max(0, self.total_bytes - len(self._head) - len(self._tail))
        return f"{head}\n... truncated {omitted} bytes ...\n{tail}"


async def _read_stream(
    stream: asyncio.StreamReader | None,
    capture: _BoundedOutput,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return
        capture.append(chunk)


async def _finish_readers(tasks: list[asyncio.Task[None]]) -> None:
    if not tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=1.0,
        )
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _send_process_signal(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    _send_process_signal(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except TimeoutError:
        pass

    _send_process_signal(process, signal.SIGKILL)
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        return


def _read_text_file_limited(path: str, max_chars: int) -> tuple[str, bool]:
    max_bytes = max(0, max_chars)
    file_path = Path(path)
    size = file_path.stat().st_size
    if size <= max_bytes:
        return file_path.read_text(encoding="utf-8", errors="replace"), False

    if max_bytes < 200:
        with file_path.open("rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="replace"), True

    half = (max_bytes - 80) // 2
    with file_path.open("rb") as handle:
        head = handle.read(half)
        handle.seek(-half, os.SEEK_END)
        tail = handle.read(half)

    omitted = max(0, size - len(head) - len(tail))
    text = (
        f"{head.decode('utf-8', errors='replace')}\n"
        f"... truncated {omitted} bytes ...\n"
        f"{tail.decode('utf-8', errors='replace')}"
    )
    return text, True


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
    stdout_capture = _BoundedOutput(max_output_chars)
    stderr_capture = _BoundedOutput(max_output_chars)
    reader_tasks = [
        asyncio.create_task(_read_stream(process.stdout, stdout_capture)),
        asyncio.create_task(_read_stream(process.stderr, stderr_capture)),
    ]

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        await _terminate_process_tree(process)

    await _finish_readers(reader_tasks)

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
        error=f"exited with code {process.returncode}"
        if status == ToolStatus.ERROR
        else None,
        stdout=stdout_raw,
        stderr=stderr_raw,
        exit_code=process.returncode if process.returncode is not None else -1,
        command=command,
        cwd=str(resolved_cwd),
        timed_out=timed_out,
        truncated=truncated,
        duration_ms=duration_ms,
    )


setattr(
    run_shell,
    "approval",
    ToolApprovalMeta(
        category="shell",
        risk_level=RiskLevel.HIGH,
        reversible=False,
        description_fn=lambda command, cwd=None, **_: (
            f"Run `{command}`" + (f" in `{cwd}`" if cwd else "")
        ),
    ),
)


async def run_shell_background(
    command: str,
    cwd: str | None = None,
) -> BackgroundProcessResult:
    """Start a long-running command in the background. Returns a process_id for polling via read_process_output."""
    resolved_cwd = Path(cwd) if cwd else Path.cwd()
    process_id = uuid.uuid4().hex[:8]

    stdout_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{process_id}.stdout",
        mode="w",
    )
    stderr_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{process_id}.stderr",
        mode="w",
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
        Path(stdout_name).unlink(missing_ok=True)
        Path(stderr_name).unlink(missing_ok=True)
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


setattr(
    run_shell_background,
    "approval",
    ToolApprovalMeta(
        category="shell",
        risk_level=RiskLevel.HIGH,
        reversible=False,
        description_fn=lambda command, cwd=None, **_: (
            f"Start background: `{command}`" + (f" in `{cwd}`" if cwd else "")
        ),
    ),
)


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


setattr(
    list_processes,
    "approval",
    ToolApprovalMeta(
        category="shell",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda **_: "List background processes",
    ),
)


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
        return ProcessOutputResult(
            status=ToolStatus.ERROR, error=str(exc), running=running
        )

    truncated = stdout_truncated or stderr_truncated
    return ProcessOutputResult(
        warnings=["output was truncated"] if truncated else [],
        stdout=stdout_raw,
        stderr=stderr_raw,
        running=running,
        exit_code=info.process.returncode,
    )


setattr(
    read_process_output,
    "approval",
    ToolApprovalMeta(
        category="shell",
        risk_level=RiskLevel.READ_ONLY,
        reversible=True,
        description_fn=lambda process_id="", **_: (
            f"Read output of process `{process_id}`"
        ),
    ),
)
