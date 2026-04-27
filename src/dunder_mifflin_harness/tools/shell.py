"""Shared shell execution helpers for user and agent commands."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShellResult:
    """Captured shell command result."""

    command: str
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    @property
    def succeeded(self) -> bool:
        """Whether the command completed with exit code 0."""
        return self.exit_code == 0 and not self.timed_out


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


async def run_shell_command(
    command: str,
    cwd: Path,
    timeout_seconds: float,
    max_output_chars: int,
) -> ShellResult:
    """Run a shell command with timeout and output capture."""
    started_at = time.monotonic()
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    duration = time.monotonic() - started_at

    return ShellResult(
        command=command,
        cwd=cwd,
        exit_code=process.returncode,
        stdout=truncate_output(stdout, max_output_chars),
        stderr=truncate_output(stderr, max_output_chars),
        timed_out=timed_out,
        duration_seconds=duration,
    )
