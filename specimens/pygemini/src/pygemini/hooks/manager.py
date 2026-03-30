"""Lifecycle hook manager.

Hooks are shell commands fired at specific points in the agent loop.
They receive JSON on stdin and communicate via exit codes:

- 0 = success (continue)
- 1 = error (log warning, continue)
- 2 = block (abort the operation)

Hooks can optionally write JSON to stdout to modify context data.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

from pygemini.core.config import Config, HookConfig

__all__ = [
    "HookEvent",
    "HookResult",
    "HookManager",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook event enum
# ---------------------------------------------------------------------------


class HookEvent(Enum):
    """Lifecycle events that hooks can subscribe to."""

    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMPRESS = "pre_compress"
    NOTIFICATION = "notification"


# ---------------------------------------------------------------------------
# Hook result
# ---------------------------------------------------------------------------


@dataclass
class HookResult:
    """Result of executing a hook."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Exit code 0 means success."""
        return self.exit_code == 0

    @property
    def should_block(self) -> bool:
        """Exit code 2 means block the operation."""
        return self.exit_code == 2


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------


class HookManager:
    """Manages lifecycle hooks — shell commands fired at agent loop events.

    Hooks are configured in settings.toml::

        [hooks]
        before_tool = [
            {command = "echo 'tool starting'", timeout = 5},
        ]

    Protocol:
    - Hook receives JSON on stdin: ``{"event": "before_tool", "data": {...}}``
    - Hook communicates via exit codes:
        0 = success (continue)
        1 = error (log warning, continue)
        2 = block (abort the operation)
    - Hook can write JSON to stdout to modify data (optional)
    """

    def __init__(self, config: Config) -> None:
        self._enabled = config.hooks_enabled
        self._hooks: dict[str, list[HookConfig]] = config.hooks

    async def emit(
        self, event: HookEvent, data: dict[str, object] | None = None
    ) -> list[HookResult]:
        """Fire all hooks registered for the given event.

        Args:
            event: The lifecycle event.
            data: Context data passed to hooks as JSON on stdin.

        Returns:
            List of results from each hook execution.
        """
        if not self._enabled:
            return []

        hook_configs = self._hooks.get(event.value, [])
        if not hook_configs:
            return []

        payload = data if data is not None else {}
        results: list[HookResult] = []

        for hook_config in hook_configs:
            result = await self._execute_hook(hook_config, event, payload)
            results.append(result)

            if result.should_block:
                logger.info(
                    "Hook blocked %s: command=%r", event.value, hook_config.command
                )
                break

        return results

    async def _execute_hook(
        self,
        hook_config: HookConfig,
        event: HookEvent,
        data: dict[str, object],
    ) -> HookResult:
        """Execute a single hook command.

        Runs the command via ``asyncio.create_subprocess_shell``, passes the
        event payload as JSON on stdin, enforces the configured timeout, and
        captures stdout/stderr.
        """
        stdin_payload = json.dumps({"event": event.value, "data": data})

        logger.debug(
            "Executing hook for %s: command=%r timeout=%d",
            event.value,
            hook_config.command,
            hook_config.timeout,
        )

        try:
            process = await asyncio.create_subprocess_shell(
                hook_config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=stdin_payload.encode()),
                    timeout=hook_config.timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Hook timed out after %ds for %s: command=%r",
                    hook_config.timeout,
                    event.value,
                    hook_config.command,
                )
                process.kill()
                await process.wait()
                return HookResult(exit_code=1, stdout="", stderr="Hook timed out")

            exit_code = process.returncode if process.returncode is not None else 1
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")

            logger.debug(
                "Hook completed for %s: command=%r exit_code=%d",
                event.value,
                hook_config.command,
                exit_code,
            )

            if exit_code == 1:
                logger.warning(
                    "Hook error for %s: command=%r stderr=%s",
                    event.value,
                    hook_config.command,
                    stderr.strip(),
                )

            return HookResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

        except OSError as exc:
            logger.warning(
                "Hook failed to start for %s: command=%r error=%s",
                event.value,
                hook_config.command,
                exc,
            )
            return HookResult(exit_code=1, stdout="", stderr=str(exc))

    def has_hooks(self, event: HookEvent) -> bool:
        """Check if any hooks are registered for an event."""
        if not self._enabled:
            return False
        return len(self._hooks.get(event.value, [])) > 0

    @property
    def enabled(self) -> bool:
        """Whether the hook system is enabled."""
        return self._enabled
