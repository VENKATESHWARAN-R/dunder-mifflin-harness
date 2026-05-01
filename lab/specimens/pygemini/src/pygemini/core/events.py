"""Core-to-CLI event bridge.

Defines the event types, event data classes, and the ``EventEmitter`` that
lets the core layer communicate with the CLI layer without importing it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

__all__ = [
    "CoreEvent",
    "StreamTextEvent",
    "ToolExecutingEvent",
    "ToolOutputEvent",
    "ConfirmRequestEvent",
    "ErrorEvent",
    "TurnCompleteEvent",
    "EventEmitter",
]


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------


class CoreEvent(Enum):
    """Enumeration of events the core layer can emit."""

    STREAM_TEXT = "stream_text"
    TOOL_EXECUTING = "tool_executing"
    TOOL_OUTPUT = "tool_output"
    CONFIRM_REQUEST = "confirm_request"
    ERROR = "error"
    TURN_COMPLETE = "turn_complete"


# ---------------------------------------------------------------------------
# Event data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamTextEvent:
    """A chunk of streamed model text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolExecutingEvent:
    """Emitted when a tool is about to execute."""

    tool_name: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolOutputEvent:
    """Result from a tool execution."""

    display_content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmRequestEvent:
    """Asks the CLI layer to confirm an action before proceeding."""

    description: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """An error that should be surfaced to the user."""

    message: str
    exception: Exception | None = None


@dataclass(frozen=True, slots=True)
class TurnCompleteEvent:
    """Signals the end of an agent turn."""


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """Async event emitter bridging core and CLI layers.

    Listeners are async callables registered per :class:`CoreEvent`.  The
    special :meth:`request_confirmation` / :meth:`respond_confirmation` pair
    implements a blocking (awaiting) handshake so the core can pause until the
    CLI layer gathers a yes/no answer from the user.
    """

    def __init__(self) -> None:
        self._listeners: dict[CoreEvent, list[Callable[..., Any]]] = {
            event: [] for event in CoreEvent
        }
        self._confirmation_queue: asyncio.Queue[bool] | None = None

    # -- listener management -------------------------------------------------

    def on(self, event: CoreEvent, callback: Callable[..., Any]) -> None:
        """Register *callback* to be invoked when *event* is emitted."""
        self._listeners[event].append(callback)

    def off(self, event: CoreEvent, callback: Callable[..., Any]) -> None:
        """Remove a previously registered *callback* for *event*."""
        try:
            self._listeners[event].remove(callback)
        except ValueError:
            pass  # Silently ignore if the callback was not registered.

    # -- emitting ------------------------------------------------------------

    async def emit(self, event: CoreEvent, data: Any = None) -> None:
        """Invoke all listeners registered for *event* with *data*.

        Listeners may be sync or async callables.  Async listeners are
        awaited; sync listeners are called directly.
        """
        for callback in self._listeners[event]:
            result = callback(data)
            if asyncio.iscoroutine(result):
                await result

    # -- confirmation handshake ----------------------------------------------

    async def request_confirmation(
        self, description: str, details: dict[str, Any]
    ) -> bool:
        """Emit a :class:`ConfirmRequestEvent` and block until the CLI responds.

        The CLI layer is expected to call :meth:`respond_confirmation` with a
        boolean after presenting the request to the user.

        Returns:
            ``True`` if the user approved, ``False`` otherwise.
        """
        self._confirmation_queue = asyncio.Queue(maxsize=1)
        event_data = ConfirmRequestEvent(description=description, details=details)
        await self.emit(CoreEvent.CONFIRM_REQUEST, event_data)
        approved = await self._confirmation_queue.get()
        self._confirmation_queue = None
        return approved

    def respond_confirmation(self, approved: bool) -> None:
        """Provide the user's confirmation answer.

        Must only be called while a :meth:`request_confirmation` is awaiting.

        Raises:
            RuntimeError: If no confirmation is currently pending.
        """
        if self._confirmation_queue is None:
            raise RuntimeError("No confirmation request is pending.")
        self._confirmation_queue.put_nowait(approved)
