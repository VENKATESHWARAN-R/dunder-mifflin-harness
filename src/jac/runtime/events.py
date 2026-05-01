"""Typed runtime events and request/response bridge."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from dunder_mifflin_harness.runtime.approvals import (
    ApprovalRequest,
    ApprovalResponse,
)
from dunder_mifflin_harness.runtime.questions import (
    QuestionRequest,
    QuestionResponse,
)

EventT = TypeVar("EventT")
EventListener = Callable[[Any], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Base class for typed events."""


@dataclass(frozen=True, slots=True)
class RunStarted(RuntimeEvent):
    run_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class RunCompleted(RuntimeEvent):
    run_id: str
    output: str = ""


@dataclass(frozen=True, slots=True)
class RunFailed(RuntimeEvent):
    run_id: str
    message: str
    exception: Exception | None = None


@dataclass(frozen=True, slots=True)
class AgentTextDelta(RuntimeEvent):
    text: str


@dataclass(frozen=True, slots=True)
class AgentMessageCompleted(RuntimeEvent):
    message: str


@dataclass(frozen=True, slots=True)
class NodeStarted(RuntimeEvent):
    node_name: str


@dataclass(frozen=True, slots=True)
class NodeCompleted(RuntimeEvent):
    node_name: str
    status: str = "success"


@dataclass(frozen=True, slots=True)
class NodeFailed(RuntimeEvent):
    node_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ToolCallRequested(RuntimeEvent):
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallCompleted(RuntimeEvent):
    tool_name: str
    display_content: str = ""
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ApprovalRequested(RuntimeEvent):
    request: ApprovalRequest


@dataclass(frozen=True, slots=True)
class ApprovalResolved(RuntimeEvent):
    response: ApprovalResponse


@dataclass(frozen=True, slots=True)
class QuestionRequested(RuntimeEvent):
    request: QuestionRequest


@dataclass(frozen=True, slots=True)
class QuestionAnswered(RuntimeEvent):
    response: QuestionResponse


@dataclass(frozen=True, slots=True)
class FileEditPreviewed(RuntimeEvent):
    path: Path
    diff: str


@dataclass(frozen=True, slots=True)
class FileEditApplied(RuntimeEvent):
    path: Path


@dataclass(frozen=True, slots=True)
class ShellCommandStarted(RuntimeEvent):
    command: str
    cwd: Path
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ShellCommandCompleted(RuntimeEvent):
    command: str
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class CostUpdated(RuntimeEvent):
    summary: str


@dataclass(frozen=True, slots=True)
class StateUpdated(RuntimeEvent):
    key: str
    value: Any


@dataclass(frozen=True, slots=True)
class WarningRaised(RuntimeEvent):
    message: str


class EventBus:
    """Async typed event bus with explicit approval/question handshakes."""

    def __init__(self) -> None:
        self._listeners: dict[type[Any], list[EventListener]] = defaultdict(list)
        self._approval_waiters: dict[str, asyncio.Future[ApprovalResponse]] = {}
        self._question_waiters: dict[str, asyncio.Future[QuestionResponse]] = {}

    def on(self, event_type: type[EventT], listener: Callable[[EventT], Any]) -> None:
        """Subscribe a sync or async listener to a concrete event type."""
        self._listeners[event_type].append(listener)

    def off(self, event_type: type[EventT], listener: Callable[[EventT], Any]) -> None:
        """Remove a listener if it is currently subscribed."""
        try:
            self._listeners[event_type].remove(listener)
        except ValueError:
            pass

    async def emit(self, event: RuntimeEvent) -> None:
        """Emit an event to listeners registered for its concrete type."""
        for listener in list(self._listeners[type(event)]):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Emit an approval request and wait for a response."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResponse] = loop.create_future()
        self._approval_waiters[request.id] = future
        await self.emit(ApprovalRequested(request=request))
        try:
            return await future
        finally:
            self._approval_waiters.pop(request.id, None)

    async def resolve_approval(self, response: ApprovalResponse) -> None:
        """Resolve a pending approval request."""
        future = self._approval_waiters.get(response.request_id)
        if future is None:
            msg = f"no pending approval request: {response.request_id}"
            raise RuntimeError(msg)
        if not future.done():
            future.set_result(response)
        await self.emit(ApprovalResolved(response=response))

    async def request_question(self, request: QuestionRequest) -> QuestionResponse:
        """Emit a question request and wait for a response."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[QuestionResponse] = loop.create_future()
        self._question_waiters[request.id] = future
        await self.emit(QuestionRequested(request=request))
        try:
            return await future
        finally:
            self._question_waiters.pop(request.id, None)

    async def answer_question(self, response: QuestionResponse) -> None:
        """Resolve a pending question request."""
        future = self._question_waiters.get(response.request_id)
        if future is None:
            msg = f"no pending question request: {response.request_id}"
            raise RuntimeError(msg)
        if not future.done():
            future.set_result(response)
        await self.emit(QuestionAnswered(response=response))
