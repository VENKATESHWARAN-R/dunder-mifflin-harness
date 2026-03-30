"""Unit tests for pygemini.core.events.

Covers:
- EventEmitter.on / off registration
- emit calls all sync listeners
- emit works with async listeners
- request_confirmation / respond_confirmation handshake
- respond_confirmation raises RuntimeError when no request is pending
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pygemini.core.events import (
    ConfirmRequestEvent,
    CoreEvent,
    ErrorEvent,
    EventEmitter,
    StreamTextEvent,
    ToolExecutingEvent,
    ToolOutputEvent,
    TurnCompleteEvent,
)


# ---------------------------------------------------------------------------
# Event dataclass sanity checks
# ---------------------------------------------------------------------------


class TestEventDataclasses:
    def test_stream_text_event(self) -> None:
        e = StreamTextEvent(text="hello")
        assert e.text == "hello"

    def test_tool_executing_event(self) -> None:
        e = ToolExecutingEvent(tool_name="read_file", params={"path": "/tmp/x"})
        assert e.tool_name == "read_file"
        assert e.params == {"path": "/tmp/x"}

    def test_tool_output_event_defaults(self) -> None:
        e = ToolOutputEvent(display_content="ok")
        assert e.display_content == "ok"
        assert e.is_error is False

    def test_tool_output_event_error(self) -> None:
        e = ToolOutputEvent(display_content="boom", is_error=True)
        assert e.is_error is True

    def test_confirm_request_event(self) -> None:
        e = ConfirmRequestEvent(description="delete file", details={"path": "/tmp/f"})
        assert e.description == "delete file"
        assert e.details["path"] == "/tmp/f"

    def test_error_event_defaults(self) -> None:
        e = ErrorEvent(message="something went wrong")
        assert e.message == "something went wrong"
        assert e.exception is None

    def test_error_event_with_exception(self) -> None:
        exc = ValueError("oops")
        e = ErrorEvent(message="err", exception=exc)
        assert e.exception is exc

    def test_turn_complete_event(self) -> None:
        e = TurnCompleteEvent()
        assert isinstance(e, TurnCompleteEvent)

    def test_frozen_stream_text(self) -> None:
        e = StreamTextEvent(text="hi")
        with pytest.raises((AttributeError, TypeError)):
            e.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EventEmitter.on / off
# ---------------------------------------------------------------------------


class TestOnOff:
    def test_on_registers_listener(self) -> None:
        emitter = EventEmitter()
        cb = MagicMock()
        emitter.on(CoreEvent.STREAM_TEXT, cb)
        assert cb in emitter._listeners[CoreEvent.STREAM_TEXT]

    def test_off_removes_listener(self) -> None:
        emitter = EventEmitter()
        cb = MagicMock()
        emitter.on(CoreEvent.STREAM_TEXT, cb)
        emitter.off(CoreEvent.STREAM_TEXT, cb)
        assert cb not in emitter._listeners[CoreEvent.STREAM_TEXT]

    def test_off_silently_ignores_unknown(self) -> None:
        emitter = EventEmitter()
        cb = MagicMock()
        # Should not raise even though cb was never registered
        emitter.off(CoreEvent.STREAM_TEXT, cb)

    def test_multiple_listeners_registered(self) -> None:
        emitter = EventEmitter()
        cb1, cb2 = MagicMock(), MagicMock()
        emitter.on(CoreEvent.ERROR, cb1)
        emitter.on(CoreEvent.ERROR, cb2)
        listeners = emitter._listeners[CoreEvent.ERROR]
        assert cb1 in listeners
        assert cb2 in listeners

    def test_off_removes_only_target_listener(self) -> None:
        emitter = EventEmitter()
        cb1, cb2 = MagicMock(), MagicMock()
        emitter.on(CoreEvent.TURN_COMPLETE, cb1)
        emitter.on(CoreEvent.TURN_COMPLETE, cb2)
        emitter.off(CoreEvent.TURN_COMPLETE, cb1)
        assert cb1 not in emitter._listeners[CoreEvent.TURN_COMPLETE]
        assert cb2 in emitter._listeners[CoreEvent.TURN_COMPLETE]

    def test_initial_listeners_empty(self) -> None:
        emitter = EventEmitter()
        for event in CoreEvent:
            assert emitter._listeners[event] == []


# ---------------------------------------------------------------------------
# emit — sync listeners
# ---------------------------------------------------------------------------


class TestEmitSync:
    async def test_emit_calls_single_listener(self) -> None:
        emitter = EventEmitter()
        received: list[Any] = []

        def cb(data: Any) -> None:
            received.append(data)

        emitter.on(CoreEvent.STREAM_TEXT, cb)
        evt = StreamTextEvent(text="world")
        await emitter.emit(CoreEvent.STREAM_TEXT, evt)
        assert received == [evt]

    async def test_emit_calls_all_listeners(self) -> None:
        emitter = EventEmitter()
        calls: list[str] = []

        emitter.on(CoreEvent.STREAM_TEXT, lambda _: calls.append("a"))
        emitter.on(CoreEvent.STREAM_TEXT, lambda _: calls.append("b"))
        await emitter.emit(CoreEvent.STREAM_TEXT, StreamTextEvent(text="x"))
        assert calls == ["a", "b"]

    async def test_emit_no_listeners_is_noop(self) -> None:
        emitter = EventEmitter()
        # Should not raise
        await emitter.emit(CoreEvent.ERROR, ErrorEvent(message="e"))

    async def test_emit_passes_data_argument(self) -> None:
        emitter = EventEmitter()
        received: list[Any] = []
        emitter.on(CoreEvent.TOOL_OUTPUT, lambda d: received.append(d))
        evt = ToolOutputEvent(display_content="result")
        await emitter.emit(CoreEvent.TOOL_OUTPUT, evt)
        assert received[0] is evt

    async def test_emit_none_data(self) -> None:
        emitter = EventEmitter()
        received: list[Any] = []
        emitter.on(CoreEvent.TURN_COMPLETE, lambda d: received.append(d))
        await emitter.emit(CoreEvent.TURN_COMPLETE, None)
        assert received == [None]

    async def test_removed_listener_not_called(self) -> None:
        emitter = EventEmitter()
        calls: list[int] = []
        cb = lambda _: calls.append(1)  # noqa: E731
        emitter.on(CoreEvent.STREAM_TEXT, cb)
        emitter.off(CoreEvent.STREAM_TEXT, cb)
        await emitter.emit(CoreEvent.STREAM_TEXT, StreamTextEvent(text="hi"))
        assert calls == []

    async def test_emit_different_events_isolated(self) -> None:
        emitter = EventEmitter()
        stream_calls: list[Any] = []
        error_calls: list[Any] = []
        emitter.on(CoreEvent.STREAM_TEXT, lambda d: stream_calls.append(d))
        emitter.on(CoreEvent.ERROR, lambda d: error_calls.append(d))
        await emitter.emit(CoreEvent.STREAM_TEXT, StreamTextEvent(text="s"))
        assert len(stream_calls) == 1
        assert error_calls == []


# ---------------------------------------------------------------------------
# emit — async listeners
# ---------------------------------------------------------------------------


class TestEmitAsync:
    async def test_emit_awaits_async_listener(self) -> None:
        emitter = EventEmitter()
        received: list[Any] = []

        async def async_cb(data: Any) -> None:
            received.append(data)

        emitter.on(CoreEvent.STREAM_TEXT, async_cb)
        evt = StreamTextEvent(text="async!")
        await emitter.emit(CoreEvent.STREAM_TEXT, evt)
        assert received == [evt]

    async def test_emit_mixed_sync_and_async(self) -> None:
        emitter = EventEmitter()
        order: list[str] = []

        def sync_cb(_: Any) -> None:
            order.append("sync")

        async def async_cb(_: Any) -> None:
            order.append("async")

        emitter.on(CoreEvent.TOOL_EXECUTING, sync_cb)
        emitter.on(CoreEvent.TOOL_EXECUTING, async_cb)
        await emitter.emit(CoreEvent.TOOL_EXECUTING, ToolExecutingEvent("t", {}))
        assert order == ["sync", "async"]

    async def test_emit_multiple_async_listeners(self) -> None:
        emitter = EventEmitter()
        results: list[int] = []

        async def cb1(_: Any) -> None:
            results.append(1)

        async def cb2(_: Any) -> None:
            results.append(2)

        emitter.on(CoreEvent.ERROR, cb1)
        emitter.on(CoreEvent.ERROR, cb2)
        await emitter.emit(CoreEvent.ERROR, ErrorEvent(message="e"))
        assert results == [1, 2]

    async def test_async_mock_listener(self) -> None:
        emitter = EventEmitter()
        mock_cb = AsyncMock()
        emitter.on(CoreEvent.TURN_COMPLETE, mock_cb)
        await emitter.emit(CoreEvent.TURN_COMPLETE, TurnCompleteEvent())
        mock_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# request_confirmation / respond_confirmation handshake
# ---------------------------------------------------------------------------


class TestConfirmationHandshake:
    async def test_approved_true(self) -> None:
        emitter = EventEmitter()

        async def responder(data: Any) -> None:
            # Respond asynchronously after the request is registered
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)
        result = await emitter.request_confirmation("delete?", {"file": "/tmp/x"})
        assert result is True

    async def test_approved_false(self) -> None:
        emitter = EventEmitter()

        async def responder(data: Any) -> None:
            emitter.respond_confirmation(False)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)
        result = await emitter.request_confirmation("delete?", {})
        assert result is False

    async def test_confirm_request_event_payload(self) -> None:
        emitter = EventEmitter()
        captured: list[ConfirmRequestEvent] = []

        async def responder(data: ConfirmRequestEvent) -> None:
            captured.append(data)
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)
        await emitter.request_confirmation("do thing", {"key": "value"})
        assert len(captured) == 1
        assert captured[0].description == "do thing"
        assert captured[0].details == {"key": "value"}

    async def test_queue_cleared_after_handshake(self) -> None:
        emitter = EventEmitter()

        async def responder(_: Any) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)
        await emitter.request_confirmation("first", {})
        # After resolution, _confirmation_queue should be None
        assert emitter._confirmation_queue is None

    async def test_respond_via_task(self) -> None:
        """Respond from an asyncio.create_task rather than from a listener."""
        emitter = EventEmitter()
        events_seen: list[ConfirmRequestEvent] = []

        def sync_listener(data: ConfirmRequestEvent) -> None:
            events_seen.append(data)
            # Schedule response as a separate task
            asyncio.get_event_loop().call_soon(emitter.respond_confirmation, True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, sync_listener)
        result = await emitter.request_confirmation("via task", {"x": 1})
        assert result is True
        assert len(events_seen) == 1

    async def test_sequential_confirmations(self) -> None:
        """Two consecutive handshakes must both complete independently."""
        emitter = EventEmitter()

        async def responder(data: Any) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)

        r1 = await emitter.request_confirmation("first", {})
        r2 = await emitter.request_confirmation("second", {})
        assert r1 is True
        assert r2 is True

    async def test_respond_confirmation_no_pending_raises(self) -> None:
        emitter = EventEmitter()
        with pytest.raises(RuntimeError, match="No confirmation request is pending"):
            emitter.respond_confirmation(True)

    async def test_respond_confirmation_after_completion_raises(self) -> None:
        """After a handshake completes the queue is cleared; a second respond must raise."""
        emitter = EventEmitter()

        async def responder(_: Any) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, responder)
        await emitter.request_confirmation("q", {})

        with pytest.raises(RuntimeError):
            emitter.respond_confirmation(False)
