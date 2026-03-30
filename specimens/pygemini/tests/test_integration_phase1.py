"""Phase 1 integration tests — end-to-end agent loop with mock Gemini API."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator


from pygemini.core.agent_loop import AgentLoop
from pygemini.core.config import Config
from pygemini.core.content_generator import FunctionCallData, StreamChunk
from pygemini.core.events import CoreEvent, EventEmitter, StreamTextEvent, ToolOutputEvent
from pygemini.core.history import ConversationHistory
from pygemini.tools.base import BaseTool, ToolConfirmation, ToolResult
from pygemini.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Mock tool for testing
# ---------------------------------------------------------------------------


class MockReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a file"

    @property
    def parameter_schema(self) -> dict:
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, params: dict, abort_signal: asyncio.Event | None = None) -> ToolResult:
        return ToolResult(
            llm_content=f"Contents of {params['path']}: Hello, World!",
            display_content=f"Read {params['path']}",
        )


class MockWriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write a file"

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

    def should_confirm(self, params: dict) -> ToolConfirmation | None:
        return ToolConfirmation(
            description=f"Write to {params.get('path', '?')}",
            details={"path": params.get("path", "?")},
        )

    async def execute(self, params: dict, abort_signal: asyncio.Event | None = None) -> ToolResult:
        return ToolResult(
            llm_content=f"Wrote to {params['path']}",
            display_content=f"Created {params['path']}",
        )


# ---------------------------------------------------------------------------
# Mock ContentGenerator
# ---------------------------------------------------------------------------


class MockContentGenerator:
    """Replacement for ContentGenerator that yields predefined responses."""

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = list(responses)  # copy
        self._call_count = 0

    async def generate_stream(
        self,
        history: list,
        tools: list[dict[str, Any]],
        system_instruction: str,
    ) -> AsyncIterator[StreamChunk]:
        if self._call_count < len(self._responses):
            chunks = self._responses[self._call_count]
            self._call_count += 1
            for chunk in chunks:
                yield chunk
        else:
            # Default: empty text response
            yield StreamChunk(text="Done.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_loop(
    mock_gen: MockContentGenerator,
    config: Config | None = None,
) -> tuple[AgentLoop, EventEmitter, ConversationHistory]:
    config = config or Config()
    emitter = EventEmitter()
    history = ConversationHistory()
    registry = ToolRegistry(config)
    registry.register(MockReadTool())
    registry.register(MockWriteTool())

    loop = AgentLoop(
        content_generator=mock_gen,  # type: ignore[arg-type]
        tool_registry=registry,
        history=history,
        event_emitter=emitter,
        config=config,
    )
    return loop, emitter, history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextOnlyResponse:
    """Model returns text, no tool calls."""

    async def test_text_response_emits_stream_events(self) -> None:
        mock_gen = MockContentGenerator([
            [StreamChunk(text="Hello "), StreamChunk(text="world!")],
        ])
        loop, emitter, history = _make_loop(mock_gen)

        collected_text: list[str] = []

        async def on_text(event: object) -> None:
            if isinstance(event, StreamTextEvent):
                collected_text.append(event.text)

        emitter.on(CoreEvent.STREAM_TEXT, on_text)
        await loop.run("Say hello")

        assert collected_text == ["Hello ", "world!"]
        # History should have: user message + model response = 2
        assert len(history) == 2

    async def test_turn_complete_emitted(self) -> None:
        mock_gen = MockContentGenerator([
            [StreamChunk(text="Response")],
        ])
        loop, emitter, _ = _make_loop(mock_gen)

        completed = False

        async def on_complete(_event: object) -> None:
            nonlocal completed
            completed = True

        emitter.on(CoreEvent.TURN_COMPLETE, on_complete)
        await loop.run("Test")

        assert completed


class TestSingleToolCall:
    """Model calls one tool, then returns text."""

    async def test_tool_call_and_response(self) -> None:
        mock_gen = MockContentGenerator([
            # Turn 1: model calls read_file
            [StreamChunk(function_calls=[
                FunctionCallData(name="read_file", args={"path": "test.py"}, id="fc1"),
            ])],
            # Turn 2: model responds with text after getting tool result
            [StreamChunk(text="The file contains Hello, World!")],
        ])
        loop, emitter, history = _make_loop(mock_gen)

        tool_outputs: list[str] = []

        async def on_tool_output(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event.display_content)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool_output)
        await loop.run("Read test.py")

        assert len(tool_outputs) == 1
        assert "Read test.py" in tool_outputs[0]


class TestToolWithConfirmation:
    """Model calls a tool that requires confirmation."""

    async def test_approved_confirmation(self) -> None:
        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="write_file",
                    args={"path": "out.py", "content": "# new"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="File written.")],
        ])
        loop, emitter, _ = _make_loop(mock_gen)

        # Auto-approve confirmations
        async def auto_approve(event: object) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, auto_approve)

        tool_outputs: list[str] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event.display_content)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Write a file")

        assert len(tool_outputs) == 1
        assert "out.py" in tool_outputs[0]

    async def test_denied_confirmation(self) -> None:
        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="write_file",
                    args={"path": "out.py", "content": "# new"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="OK, I won't write the file.")],
        ])
        loop, emitter, _ = _make_loop(mock_gen)

        # Auto-deny confirmations
        async def auto_deny(event: object) -> None:
            emitter.respond_confirmation(False)

        emitter.on(CoreEvent.CONFIRM_REQUEST, auto_deny)

        tool_outputs: list[str] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event.display_content)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Write a file")

        # Tool should not have executed (denial)
        assert len(tool_outputs) == 0


class TestToolNotFound:
    """Model calls a tool that doesn't exist."""

    async def test_unknown_tool_returns_error(self) -> None:
        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(name="nonexistent_tool", args={}, id="fc1"),
            ])],
            [StreamChunk(text="Sorry, that tool doesn't exist.")],
        ])
        loop, emitter, _ = _make_loop(mock_gen)
        await loop.run("Use a nonexistent tool")
        # Should not crash — error is fed back to model


class TestChainedToolCalls:
    """Model calls multiple tools in sequence."""

    async def test_two_tool_calls_in_sequence(self) -> None:
        mock_gen = MockContentGenerator([
            # Turn 1: read_file
            [StreamChunk(function_calls=[
                FunctionCallData(name="read_file", args={"path": "a.py"}, id="fc1"),
            ])],
            # Turn 2: another read_file
            [StreamChunk(function_calls=[
                FunctionCallData(name="read_file", args={"path": "b.py"}, id="fc2"),
            ])],
            # Turn 3: final text
            [StreamChunk(text="Both files read successfully.")],
        ])
        loop, emitter, history = _make_loop(mock_gen)

        tool_count = 0

        async def on_tool(event: object) -> None:
            nonlocal tool_count
            if isinstance(event, ToolOutputEvent):
                tool_count += 1

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Read both files")

        assert tool_count == 2
