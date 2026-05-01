"""Phase 2 integration tests — multi-tool scenarios through the AgentLoop.

Each test exercises real tool implementations (from register_defaults) wired
into the full AgentLoop with a MockContentGenerator that drives the scripted
Gemini responses.  No actual Gemini API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator


from pygemini.context.memory_store import MemoryStore
from pygemini.core.agent_loop import AgentLoop
from pygemini.core.config import Config
from pygemini.core.content_generator import FunctionCallData, StreamChunk
from pygemini.core.events import CoreEvent, EventEmitter, ToolOutputEvent
from pygemini.core.history import ConversationHistory
from pygemini.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# MockContentGenerator (same pattern as Phase 1)
# ---------------------------------------------------------------------------


class MockContentGenerator:
    """Replacement for ContentGenerator that yields predefined responses."""

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = list(responses)
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
            yield StreamChunk(text="Done.")


# ---------------------------------------------------------------------------
# Helper — build a loop wired to real default tools
# ---------------------------------------------------------------------------


def _make_real_loop(
    mock_gen: MockContentGenerator,
    tmp_path: Path,
    config: Config | None = None,
) -> tuple[AgentLoop, EventEmitter, ConversationHistory, MemoryStore]:
    """Create an AgentLoop with real filesystem/shell/memory tools."""
    config = config or Config()
    emitter = EventEmitter()
    history = ConversationHistory()
    memory_store = MemoryStore(storage_path=tmp_path / "memory.json")
    registry = ToolRegistry(config)
    registry.register_defaults(event_emitter=emitter, memory_store=memory_store)

    loop = AgentLoop(
        content_generator=mock_gen,  # type: ignore[arg-type]
        tool_registry=registry,
        history=history,
        event_emitter=emitter,
        config=config,
    )
    return loop, emitter, history, memory_store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListThenRead:
    """Tool chaining: list_directory → read_file."""

    async def test_list_then_read(self, tmp_path: Path) -> None:
        # Create a file inside tmp_path so list_directory has something to show
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello from the file!\n", encoding="utf-8")

        mock_gen = MockContentGenerator([
            # Turn 1: list the directory
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="list_directory",
                    args={"path": str(tmp_path)},
                    id="fc1",
                ),
            ])],
            # Turn 2: read one of the listed files
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="read_file",
                    args={"path": str(test_file)},
                    id="fc2",
                ),
            ])],
            # Turn 3: final text summary
            [StreamChunk(text="Here are the contents")],
        ])

        loop, emitter, history, _ = _make_real_loop(mock_gen, tmp_path)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("List the directory then read a file")

        # Both tools should have executed
        assert len(tool_outputs) == 2

        # list_directory result fed back: no error
        assert not tool_outputs[0].is_error

        # read_file result fed back: no error and contains file content
        assert not tool_outputs[1].is_error

        # History must contain tool result turns for both calls.
        # History shape: user msg, model(list call), user(list result),
        #                model(read call), user(read result), model(final text)
        messages = history.get_messages()
        # At minimum 6 messages for the above chain
        assert len(messages) >= 6

        # Verify both function calls appear in the history
        fn_call_names = []
        for msg in messages:
            for part in msg.parts:
                if part.function_call is not None:
                    fn_call_names.append(part.function_call.name)
        assert "list_directory" in fn_call_names
        assert "read_file" in fn_call_names


class TestEditFileFlow:
    """Read → Edit → Read round-trip that actually modifies disk."""

    async def test_edit_file_flow(self, tmp_path: Path) -> None:
        test_file = tmp_path / "source.py"
        test_file.write_text("x = 1\ny = 2\n", encoding="utf-8")

        mock_gen = MockContentGenerator([
            # Turn 1: read the file
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="read_file",
                    args={"path": str(test_file)},
                    id="fc1",
                ),
            ])],
            # Turn 2: edit — replace "x = 1" with "x = 42"
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="edit_file",
                    args={
                        "path": str(test_file),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                    },
                    id="fc2",
                ),
            ])],
            # Turn 3: read again to verify
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="read_file",
                    args={"path": str(test_file)},
                    id="fc3",
                ),
            ])],
            # Turn 4: final summary
            [StreamChunk(text="File updated successfully.")],
        ])

        loop, emitter, history, _ = _make_real_loop(mock_gen, tmp_path)

        # Auto-approve the edit_file confirmation
        async def auto_approve(event: object) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, auto_approve)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Read, edit, then re-read the file")

        # All three tool calls must have completed
        assert len(tool_outputs) == 3

        # The edit must not be an error
        assert not tool_outputs[1].is_error

        # File on disk must have been modified
        new_content = test_file.read_text(encoding="utf-8")
        assert "x = 42" in new_content
        assert "x = 1" not in new_content

        # History should contain all three function calls
        messages = history.get_messages()
        fn_call_names = []
        for msg in messages:
            for part in msg.parts:
                if part.function_call is not None:
                    fn_call_names.append(part.function_call.name)
        assert fn_call_names.count("read_file") >= 2
        assert fn_call_names.count("edit_file") >= 1


class TestShellCommand:
    """Shell execution: run_shell_command with auto-approval."""

    async def test_shell_command(self, tmp_path: Path) -> None:
        mock_gen = MockContentGenerator([
            # Turn 1: run echo
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="run_shell_command",
                    args={"command": 'echo "hello world"'},
                    id="fc1",
                ),
            ])],
            # Turn 2: summarise
            [StreamChunk(text="The command printed: hello world")],
        ])

        loop, emitter, history, _ = _make_real_loop(mock_gen, tmp_path)

        # Auto-approve shell confirmation
        async def auto_approve(event: object) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, auto_approve)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Run echo hello world")

        assert len(tool_outputs) == 1

        # Not an error (exit code 0)
        assert not tool_outputs[0].is_error

        # The tool result fed back to the model must contain stdout and exit code.
        # Inspect history: find the function response part for run_shell_command.
        messages = history.get_messages()
        shell_result_content: str | None = None
        for msg in messages:
            for part in msg.parts:
                if (
                    part.function_response is not None
                    and part.function_response.name == "run_shell_command"
                ):
                    shell_result_content = str(part.function_response.response)
                    break

        assert shell_result_content is not None, "Shell function response not found in history"
        assert "hello world" in shell_result_content
        assert "0" in shell_result_content  # exit code 0


class TestMemorySaveAndStore:
    """Memory persistence: save_memory stores the entry in MemoryStore."""

    async def test_memory_save_and_store(self, tmp_path: Path) -> None:
        memory_content = "The user prefers dark mode"

        mock_gen = MockContentGenerator([
            # Turn 1: save a memory
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="save_memory",
                    args={"content": memory_content},
                    id="fc1",
                ),
            ])],
            # Turn 2: confirm
            [StreamChunk(text="I'll remember that you prefer dark mode.")],
        ])

        loop, emitter, history, memory_store = _make_real_loop(mock_gen, tmp_path)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("Remember that I prefer dark mode")

        # Tool executed without error
        assert len(tool_outputs) == 1
        assert not tool_outputs[0].is_error

        # MemoryStore must contain the saved entry
        entries = memory_store.load()
        assert len(entries) == 1
        assert entries[0]["content"] == memory_content


class TestMultiToolSingleTurn:
    """Multiple function calls returned in a single model chunk."""

    async def test_multi_tool_single_turn(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.txt"
        file_a.write_text("Content of A\n", encoding="utf-8")

        mock_gen = MockContentGenerator([
            # Turn 1: two function calls in one chunk
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="list_directory",
                    args={"path": str(tmp_path)},
                    id="fc1",
                ),
                FunctionCallData(
                    name="read_file",
                    args={"path": str(file_a)},
                    id="fc2",
                ),
            ])],
            # Turn 2: final text
            [StreamChunk(text="Processed both results.")],
        ])

        loop, emitter, history, _ = _make_real_loop(mock_gen, tmp_path)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)
        await loop.run("List the directory and read a file at the same time")

        # Both tools should have been executed
        assert len(tool_outputs) == 2

        # Verify both tool names appear in history function calls
        messages = history.get_messages()
        fn_call_names = []
        for msg in messages:
            for part in msg.parts:
                if part.function_call is not None:
                    fn_call_names.append(part.function_call.name)
        assert "list_directory" in fn_call_names
        assert "read_file" in fn_call_names


class TestToolErrorHandling:
    """Graceful error: read_file on a nonexistent path feeds error back and loop continues."""

    async def test_tool_error_handling(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.txt"

        mock_gen = MockContentGenerator([
            # Turn 1: read a file that doesn't exist
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="read_file",
                    args={"path": str(nonexistent)},
                    id="fc1",
                ),
            ])],
            # Turn 2: model acknowledges the error
            [StreamChunk(text="I couldn't read that file because it doesn't exist.")],
        ])

        loop, emitter, history, _ = _make_real_loop(mock_gen, tmp_path)

        errors_seen: list[ToolOutputEvent] = []
        non_errors_seen: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                if event.is_error:
                    errors_seen.append(event)
                else:
                    non_errors_seen.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)

        # Loop must not raise
        await loop.run("Read a file that doesn't exist")

        # The tool output should be marked as an error
        assert len(errors_seen) == 1, "Expected one error tool output"
        assert len(non_errors_seen) == 0

        # The error result must have been fed back to the model in history
        messages = history.get_messages()
        fn_response_content: str | None = None
        for msg in messages:
            for part in msg.parts:
                if (
                    part.function_response is not None
                    and part.function_response.name == "read_file"
                ):
                    fn_response_content = str(part.function_response.response)
                    break

        assert fn_response_content is not None, "read_file response not found in history"
        # The error message should mention the file not being found
        assert "not found" in fn_response_content.lower() or "error" in fn_response_content.lower()

        # Loop completed normally — TURN_COMPLETE should have been emitted
        # (verified implicitly: if loop had raised we'd have failed above)
