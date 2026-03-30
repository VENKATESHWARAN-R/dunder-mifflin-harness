"""Final integration tests — approval, policy, hooks, and full pipeline.

Exercises the complete system with ApprovalManager, PolicyEngine,
HookManager, and context injection wired into the AgentLoop with
real tool implementations via MockContentGenerator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator
from pygemini.context.memory_store import MemoryStore
from pygemini.core.agent_loop import AgentLoop
from pygemini.core.config import Config, HookConfig
from pygemini.core.content_generator import FunctionCallData, StreamChunk
from pygemini.core.events import (
    ConfirmRequestEvent,
    CoreEvent,
    EventEmitter,
    ToolOutputEvent,
)
from pygemini.core.history import ConversationHistory
from pygemini.hooks.manager import HookEvent, HookManager, HookResult
from pygemini.safety.approval import ApprovalManager
from pygemini.safety.policy import PolicyEngine, PolicyRule
from pygemini.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# MockContentGenerator (same pattern as phase 2)
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
# Helper — build a loop with safety components
# ---------------------------------------------------------------------------


def _make_loop(
    mock_gen: MockContentGenerator,
    tmp_path: Path,
    config: Config | None = None,
    hook_manager: object | None = None,
    approval_manager: object | None = None,
    policy_engine: object | None = None,
    context_content: str = "",
) -> tuple[AgentLoop, EventEmitter, ConversationHistory, MemoryStore]:
    """Create an AgentLoop with real tools and optional safety components."""
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
        context_content=context_content,
        hook_manager=hook_manager,
        approval_manager=approval_manager,
        policy_engine=policy_engine,
    )
    return loop, emitter, history, memory_store


# ---------------------------------------------------------------------------
# 1. TestApprovalModeYolo
# ---------------------------------------------------------------------------


class TestApprovalModeYolo:
    """Yolo mode auto-approves everything — no confirmation prompts."""

    async def test_write_file_no_confirmation(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        config = Config(approval_mode="yolo")
        approval = ApprovalManager(config)

        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="write_file",
                    args={"path": str(target), "content": "yolo content"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="File written.")],
        ])

        loop, emitter, _, _ = _make_loop(
            mock_gen, tmp_path, config=config, approval_manager=approval,
        )

        confirm_requests: list[ConfirmRequestEvent] = []

        async def track_confirm(event: object) -> None:
            if isinstance(event, ConfirmRequestEvent):
                confirm_requests.append(event)
                # Respond just in case, but we expect this NOT to be called.
                emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, track_confirm)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)

        await loop.run("Write a file")

        # Tool executed successfully
        assert len(tool_outputs) == 1
        assert not tool_outputs[0].is_error

        # File was actually written
        assert target.read_text(encoding="utf-8") == "yolo content"

        # No confirmation was requested
        assert len(confirm_requests) == 0


# ---------------------------------------------------------------------------
# 2. TestApprovalModeAutoEdit
# ---------------------------------------------------------------------------


class TestApprovalModeAutoEdit:
    """auto_edit mode auto-approves file edits but requires confirmation for shell."""

    async def test_edit_auto_approved_shell_needs_confirm(self, tmp_path: Path) -> None:
        source = tmp_path / "src.py"
        source.write_text("x = 1\n", encoding="utf-8")

        config = Config(approval_mode="auto_edit")
        approval = ApprovalManager(config)

        mock_gen = MockContentGenerator([
            # Turn 1: edit_file (should auto-approve)
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="edit_file",
                    args={
                        "path": str(source),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                    },
                    id="fc1",
                ),
            ])],
            # Turn 2: run_shell_command (should require confirmation)
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="run_shell_command",
                    args={"command": "echo done"},
                    id="fc2",
                ),
            ])],
            # Turn 3: summary
            [StreamChunk(text="All done.")],
        ])

        loop, emitter, _, _ = _make_loop(
            mock_gen, tmp_path, config=config, approval_manager=approval,
        )

        confirm_requests: list[ConfirmRequestEvent] = []

        async def track_confirm(event: object) -> None:
            if isinstance(event, ConfirmRequestEvent):
                confirm_requests.append(event)
                emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, track_confirm)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)

        await loop.run("Edit the file then run a command")

        # Both tools executed
        assert len(tool_outputs) == 2

        # edit_file did NOT trigger a confirmation; shell DID
        # Only one confirmation should have been requested (for shell)
        assert len(confirm_requests) == 1

        # File was actually edited
        assert "x = 42" in source.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. TestPolicyDeny
# ---------------------------------------------------------------------------


class TestPolicyDeny:
    """Policy engine denies a tool — returns denial without execution."""

    async def test_shell_denied_by_policy(self, tmp_path: Path) -> None:
        config = Config()
        policy = PolicyEngine(config)
        policy.add_rule(PolicyRule(
            tool_pattern="run_shell_command",
            action="deny",
        ))

        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="run_shell_command",
                    args={"command": "rm -rf /"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="Could not run the command.")],
        ])

        loop, emitter, history, _ = _make_loop(
            mock_gen, tmp_path, config=config, policy_engine=policy,
        )

        await loop.run("Run a dangerous command")

        # Check history for the denial response — policy denial returns
        # the response dict directly without emitting TOOL_OUTPUT.
        messages = history.get_messages()
        denied = False
        for msg in messages:
            for part in msg.parts:
                if (
                    part.function_response is not None
                    and part.function_response.name == "run_shell_command"
                ):
                    resp = str(part.function_response.response)
                    if "denied by policy" in resp.lower():
                        denied = True
        assert denied, "Expected 'denied by policy' in function response"


# ---------------------------------------------------------------------------
# 4. TestPolicyAllow
# ---------------------------------------------------------------------------


class TestPolicyAllow:
    """Policy engine allows a tool — executes without confirmation."""

    async def test_write_file_allowed_by_policy(self, tmp_path: Path) -> None:
        target = tmp_path / "policy_allowed.txt"
        config = Config()
        policy = PolicyEngine(config)
        policy.add_rule(PolicyRule(
            tool_pattern="write_file",
            action="allow",
        ))

        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="write_file",
                    args={"path": str(target), "content": "policy allowed"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="Written.")],
        ])

        loop, emitter, _, _ = _make_loop(
            mock_gen, tmp_path, config=config, policy_engine=policy,
        )

        confirm_requests: list[ConfirmRequestEvent] = []

        async def track_confirm(event: object) -> None:
            if isinstance(event, ConfirmRequestEvent):
                confirm_requests.append(event)
                emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, track_confirm)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)

        await loop.run("Write a file")

        # Tool executed
        assert len(tool_outputs) == 1
        assert not tool_outputs[0].is_error

        # No confirmation prompt (policy said "allow")
        assert len(confirm_requests) == 0

        # File written
        assert target.read_text(encoding="utf-8") == "policy allowed"


# ---------------------------------------------------------------------------
# 5. TestHooksBeforeAfter
# ---------------------------------------------------------------------------


class TestHooksBeforeAfter:
    """Hook manager fires before/after agent hooks."""

    async def test_hooks_called(self, tmp_path: Path) -> None:
        config = Config(
            hooks_enabled=True,
            hooks={
                "before_agent": [HookConfig(command="echo before")],
                "after_agent": [HookConfig(command="echo after")],
            },
        )
        hook_manager = HookManager(config)

        mock_gen = MockContentGenerator([
            [StreamChunk(text="Hello from model.")],
        ])

        emitted_events: list[str] = []
        original_emit = hook_manager.emit

        async def tracking_emit(
            event: HookEvent, data: dict[str, object] | None = None
        ) -> list[HookResult]:
            emitted_events.append(event.value)
            return await original_emit(event, data)

        hook_manager.emit = tracking_emit  # type: ignore[assignment]

        loop, emitter, _, _ = _make_loop(
            mock_gen, tmp_path, config=config, hook_manager=hook_manager,
        )

        await loop.run("Say hello")

        # before_agent and after_agent hooks should have been emitted
        assert "before_agent" in emitted_events
        assert "after_agent" in emitted_events


# ---------------------------------------------------------------------------
# 6. TestHookBlocks
# ---------------------------------------------------------------------------


class TestHookBlocks:
    """A hook returning exit_code=2 blocks the tool."""

    async def test_hook_blocks_tool(self, tmp_path: Path) -> None:
        config = Config(
            hooks_enabled=True,
            hooks={
                "before_tool": [HookConfig(command="exit 2", timeout=5)],
            },
        )
        hook_manager = HookManager(config)

        target = tmp_path / "blocked.txt"

        mock_gen = MockContentGenerator([
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="write_file",
                    args={"path": str(target), "content": "should not write"},
                    id="fc1",
                ),
            ])],
            [StreamChunk(text="Blocked.")],
        ])

        loop, emitter, history, _ = _make_loop(
            mock_gen, tmp_path, config=config, hook_manager=hook_manager,
        )

        await loop.run("Write a file")

        # File should NOT have been created
        assert not target.exists()

        # History should contain "Blocked by hook" — hook blocking returns
        # the response dict directly without emitting TOOL_OUTPUT.
        messages = history.get_messages()
        blocked = False
        for msg in messages:
            for part in msg.parts:
                if (
                    part.function_response is not None
                    and part.function_response.name == "write_file"
                ):
                    resp = str(part.function_response.response)
                    if "blocked by hook" in resp.lower():
                        blocked = True
        assert blocked, "Expected 'Blocked by hook' in function response"


# ---------------------------------------------------------------------------
# 7. TestContextInjection
# ---------------------------------------------------------------------------


class TestContextInjection:
    """Context content appears in the system prompt passed to the model."""

    async def test_system_prompt_includes_context(self, tmp_path: Path) -> None:
        context_text = "This project uses FastAPI and PostgreSQL."

        captured_system_prompts: list[str] = []

        class CapturingMockGen:
            """Mock that captures the system_instruction argument."""

            async def generate_stream(
                self,
                history: list,
                tools: list[dict[str, Any]],
                system_instruction: str,
            ) -> AsyncIterator[StreamChunk]:
                captured_system_prompts.append(system_instruction)
                yield StreamChunk(text="Understood.")

        loop, emitter, _, _ = _make_loop(
            CapturingMockGen(),  # type: ignore[arg-type]
            tmp_path,
            context_content=context_text,
        )

        await loop.run("Tell me about the project")

        assert len(captured_system_prompts) == 1
        assert context_text in captured_system_prompts[0]


# ---------------------------------------------------------------------------
# 8. TestFullPipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end pipeline: read → edit → shell with interactive approval."""

    async def test_full_read_edit_shell(self, tmp_path: Path) -> None:
        source = tmp_path / "app.py"
        source.write_text("version = '1.0'\n", encoding="utf-8")

        config = Config(approval_mode="interactive")
        approval = ApprovalManager(config)
        policy = PolicyEngine(config)  # no rules — defaults to confirm
        # No hooks
        hook_config = Config(hooks_enabled=False)
        hook_manager = HookManager(hook_config)

        mock_gen = MockContentGenerator([
            # Turn 1: read the file
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="read_file",
                    args={"path": str(source)},
                    id="fc1",
                ),
            ])],
            # Turn 2: edit the file
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="edit_file",
                    args={
                        "path": str(source),
                        "old_string": "version = '1.0'",
                        "new_string": "version = '2.0'",
                    },
                    id="fc2",
                ),
            ])],
            # Turn 3: run shell to verify
            [StreamChunk(function_calls=[
                FunctionCallData(
                    name="run_shell_command",
                    args={"command": f"cat {source}"},
                    id="fc3",
                ),
            ])],
            # Turn 4: summary
            [StreamChunk(text="Version bumped from 1.0 to 2.0.")],
        ])

        loop, emitter, history, _ = _make_loop(
            mock_gen,
            tmp_path,
            config=config,
            approval_manager=approval,
            policy_engine=policy,
            hook_manager=hook_manager,
        )

        # Auto-approve all confirmations
        async def auto_approve(event: object) -> None:
            emitter.respond_confirmation(True)

        emitter.on(CoreEvent.CONFIRM_REQUEST, auto_approve)

        tool_outputs: list[ToolOutputEvent] = []

        async def on_tool(event: object) -> None:
            if isinstance(event, ToolOutputEvent):
                tool_outputs.append(event)

        emitter.on(CoreEvent.TOOL_OUTPUT, on_tool)

        await loop.run("Bump the version and verify")

        # All three tools executed
        assert len(tool_outputs) == 3
        assert all(not o.is_error for o in tool_outputs)

        # File was actually modified
        content = source.read_text(encoding="utf-8")
        assert "version = '2.0'" in content
        assert "version = '1.0'" not in content

        # History has all three function calls
        messages = history.get_messages()
        fn_call_names = []
        for msg in messages:
            for part in msg.parts:
                if part.function_call is not None:
                    fn_call_names.append(part.function_call.name)
        assert "read_file" in fn_call_names
        assert "edit_file" in fn_call_names
        assert "run_shell_command" in fn_call_names

        # Shell result in history contains the updated version
        for msg in messages:
            for part in msg.parts:
                if (
                    part.function_response is not None
                    and part.function_response.name == "run_shell_command"
                ):
                    resp = str(part.function_response.response)
                    assert "2.0" in resp
