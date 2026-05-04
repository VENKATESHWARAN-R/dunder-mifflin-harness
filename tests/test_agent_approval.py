"""Tests for the agent tool approval middleware.

These exercise `make_approval_wrapper` directly — building a tiny EventBus
and ApprovalPolicy, wrapping a real tool, and asserting the gate fires (or
doesn't) for each combination of approval mode and risk level. The tests do
not stand up a full pydantic_ai Agent because the wrapper's contract is
purely between the tool function, the event bus, and the policy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


from jac.agents.approval import make_approval_wrapper
from jac.runtime.approvals import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.runtime.events import (
    ApprovalRequested,
    EventBus,
    FileEditApplied,
    FileEditPreviewed,
)
from jac.tools.filesystem import edit_file, read_file, write_file
from jac.tools.shell import run_shell
from jac.tools.types import ToolStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


class _ApprovalProbe:
    """Listener that records ApprovalRequested events and replies with a
    pre-set decision so the wrapper's `await events.request_approval(...)`
    resolves without a real UI."""

    def __init__(self, events: EventBus, decision: ApprovalDecision) -> None:
        self.events = events
        self.decision = decision
        self.requests: list[ApprovalRequest] = []
        events.on(ApprovalRequested, self._on_request)

    async def _on_request(self, event: ApprovalRequested) -> None:
        self.requests.append(event.request)
        await self.events.resolve_approval(
            ApprovalResponse(request_id=event.request.id, decision=self.decision)
        )

    @property
    def called(self) -> bool:
        return bool(self.requests)


def _record(events: EventBus, event_type: type) -> list:
    """Append every emitted event of `event_type` to a list and return it."""
    captured: list = []

    async def listener(event):
        captured.append(event)

    events.on(event_type, listener)
    return captured


# ---------------------------------------------------------------------------
# read_only path: gate is skipped entirely
# ---------------------------------------------------------------------------


def test_read_only_tool_skips_approval_gate(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hi")

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(read_file, events, policy)

        result = await wrapped(path=str(f))
        return probe.called, result

    called, result = _run(scenario())
    assert called is False
    assert result.status == ToolStatus.OK
    assert result.content == "hi"


# ---------------------------------------------------------------------------
# write_file: interactive approve / deny
# ---------------------------------------------------------------------------


def test_write_file_prompts_in_interactive_mode(tmp_path: Path) -> None:
    target = tmp_path / "out.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _ApprovalProbe(events, ApprovalDecision.APPROVE_ONCE)
        wrapped = make_approval_wrapper(write_file, events, policy)

        result = await wrapped(path=str(target), content="print('hi')\n")
        return probe.requests, result

    requests, result = _run(scenario())
    assert len(requests) == 1
    assert requests[0].tool_name == "write_file"
    assert result.status == ToolStatus.OK
    assert target.read_text() == "print('hi')\n"


def test_write_file_denied_does_not_write(tmp_path: Path) -> None:
    target = tmp_path / "should_not_exist.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(write_file, events, policy)

        result = await wrapped(path=str(target), content="print('hi')\n")
        return result

    result = _run(scenario())
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert not target.exists()


# ---------------------------------------------------------------------------
# Preview-before-write ordering
# ---------------------------------------------------------------------------


def test_file_edit_previewed_emitted_before_write(tmp_path: Path) -> None:
    target = tmp_path / "preview.txt"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        previews = _record(events, FileEditPreviewed)
        applied = _record(events, FileEditApplied)
        sequence: list[str] = []

        async def on_preview(_event):
            sequence.append(f"preview:exists={target.exists()}")

        events.on(FileEditPreviewed, on_preview)

        async def on_request(event):
            sequence.append(f"approval:exists={target.exists()}")
            await events.resolve_approval(
                ApprovalResponse(
                    request_id=event.request.id,
                    decision=ApprovalDecision.APPROVE_ONCE,
                )
            )

        events.on(ApprovalRequested, on_request)
        wrapped = make_approval_wrapper(write_file, events, policy)
        await wrapped(path=str(target), content="hello\n")
        return previews, applied, sequence

    previews, applied, sequence = _run(scenario())
    assert len(previews) == 1
    assert previews[0].path == target
    assert "+hello" in previews[0].diff
    assert sequence == [
        "preview:exists=False",
        "approval:exists=False",
    ]
    assert len(applied) == 1
    assert applied[0].path == target


def test_edit_file_previewed_diff_matches_actual_change(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("hello world\n")

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        previews = _record(events, FileEditPreviewed)
        _ApprovalProbe(events, ApprovalDecision.APPROVE_ONCE)
        wrapped = make_approval_wrapper(edit_file, events, policy)
        result = await wrapped(
            path=str(target), old_string="world", new_string="python"
        )
        return previews, result

    previews, result = _run(scenario())
    assert len(previews) == 1
    assert "-hello world" in previews[0].diff
    assert "+hello python" in previews[0].diff
    assert result.status == ToolStatus.OK
    assert target.read_text() == "hello python\n"


def test_edit_file_denied_does_not_modify(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("hello world\n")

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(edit_file, events, policy)
        result = await wrapped(
            path=str(target), old_string="world", new_string="python"
        )
        return result

    result = _run(scenario())
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert target.read_text() == "hello world\n"


# ---------------------------------------------------------------------------
# Mode-specific behaviour
# ---------------------------------------------------------------------------


def test_yolo_mode_auto_approves_without_prompting(tmp_path: Path) -> None:
    target = tmp_path / "yolo.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.YOLO)
        probe = _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(write_file, events, policy)
        result = await wrapped(path=str(target), content="x")
        return probe.called, result

    called, result = _run(scenario())
    assert called is False
    assert result.status == ToolStatus.OK
    assert target.read_text() == "x"


def test_auto_edit_approves_file_writes_without_prompting(tmp_path: Path) -> None:
    target = tmp_path / "auto.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.AUTO_EDIT)
        probe = _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(write_file, events, policy)
        result = await wrapped(path=str(target), content="x")
        return probe.called, result

    called, result = _run(scenario())
    assert called is False
    assert result.status == ToolStatus.OK


def test_auto_edit_does_not_auto_approve_shell() -> None:
    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.AUTO_EDIT)
        probe = _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(run_shell, events, policy)
        result = await wrapped(command="echo hi")
        return probe.called, result

    called, result = _run(scenario())
    assert called is True  # shell still gates in auto-edit
    assert result.status == ToolStatus.PERMISSION_DENIED


def test_yolo_auto_approves_shell() -> None:
    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.YOLO)
        probe = _ApprovalProbe(events, ApprovalDecision.DENY)
        wrapped = make_approval_wrapper(run_shell, events, policy)
        result = await wrapped(command="echo gated")
        return probe.called, result

    called, result = _run(scenario())
    assert called is False
    assert result.status == ToolStatus.OK
    assert "gated" in result.stdout


# ---------------------------------------------------------------------------
# Wrapper preserves tool identity (pydantic_ai schema introspection)
# ---------------------------------------------------------------------------


def test_wrapper_preserves_name_and_signature() -> None:
    events = EventBus()
    policy = ApprovalPolicy()
    wrapped = make_approval_wrapper(write_file, events, policy)
    assert wrapped.__name__ == "write_file"
    assert wrapped.__wrapped__ is write_file


# ---------------------------------------------------------------------------
# Session-scoped allowance still works after wrapping
# ---------------------------------------------------------------------------


def test_allow_tool_for_session_skips_subsequent_prompts(tmp_path: Path) -> None:
    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _ApprovalProbe(events, ApprovalDecision.ALLOW_TOOL_FOR_SESSION)
        wrapped = make_approval_wrapper(write_file, events, policy)
        await wrapped(path=str(target_a), content="a")
        await wrapped(path=str(target_b), content="b")
        return probe.requests

    requests = _run(scenario())
    assert len(requests) == 1  # second call uses session allowance
    assert target_a.read_text() == "a"
    assert target_b.read_text() == "b"


# ---------------------------------------------------------------------------
# Compute-edit error paths surface without prompting
# ---------------------------------------------------------------------------


def test_edit_file_compute_error_short_circuits_without_gate(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"  # never created

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _ApprovalProbe(events, ApprovalDecision.APPROVE_ONCE)
        wrapped = make_approval_wrapper(edit_file, events, policy)
        result = await wrapped(path=str(target), old_string="x", new_string="y")
        return probe.called, result

    called, result = _run(scenario())
    # File missing → compute_edit returns NOT_FOUND. There's no action to
    # approve, so the wrapper short-circuits before prompting.
    assert result.status == ToolStatus.NOT_FOUND
    assert called is False


# ---------------------------------------------------------------------------
# REDIRECT decision: tool not executed; feedback returned as tool result
# ---------------------------------------------------------------------------


class _RedirectProbe:
    """Replies to every ApprovalRequested with a REDIRECT decision carrying a
    specific feedback message."""

    def __init__(self, events: EventBus, redirect_message: str) -> None:
        self.events = events
        self.redirect_message = redirect_message
        self.requests: list[ApprovalRequest] = []
        events.on(ApprovalRequested, self._on_request)

    async def _on_request(self, event: ApprovalRequested) -> None:
        self.requests.append(event.request)
        await self.events.resolve_approval(
            ApprovalResponse(
                request_id=event.request.id,
                decision=ApprovalDecision.REDIRECT,
                redirect_message=self.redirect_message,
            )
        )


def test_redirect_skips_execution_and_returns_feedback(tmp_path: Path) -> None:
    target = tmp_path / "out.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _RedirectProbe(events, "use append mode instead of overwrite")
        wrapped = make_approval_wrapper(write_file, events, policy)

        result = await wrapped(path=str(target), content="print('hi')\n")
        return probe.requests, result

    requests, result = _run(scenario())
    assert len(requests) == 1
    # Tool must NOT have executed
    assert not target.exists()
    # Result carries the user's feedback for the model
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert "use append mode instead of overwrite" in result.error


def test_redirect_without_message_returns_generic_feedback(tmp_path: Path) -> None:
    target = tmp_path / "out.py"

    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        # redirect_message omitted → wrapper fills in a generic message
        events_bus = events

        async def _on_request(event: ApprovalRequested) -> None:
            await events_bus.resolve_approval(
                ApprovalResponse(
                    request_id=event.request.id,
                    decision=ApprovalDecision.REDIRECT,
                )
            )

        events.on(ApprovalRequested, _on_request)
        wrapped = make_approval_wrapper(write_file, events, policy)
        result = await wrapped(path=str(target), content="x")
        return result

    result = _run(scenario())
    assert not target.exists()
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert "adjust" in result.error.lower() or "feedback" in result.error.lower()


def test_redirect_on_shell_tool_skips_execution() -> None:
    async def scenario():
        events = EventBus()
        policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
        probe = _RedirectProbe(events, "use tail -20 instead")
        wrapped = make_approval_wrapper(run_shell, events, policy)

        result = await wrapped(command="cat huge_file.log")
        return probe.requests, result

    requests, result = _run(scenario())
    assert len(requests) == 1
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert "tail -20" in result.error
