"""Tests for pygemini.hooks.manager (HookManager, HookEvent, HookResult)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from pygemini.core.config import Config, HookConfig
from pygemini.hooks.manager import HookEvent, HookManager, HookResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    hooks: dict[str, list[HookConfig]] | None = None,
    hooks_enabled: bool = True,
) -> HookManager:
    """Build a HookManager from a minimal Config."""
    config = Config(
        hooks_enabled=hooks_enabled,
        hooks=hooks or {},
    )
    return HookManager(config)


# ---------------------------------------------------------------------------
# TestHookEvent
# ---------------------------------------------------------------------------


class TestHookEvent:
    """HookEvent enum should have the correct string values."""

    def test_before_agent_value(self) -> None:
        assert HookEvent.BEFORE_AGENT.value == "before_agent"

    def test_after_agent_value(self) -> None:
        assert HookEvent.AFTER_AGENT.value == "after_agent"

    def test_before_tool_value(self) -> None:
        assert HookEvent.BEFORE_TOOL.value == "before_tool"

    def test_after_tool_value(self) -> None:
        assert HookEvent.AFTER_TOOL.value == "after_tool"

    def test_before_model_value(self) -> None:
        assert HookEvent.BEFORE_MODEL.value == "before_model"

    def test_after_model_value(self) -> None:
        assert HookEvent.AFTER_MODEL.value == "after_model"

    def test_session_start_value(self) -> None:
        assert HookEvent.SESSION_START.value == "session_start"

    def test_session_end_value(self) -> None:
        assert HookEvent.SESSION_END.value == "session_end"

    def test_pre_compress_value(self) -> None:
        assert HookEvent.PRE_COMPRESS.value == "pre_compress"

    def test_notification_value(self) -> None:
        assert HookEvent.NOTIFICATION.value == "notification"


# ---------------------------------------------------------------------------
# TestHookResult
# ---------------------------------------------------------------------------


class TestHookResult:
    """HookResult dataclass and its convenience properties."""

    def test_exit_code_zero_is_success(self) -> None:
        result = HookResult(exit_code=0, stdout="ok", stderr="")
        assert result.success is True
        assert result.should_block is False

    def test_exit_code_one_is_not_success(self) -> None:
        result = HookResult(exit_code=1, stdout="", stderr="error")
        assert result.success is False
        assert result.should_block is False

    def test_exit_code_two_is_block(self) -> None:
        result = HookResult(exit_code=2, stdout="", stderr="blocked")
        assert result.should_block is True
        assert result.success is False

    def test_exit_code_nonzero_not_two_is_not_success_not_block(self) -> None:
        result = HookResult(exit_code=127, stdout="", stderr="command not found")
        assert result.success is False
        assert result.should_block is False

    def test_stdout_and_stderr_stored(self) -> None:
        result = HookResult(exit_code=0, stdout="some output", stderr="some error")
        assert result.stdout == "some output"
        assert result.stderr == "some error"


# ---------------------------------------------------------------------------
# TestEmit
# ---------------------------------------------------------------------------


class TestEmit:
    """emit() should orchestrate hook execution and handle edge cases."""

    async def test_disabled_returns_empty_list(self) -> None:
        mgr = _make_manager(
            hooks={"before_tool": [HookConfig(command="echo hi")]},
            hooks_enabled=False,
        )
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert results == []

    async def test_no_hooks_for_event_returns_empty_list(self) -> None:
        mgr = _make_manager(hooks={})
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert results == []

    async def test_fires_registered_hook(self) -> None:
        """A hook that exits 0 should produce one success result."""
        hooks = {"before_tool": [HookConfig(command="exit 0", timeout=5)]}
        mgr = _make_manager(hooks=hooks)
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert len(results) == 1
        assert results[0].success is True

    async def test_fires_multiple_hooks(self) -> None:
        """All hooks for an event should fire (unless blocked)."""
        hooks = {
            "before_tool": [
                HookConfig(command="exit 0", timeout=5),
                HookConfig(command="exit 0", timeout=5),
            ]
        }
        mgr = _make_manager(hooks=hooks)
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert len(results) == 2

    async def test_blocking_hook_stops_further_hooks(self) -> None:
        """A hook that exits 2 should stop subsequent hooks from firing."""
        hooks = {
            "before_tool": [
                HookConfig(command="exit 2", timeout=5),
                HookConfig(command="exit 0", timeout=5),
            ]
        }
        mgr = _make_manager(hooks=hooks)
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert len(results) == 1
        assert results[0].should_block is True

    async def test_data_passed_as_json_on_stdin(self) -> None:
        """Hook should receive JSON-encoded event data on stdin."""
        received_stdin: list[bytes] = []

        async def fake_subprocess(cmd, **kwargs):
            proc = MagicMock()

            async def communicate(input=None):
                if input:
                    received_stdin.append(input)
                return (b"", b"")

            proc.communicate = communicate
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_shell", side_effect=fake_subprocess):
            hooks = {"before_tool": [HookConfig(command="cat", timeout=5)]}
            mgr = _make_manager(hooks=hooks)
            await mgr.emit(HookEvent.BEFORE_TOOL, data={"tool": "write_file"})

        assert len(received_stdin) == 1
        payload = json.loads(received_stdin[0])
        assert payload["event"] == "before_tool"
        assert payload["data"]["tool"] == "write_file"

    async def test_timeout_returns_error_result(self) -> None:
        """A hook that exceeds its timeout should return exit_code=1."""
        hooks = {"before_tool": [HookConfig(command="sleep 100", timeout=1)]}
        mgr = _make_manager(hooks=hooks)
        results = await mgr.emit(HookEvent.BEFORE_TOOL)
        assert len(results) == 1
        assert results[0].exit_code == 1
        assert "timed out" in results[0].stderr.lower() or results[0].exit_code != 0

    async def test_emit_with_none_data_uses_empty_dict(self) -> None:
        """emit(event, data=None) should not crash — defaults to {}."""
        received_stdin: list[bytes] = []

        async def fake_subprocess(cmd, **kwargs):
            proc = MagicMock()

            async def communicate(input=None):
                if input:
                    received_stdin.append(input)
                return (b"", b"")

            proc.communicate = communicate
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_shell", side_effect=fake_subprocess):
            hooks = {"before_tool": [HookConfig(command="cat", timeout=5)]}
            mgr = _make_manager(hooks=hooks)
            await mgr.emit(HookEvent.BEFORE_TOOL, data=None)

        payload = json.loads(received_stdin[0])
        assert payload["data"] == {}


# ---------------------------------------------------------------------------
# TestExecuteHook
# ---------------------------------------------------------------------------


class TestExecuteHook:
    """_execute_hook() should correctly spawn the process and parse output."""

    async def test_exit_code_captured(self) -> None:
        """A real shell exit with known code should be captured correctly."""
        hook_config = HookConfig(command="exit 0", timeout=5)
        mgr = _make_manager()
        result = await mgr._execute_hook(hook_config, HookEvent.BEFORE_TOOL, {})
        assert result.exit_code == 0

    async def test_stdout_captured(self) -> None:
        hook_config = HookConfig(command="echo captured_output", timeout=5)
        mgr = _make_manager()
        result = await mgr._execute_hook(hook_config, HookEvent.BEFORE_TOOL, {})
        assert "captured_output" in result.stdout

    async def test_stderr_captured(self) -> None:
        hook_config = HookConfig(command="echo err >&2", timeout=5)
        mgr = _make_manager()
        result = await mgr._execute_hook(hook_config, HookEvent.BEFORE_TOOL, {})
        assert "err" in result.stderr

    async def test_os_error_returns_exit_code_one(self) -> None:
        """If the subprocess fails to start, exit_code should be 1."""
        hook_config = HookConfig(command="nonexistent_command_xyz", timeout=5)
        mgr = _make_manager()

        with patch(
            "asyncio.create_subprocess_shell",
            side_effect=OSError("No such file"),
        ):
            result = await mgr._execute_hook(hook_config, HookEvent.BEFORE_TOOL, {})

        assert result.exit_code == 1

    async def test_json_stdin_includes_event_name(self) -> None:
        """stdin payload should carry the event value."""
        received: list[bytes] = []

        async def fake_subprocess(cmd, **kwargs):
            proc = MagicMock()

            async def communicate(input=None):
                if input:
                    received.append(input)
                return (b"stdout_val", b"")

            proc.communicate = communicate
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_shell", side_effect=fake_subprocess):
            mgr = _make_manager()
            hook_config = HookConfig(command="cat", timeout=5)
            await mgr._execute_hook(hook_config, HookEvent.SESSION_START, {"key": "val"})

        payload = json.loads(received[0])
        assert payload["event"] == "session_start"
        assert payload["data"] == {"key": "val"}


# ---------------------------------------------------------------------------
# TestHasHooks
# ---------------------------------------------------------------------------


class TestHasHooks:
    """has_hooks() should reflect whether any hooks are registered for an event."""

    def test_no_hooks_returns_false(self) -> None:
        mgr = _make_manager(hooks={})
        assert mgr.has_hooks(HookEvent.BEFORE_TOOL) is False

    def test_with_hook_returns_true(self) -> None:
        hooks = {"before_tool": [HookConfig(command="echo hi")]}
        mgr = _make_manager(hooks=hooks)
        assert mgr.has_hooks(HookEvent.BEFORE_TOOL) is True

    def test_different_event_returns_false(self) -> None:
        hooks = {"before_tool": [HookConfig(command="echo hi")]}
        mgr = _make_manager(hooks=hooks)
        assert mgr.has_hooks(HookEvent.AFTER_TOOL) is False

    def test_disabled_hooks_returns_false_even_if_registered(self) -> None:
        """When hooks_enabled is False, has_hooks() should always return False."""
        hooks = {"before_tool": [HookConfig(command="echo hi")]}
        mgr = _make_manager(hooks=hooks, hooks_enabled=False)
        assert mgr.has_hooks(HookEvent.BEFORE_TOOL) is False

    def test_enabled_property_true_when_enabled(self) -> None:
        mgr = _make_manager(hooks_enabled=True)
        assert mgr.enabled is True

    def test_enabled_property_false_when_disabled(self) -> None:
        mgr = _make_manager(hooks_enabled=False)
        assert mgr.enabled is False
