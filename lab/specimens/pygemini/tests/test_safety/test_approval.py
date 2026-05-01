"""Tests for pygemini.safety.approval (ApprovalManager)."""

from __future__ import annotations

import pytest

from pygemini.core.config import Config
from pygemini.safety.approval import ApprovalManager
from pygemini.tools.base import ToolConfirmation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    approval_mode: str = "interactive",
    allowed_tools: list[str] | None = None,
) -> ApprovalManager:
    """Build an ApprovalManager from a minimal Config."""
    config = Config(
        approval_mode=approval_mode,  # type: ignore[arg-type]
        allowed_tools=allowed_tools or [],
    )
    return ApprovalManager(config)


_CONFIRMATION = ToolConfirmation(description="About to run something dangerous.")


# ---------------------------------------------------------------------------
# TestNeedsConfirmation
# ---------------------------------------------------------------------------


class TestNeedsConfirmation:
    """needs_confirmation() should follow mode/allowed-tool precedence rules."""

    def test_no_confirmation_payload_returns_false(self) -> None:
        """Tool that returns None confirmation → no prompt needed."""
        mgr = _make_manager("interactive")
        assert mgr.needs_confirmation("any_tool", None) is False

    def test_yolo_mode_returns_false(self) -> None:
        """In yolo mode, nothing needs confirmation."""
        mgr = _make_manager("yolo")
        assert mgr.needs_confirmation("run_shell_command", _CONFIRMATION) is False

    def test_allowed_tools_returns_false(self) -> None:
        """Tool in allowed_tools list is auto-approved."""
        mgr = _make_manager("interactive", allowed_tools=["my_risky_tool"])
        assert mgr.needs_confirmation("my_risky_tool", _CONFIRMATION) is False

    def test_auto_edit_for_write_file_returns_false(self) -> None:
        """auto_edit mode auto-approves write_file."""
        mgr = _make_manager("auto_edit")
        assert mgr.needs_confirmation("write_file", _CONFIRMATION) is False

    def test_auto_edit_for_edit_file_returns_false(self) -> None:
        """auto_edit mode auto-approves edit_file."""
        mgr = _make_manager("auto_edit")
        assert mgr.needs_confirmation("edit_file", _CONFIRMATION) is False

    def test_auto_edit_for_shell_returns_true(self) -> None:
        """auto_edit mode still prompts for non-file-edit tools."""
        mgr = _make_manager("auto_edit")
        assert mgr.needs_confirmation("run_shell_command", _CONFIRMATION) is True

    def test_interactive_mode_returns_true(self) -> None:
        """In interactive mode, any tool with a confirmation payload needs prompting."""
        mgr = _make_manager("interactive")
        assert mgr.needs_confirmation("run_shell_command", _CONFIRMATION) is True

    def test_interactive_mode_non_edit_tool_returns_true(self) -> None:
        """In interactive mode, even read_file with a confirmation needs prompting."""
        mgr = _make_manager("interactive")
        assert mgr.needs_confirmation("read_file", _CONFIRMATION) is True

    def test_yolo_ignores_allowed_tools_distinction(self) -> None:
        """In yolo mode, tools NOT in allowed_tools also return False."""
        mgr = _make_manager("yolo", allowed_tools=[])
        assert mgr.needs_confirmation("unknown_dangerous_tool", _CONFIRMATION) is False

    def test_allowed_tool_overrides_interactive(self) -> None:
        """allowed_tools takes precedence over interactive mode."""
        mgr = _make_manager("interactive", allowed_tools=["dangerous"])
        assert mgr.needs_confirmation("dangerous", _CONFIRMATION) is False


# ---------------------------------------------------------------------------
# TestSetMode
# ---------------------------------------------------------------------------


class TestSetMode:
    """set_mode() should update the mode or raise ValueError for invalid values."""

    def test_set_valid_mode_interactive(self) -> None:
        mgr = _make_manager("yolo")
        mgr.set_mode("interactive")
        assert mgr.mode == "interactive"

    def test_set_valid_mode_auto_edit(self) -> None:
        mgr = _make_manager("interactive")
        mgr.set_mode("auto_edit")
        assert mgr.mode == "auto_edit"

    def test_set_valid_mode_yolo(self) -> None:
        mgr = _make_manager("interactive")
        mgr.set_mode("yolo")
        assert mgr.mode == "yolo"

    def test_invalid_mode_raises_value_error(self) -> None:
        mgr = _make_manager("interactive")
        with pytest.raises(ValueError, match="Invalid approval mode"):
            mgr.set_mode("turbo")

    def test_invalid_mode_preserves_existing_mode(self) -> None:
        """Mode should not change if the new mode is invalid."""
        mgr = _make_manager("interactive")
        try:
            mgr.set_mode("bad_mode")
        except ValueError:
            pass
        assert mgr.mode == "interactive"

    def test_mode_property_reflects_initial_value(self) -> None:
        mgr = _make_manager("auto_edit")
        assert mgr.mode == "auto_edit"


# ---------------------------------------------------------------------------
# TestAddRemoveAllowed
# ---------------------------------------------------------------------------


class TestAddRemoveAllowed:
    """add_allowed_tool / remove_allowed_tool should mutate the allowed set."""

    def test_add_tool_enables_auto_approval(self) -> None:
        mgr = _make_manager("interactive")
        mgr.add_allowed_tool("my_tool")
        assert mgr.needs_confirmation("my_tool", _CONFIRMATION) is False

    def test_remove_tool_restores_confirmation(self) -> None:
        mgr = _make_manager("interactive", allowed_tools=["my_tool"])
        mgr.remove_allowed_tool("my_tool")
        assert mgr.needs_confirmation("my_tool", _CONFIRMATION) is True

    def test_remove_non_existent_tool_is_noop(self) -> None:
        """Removing a tool that was never added should not raise."""
        mgr = _make_manager("interactive")
        mgr.remove_allowed_tool("ghost_tool")  # should not raise

    def test_add_then_remove_round_trip(self) -> None:
        mgr = _make_manager("interactive")
        mgr.add_allowed_tool("transient_tool")
        assert mgr.needs_confirmation("transient_tool", _CONFIRMATION) is False
        mgr.remove_allowed_tool("transient_tool")
        assert mgr.needs_confirmation("transient_tool", _CONFIRMATION) is True

    def test_add_duplicate_tool_is_idempotent(self) -> None:
        """Adding the same tool twice should not raise or cause issues."""
        mgr = _make_manager("interactive")
        mgr.add_allowed_tool("dup_tool")
        mgr.add_allowed_tool("dup_tool")
        assert mgr.needs_confirmation("dup_tool", _CONFIRMATION) is False

    def test_initial_allowed_tools_from_config(self) -> None:
        """Tools in Config.allowed_tools are auto-approved from the start."""
        mgr = _make_manager("interactive", allowed_tools=["preloaded"])
        assert mgr.needs_confirmation("preloaded", _CONFIRMATION) is False
