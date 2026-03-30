"""Approval manager for tool execution confirmation.

Determines whether a tool requires user confirmation based on the configured
approval mode and the per-tool allowed list.
"""

from __future__ import annotations

import logging

from pygemini.core.config import Config
from pygemini.tools.base import ToolConfirmation

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"interactive", "auto_edit", "yolo"})


class ApprovalManager:
    """Manages tool execution approval based on configured mode.

    Approval modes:
    - "interactive": Always ask user for confirmation on tools that require it.
    - "auto_edit": Auto-approve file edit tools (edit_file, write_file), ask
      for confirmation on all other tools that require it.
    - "yolo": Auto-approve everything (no confirmations).

    Additionally, tools listed in ``config.allowed_tools`` are always
    auto-approved regardless of mode.
    """

    _auto_edit_tools: frozenset[str] = frozenset({"write_file", "edit_file"})

    def __init__(self, config: Config) -> None:
        self._mode: str = config.approval_mode
        self._allowed_tools: set[str] = set(config.allowed_tools)
        logger.debug(
            "ApprovalManager initialised: mode=%r, allowed_tools=%r",
            self._mode,
            self._allowed_tools,
        )

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def needs_confirmation(
        self, tool_name: str, confirmation: ToolConfirmation | None
    ) -> bool:
        """Determine if a tool execution needs user confirmation.

        Returns ``False`` (auto-approve) when any of the following hold:

        - The tool carries no confirmation prompt (``confirmation is None``).
        - Mode is ``"yolo"``.
        - The tool is in the always-allowed list.
        - Mode is ``"auto_edit"`` and the tool is a file-edit tool.

        Returns ``True`` (needs confirmation) otherwise.

        Args:
            tool_name: The registered name of the tool about to execute.
            confirmation: The ``ToolConfirmation`` returned by the tool's
                ``should_confirm()`` method, or ``None`` if the tool requires
                no confirmation by default.

        Returns:
            ``True`` if the user must be prompted; ``False`` to proceed
            automatically.
        """
        # No confirmation payload — tool opted out of prompting.
        if confirmation is None:
            logger.debug("tool=%r: no confirmation required by tool", tool_name)
            return False

        # Yolo — never ask.
        if self._mode == "yolo":
            logger.debug("tool=%r: auto-approved (yolo mode)", tool_name)
            return False

        # Explicitly allowed tool.
        if tool_name in self._allowed_tools:
            logger.debug("tool=%r: auto-approved (in allowed_tools)", tool_name)
            return False

        # auto_edit — silently approve file-edit tools.
        if self._mode == "auto_edit" and tool_name in self._auto_edit_tools:
            logger.debug("tool=%r: auto-approved (auto_edit mode)", tool_name)
            return False

        logger.debug("tool=%r: confirmation required (mode=%r)", tool_name, self._mode)
        return True

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current approval mode."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Change the approval mode.

        Args:
            mode: One of ``"interactive"``, ``"auto_edit"``, or ``"yolo"``.

        Raises:
            ValueError: If *mode* is not a recognised value.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid approval mode: {mode!r}. "
                f"Must be one of {sorted(_VALID_MODES)}."
            )
        logger.debug("ApprovalManager: mode changed %r -> %r", self._mode, mode)
        self._mode = mode

    # ------------------------------------------------------------------
    # Allowed-tool list management
    # ------------------------------------------------------------------

    def add_allowed_tool(self, tool_name: str) -> None:
        """Add a tool to the always-approved list.

        Args:
            tool_name: The tool's registered name.
        """
        self._allowed_tools.add(tool_name)
        logger.debug("ApprovalManager: added %r to allowed_tools", tool_name)

    def remove_allowed_tool(self, tool_name: str) -> None:
        """Remove a tool from the always-approved list.

        No-op if the tool is not currently in the list.

        Args:
            tool_name: The tool's registered name.
        """
        self._allowed_tools.discard(tool_name)
        logger.debug("ApprovalManager: removed %r from allowed_tools", tool_name)
