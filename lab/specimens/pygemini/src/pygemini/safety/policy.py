"""TOML-based policy engine for fine-grained tool execution control.

Loads rules from two layers (later overrides earlier):
1. ``~/.pygemini/policies.toml``  (user-level defaults)
2. ``.pygemini/policies.toml``    (project-level overrides)

Rule format in TOML::

    [[rules]]
    tool = "run_shell_command"
    action = "deny"
    command_pattern = "rm -rf.*"

    [[rules]]
    tool = "write_file"
    action = "allow"
    path_pattern = "*.test.py"
"""

from __future__ import annotations

import fnmatch
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygemini.core.config import Config

logger = logging.getLogger(__name__)

# Valid actions a rule can specify.
_VALID_ACTIONS = frozenset({"allow", "deny", "confirm"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PolicyRule:
    """A single policy rule."""

    tool_pattern: str
    """Glob pattern matching tool names (e.g. ``"run_shell_*"``, ``"web_*"``)."""

    action: str
    """One of ``"allow"``, ``"deny"``, ``"confirm"``."""

    path_pattern: str | None = None
    """Optional glob for file-path restrictions (matched against ``params["path"]``)."""

    command_pattern: str | None = None
    """Optional regex for command restrictions (matched against ``params["command"]``)."""


@dataclass
class PolicyDecision:
    """Result of evaluating policies for a tool call."""

    action: str
    """One of ``"allow"``, ``"deny"``, ``"confirm"``."""

    matched_rule: PolicyRule | None = None
    """The rule that produced this decision, or ``None`` for the default."""

    reason: str = ""
    """Human-readable explanation of why this decision was reached."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """TOML-based policy engine for tool execution control.

    Rules are evaluated in order; the first matching rule wins.
    If no rule matches the default decision is ``"confirm"`` (defer to the
    approval manager).
    """

    def __init__(self, config: Config) -> None:
        self._rules: list[PolicyRule] = []
        self._load_policies(config)

    # -- public API ---------------------------------------------------------

    def evaluate(self, tool_name: str, params: dict[str, object]) -> PolicyDecision:
        """Evaluate all rules against a tool call.

        Matching logic:

        * ``tool_pattern`` — ``fnmatch`` against *tool_name* (always checked).
        * ``path_pattern`` — ``fnmatch`` against ``params["path"]`` when present
          on both the rule **and** the call.
        * ``command_pattern`` — ``re.search`` against ``params["command"]`` when
          present on both the rule **and** the call.

        Returns the decision from the first matching rule, or a default
        ``"confirm"`` decision if nothing matches.
        """
        for rule in self._rules:
            if self._matches(rule, tool_name, params):
                return PolicyDecision(
                    action=rule.action,
                    matched_rule=rule,
                    reason=f"Matched rule: tool={rule.tool_pattern!r}, action={rule.action}",
                )

        return PolicyDecision(
            action="confirm",
            reason="No policy rule matched; defaulting to confirm",
        )

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule at the end of the rule list."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[PolicyRule]:
        """Return a *copy* of the current rule list (for inspection/testing)."""
        return list(self._rules)

    # -- loading ------------------------------------------------------------

    def _load_policies(self, config: Config) -> None:
        """Load policy rules from TOML files.

        User-level rules are loaded first, then project-level rules are
        appended so they are evaluated first-match after user-level rules.
        Because project-level rules come *after* user-level ones and rules
        are evaluated in order, the project can effectively override the user
        defaults by placing more-specific rules earlier.
        """
        user_path = config.config_dir / "policies.toml"
        project_path = Path(".pygemini/policies.toml")

        self._rules = self._parse_toml(user_path) + self._parse_toml(project_path)

    def _parse_toml(self, path: Path) -> list[PolicyRule]:
        """Parse a ``policies.toml`` file into a list of :class:`PolicyRule`.

        Gracefully handles:
        * Missing files (returns ``[]``).
        * Malformed TOML (logs a warning, returns ``[]``).
        * Individual rules with missing/invalid fields (logs a warning, skips).
        """
        if not path.is_file():
            return []

        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            return []

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            logger.warning("'rules' in %s is not an array; skipping", path)
            return []

        parsed: list[PolicyRule] = []
        for idx, entry in enumerate(raw_rules):
            if not isinstance(entry, dict):
                logger.warning("Rule #%d in %s is not a table; skipping", idx, path)
                continue

            tool = entry.get("tool")
            action = entry.get("action")

            if not tool or not isinstance(tool, str):
                logger.warning(
                    "Rule #%d in %s missing 'tool' string; skipping", idx, path
                )
                continue

            if action not in _VALID_ACTIONS:
                logger.warning(
                    "Rule #%d in %s has invalid action %r; skipping", idx, path, action
                )
                continue

            # Optional fields — accept only strings, ignore anything else.
            path_pattern = entry.get("path_pattern")
            if path_pattern is not None and not isinstance(path_pattern, str):
                logger.warning(
                    "Rule #%d in %s: path_pattern is not a string; ignoring it",
                    idx,
                    path,
                )
                path_pattern = None

            command_pattern = entry.get("command_pattern")
            if command_pattern is not None and not isinstance(command_pattern, str):
                logger.warning(
                    "Rule #%d in %s: command_pattern is not a string; ignoring it",
                    idx,
                    path,
                )
                command_pattern = None

            # Validate regex early so a bad pattern doesn't explode at runtime.
            if command_pattern is not None:
                try:
                    re.compile(command_pattern)
                except re.error as exc:
                    logger.warning(
                        "Rule #%d in %s: bad regex %r: %s; skipping rule",
                        idx,
                        path,
                        command_pattern,
                        exc,
                    )
                    continue

            parsed.append(
                PolicyRule(
                    tool_pattern=tool,
                    action=action,
                    path_pattern=path_pattern,
                    command_pattern=command_pattern,
                )
            )

        return parsed

    # -- matching -----------------------------------------------------------

    @staticmethod
    def _matches(
        rule: PolicyRule, tool_name: str, params: dict[str, object]
    ) -> bool:
        """Return ``True`` if *rule* matches the given tool call."""
        # 1. Tool name must match (always required).
        if not fnmatch.fnmatch(tool_name, rule.tool_pattern):
            return False

        # 2. If the rule has a path_pattern, it must match params["path"].
        if rule.path_pattern is not None:
            param_path = params.get("path")
            if not isinstance(param_path, str):
                return False
            if not fnmatch.fnmatch(param_path, rule.path_pattern):
                return False

        # 3. If the rule has a command_pattern, it must match params["command"].
        if rule.command_pattern is not None:
            param_command = params.get("command")
            if not isinstance(param_command, str):
                return False
            if not re.search(rule.command_pattern, param_command):
                return False

        return True
