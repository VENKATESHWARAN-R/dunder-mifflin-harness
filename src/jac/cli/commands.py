"""Slash command registry for the interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

_KEYBOARD_SHORTCUTS = """\
Keyboard shortcuts:
  ctrl+r           — search history
  ctrl+c           — cancel current input
  ctrl+d           — exit (on empty line)
  esc+enter        — insert newline (multiline input)
  tab              — complete slash commands and @file paths"""


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Registered slash command metadata."""

    name: str
    handler: Callable[[str], Awaitable[None]]
    description: str
    example: str = field(default="")


class SlashCommandRegistry:
    """Small async registry for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[str], Awaitable[None]],
        description: str,
        example: str = "",
    ) -> None:
        """Register a command handler."""
        self._commands[name] = SlashCommand(name, handler, description, example)

    def alias(self, short: str, target: str) -> None:
        """Register a short alias that redirects to an existing command."""
        self._aliases[short] = target

    def get(self, name: str) -> SlashCommand | None:
        """Return a command by name (or alias), if registered."""
        if name in self._commands:
            return self._commands[name]
        target = self._aliases.get(name)
        if target is not None:
            return self._commands.get(target)
        return None

    def descriptions(self) -> dict[str, str]:
        """Return {name: description} for all registered commands (used by completer)."""
        return {name: cmd.description for name, cmd in self._commands.items()}

    async def dispatch(self, name: str, args: str = "") -> bool:
        """Dispatch a command. Returns False when the command is unknown."""
        command = self.get(name)
        if command is None:
            return False
        await command.handler(args)
        return True

    def help_text(self) -> str:
        """Format command help text with examples and keyboard shortcuts."""
        if not self._commands:
            return "No slash commands registered."

        lines = ["Slash commands:", ""]
        for command in sorted(self._commands.values(), key=lambda c: c.name):
            line = f"  /{command.name:<14} {command.description}"
            if command.example:
                line += f"  (e.g. {command.example})"
            lines.append(line)

        if self._aliases:
            lines.append("")
            lines.append("Aliases:")
            for short, target in sorted(self._aliases.items()):
                lines.append(f"  /{short:<14} → /{target}")

        lines.append("")
        lines.append(_KEYBOARD_SHORTCUTS)
        lines.append("")
        lines.append("Input prefixes:  @path — attach file   !cmd — run shell command")

        return "\n".join(lines)
