"""Slash command registry for the interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Registered slash command metadata."""

    name: str
    handler: Callable[[str], Awaitable[None]]
    description: str


class SlashCommandRegistry:
    """Small async registry for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(
        self,
        name: str,
        handler: Callable[[str], Awaitable[None]],
        description: str,
    ) -> None:
        """Register a command handler."""
        self._commands[name] = SlashCommand(name, handler, description)

    def get(self, name: str) -> SlashCommand | None:
        """Return a command by name, if registered."""
        return self._commands.get(name)

    async def dispatch(self, name: str, args: str = "") -> bool:
        """Dispatch a command. Returns False when the command is unknown."""
        command = self.get(name)
        if command is None:
            return False
        await command.handler(args)
        return True

    def help_text(self) -> str:
        """Format command help text."""
        if not self._commands:
            return "No slash commands registered."
        lines = ["Available commands:", ""]
        for command in sorted(self._commands.values(), key=lambda item: item.name):
            lines.append(f"  /{command.name:<10} {command.description}")
        return "\n".join(lines)
