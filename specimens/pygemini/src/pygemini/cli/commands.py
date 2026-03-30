"""Slash command registry for PyGeminiCLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable


@dataclass
class SlashCommand:
    """A registered slash command."""

    name: str
    handler: Callable[..., Awaitable[Any]]
    description: str


class SlashCommandRegistry:
    """Registry for slash commands dispatched from the REPL."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        description: str,
    ) -> None:
        """Register a slash command."""
        self._commands[name] = SlashCommand(
            name=name, handler=handler, description=description
        )

    def get(self, name: str) -> SlashCommand | None:
        """Look up a command by name."""
        return self._commands.get(name)

    async def dispatch(self, input_text: str) -> bool:
        """Try to dispatch input as a slash command.

        Returns True if input was a slash command (even if not found),
        False if input is not a slash command.
        """
        if not input_text.startswith("/"):
            return False

        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        command = self._commands.get(cmd_name)
        if command is None:
            return True  # Was a slash command, just unknown

        await command.handler(args)
        return True

    def get_help_text(self) -> str:
        """Generate help text listing all commands."""
        if not self._commands:
            return "No commands registered."

        lines = ["Available commands:", ""]
        for cmd in sorted(self._commands.values(), key=lambda c: c.name):
            lines.append(f"  /{cmd.name:<15} {cmd.description}")
        return "\n".join(lines)
