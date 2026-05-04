"""prompt_toolkit input session for interactive chat."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _insert_newline(event: object) -> None:
        buffer = getattr(event, "current_buffer", None)
        if buffer is not None:
            buffer.insert_text("\n")

    return bindings


class InputSession:
    """Async prompt_toolkit wrapper."""

    def __init__(self, history_path: Path) -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self._session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            key_bindings=_key_bindings(),
            multiline=False,
            style=Style.from_dict({"prompt": "bold green"}),
        )

    async def read(self) -> str | None:
        """Read one input line, returning None for EOF or interruption."""
        try:
            value = await self._session.prompt_async([("class:prompt", "jac › ")])
        except (EOFError, KeyboardInterrupt):
            return None
        return value
