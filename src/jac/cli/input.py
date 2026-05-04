"""prompt_toolkit input session for interactive chat."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


class DedupFileHistory(FileHistory):
    """FileHistory that skips consecutive duplicate entries."""

    def append_string(self, string: str) -> None:
        existing = list(self.load_history_strings())
        if existing and existing[0] == string:
            return
        super().append_string(string)


class JacCompleter(Completer):
    """Completion for slash commands, command args, and @file paths."""

    _ARG_COMPLETIONS: dict[str, list[str]] = {
        "tier": ["scout", "worker", "architect"],
        "mode": ["autopilot", "hitl"],
        "approval": ["interactive", "auto-edit", "yolo"],
        "params": ["temperature", "max_tokens"],
    }

    def __init__(self, command_source: Callable[[], dict[str, str]]) -> None:
        self._command_source = command_source
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor

        # @path completion — match the last @reference before the cursor
        at_match = re.search(r'@(?:"([^"]*)|(\S*))$', text)
        if at_match:
            path_text = at_match.group(1) or at_match.group(2) or ""
            path_doc = Document(path_text, len(path_text))
            yield from self._path_completer.get_completions(path_doc, complete_event)
            return

        # Slash command completion
        if text.startswith("/"):
            parts = text[1:].split(" ", 1)
            cmd_part = parts[0]

            if len(parts) == 1:
                # Completing the command name itself
                for name, description in sorted(self._command_source().items()):
                    if name.startswith(cmd_part):
                        yield Completion(
                            name[len(cmd_part) :],
                            display=f"/{name}",
                            display_meta=description,
                        )
            else:
                # Completing the argument for a known command
                args_part = parts[1]
                for value in self._ARG_COMPLETIONS.get(cmd_part, []):
                    if value.startswith(args_part):
                        yield Completion(value[len(args_part) :], display=value)


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _insert_newline(event: object) -> None:
        buffer = getattr(event, "current_buffer", None)
        if buffer is not None:
            buffer.insert_text("\n")

    return bindings


class InputSession:
    """Async prompt_toolkit wrapper with completions, toolbar, and dedup history."""

    def __init__(
        self,
        history_path: Path,
        command_source: Callable[[], dict[str, str]] | None = None,
        session_config_source: Callable[[], object] | None = None,
    ) -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        completer = JacCompleter(command_source) if command_source else None

        def _toolbar() -> str:
            if session_config_source is None:
                return ""
            config = session_config_source()
            model = getattr(config, "model", None) or "default"
            tier = str(getattr(config, "tier", None) or "worker")
            mode = str(getattr(config, "mode", "autopilot"))
            approval = str(getattr(config, "approval_mode", "interactive"))
            return (
                f" model: {model}"
                f" · tier: {tier}"
                f" · mode: {mode}"
                f" · approval: {approval}"
            )

        placeholder = FormattedText([("class:placeholder", "  (esc+enter for newline)")])

        self._session: PromptSession[str] = PromptSession(
            history=DedupFileHistory(str(history_path)),
            key_bindings=_key_bindings(),
            multiline=False,
            style=Style.from_dict({"prompt": "bold green", "placeholder": "italic dim"}),
            completer=completer,
            complete_while_typing=False,
            bottom_toolbar=_toolbar if session_config_source else None,
            placeholder=placeholder,
        )

    async def read(self) -> str | None:
        """Read one input line, returning None for EOF or interruption."""
        try:
            value = await self._session.prompt_async([("class:prompt", "jac › ")])
        except (EOFError, KeyboardInterrupt):
            return None
        return value
