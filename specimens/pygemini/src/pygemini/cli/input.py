"""CLI input handler using prompt_toolkit.

Supports special input shortcuts:
- ``@path/to/file`` — read file and inline its contents
- ``!command`` — run shell command and inline its output
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Pattern to match @file references (not preceded by alphanumeric — avoids emails)
_FILE_REF_PATTERN = re.compile(r"(?<![a-zA-Z0-9])@(\S+)")

# Pattern to match !command at the start of input
_SHELL_CMD_PATTERN = re.compile(r"^!(.+)$", re.MULTILINE)


def _create_key_bindings() -> KeyBindings:
    """Create key bindings for the input prompt."""
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event: object) -> None:
        """Insert newline on Escape+Enter."""
        buf = getattr(event, "current_buffer", None)
        if buf is not None:
            buf.insert_text("\n")

    return bindings


def expand_file_references(text: str) -> str:
    """Replace ``@path/to/file`` references with the file's contents.

    If the file cannot be read, the reference is replaced with an error
    message so the model still knows what was intended.
    """

    def _replace(match: re.Match[str]) -> str:
        file_path = Path(match.group(1)).expanduser()
        try:
            content = file_path.read_text(encoding="utf-8")
            return f"\n--- Contents of {file_path} ---\n{content}\n--- End of {file_path} ---\n"
        except FileNotFoundError:
            return f"[Error: File not found: {file_path}]"
        except PermissionError:
            return f"[Error: Permission denied: {file_path}]"
        except UnicodeDecodeError:
            return f"[Error: Binary file cannot be read: {file_path}]"
        except OSError as e:
            return f"[Error reading {file_path}: {e}]"

    return _FILE_REF_PATTERN.sub(_replace, text)


def expand_shell_commands(text: str) -> str:
    """Replace ``!command`` lines with the command's output.

    Only matches lines that start with ``!``.  The command is executed
    synchronously with a 10-second timeout.
    """

    def _replace(match: re.Match[str]) -> str:
        command = match.group(1).strip()
        if not command:
            return match.group(0)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            return f"\n--- Output of `{command}` ---\n{output.rstrip()}\n--- End of command output ---\n"
        except subprocess.TimeoutExpired:
            return f"[Error: Command timed out after 10s: {command}]"
        except OSError as e:
            return f"[Error running command: {e}]"

    return _SHELL_CMD_PATTERN.sub(_replace, text)


def process_input(text: str) -> str:
    """Expand all special input shortcuts in *text*.

    Currently handles:
    - ``@path/to/file`` — inline file contents
    - ``!command`` — inline shell command output
    """
    text = expand_file_references(text)
    text = expand_shell_commands(text)
    return text


class InputHandler:
    """Handles user input via prompt_toolkit."""

    def __init__(self, config_dir: Path | None = None) -> None:
        history_path = (config_dir or Path.home() / ".pygemini") / "input_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self._style = Style.from_dict({
            "prompt": "bold green",
        })

        self._session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            style=self._style,
            key_bindings=_create_key_bindings(),
            multiline=False,
        )

    async def get_input(self) -> str | None:
        """Prompt the user for input.

        Returns the processed input string (with @file and !command expanded),
        or None on EOF (Ctrl+D).
        """
        try:
            result = await self._session.prompt_async(
                [("class:prompt", ">>> ")],
            )
            text = result.strip()
            if not text:
                return None
            # Expand @file references and !command shortcuts
            text = process_input(text)
            return text
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None
