"""Rich-based terminal renderer for PyGeminiCLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pygemini.cli.themes import Theme, get_theme
from pygemini.core.events import (
    ConfirmRequestEvent,
    CoreEvent,
    ErrorEvent,
    EventEmitter,
    StreamTextEvent,
    ToolExecutingEvent,
    ToolOutputEvent,
    TurnCompleteEvent,
)


class Renderer:
    """Renders agent output to the terminal using Rich."""

    def __init__(
        self,
        console: Console | None = None,
        theme: Theme | None = None,
    ) -> None:
        self._console = console or Console()
        self._theme = theme or get_theme()
        self._stream_buffer: list[str] = []

    def wire_events(self, emitter: EventEmitter) -> None:
        """Register event listeners on the emitter."""
        emitter.on(CoreEvent.STREAM_TEXT, self._on_stream_text)
        emitter.on(CoreEvent.TOOL_EXECUTING, self._on_tool_executing)
        emitter.on(CoreEvent.TOOL_OUTPUT, self._on_tool_output)
        emitter.on(CoreEvent.ERROR, self._on_error)
        emitter.on(CoreEvent.TURN_COMPLETE, self._on_turn_complete)
        # CONFIRM_REQUEST is handled separately via request_confirmation flow

    # -- Event handlers -------------------------------------------------------

    async def _on_stream_text(self, event: Any) -> None:
        if isinstance(event, StreamTextEvent):
            self._stream_buffer.append(event.text)

    async def _on_tool_executing(self, event: Any) -> None:
        if isinstance(event, ToolExecutingEvent):
            self._flush_stream()
            self._console.print(
                f"[{self._theme.tool_color}]> Running tool:[/{self._theme.tool_color}] "
                f"{event.tool_name}",
            )

    async def _on_tool_output(self, event: Any) -> None:
        if isinstance(event, ToolOutputEvent):
            style = self._theme.error_color if event.is_error else self._theme.dim_color
            self._console.print(
                Panel(
                    event.display_content,
                    border_style=style,
                    expand=False,
                )
            )

    async def _on_error(self, event: Any) -> None:
        if isinstance(event, ErrorEvent):
            self._flush_stream()
            self._console.print(
                f"[{self._theme.error_color}]Error: {event.message}[/{self._theme.error_color}]"
            )

    async def _on_turn_complete(self, event: Any) -> None:
        if isinstance(event, TurnCompleteEvent):
            self._flush_stream()
            self._console.print()  # blank line after turn

    # -- Stream management ----------------------------------------------------

    def _flush_stream(self) -> None:
        """Render accumulated streamed text as markdown."""
        if not self._stream_buffer:
            return
        full_text = "".join(self._stream_buffer)
        self._stream_buffer.clear()
        if full_text.strip():
            self._console.print(Markdown(full_text))

    # -- Confirmation ---------------------------------------------------------

    def render_confirmation(self, event: ConfirmRequestEvent) -> bool:
        """Show a confirmation prompt and return the user's choice."""
        self._flush_stream()
        self._console.print()
        self._console.print(
            Panel(
                event.description,
                title="[bold yellow]Confirm?[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )
        if event.details:
            for key, value in event.details.items():
                if key == "preview":
                    self._console.print(
                        Panel(str(value), title="Preview", border_style="dim", expand=False)
                    )
                else:
                    self._console.print(f"  [dim]{key}:[/dim] {value}")

        try:
            answer = self._console.input("[bold yellow]Allow? (y/n): [/bold yellow]")
            return answer.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    # -- Welcome --------------------------------------------------------------

    def render_welcome(self) -> None:
        """Display the startup banner."""
        self._console.print()
        self._console.print(
            Text("PyGeminiCLI", style="bold cyan"),
            Text(" — AI coding assistant in your terminal", style="dim"),
        )
        self._console.print(
            Text("Type /help for commands, Ctrl+D to exit.", style="dim"),
        )
        self._console.print()

    # -- Utilities ------------------------------------------------------------

    def print_info(self, message: str) -> None:
        """Print an informational message."""
        self._console.print(f"[{self._theme.info_color}]{message}[/{self._theme.info_color}]")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self._console.print(f"[{self._theme.error_color}]{message}[/{self._theme.error_color}]")
