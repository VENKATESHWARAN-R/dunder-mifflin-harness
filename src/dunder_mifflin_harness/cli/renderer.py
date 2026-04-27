"""Rich renderer for runtime events."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from dunder_mifflin_harness.runtime.events import (
    AgentMessageCompleted,
    AgentTextDelta,
    CostUpdated,
    EventBus,
    FileEditApplied,
    FileEditPreviewed,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    RunFailed,
    ShellCommandCompleted,
    ShellCommandStarted,
    ToolCallCompleted,
    ToolCallRequested,
    WarningRaised,
)


class Renderer:
    """Subscribe to runtime events and render them to a terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._stream_buffer: list[str] = []

    def wire(self, events: EventBus) -> None:
        """Register event handlers."""
        events.on(AgentTextDelta, self._on_agent_text)
        events.on(AgentMessageCompleted, self._on_agent_done)
        events.on(ToolCallRequested, self._on_tool_requested)
        events.on(ToolCallCompleted, self._on_tool_completed)
        events.on(NodeStarted, self._on_node_started)
        events.on(NodeCompleted, self._on_node_completed)
        events.on(NodeFailed, self._on_node_failed)
        events.on(FileEditPreviewed, self._on_file_edit_previewed)
        events.on(FileEditApplied, self._on_file_edit_applied)
        events.on(ShellCommandStarted, self._on_shell_started)
        events.on(ShellCommandCompleted, self._on_shell_completed)
        events.on(CostUpdated, self._on_cost_updated)
        events.on(WarningRaised, self._on_warning)
        events.on(RunFailed, self._on_run_failed)

    async def _on_agent_text(self, event: AgentTextDelta) -> None:
        self._stream_buffer.append(event.text)

    async def _on_agent_done(self, _event: AgentMessageCompleted) -> None:
        self.flush_stream()
        self.console.print()

    async def _on_tool_requested(self, event: ToolCallRequested) -> None:
        self.flush_stream()
        self.console.print(f"[cyan]tool[/cyan] {event.tool_name}")
        if event.params:
            self.console.print(Panel(str(event.params), border_style="cyan", expand=False))

    async def _on_tool_completed(self, event: ToolCallCompleted) -> None:
        style = "red" if event.is_error else "dim"
        if event.display_content:
            self.console.print(
                Panel(event.display_content, border_style=style, expand=False)
            )

    async def _on_node_started(self, event: NodeStarted) -> None:
        self.flush_stream()
        self.console.print(f"[dim]node started:[/dim] {event.node_name}")

    async def _on_node_completed(self, event: NodeCompleted) -> None:
        self.console.print(
            f"[dim]node completed:[/dim] {event.node_name} ({event.status})"
        )

    async def _on_node_failed(self, event: NodeFailed) -> None:
        self.console.print(f"[red]node failed:[/red] {event.node_name}: {event.message}")

    async def _on_file_edit_previewed(self, event: FileEditPreviewed) -> None:
        self.flush_stream()
        syntax = Syntax(event.diff, "diff", word_wrap=True)
        self.console.print(
            Panel(syntax, title=str(event.path), border_style="yellow", expand=False)
        )

    async def _on_file_edit_applied(self, event: FileEditApplied) -> None:
        self.console.print(f"[green]file edited:[/green] {event.path}")

    async def _on_shell_started(self, event: ShellCommandStarted) -> None:
        self.flush_stream()
        body = (
            f"command: {event.command}\n"
            f"cwd: {event.cwd}\n"
            f"timeout: {event.timeout_seconds:g}s"
        )
        self.console.print(Panel(body, title="Shell", border_style="cyan", expand=False))

    async def _on_shell_completed(self, event: ShellCommandCompleted) -> None:
        status = "timed out" if event.timed_out else f"exit {event.exit_code}"
        body = [f"status: {status}"]
        if event.stdout:
            body.append(f"\nstdout:\n{event.stdout.rstrip()}")
        if event.stderr:
            body.append(f"\nstderr:\n{event.stderr.rstrip()}")
        self.console.print(
            Panel("\n".join(body), title="Shell Result", border_style="dim", expand=False)
        )

    async def _on_cost_updated(self, event: CostUpdated) -> None:
        self.console.print(Panel(event.summary, title="Cost", border_style="green"))

    async def _on_warning(self, event: WarningRaised) -> None:
        self.console.print(f"[yellow]warning:[/yellow] {event.message}")

    async def _on_run_failed(self, event: RunFailed) -> None:
        self.flush_stream()
        self.console.print(f"[red]run failed:[/red] {event.message}")

    def flush_stream(self) -> None:
        """Render buffered model text as Markdown."""
        if not self._stream_buffer:
            return
        text = "".join(self._stream_buffer)
        self._stream_buffer.clear()
        if text.strip():
            self.console.print(Markdown(text))

    def render_welcome(self) -> None:
        """Display the chat welcome text."""
        self.console.print()
        self.console.print("[bold cyan]dunder-mifflin-harness[/bold cyan]")
        self.console.print("[dim]Type /help for commands, /quit to exit.[/dim]")
        self.console.print()

    def print_info(self, message: str) -> None:
        """Print informational text."""
        self.console.print(f"[dim]{message}[/dim]")

    def print_error(self, message: str) -> None:
        """Print an error."""
        self.console.print(f"[red]{message}[/red]")

    def print_warning(self, message: str) -> None:
        """Print a warning."""
        self.console.print(f"[yellow]{message}[/yellow]")

    def print_value(self, title: str, value: Any) -> None:
        """Render a small titled value panel."""
        self.console.print(Panel(str(value), title=title, expand=False))
