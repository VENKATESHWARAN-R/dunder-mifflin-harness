"""Rich renderer for runtime events."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from jac import __version__
from jac.agents.plans import Plan
from jac.runtime.events import (
    AgentDelegated,
    AgentMessageCompleted,
    AgentTextDelta,
    AttemptRecorded,
    CostUpdated,
    EventBus,
    FileEditApplied,
    FileEditPreviewed,
    LlmCallCompleted,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    RunFailed,
    ShellCommandCompleted,
    ShellCommandStarted,
    ToolCallCompleted,
    ToolCallRequested,
    WarningRaised,
    WorkspaceSurveyCompleted,
)


class Renderer:
    """Subscribe to runtime events and render them to a terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._stream_buffer: list[str] = []
        self._debug = False

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
        events.on(AgentDelegated, self._on_agent_delegated)
        events.on(AttemptRecorded, self._on_attempt_recorded)
        events.on(LlmCallCompleted, self._on_llm_call_completed)
        events.on(WorkspaceSurveyCompleted, self._on_workspace_survey_completed)

    async def _on_agent_text(self, event: AgentTextDelta) -> None:
        self._stream_buffer.append(event.text)

    async def _on_agent_done(self, _event: AgentMessageCompleted) -> None:
        self.flush_stream()
        self.console.print()

    async def _on_tool_requested(self, event: ToolCallRequested) -> None:
        self.flush_stream()
        if self._debug:
            self.console.print(f"[cyan]tool start[/cyan] {event.tool_name}")
        else:
            self.console.print(f"[cyan]running tool:[/cyan] {event.tool_name}")
        if event.params:
            self.console.print(
                Panel(str(event.params), border_style="cyan", expand=False)
            )

    async def _on_tool_completed(self, event: ToolCallCompleted) -> None:
        style = "red" if event.is_error else "dim"
        if event.display_content:
            self.console.print(
                Panel(event.display_content, border_style=style, expand=False)
            )
        elif self._debug:
            status = "error" if event.is_error else "ok"
            self.console.print(f"[dim]tool done:[/dim] {event.tool_name} ({status})")

    async def _on_node_started(self, event: NodeStarted) -> None:
        self.flush_stream()
        self.console.print(f"[dim]node started:[/dim] {event.node_name}")

    async def _on_node_completed(self, event: NodeCompleted) -> None:
        self.console.print(
            f"[dim]node completed:[/dim] {event.node_name} ({event.status})"
        )

    async def _on_node_failed(self, event: NodeFailed) -> None:
        self.console.print(
            f"[red]node failed:[/red] {event.node_name}: {event.message}"
        )

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
        self.console.print(
            Panel(body, title="Shell", border_style="cyan", expand=False)
        )

    async def _on_shell_completed(self, event: ShellCommandCompleted) -> None:
        status = "timed out" if event.timed_out else f"exit {event.exit_code}"
        body = [f"status: {status}"]
        if event.stdout:
            body.append(f"\nstdout:\n{event.stdout.rstrip()}")
        if event.stderr:
            body.append(f"\nstderr:\n{event.stderr.rstrip()}")
        self.console.print(
            Panel(
                "\n".join(body), title="Shell Result", border_style="dim", expand=False
            )
        )

    async def _on_cost_updated(self, event: CostUpdated) -> None:
        self.console.print(f"[dim]  ↳ {event.summary}[/dim]")

    async def _on_attempt_recorded(self, event: AttemptRecorded) -> None:
        if not self._debug:
            return
        parent = event.parent_attempt_id[:8] if event.parent_attempt_id else "-"
        self.console.print(
            "[dim]attempt:[/dim] "
            f"{event.role} {event.call_type} id={event.attempt_id[:8]} parent={parent}"
        )

    async def _on_llm_call_completed(self, event: LlmCallCompleted) -> None:
        if not self._debug:
            return
        self.console.print(
            "[dim]llm:[/dim] "
            f"{event.role} {event.model} ({event.tier}) "
            f"in={event.input_tokens} out={event.output_tokens} "
            f"req={event.requests} tools={event.tool_calls} "
            f"dur={event.duration_ms}ms type={event.call_type}"
        )

    async def _on_warning(self, event: WarningRaised) -> None:
        self.console.print(f"[yellow]warning:[/yellow] {event.message}")

    async def _on_workspace_survey_completed(
        self, event: WorkspaceSurveyCompleted
    ) -> None:
        self.print_info(
            f"AGENTS.md updated: {event.agents_md_path} ({event.line_count} lines)"
        )

    async def _on_run_failed(self, event: RunFailed) -> None:
        self.flush_stream()
        self.console.print(f"[red]run failed:[/red] {event.message}")

    async def _on_agent_delegated(self, event: AgentDelegated) -> None:
        self.flush_stream()
        self.console.print(f"[dim]→ Handing to {event.display_name}…[/dim]")

    def flush_stream(self) -> None:
        """Render buffered model text as Markdown."""
        if not self._stream_buffer:
            return
        text = "".join(self._stream_buffer)
        self._stream_buffer.clear()
        if text.strip():
            self.console.print(Markdown(text))

    def render_welcome(self, model: str | None = None, tier: str | None = None, mode: str | None = None) -> None:
        """Display the chat welcome text with current session config."""
        self.console.print()
        self.console.print(
            f"[bold cyan]JAC[/bold cyan] [dim]v{__version__} — Just Another CLI[/dim]"
        )
        config_parts = [
            f"model: {model or 'default'}",
            f"tier: {tier or 'worker'}",
            f"mode: {mode or 'hitl'}",
        ]
        if self._debug:
            config_parts.append("debug: on")
        self.console.print(f"[dim]{' · '.join(config_parts)}[/dim]")
        self.console.print("[dim]Type a message to start · /help for commands · ctrl+d to exit[/dim]")
        self.console.print()

    def render_resume_context(self, messages: list[Any]) -> None:
        """Show a compact preview of recent messages when resuming a session."""
        if not messages:
            return
        self.console.print("[dim]— resuming session —[/dim]")
        self.console.print()
        for msg in messages:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "")
            if role == "user":
                prefix = "[bold green]you[/bold green]"
            else:
                prefix = "[bold cyan]jac[/bold cyan]"
            preview = content[:300].replace("\n", " ")
            if len(content) > 300:
                preview += "…"
            self.console.print(f"{prefix}: {preview}")
        self.console.print()

    def render_message_history(self, messages: list[Any], n: int) -> None:
        """Render last n messages from session history."""
        shown = messages[-n:] if len(messages) > n else messages
        if not shown:
            self.console.print("[dim]No messages yet.[/dim]")
            return

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold", width=5)
        table.add_column()

        for msg in shown:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "")
            preview = content[:200].replace("\n", " ")
            if len(content) > 200:
                preview += "…"
            style = "green" if role == "user" else "cyan"
            table.add_row(f"[{style}]{role}[/{style}]", preview)

        self.console.print(table)

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

    def render_plan(self, plan: Plan) -> None:
        """Render a planner output in a compact panel + table."""
        self.console.print(
            Panel(
                f"{plan.summary}\n\n[bold]Strategy:[/bold] {plan.dev_strategy}",
                title="Plan",
                expand=False,
            )
        )
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title")
        table.add_column("Complexity", width=10)
        table.add_column("Acceptance")
        for index, task in enumerate(plan.tasks, start=1):
            table.add_row(
                str(index),
                task.title,
                task.complexity,
                task.acceptance_criteria,
            )
        self.console.print(table)

    def set_debug(self, enabled: bool) -> None:
        """Enable verbose developer-oriented runtime tracing."""
        self._debug = enabled
