"""CLI App — REPL orchestrator that wires all components together."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console

from pygemini.cli.commands import SlashCommandRegistry
from pygemini.cli.input import InputHandler
from pygemini.cli.renderer import Renderer
from pygemini.cli.themes import get_theme
from pygemini.context.gemini_md import GeminiMDDiscovery
from pygemini.context.memory_store import MemoryStore
from pygemini.core.agent_loop import AgentLoop
from pygemini.core.config import Config, load_config
from pygemini.core.content_generator import ContentGenerator
from pygemini.core.events import ConfirmRequestEvent, CoreEvent, EventEmitter
from pygemini.core.history import ConversationHistory
from pygemini.hooks.manager import HookManager
from pygemini.safety.approval import ApprovalManager
from pygemini.safety.policy import PolicyEngine
from pygemini.session.compressor import ConversationCompressor
from pygemini.session.manager import SessionManager
from pygemini.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class App:
    """Main application that wires all components and runs the REPL."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()

        # Core components
        self._event_emitter = EventEmitter()
        self._history = ConversationHistory()
        self._memory_store = MemoryStore()
        self._tool_registry = ToolRegistry(self._config)
        self._tool_registry.register_defaults(
            event_emitter=self._event_emitter,
            memory_store=self._memory_store,
        )

        # Context discovery
        self._gemini_md = GeminiMDDiscovery()
        self._context_content = self._load_context()

        # Session management
        self._session_manager = SessionManager(self._config)
        self._compressor = ConversationCompressor(self._config)

        # Safety & hooks
        self._approval_manager = ApprovalManager(self._config)
        self._policy_engine = PolicyEngine(self._config)
        self._hook_manager = HookManager(self._config)

        # Content generator (deferred — needs API key)
        self._content_generator: ContentGenerator | None = None

        # CLI components
        self._console = Console()
        self._renderer = Renderer(
            console=self._console,
            theme=get_theme(self._config.theme),
        )
        self._input_handler = InputHandler(config_dir=self._config.config_dir)
        self._commands = SlashCommandRegistry()

        # Agent loop (deferred — needs content generator)
        self._agent_loop: AgentLoop | None = None

        # Wire events
        self._renderer.wire_events(self._event_emitter)
        self._wire_confirmation()
        self._register_commands()

    def _load_context(self) -> str:
        """Load GEMINI.md context files and memory."""
        parts: list[str] = []

        # GEMINI.md files
        gemini_context = self._gemini_md.load_context(
            cwd=Path.cwd(),
            config_dir=self._config.config_dir,
        )
        if gemini_context:
            parts.append(gemini_context)

        # Memory
        memory_text = self._memory_store.get_formatted()
        if memory_text:
            parts.append(f"## Memories\n{memory_text}")

        return "\n\n".join(parts)

    def _wire_confirmation(self) -> None:
        """Connect the confirmation flow: core asks -> renderer prompts -> core gets answer."""

        async def on_confirm(event: object) -> None:
            if isinstance(event, ConfirmRequestEvent):
                approved = self._renderer.render_confirmation(event)
                self._event_emitter.respond_confirmation(approved)

        self._event_emitter.on(CoreEvent.CONFIRM_REQUEST, on_confirm)

    def _register_commands(self) -> None:
        """Register built-in slash commands."""

        async def cmd_help(_args: str) -> None:
            self._console.print(self._commands.get_help_text())

        async def cmd_quit(_args: str) -> None:
            self._console.print("[dim]Goodbye![/dim]")
            sys.exit(0)

        async def cmd_clear(_args: str) -> None:
            self._history.clear()
            self._console.print("[dim]Conversation cleared.[/dim]")

        async def cmd_memory(args: str) -> None:
            subcmd = args.strip().lower() if args else "show"
            if subcmd == "show":
                memory_text = self._memory_store.get_formatted()
                if memory_text:
                    self._console.print(memory_text)
                else:
                    self._console.print("[dim]No memories saved yet.[/dim]")
            elif subcmd == "refresh":
                self._context_content = self._load_context()
                if self._agent_loop:
                    self._agent_loop.set_context_content(self._context_content)
                self._console.print("[dim]Context refreshed.[/dim]")
            elif subcmd == "clear":
                self._memory_store.clear()
                self._console.print("[dim]All memories cleared.[/dim]")
            else:
                self._console.print("[dim]Usage: /memory [show|refresh|clear][/dim]")

        async def cmd_compress(_args: str) -> None:
            if self._history.message_count == 0:
                self._console.print("[dim]Nothing to compress.[/dim]")
                return
            self._console.print("[dim]Compressing conversation...[/dim]")
            try:
                result = await self._compressor.compress(self._history)
                keep = self._compressor._keep_recent
                msg_count = self._history.message_count
                compress_end = msg_count - keep
                if compress_end > 0:
                    from google.genai import types as genai_types

                    summary_content = genai_types.Content(
                        role="user",
                        parts=[genai_types.Part.from_text(
                            text=f"[Conversation summary]\n{result.summary}"
                        )],
                    )
                    self._history.replace_messages(0, compress_end, [summary_content])
                self._console.print(
                    f"[green]Compressed {result.original_message_count} messages "
                    f"into summary ({result.compressed_token_estimate} tokens est.)[/green]"
                )
            except Exception as e:
                self._console.print(f"[red]Compression failed: {e}[/red]")

        async def cmd_chat(args: str) -> None:
            parts = args.strip().split(maxsplit=1) if args else []
            subcmd = parts[0] if parts else "list"
            name = parts[1] if len(parts) > 1 else ""

            if subcmd == "list":
                sessions = self._session_manager.list_sessions()
                if not sessions:
                    self._console.print("[dim]No saved sessions.[/dim]")
                else:
                    self._console.print("[bold]Saved sessions:[/bold]")
                    for s in sessions:
                        self._console.print(
                            f"  {s.name} ({s.message_count} messages, {s.updated_at})"
                        )
            elif subcmd == "save" and name:
                info = self._session_manager.save(name, self._history)
                self._console.print(
                    f"[green]Session saved:[/green] {info.name} ({info.message_count} messages)"
                )
            elif subcmd == "load" and name:
                try:
                    messages = self._session_manager.load(name)
                    self._history.clear()
                    # Re-add messages from saved session
                    for msg in messages:
                        self._history._messages.append(msg)  # noqa: SLF001
                    self._console.print(
                        f"[green]Session loaded:[/green] {name} ({len(messages)} messages)"
                    )
                except FileNotFoundError:
                    self._console.print(f"[red]Session not found: {name}[/red]")
            elif subcmd == "delete" and name:
                if self._session_manager.delete(name):
                    self._console.print(f"[green]Session deleted:[/green] {name}")
                else:
                    self._console.print(f"[red]Session not found: {name}[/red]")
            else:
                self._console.print(
                    "[dim]Usage: /chat [list|save <name>|load <name>|delete <name>][/dim]"
                )

        async def cmd_model(args: str) -> None:
            model_name = args.strip() if args else ""
            if not model_name:
                self._console.print(f"[dim]Current model: {self._config.model}[/dim]")
            else:
                self._config.model = model_name
                # Reset content generator to pick up new model
                self._content_generator = None
                self._agent_loop = None
                self._console.print(f"[green]Model switched to:[/green] {model_name}")

        async def cmd_restore(_args: str) -> None:
            self._console.print(
                "[dim]Checkpointing requires --checkpointing flag. "
                "Use /chat load <name> for session restore.[/dim]"
            )

        self._commands.register("help", cmd_help, "Show available commands")
        self._commands.register("quit", cmd_quit, "Exit PyGeminiCLI")
        self._commands.register("clear", cmd_clear, "Clear conversation history")
        self._commands.register("memory", cmd_memory, "Show/refresh/clear memories")
        self._commands.register("compress", cmd_compress, "Compress conversation history")
        self._commands.register("chat", cmd_chat, "Save/load/list sessions")
        self._commands.register("model", cmd_model, "Show or switch model")
        self._commands.register("restore", cmd_restore, "Restore from checkpoint")

    def _ensure_content_generator(self) -> ContentGenerator:
        """Lazily create the ContentGenerator (validates API key)."""
        if self._content_generator is None:
            self._content_generator = ContentGenerator(self._config)
        return self._content_generator

    def _ensure_agent_loop(self) -> AgentLoop:
        """Lazily create the AgentLoop."""
        if self._agent_loop is None:
            self._agent_loop = AgentLoop(
                content_generator=self._ensure_content_generator(),
                tool_registry=self._tool_registry,
                history=self._history,
                event_emitter=self._event_emitter,
                config=self._config,
                context_content=self._context_content,
                compressor=self._compressor,
                hook_manager=self._hook_manager,
                approval_manager=self._approval_manager,
                policy_engine=self._policy_engine,
            )
        return self._agent_loop

    async def run(self) -> None:
        """Run the interactive REPL."""
        self._renderer.render_welcome()

        while True:
            try:
                user_input = await self._input_handler.get_input()

                if user_input is None:
                    # EOF or empty
                    continue

                # Check for slash commands
                if user_input.startswith("/"):
                    was_command = await self._commands.dispatch(user_input)
                    if was_command:
                        cmd_name = user_input.split()[0][1:]
                        if not self._commands.get(cmd_name):
                            self._renderer.print_error(
                                f"Unknown command: {user_input.split()[0]}"
                            )
                        continue

                # Run agent loop
                try:
                    agent_loop = self._ensure_agent_loop()
                    await agent_loop.run(user_input)
                except ValueError as e:
                    # API key missing etc.
                    self._renderer.print_error(str(e))

            except KeyboardInterrupt:
                # Abort current turn if running
                if self._agent_loop:
                    self._agent_loop.abort_signal.set()
                self._console.print("\n[dim]Interrupted.[/dim]")
                continue
            except EOFError:
                self._console.print("\n[dim]Goodbye![/dim]")
                break

    async def run_headless(self, prompt: str) -> None:
        """Run a single prompt in headless mode (no REPL)."""
        try:
            agent_loop = self._ensure_agent_loop()
            await agent_loop.run(prompt)
        except ValueError as e:
            self._renderer.print_error(str(e))
            sys.exit(1)
