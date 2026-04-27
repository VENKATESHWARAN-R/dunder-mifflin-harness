"""Interactive CLI composition root."""

from __future__ import annotations

from dunder_mifflin_harness.cli.commands import SlashCommandRegistry
from dunder_mifflin_harness.cli.input import InputSession
from dunder_mifflin_harness.cli.parser import ParsedInputKind, parse_input
from dunder_mifflin_harness.cli.prompts import PromptViews
from dunder_mifflin_harness.cli.renderer import Renderer
from dunder_mifflin_harness.config import Settings
from dunder_mifflin_harness.runtime.approvals import (
    ApprovalMode,
    ApprovalPolicy,
)
from dunder_mifflin_harness.runtime.events import (
    ApprovalRequested,
    EventBus,
    QuestionRequested,
    ShellCommandCompleted,
    ShellCommandStarted,
    WarningRaised,
)
from dunder_mifflin_harness.runtime.runner import RunCoordinator, UserMessage
from dunder_mifflin_harness.runtime.session import ModelTier, RunMode, SessionConfig, SessionState
from dunder_mifflin_harness.tools.shell import run_shell_command


class ChatApp:
    """Interactive harness chat application."""

    def __init__(
        self,
        settings: Settings | None = None,
        events: EventBus | None = None,
        renderer: Renderer | None = None,
    ) -> None:
        self.settings = settings or Settings()
        config = SessionConfig(
            model=self.settings.model,
            max_attachment_bytes=self.settings.max_attachment_bytes,
            shell_timeout_seconds=self.settings.shell_timeout_seconds,
            shell_max_output_chars=self.settings.shell_max_output_chars,
        )
        self.session = SessionState(config=config)
        self.events = events or EventBus()
        self.renderer = renderer or Renderer()
        self.prompts = PromptViews(self.renderer.console)
        self.approvals = ApprovalPolicy(mode=self.session.config.approval_mode)
        self.coordinator = RunCoordinator(
            settings=self.settings,
            session=self.session,
            events=self.events,
        )
        self.input = InputSession(self.settings.config_dir / "input_history")
        self.commands = SlashCommandRegistry()
        self._should_exit = False

        self.renderer.wire(self.events)
        self._wire_requests()
        self._register_commands()

    def _wire_requests(self) -> None:
        async def on_approval(event: ApprovalRequested) -> None:
            auto_response = self.approvals.auto_response_for(event.request)
            response = auto_response or self.prompts.ask_approval(event.request)
            self.approvals.record_response(event.request, response)
            await self.events.resolve_approval(response)

        async def on_question(event: QuestionRequested) -> None:
            response = self.prompts.ask_question(event.request)
            await self.events.answer_question(response)

        self.events.on(ApprovalRequested, on_approval)
        self.events.on(QuestionRequested, on_question)

    def _register_commands(self) -> None:
        async def help_command(_args: str) -> None:
            self.renderer.print_value("Slash Commands", self.commands.help_text())

        async def quit_command(_args: str) -> None:
            self._should_exit = True

        async def model_command(args: str) -> None:
            model = args.strip()
            if not model:
                self.renderer.print_info(f"Current model: {self.session.config.model}")
                return
            self.session.config.model = model
            self.coordinator.reset_agent()
            self.renderer.print_info(f"Model set to: {model}")

        async def tier_command(args: str) -> None:
            value = args.strip()
            if not value:
                current = self.session.config.tier or "(none)"
                self.renderer.print_info(f"Current preferred tier: {current}")
                return
            try:
                self.session.config.tier = ModelTier(value)
            except ValueError:
                self.renderer.print_error("Tier must be scout, worker, or architect.")
                return
            self.renderer.print_info(f"Preferred tier set to: {value}")

        async def mode_command(args: str) -> None:
            value = args.strip()
            if not value:
                self.renderer.print_info(f"Current mode: {self.session.config.mode}")
                return
            try:
                self.session.config.mode = RunMode(value)
            except ValueError:
                self.renderer.print_error("Mode must be autopilot or hitl.")
                return
            self.renderer.print_info(f"Mode set to: {value}")

        async def approval_command(args: str) -> None:
            value = args.strip()
            if not value:
                self.renderer.print_info(f"Current approval mode: {self.approvals.mode}")
                return
            try:
                mode = ApprovalMode(value)
            except ValueError:
                self.renderer.print_error("Approval mode must be interactive, auto-edit, or yolo.")
                return
            self.session.config.approval_mode = mode
            self.approvals.mode = mode
            self.renderer.print_info(f"Approval mode set to: {mode}")

        async def params_command(args: str) -> None:
            parts = args.split(maxsplit=1)
            if not parts:
                self.renderer.print_value("Model Params", self.session.config.model_params)
                return
            if len(parts) != 2:
                self.renderer.print_error("Usage: /params <key> <value>")
                return
            key, value = parts
            if key not in {"temperature", "max_tokens"}:
                self.renderer.print_error("Supported params: temperature, max_tokens")
                return
            self.session.config.model_params[key] = value
            self.coordinator.reset_agent()
            self.renderer.print_info(f"Parameter set: {key}={value}")

        async def context_command(_args: str) -> None:
            paths = "\n".join(str(path) for path in self.session.attached_paths)
            body = f"cwd: {self.session.config.cwd}"
            if paths:
                body += f"\n\nattached files:\n{paths}"
            self.renderer.print_value("Context", body)

        async def cost_command(_args: str) -> None:
            summary = self.session.latest_cost_summary or "No cost data reported yet."
            self.renderer.print_value("Cost", summary)

        self.commands.register("help", help_command, "Show available commands")
        self.commands.register("quit", quit_command, "Exit the chat loop")
        self.commands.register("model", model_command, "Show or set the active model")
        self.commands.register("tier", tier_command, "Show or set preferred model tier")
        self.commands.register("mode", mode_command, "Show or set run mode")
        self.commands.register("approval", approval_command, "Show or set approval mode")
        self.commands.register("params", params_command, "Show or set model parameters")
        self.commands.register("context", context_command, "Show session context")
        self.commands.register("cost", cost_command, "Show current cost summary")

    async def run(self) -> None:
        """Run the prompt_toolkit chat loop."""
        self.renderer.render_welcome()
        while not self._should_exit:
            raw = await self.input.read()
            if raw is None:
                continue
            await self.handle_input(raw)

    async def handle_input(self, raw: str) -> None:
        """Handle one raw user input."""
        parsed = parse_input(
            raw,
            cwd=self.session.config.cwd,
            max_attachment_bytes=self.session.config.max_attachment_bytes,
        )

        if parsed.kind == ParsedInputKind.EMPTY:
            return

        for warning in parsed.warnings:
            await self.events.emit(WarningRaised(message=warning.message))

        if parsed.kind == ParsedInputKind.SLASH:
            assert parsed.slash is not None
            dispatched = await self.commands.dispatch(
                parsed.slash.command,
                parsed.slash.args,
            )
            if not dispatched:
                self.renderer.print_error(f"Unknown command: /{parsed.slash.command}")
            return

        if parsed.kind == ParsedInputKind.SHELL:
            await self._handle_shell(parsed.shell_command or "")
            return

        await self.coordinator.submit_message(
            UserMessage(text=parsed.text, attachments=parsed.attachments)
        )

    async def _handle_shell(self, command: str) -> None:
        if not command:
            await self.events.emit(WarningRaised(message="empty shell command"))
            return

        await self.events.emit(
            ShellCommandStarted(
                command=command,
                cwd=self.session.config.cwd,
                timeout_seconds=self.session.config.shell_timeout_seconds,
            )
        )
        result = await run_shell_command(
            command=command,
            cwd=self.session.config.cwd,
            timeout_seconds=self.session.config.shell_timeout_seconds,
            max_output_chars=self.session.config.shell_max_output_chars,
        )
        await self.events.emit(
            ShellCommandCompleted(
                command=result.command,
                cwd=result.cwd,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            )
        )
