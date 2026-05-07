"""Interactive CLI composition root."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jac.cli.commands import SlashCommandRegistry
from jac.cli.input import InputSession
from jac.cli.parser import ParsedInputKind, parse_input
from jac.cli.prompts import PromptViews
from jac.cli.renderer import Renderer
from jac.config import Settings
from jac.agents.plans import Plan
from jac.runtime.approvals import (
    ApprovalMode,
    ApprovalPolicy,
)
from jac.runtime.events import (
    PlanGenerated,
    ApprovalRequested,
    EventBus,
    FileEditPreviewed,
    QuestionRequested,
    ShellCommandCompleted,
    ShellCommandStarted,
    WarningRaised,
    WorkspaceSurveyCompleted,
)
from jac.runtime.coordinator import RunCoordinator, UserMessage, resume_run
from jac.runtime.session import ModelTier, RunMode, SessionConfig, SessionState
from jac.state import StateStore, open_state_store, seed_workspace
from jac.tools.shell import run_shell
from jac.workspace import discover_workspace

_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b.*--force", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
]
_UNDO_STACK_LIMIT = 20


class ChatApp:
    """Interactive JAC chat application."""

    def __init__(
        self,
        settings: Settings | None = None,
        events: EventBus | None = None,
        renderer: Renderer | None = None,
        state: StateStore | None = None,
        session: SessionState | None = None,
        coordinator: RunCoordinator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if session is None:
            config = SessionConfig(
                max_attachment_bytes=self.settings.max_attachment_bytes,
                shell_timeout_seconds=self.settings.shell_timeout_seconds,
                shell_max_output_chars=self.settings.shell_max_output_chars,
            )
            session = SessionState(config=config)
        self.session = session
        self.events = events or EventBus()
        self.renderer = renderer or Renderer()
        if hasattr(self.renderer, "set_debug"):
            self.renderer.set_debug(self.session.config.debug)
        self.prompts = PromptViews(self.renderer.console)
        self.state = state
        if coordinator is None:
            self.approvals = ApprovalPolicy(mode=self.session.config.approval_mode)
            self.coordinator = RunCoordinator(
                settings=self.settings,
                session=self.session,
                events=self.events,
                state=self.state,
                approval_policy=self.approvals,
            )
        else:
            # Share the coordinator's policy so /approval mutations flow through
            # to the approval wrapper.
            self.coordinator = coordinator
            self.approvals = coordinator.approval_policy
        self.commands = SlashCommandRegistry()
        self._should_exit = False
        # Undo stack: (path, original_bytes) snapshots captured before file edits
        self._undo_stack: list[tuple[Path, bytes]] = []

        self.input = InputSession(
            self.settings.config_dir / "input_history",
            command_source=lambda: self.commands.descriptions(),
            session_config_source=lambda: self.session.config,
        )

        self.renderer.wire(self.events)
        self._wire_requests()
        self._register_commands()

    @classmethod
    async def open(cls, settings: Settings | None = None) -> "ChatApp":
        """Build a ChatApp with a freshly-opened StateStore for the current cwd."""
        resolved_settings = settings or Settings()
        config = SessionConfig(
            max_attachment_bytes=resolved_settings.max_attachment_bytes,
            shell_timeout_seconds=resolved_settings.shell_timeout_seconds,
            shell_max_output_chars=resolved_settings.shell_max_output_chars,
        )
        session = SessionState(config=config)
        workspace = discover_workspace(session.config.cwd)
        state = await open_state_store(workspace.state_db_path)
        await seed_workspace(workspace, state)
        return cls(settings=resolved_settings, state=state, session=session)

    @classmethod
    async def from_resumed(
        cls, run_id: str, settings: Settings | None = None
    ) -> "ChatApp":
        """Build a ChatApp pre-loaded from a prior run."""
        resolved_settings = settings or Settings()
        workspace = discover_workspace(Path.cwd())
        state = await open_state_store(workspace.state_db_path)
        await seed_workspace(workspace, state)
        try:
            coordinator = await resume_run(
                state=state, settings=resolved_settings, run_id=run_id
            )
        except LookupError:
            await state.close()
            raise
        return cls(
            settings=resolved_settings,
            state=state,
            session=coordinator.session,
            coordinator=coordinator,
            events=coordinator.events,
        )

    async def aclose(self) -> None:
        if self.state is not None:
            await self.state.close()
            self.state = None

    def _wire_requests(self) -> None:
        async def on_approval(event: ApprovalRequested) -> None:
            auto_response = self.approvals.auto_response_for(event.request)
            response = auto_response or await self.prompts.ask_approval(event.request)
            self.approvals.record_response(event.request, response)
            await self.events.resolve_approval(response)

        async def on_question(event: QuestionRequested) -> None:
            response = self.prompts.ask_question(event.request)
            await self.events.answer_question(response)

        async def on_file_edit_previewed(event: FileEditPreviewed) -> None:
            path = event.path
            if path.exists() and path.is_file():
                try:
                    original = path.read_bytes()
                    self._undo_stack.append((path, original))
                    if len(self._undo_stack) > _UNDO_STACK_LIMIT:
                        self._undo_stack.pop(0)
                except OSError:
                    pass

        self.events.on(ApprovalRequested, on_approval)
        self.events.on(QuestionRequested, on_question)
        self.events.on(FileEditPreviewed, on_file_edit_previewed)

    def _register_commands(self) -> None:
        async def help_command(_args: str) -> None:
            self.renderer.print_value("Help", self.commands.help_text())

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
            valid = [t.value for t in ModelTier]
            if value not in valid:
                self.renderer.print_error(
                    f"Unknown tier: '{value}'\n"
                    f"Valid options: {', '.join(valid)}\n"
                    f"Example: /tier worker"
                )
                return
            self.session.config.tier = ModelTier(value)
            self.coordinator.reset_agent()
            self.renderer.print_info(f"Preferred tier set to: {value}")

        async def mode_command(args: str) -> None:
            value = args.strip()
            if not value:
                self.renderer.print_info(f"Current mode: {self.session.config.mode}")
                return
            valid = [m.value for m in RunMode]
            if value not in valid:
                self.renderer.print_error(
                    f"Unknown mode: '{value}'\n"
                    f"Valid options: {', '.join(valid)}\n"
                    f"Example: /mode autopilot"
                )
                return
            self.session.config.mode = RunMode(value)
            self.renderer.print_info(f"Mode set to: {value}")

        async def debug_command(args: str) -> None:
            value = args.strip().lower()
            if not value:
                current = "on" if self.session.config.debug else "off"
                self.renderer.print_info(f"Debug mode: {current}")
                return
            if value not in {"on", "off"}:
                self.renderer.print_error(
                    "Unknown debug mode.\nValid options: on, off\nExample: /debug on"
                )
                return
            enabled = value == "on"
            self.session.config.debug = enabled
            if hasattr(self.renderer, "set_debug"):
                self.renderer.set_debug(enabled)
            self.renderer.print_info(f"Debug mode set to: {'on' if enabled else 'off'}")

        async def approval_command(args: str) -> None:
            value = args.strip()
            if not value:
                self.renderer.print_info(
                    f"Current approval mode: {self.approvals.mode}"
                )
                return
            valid = [m.value for m in ApprovalMode]
            if value not in valid:
                self.renderer.print_error(
                    f"Unknown approval mode: '{value}'\n"
                    f"Valid options: {', '.join(valid)}\n"
                    f"Example: /approval auto-edit"
                )
                return
            mode = ApprovalMode(value)
            self.session.config.approval_mode = mode
            self.approvals.mode = mode
            self.renderer.print_info(f"Approval mode set to: {mode}")

        async def params_command(args: str) -> None:
            parts = args.split(maxsplit=1)
            if not parts:
                self.renderer.print_value(
                    "Model Params", self.session.config.model_params
                )
                return
            if len(parts) != 2:
                self.renderer.print_error(
                    "Usage: /params <key> <value>\n"
                    "Supported keys: temperature, max_tokens\n"
                    "Example: /params temperature 0.2"
                )
                return
            key, value = parts
            valid_keys = {"temperature", "max_tokens"}
            if key not in valid_keys:
                self.renderer.print_error(
                    f"Unknown parameter: '{key}'\n"
                    f"Supported: {', '.join(sorted(valid_keys))}"
                )
                return
            self.session.config.model_params[key] = value
            self.coordinator.reset_agent()
            self.renderer.print_info(f"Parameter set: {key}={value}")

        async def context_command(_args: str) -> None:
            run_id = self.session.run_id
            message_count: int | None = None
            if self.state is not None:
                message_count = await self.state.messages.count_for_run(run_id)
            header = f"run_id: {run_id}"
            if message_count is not None:
                header += f" ({message_count} messages)"
            body = f"{header}\ncwd: {self.session.config.cwd}"
            paths = "\n".join(str(path) for path in self.session.attached_paths)
            if paths:
                body += f"\n\nattached files:\n{paths}"
            self.renderer.print_value("Context", body)

        async def cost_command(_args: str) -> None:
            summary = self.session.latest_cost_summary or "No cost data reported yet."
            self.renderer.print_value("Cost", summary)

        async def clear_command(_args: str) -> None:
            self.renderer.console.clear()

        async def history_command(args: str) -> None:
            n = 10
            stripped = args.strip()
            if stripped.isdigit():
                n = int(stripped)
            if self.state is None:
                self.renderer.print_info("No state store — history unavailable.")
                return
            messages = await self.state.messages.list_for_run(self.session.run_id)
            self.renderer.render_message_history(messages, n)

        async def save_command(args: str) -> None:
            filename = args.strip() or f"jac-session-{self.session.run_id[:8]}.md"
            if self.state is None:
                self.renderer.print_info("No state store — nothing to save.")
                return
            messages = await self.state.messages.list_for_run(self.session.run_id)
            if not messages:
                self.renderer.print_info("No messages to save.")
                return
            lines = [f"# JAC Session {self.session.run_id[:8]}\n"]
            for msg in messages:
                role = msg.role.capitalize()
                lines.append(f"**{role}:** {msg.content}\n")
            Path(filename).write_text("\n".join(lines))
            self.renderer.print_info(f"Session saved to: {filename}")

        async def undo_command(_args: str) -> None:
            if not self._undo_stack:
                self.renderer.print_info("Nothing to undo.")
                return
            path, original = self._undo_stack.pop()
            try:
                path.write_bytes(original)
                self.renderer.print_info(f"Reverted: {path}")
            except OSError as exc:
                self.renderer.print_error(f"Could not revert {path}: {exc}")

        async def capabilities_command(_args: str) -> None:
            config = self.session.config
            lines = [
                f"model:    {config.model or '(from settings)'}",
                f"tier:     {config.tier or '(from settings)'}",
                f"mode:     {config.mode}",
                f"approval: {config.approval_mode}",
            ]
            if self.state is not None:
                agent_cfg = await self.state.agent_configs.get_by_run_and_role(
                    self.session.run_id, config.role
                )
                if agent_cfg:
                    try:
                        tools = json.loads(agent_cfg.allowed_tools)
                        if tools:
                            lines.append(f"\ntools: {', '.join(tools)}")
                    except (ValueError, TypeError):
                        pass

                mcp_rows = await self.state.run_mcp_servers.list_active_for_run(
                    self.session.run_id
                )
                if mcp_rows:
                    names = [getattr(r, "name", str(r)) for r in mcp_rows]
                    lines.append(f"mcp: {', '.join(names)}")

                skill_rows = await self.state.run_skills.list_active_for_run(
                    self.session.run_id
                )
                if skill_rows:
                    names = [getattr(r, "name", str(r)) for r in skill_rows]
                    lines.append(f"skills: {', '.join(names)}")

            self.renderer.print_value("Capabilities", "\n".join(lines))

        async def plan_command(args: str) -> None:
            task = args.strip()
            if not task:
                self.renderer.print_error("Usage: /plan <task description>")
                return
            self.session.config.slash_mode = "plan"
            try:
                plan = await self.coordinator.submit_slash_run(
                    role="planner",
                    prompt=task,
                    addendum_mode="plan",
                    output_type=Plan,
                    persist_user_prompt=f"/plan {task}",
                )
            finally:
                self.session.config.slash_mode = None

            assert isinstance(plan, Plan)
            if self.state is not None:
                await self.state.tasks.create_many(
                    self.session.run_id,
                    [planned.model_dump() for planned in plan.tasks],
                )
            await self.events.emit(
                PlanGenerated(
                    summary=plan.summary,
                    dev_strategy=plan.dev_strategy,
                    task_count=len(plan.tasks),
                )
            )
            self.renderer.render_plan(plan)

        async def init_command(_args: str) -> None:
            self.session.config.slash_mode = "init"
            try:
                result = await self.coordinator.submit_slash_run(
                    role="manager",
                    prompt="Survey this workspace and write an AGENTS.md at the project root.",
                    addendum_mode="init",
                    persist_user_prompt="/init",
                )
            finally:
                self.session.config.slash_mode = None

            agents_path = self.session.config.cwd / "AGENTS.md"
            line_count = 0
            if agents_path.exists():
                try:
                    line_count = len(agents_path.read_text().splitlines())
                except OSError:
                    line_count = 0
            await self.events.emit(
                WorkspaceSurveyCompleted(
                    agents_md_path=agents_path,
                    line_count=line_count,
                )
            )
            self.renderer.print_info(str(result))

        self.commands.register("help", help_command, "Show available commands")
        self.commands.register("quit", quit_command, "Exit the chat loop")
        self.commands.register(
            "model",
            model_command,
            "Show or set the active model",
            example="/model claude-sonnet-4-6",
        )
        self.commands.register(
            "tier",
            tier_command,
            "Show or set preferred model tier",
            example="/tier worker",
        )
        self.commands.register(
            "mode",
            mode_command,
            "Show or set run mode",
            example="/mode autopilot",
        )
        self.commands.register(
            "debug",
            debug_command,
            "Show or set verbose debug tracing",
            example="/debug on",
        )
        self.commands.register(
            "approval",
            approval_command,
            "Show or set approval mode",
            example="/approval auto-edit",
        )
        self.commands.register(
            "params",
            params_command,
            "Show or set model parameters",
            example="/params temperature 0.2",
        )
        self.commands.register("context", context_command, "Show session context")
        self.commands.register("cost", cost_command, "Show current cost summary")
        self.commands.register("clear", clear_command, "Clear the terminal screen")
        self.commands.register(
            "history",
            history_command,
            "Show recent messages",
            example="/history 5",
        )
        self.commands.register(
            "save",
            save_command,
            "Save session transcript to a file",
            example="/save transcript.md",
        )
        self.commands.register("undo", undo_command, "Revert the last file edit")
        self.commands.register(
            "capabilities", capabilities_command, "Show active tools and configuration"
        )
        self.commands.register(
            "plan",
            plan_command,
            "Generate a structured implementation plan",
            example="/plan add a /search command",
        )
        self.commands.register(
            "init",
            init_command,
            "Survey workspace and write AGENTS.md",
            example="/init",
        )

        # Short aliases
        self.commands.alias("h", "help")
        self.commands.alias("q", "quit")
        self.commands.alias("m", "model")
        self.commands.alias("t", "tier")
        self.commands.alias("x", "context")
        self.commands.alias("?", "help")

    async def run(self) -> None:
        """Run the prompt_toolkit chat loop."""
        config = self.session.config
        self.renderer.render_welcome(
            model=config.model,
            tier=str(config.tier) if config.tier else None,
            mode=str(config.mode),
        )
        while not self._should_exit:
            raw = await self.input.read()
            if raw is None:
                continue
            await self.handle_input(raw)

    async def run_resumed(self) -> None:
        """Run after resuming a prior session — shows context preview first."""
        if self.state is not None:
            messages = await self.state.messages.list_for_run(self.session.run_id)
            self.renderer.render_resume_context(messages[-6:])
        await self.run()

    async def handle_input(self, raw: str) -> None:
        """Handle one raw user input."""
        parsed = parse_input(
            raw,
            cwd=self.session.config.cwd,
            max_attachment_bytes=self.session.config.max_attachment_bytes,
        )

        if parsed.kind == ParsedInputKind.EMPTY:
            return

        # Consolidate attachment warnings into a single message
        if len(parsed.warnings) == 1:
            await self.events.emit(WarningRaised(message=parsed.warnings[0].message))
        elif len(parsed.warnings) > 1:
            lines = ["Attachment warnings:"]
            for w in parsed.warnings:
                lines.append(f"  {w.message}")
            await self.events.emit(WarningRaised(message="\n".join(lines)))

        if parsed.kind == ParsedInputKind.SLASH:
            assert parsed.slash is not None
            dispatched = await self.commands.dispatch(
                parsed.slash.command,
                parsed.slash.args,
            )
            if not dispatched:
                self.renderer.print_error(
                    f"Unknown command: /{parsed.slash.command}  (try /help)"
                )
            return

        if parsed.kind == ParsedInputKind.SHELL:
            await self._handle_shell(parsed.shell_command or "")
            return

        await self._submit_message(
            UserMessage(text=parsed.text, attachments=parsed.attachments)
        )

    async def _submit_message(self, message: UserMessage) -> None:
        """Submit a message to the coordinator with retry-on-failure."""
        try:
            await self.coordinator.submit_message(message)
        except Exception:
            # RunFailed event already fired and was rendered; offer retry
            if await self.prompts.ask_yn("Retry with the same input?"):
                try:
                    await self.coordinator.submit_message(message)
                except Exception:
                    pass  # second failure: already shown by RunFailed event

    async def _handle_shell(self, command: str) -> None:
        if not command:
            await self.events.emit(WarningRaised(message="empty shell command"))
            return

        if any(p.search(command) for p in _DESTRUCTIVE_PATTERNS):
            self.renderer.print_warning(f"Potentially destructive command: {command}")
            if not await self.prompts.ask_yn("Run anyway?"):
                self.renderer.print_info("Cancelled.")
                return

        await self.events.emit(
            ShellCommandStarted(
                command=command,
                cwd=self.session.config.cwd,
                timeout_seconds=self.session.config.shell_timeout_seconds,
            )
        )
        result = await run_shell(
            command=command,
            cwd=str(self.session.config.cwd),
            timeout_seconds=self.session.config.shell_timeout_seconds,
            max_output_chars=self.session.config.shell_max_output_chars,
        )
        await self.events.emit(
            ShellCommandCompleted(
                command=result.command,
                cwd=Path(result.cwd),
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            )
        )
