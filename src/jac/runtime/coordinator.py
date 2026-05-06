"""Runtime coordinator facade used by CLI and future UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import logfire
from logfire.exceptions import LogfireConfigError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from jac.config import Settings
from jac.runtime.approvals import ApprovalPolicy
from jac.runtime.events import (
    AgentMessageCompleted,
    AgentTextDelta,
    CostUpdated,
    EventBus,
    LlmCallCompleted,
    RunCompleted,
    RunFailed,
    RunStarted,
    WarningRaised,
)
from jac.runtime.models import build_pydantic_model
from jac.runtime.session import SessionState
from jac.state import StateStore
from jac.tools.filesystem import (
    FileAttachment,
    format_attachments_for_prompt,
)


@dataclass(frozen=True, slots=True)
class UserMessage:
    """A user message plus structured attachments."""

    text: str
    attachments: list[FileAttachment] = field(default_factory=list)

    def as_prompt(self) -> str:
        """Materialize the message for the current plain LLM backend."""
        return self.text + format_attachments_for_prompt(self.attachments)


class RunCoordinator:
    """Small facade over the current agent implementation.

    This is the adapter that the CLI talks to today. As graph workflows arrive,
    this class should delegate to workflow runners while keeping the public event
    contract stable.
    """

    def __init__(
        self,
        settings: Settings,
        session: SessionState | None = None,
        events: EventBus | None = None,
        state: StateStore | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or SessionState()
        self.events = events or EventBus()
        self.state = state
        self.approval_policy = approval_policy or ApprovalPolicy(
            mode=self.session.config.approval_mode
        )
        self._agent: Agent | None = None
        self._logfire_configured = False
        self._run_persisted = False
        self._message_history: list[ModelMessage] = []

    def _build_fallback_agent(self) -> Agent:
        """Build the agent inline when no state store is available."""
        selection = self.settings.resolve_model_selection(
            model_override=self.session.config.model,
            tier=self.session.config.tier,
        )
        model = build_pydantic_model(selection, self.settings)
        temperature = float(self.session.config.model_params.get("temperature", "0"))
        return Agent(
            model,
            instructions=(
                "You are a helpful assistant inside JAC, a research CLI. "
                "Answer clearly and keep implementation details "
                "grounded in the user's workspace."
            ),
            output_type=str,
            model_settings={"temperature": temperature},
        )

    async def build_agent(self) -> Agent:
        """Build the current single-agent backend.

        When a state store is available this delegates to the agent factory;
        otherwise it falls back to an inline build.
        """
        if self.state is None:
            return self._build_fallback_agent()
        from jac.agents import config_loader
        from jac.agents.tools import make_summon_jim_tool

        role = self.session.config.role
        extra_tools = None
        if role == "manager":
            extra_tools = [
                make_summon_jim_tool(
                    self.state,
                    self.settings,
                    self.session,
                    self.events,
                    self.approval_policy,
                )
            ]

        return await config_loader(
            state=self.state,
            settings=self.settings,
            run_id=self.session.run_id,
            role=role,
            events=self.events,
            approval_policy=self.approval_policy,
            extra_tools=extra_tools,
            model_settings={
                "temperature": float(
                    self.session.config.model_params.get("temperature", "0")
                )
            },
        )

    async def _ensure_agent(self) -> Agent:
        if self._agent is None:
            if self.state is not None:
                from jac.agents import ensure_builder_config, ensure_manager_config

                expected_tier = str(
                    self.session.config.tier or self.settings.default_tier
                )
                cfg = await ensure_manager_config(
                    self.state,
                    self.session.run_id,
                    model_tier=expected_tier,
                    model_override=self.session.config.model,
                )
                await ensure_builder_config(
                    self.state,
                    self.session.run_id,
                    model_tier=expected_tier,
                    model_override=self.session.config.model,
                )
                if (
                    cfg.model_override != self.session.config.model
                    or cfg.model_tier != expected_tier
                ):
                    await self.state.agent_configs.update(
                        cfg.config_id,
                        model_tier=expected_tier,
                        model_override=self.session.config.model,
                    )
                b_row = await self.state.agent_configs.get_by_run_and_role(
                    self.session.run_id, "builder"
                )
                if b_row is not None and (
                    b_row.model_override != self.session.config.model
                    or b_row.model_tier != expected_tier
                ):
                    await self.state.agent_configs.update(
                        b_row.config_id,
                        model_tier=expected_tier,
                        model_override=self.session.config.model,
                    )
            self._agent = await self.build_agent()
        return self._agent

    def reset_agent(self) -> None:
        """Drop the cached agent after a model or parameter change."""
        self._agent = None

    def seed_message_history(self, history: list[ModelMessage]) -> None:
        """Pre-load message history (used by resume)."""
        self._message_history = list(history)
        self._run_persisted = True

    def _configure_observability(self) -> str | None:
        if self._logfire_configured:
            return None
        try:
            logfire.configure(send_to_logfire="if-token-present")
            logfire.instrument_pydantic_ai()
        except (LogfireConfigError, RuntimeError) as exc:
            self._logfire_configured = True
            return (
                "Observability is disabled for this session because Logfire "
                f"configuration failed: {exc}"
            )
        self._logfire_configured = True
        return None

    async def _ensure_run_persisted(self, prompt: str) -> None:
        if self._run_persisted or self.state is None:
            return
        await self.state.runs.create(
            run_id=self.session.run_id,
            prompt=prompt,
            workflow_mode=str(self.session.config.mode),
        )
        self._run_persisted = True

    async def _start_manager_attempt(self, run_id: str) -> str | None:
        """Create the manager attempt row for the current turn."""
        if self.state is None:
            return None
        tier = str(self.session.config.tier or self.settings.default_tier)
        selection = self.settings.resolve_model_selection(
            model_override=self.session.config.model,
            tier=tier,
        )
        scott_row = await self.state.attempts.create(
            run_id=run_id,
            role="manager",
            model=selection.model_ref,
            tier=tier,
            call_type="agent",
        )
        self.session.active_attempt_id = scott_row.attempt_id
        return scott_row.attempt_id

    async def _finish_manager_attempt(self, attempt_id: str | None, *, passed: bool) -> None:
        """Update the manager attempt status for the current turn."""
        if self.state is None or attempt_id is None:
            return
        await self.state.attempts.update_status(
            attempt_id,
            "passed" if passed else "failed",
        )

    async def submit_message(self, message: UserMessage) -> str:
        """Run one prompt through the backend and emit runtime events."""
        run_id = self.session.run_id
        prompt = message.as_prompt()
        await self._ensure_run_persisted(message.text)
        await self.events.emit(RunStarted(run_id=run_id, prompt=message.text))
        self.session.remember_attachments([item.path for item in message.attachments])

        if self.state is not None:
            await self.state.messages.append(run_id, "user", message.text)

        scott_attempt_id: str | None = None
        started_at = perf_counter()
        try:
            warning_message = self._configure_observability()
            if warning_message:
                await self.events.emit(WarningRaised(message=warning_message))
            agent = await self._ensure_agent()
            scott_attempt_id = await self._start_manager_attempt(run_id)
            async with agent:
                result = await agent.run(
                    prompt, message_history=self._message_history or None
                )
        except Exception as exc:
            await self.events.emit(
                RunFailed(run_id=run_id, message=str(exc), exception=exc)
            )
            if self.state is not None:
                await self.state.runs.update_status(run_id, "failed")
                await self._finish_manager_attempt(scott_attempt_id, passed=False)
            raise
        finally:
            self.session.active_attempt_id = None

        await self._finish_manager_attempt(scott_attempt_id, passed=True)

        output = result.output
        usage = result.usage() if hasattr(result, "usage") else None
        tier = str(self.session.config.tier or self.settings.default_tier)
        selection = self.settings.resolve_model_selection(
            model_override=self.session.config.model,
            tier=tier,
        )
        duration_ms = int((perf_counter() - started_at) * 1000)
        if usage is not None:
            summary = (
                f"{usage.input_tokens} in · {usage.output_tokens} out · "
                f"{usage.requests} req · {usage.tool_calls} tool calls"
            )
            self.session.latest_cost_summary = summary
            await self.events.emit(CostUpdated(summary=summary))
            await self.events.emit(
                LlmCallCompleted(
                    role="manager",
                    model=selection.model_ref,
                    tier=tier,
                    call_type="agent",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    requests=usage.requests,
                    tool_calls=usage.tool_calls,
                    duration_ms=duration_ms,
                )
            )
        self._message_history = list(result.all_messages())
        await self.events.emit(AgentTextDelta(text=output))
        await self.events.emit(AgentMessageCompleted(message=output))
        await self.events.emit(RunCompleted(run_id=run_id, output=output))

        if self.state is not None:
            await self.state.messages.append(run_id, "assistant", output)
            await self.state.runs.update_status(run_id, "running")
        return output


async def resume_run(
    state: StateStore,
    settings: Settings,
    run_id: str,
    events: EventBus | None = None,
    session: SessionState | None = None,
) -> RunCoordinator:
    """Build a coordinator pre-loaded with prior messages from `run_id`."""
    row = await state.runs.get(run_id)
    if row is None:
        raise LookupError(f"no run with id {run_id}")
    target_session = session or SessionState()
    target_session.run_id = run_id
    coordinator = RunCoordinator(
        settings=settings,
        session=target_session,
        events=events,
        state=state,
    )
    history: list[ModelMessage] = []
    for message in await state.messages.list_for_run(run_id):
        if message.role == "user":
            history.append(
                ModelRequest(parts=[UserPromptPart(content=message.content)])
            )
        elif message.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    coordinator.seed_message_history(history)
    return coordinator
