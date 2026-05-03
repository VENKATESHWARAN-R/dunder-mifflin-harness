"""Runtime coordinator facade used by CLI and future UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from jac.config import Settings
from jac.runtime.events import (
    AgentMessageCompleted,
    AgentTextDelta,
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
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
    ) -> None:
        self.settings = settings
        self.session = session or SessionState()
        self.events = events or EventBus()
        self.state = state
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

        return await config_loader(
            state=self.state,
            settings=self.settings,
            run_id=self.session.run_id,
            role=self.session.config.role,
            events=self.events,
            model_settings={
                "temperature": float(
                    self.session.config.model_params.get("temperature", "0")
                )
            },
        )

    async def _ensure_agent(self) -> Agent:
        if self._agent is None:
            if self.state is not None:
                from jac.agents import ensure_default_run_config

                cfg = await ensure_default_run_config(
                    self.state,
                    self.session.run_id,
                    role=self.session.config.role,
                    model_tier=str(
                        self.session.config.tier or self.settings.default_tier
                    ),
                    model_override=self.session.config.model,
                )
                # Sync DB row if session config drifted since creation.
                expected_tier = str(
                    self.session.config.tier or self.settings.default_tier
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
            self._agent = await self.build_agent()
        return self._agent

    def reset_agent(self) -> None:
        """Drop the cached agent after a model or parameter change."""
        self._agent = None

    def seed_message_history(self, history: list[ModelMessage]) -> None:
        """Pre-load message history (used by resume)."""
        self._message_history = list(history)
        self._run_persisted = True

    def _configure_observability(self) -> None:
        if self._logfire_configured:
            return
        logfire.configure()
        logfire.instrument_pydantic_ai()
        self._logfire_configured = True

    async def _ensure_run_persisted(self, prompt: str) -> None:
        if self._run_persisted or self.state is None:
            return
        await self.state.runs.create(
            run_id=self.session.run_id,
            prompt=prompt,
            workflow_mode=str(self.session.config.mode),
        )
        self._run_persisted = True

    async def submit_message(self, message: UserMessage) -> str:
        """Run one prompt through the backend and emit runtime events."""
        run_id = self.session.run_id
        prompt = message.as_prompt()
        await self._ensure_run_persisted(message.text)
        await self.events.emit(RunStarted(run_id=run_id, prompt=message.text))
        self.session.remember_attachments([item.path for item in message.attachments])

        if self.state is not None:
            await self.state.messages.append(run_id, "user", message.text)

        try:
            self._configure_observability()
            agent = await self._ensure_agent()
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
            raise

        output = result.output
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
            history.append(
                ModelResponse(parts=[TextPart(content=message.content)])
            )
    coordinator.seed_message_history(history)
    return coordinator
