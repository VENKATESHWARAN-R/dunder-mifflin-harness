"""Runtime coordinator facade used by CLI and future UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import logfire
from pydantic_ai import Agent

from dunder_mifflin_harness.config import Settings
from dunder_mifflin_harness.runtime.events import (
    AgentMessageCompleted,
    AgentTextDelta,
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from dunder_mifflin_harness.runtime.session import SessionState
from dunder_mifflin_harness.tools.filesystem import (
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
    ) -> None:
        self.settings = settings
        self.session = session or SessionState()
        self.events = events or EventBus()
        self._agent: Agent | None = None
        self._logfire_configured = False

    def build_agent(self) -> Agent:
        """Build the current single-agent backend."""
        model = self.session.config.model or self.settings.model
        temperature = float(self.session.config.model_params.get("temperature", "0"))
        return Agent(
            model,
            instructions=(
                "You are a helpful assistant inside the dunder-mifflin-harness "
                "research CLI. Answer clearly and keep implementation details "
                "grounded in the user's workspace."
            ),
            output_type=str,
            model_settings={"temperature": temperature},
        )

    def _ensure_agent(self) -> Agent:
        if self._agent is None:
            self._agent = self.build_agent()
        return self._agent

    def reset_agent(self) -> None:
        """Drop the cached agent after a model or parameter change."""
        self._agent = None

    def _configure_observability(self) -> None:
        if self._logfire_configured:
            return
        logfire.configure()
        logfire.instrument_pydantic_ai()
        self._logfire_configured = True

    async def submit_message(self, message: UserMessage) -> str:
        """Run one prompt through the backend and emit runtime events."""
        run_id = uuid4().hex
        prompt = message.as_prompt()
        await self.events.emit(RunStarted(run_id=run_id, prompt=message.text))
        self.session.remember_attachments([item.path for item in message.attachments])

        try:
            self._configure_observability()
            result = await self._ensure_agent().run(prompt)
        except Exception as exc:
            await self.events.emit(
                RunFailed(run_id=run_id, message=str(exc), exception=exc)
            )
            raise

        output = result.output
        await self.events.emit(AgentTextDelta(text=output))
        await self.events.emit(AgentMessageCompleted(message=output))
        await self.events.emit(RunCompleted(run_id=run_id, output=output))
        return output
