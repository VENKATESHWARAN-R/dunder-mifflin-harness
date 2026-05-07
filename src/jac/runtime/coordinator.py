"""Runtime coordinator facade used by CLI and future UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
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
    EventBus,
    LlmCallCompleted,
    RunCompleted,
    RunFailed,
    RunStarted,
    SessionUsageUpdated,
    WarningRaised,
)
from jac.runtime.model_specs import spec_for
from jac.runtime.usage import snapshot_usage, sum_child_usage, vector_sub
from jac.runtime.history import filter_tool_noise
from jac.runtime.models import build_pydantic_model
from jac.runtime.session import SessionState
from jac.state import StateStore
from jac.tools.filesystem import (
    FileAttachment,
    format_attachments_for_prompt,
)
from jac.tools.cache import ToolResultCache
from jac.tools.summarize import build_scout_summariser


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
        self._tool_result_cache = ToolResultCache()
        self._summariser = build_scout_summariser(
            settings, state=self.state, session=self.session
        )

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
        from jac.agents.spawn import native_agent_extras
        from jac.agents.tools import make_summon_jim_tool

        role = self.session.config.role
        cfg = await self.state.agent_configs.get_by_run_and_role(
            self.session.run_id, role
        )
        parent_allowed = json.loads(cfg.allowed_tools) if cfg else []
        extra_tools = native_agent_extras(
            state=self.state,
            settings=self.settings,
            session=self.session,
            events=self.events,
            approval_policy=self.approval_policy,
            cache=self._tool_result_cache,
            summariser=self._summariser,
            parent_role=role,
            parent_depth=0,
            parent_allowed_tools=parent_allowed,
        )
        if role == "manager":
            extra_tools.append(
                make_summon_jim_tool(
                    self.state,
                    self.settings,
                    self.session,
                    self.events,
                    self.approval_policy,
                    tool_result_cache=self._tool_result_cache,
                    summariser=self._summariser,
                )
            )

        return await config_loader(
            state=self.state,
            settings=self.settings,
            run_id=self.session.run_id,
            role=role,
            events=self.events,
            approval_policy=self.approval_policy,
            extra_tools=extra_tools,
            tool_result_cache=self._tool_result_cache,
            summariser=self._summariser,
            model_settings={
                "temperature": float(
                    self.session.config.model_params.get("temperature", "0")
                )
            },
        )

    async def _ensure_agent(self) -> Agent:
        if self._agent is None:
            if self.state is not None:
                from jac.agents import (
                    ensure_builder_config,
                    ensure_manager_config,
                    ensure_planner_config,
                )

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
                await ensure_planner_config(
                    self.state,
                    self.session.run_id,
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
                p_row = await self.state.agent_configs.get_by_run_and_role(
                    self.session.run_id, "planner"
                )
                if (
                    p_row is not None
                    and p_row.model_override != self.session.config.model
                ):
                    await self.state.agent_configs.update(
                        p_row.config_id,
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

    async def _finish_manager_attempt(
        self, attempt_id: str | None, *, passed: bool
    ) -> None:
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
            R = snapshot_usage(usage)
            own = R
            if self.state is not None and scott_attempt_id is not None:
                children = await self.state.attempts.list_direct_children(
                    scott_attempt_id
                )
                own = vector_sub(R, sum_child_usage(children))
                await self.state.attempts.update_usage(
                    scott_attempt_id,
                    tokens_in=own[0],
                    tokens_out=own[1],
                    requests=own[2],
                    tool_calls=own[3],
                    duration_ms=duration_ms,
                )
            self.session.cumulative_tokens_in += R[0]
            self.session.cumulative_tokens_out += R[1]
            self.session.cumulative_requests += R[2]
            self.session.cumulative_tool_calls += R[3]
            self.session.last_context_tokens = R[0]
            self.session.last_model = selection.model_ref
            mx = spec_for(selection.model_ref).max_context
            pct = (R[0] / mx) if mx else 0.0
            await self.events.emit(
                SessionUsageUpdated(
                    tokens_in=self.session.cumulative_tokens_in,
                    tokens_out=self.session.cumulative_tokens_out,
                    requests=self.session.cumulative_requests,
                    tool_calls=self.session.cumulative_tool_calls,
                    last_context_tokens=R[0],
                    context_max=mx,
                    context_pct=pct,
                    model=selection.model_ref,
                )
            )
            await self.events.emit(
                LlmCallCompleted(
                    role="manager",
                    model=selection.model_ref,
                    tier=tier,
                    call_type="agent",
                    input_tokens=own[0],
                    output_tokens=own[1],
                    requests=own[2],
                    tool_calls=own[3],
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

    async def submit_slash_run(
        self,
        *,
        role: str,
        prompt: str,
        addendum_mode: str,
        output_type: type | None = None,
        persist_user_prompt: str | None = None,
    ) -> object:
        """Run one prompt through a specified role with a mode addendum."""
        from jac.agents import config_loader
        from jac.agents.spawn import native_agent_extras
        from jac.agents.modes import MODE_PROMPTS
        from jac.agents.tools import make_summon_jim_tool

        if self.state is None:
            raise RuntimeError("slash runs require a state-backed session")

        run_id = self.session.run_id
        persisted_prompt = persist_user_prompt or prompt
        await self._ensure_run_persisted(persisted_prompt)
        await self.events.emit(RunStarted(run_id=run_id, prompt=persisted_prompt))
        if self.state is not None:
            await self.state.messages.append(run_id, "user", persisted_prompt)

        started_at = perf_counter()
        attempt_id: str | None = None
        try:
            warning_message = self._configure_observability()
            if warning_message:
                await self.events.emit(WarningRaised(message=warning_message))
            cfg = await self.state.agent_configs.get_by_run_and_role(run_id, role)
            parent_allowed = json.loads(cfg.allowed_tools) if cfg else []
            extra_tools = native_agent_extras(
                state=self.state,
                settings=self.settings,
                session=self.session,
                events=self.events,
                approval_policy=self.approval_policy,
                cache=self._tool_result_cache,
                summariser=self._summariser,
                parent_role=role,
                parent_depth=0,
                parent_allowed_tools=parent_allowed,
            )
            if role == "manager":
                extra_tools.append(
                    make_summon_jim_tool(
                        self.state,
                        self.settings,
                        self.session,
                        self.events,
                        self.approval_policy,
                        tool_result_cache=self._tool_result_cache,
                        summariser=self._summariser,
                    )
                )
            temperature = float(
                self.session.config.model_params.get("temperature", "0")
            )
            agent = await config_loader(
                state=self.state,
                settings=self.settings,
                run_id=run_id,
                role=role,
                output_type=output_type,
                events=self.events,
                approval_policy=self.approval_policy,
                model_settings={"temperature": temperature},
                extra_tools=extra_tools,
                instructions_addendum=MODE_PROMPTS.get(addendum_mode),
                tool_result_cache=self._tool_result_cache,
                summariser=self._summariser,
            )
            if self.state is not None:
                cfg = await self.state.agent_configs.get_by_run_and_role(run_id, role)
                if cfg is not None:
                    selection = self.settings.resolve_model_selection(
                        model_override=cfg.model_override,
                        tier=cfg.model_tier,
                    )
                    attempt = await self.state.attempts.create(
                        run_id=run_id,
                        role=role,
                        model=selection.model_ref,
                        tier=cfg.model_tier,
                        call_type="agent",
                        parent_attempt_id=self.session.active_attempt_id,
                    )
                    attempt_id = attempt.attempt_id
            prev_active = self.session.active_attempt_id
            if attempt_id is not None:
                self.session.active_attempt_id = attempt_id
            try:
                async with agent:
                    result = await agent.run(
                        prompt, message_history=self._message_history or None
                    )
            finally:
                self.session.active_attempt_id = prev_active
        except Exception as exc:
            await self.events.emit(
                RunFailed(run_id=run_id, message=str(exc), exception=exc)
            )
            if self.state is not None:
                await self.state.runs.update_status(run_id, "failed")
                if attempt_id is not None:
                    await self.state.attempts.update_status(attempt_id, "failed")
            raise

        if self.state is not None and attempt_id is not None:
            await self.state.attempts.update_status(attempt_id, "passed")

        output = result.output
        output_text = output if isinstance(output, str) else str(output)
        usage = result.usage() if hasattr(result, "usage") else None
        if usage is not None:
            R = snapshot_usage(usage)
            own = R
            if self.state is not None and attempt_id is not None:
                children = await self.state.attempts.list_direct_children(attempt_id)
                own = vector_sub(R, sum_child_usage(children))
                duration_ms = int((perf_counter() - started_at) * 1000)
                await self.state.attempts.update_usage(
                    attempt_id,
                    tokens_in=own[0],
                    tokens_out=own[1],
                    requests=own[2],
                    tool_calls=own[3],
                    duration_ms=duration_ms,
                )
            else:
                duration_ms = int((perf_counter() - started_at) * 1000)
            self.session.cumulative_tokens_in += R[0]
            self.session.cumulative_tokens_out += R[1]
            self.session.cumulative_requests += R[2]
            self.session.cumulative_tool_calls += R[3]
            self.session.last_context_tokens = R[0]
            tier = str(self.session.config.tier or self.settings.default_tier)
            selection = self.settings.resolve_model_selection(
                model_override=self.session.config.model,
                tier=tier,
            )
            self.session.last_model = selection.model_ref
            mx = spec_for(selection.model_ref).max_context
            pct = (R[0] / mx) if mx else 0.0
            await self.events.emit(
                SessionUsageUpdated(
                    tokens_in=self.session.cumulative_tokens_in,
                    tokens_out=self.session.cumulative_tokens_out,
                    requests=self.session.cumulative_requests,
                    tool_calls=self.session.cumulative_tool_calls,
                    last_context_tokens=R[0],
                    context_max=mx,
                    context_pct=pct,
                    model=selection.model_ref,
                )
            )
            await self.events.emit(
                LlmCallCompleted(
                    role=role,
                    model=selection.model_ref,
                    tier=tier,
                    call_type="agent",
                    input_tokens=own[0],
                    output_tokens=own[1],
                    requests=own[2],
                    tool_calls=own[3],
                    duration_ms=duration_ms,
                )
            )

        self._message_history = filter_tool_noise(list(result.all_messages()))
        await self.events.emit(AgentTextDelta(text=output_text))
        await self.events.emit(AgentMessageCompleted(message=output_text))
        await self.events.emit(RunCompleted(run_id=run_id, output=output_text))
        if self.state is not None:
            await self.state.messages.append(run_id, "assistant", output_text)
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
