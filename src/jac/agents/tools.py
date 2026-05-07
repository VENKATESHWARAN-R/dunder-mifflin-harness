"""Delegation tools for the manager (Scott) agent."""

from __future__ import annotations

import json
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jac.config import Settings
    from jac.runtime.approvals import ApprovalPolicy
    from jac.runtime.events import EventBus
    from jac.runtime.session import SessionState
    from jac.state import StateStore


def make_summon_jim_tool(
    state: StateStore,
    settings: Settings,
    session: SessionState,
    events: EventBus,
    approval_policy: ApprovalPolicy,
    *,
    tool_result_cache,
    summariser,
):
    """Return a tool function that Scott can call to delegate to Jim."""

    async def summon_jim(task: str) -> str:
        """Delegate a coding task to Jim Halpert (builder).

        Args:
            task: Full description of the coding task for Jim to execute.
        """
        from jac.agents.base import config_loader
        from jac.agents.spawn import native_agent_extras
        from jac.agents.personas import PERSONAS
        from jac.runtime.events import AgentDelegated, AttemptRecorded, LlmCallCompleted

        jim_persona = PERSONAS["builder"]

        await events.emit(
            AgentDelegated(
                from_role="manager",
                to_role="builder",
                persona=jim_persona.persona,
                display_name=jim_persona.display_name,
                task_summary=task[:120],
            )
        )

        parent_id = session.active_attempt_id
        tier = str(session.config.tier or settings.default_tier)
        selection = settings.resolve_model_selection(
            model_override=session.config.model,
            tier=tier,
        )
        jim_attempt = await state.attempts.create(
            run_id=session.run_id,
            role="builder",
            model=selection.model_ref,
            tier=tier,
            parent_attempt_id=parent_id,
            call_type="agent",
        )
        await events.emit(
            AttemptRecorded(
                attempt_id=jim_attempt.attempt_id,
                role="builder",
                call_type="agent",
                parent_attempt_id=jim_attempt.parent_attempt_id,
            )
        )

        jim_cfg = await state.agent_configs.get_by_run_and_role(session.run_id, "builder")
        jim_allowed = json.loads(jim_cfg.allowed_tools) if jim_cfg else []
        extras = native_agent_extras(
            state=state,
            settings=settings,
            session=session,
            events=events,
            approval_policy=approval_policy,
            cache=tool_result_cache,
            summariser=summariser,
            parent_role="builder",
            parent_depth=0,
            parent_allowed_tools=jim_allowed,
        )

        jim_agent = await config_loader(
            state=state,
            settings=settings,
            run_id=session.run_id,
            role="builder",
            events=events,
            approval_policy=approval_policy,
            extra_tools=extras,
            tool_result_cache=tool_result_cache,
            summariser=summariser,
            model_settings={
                "temperature": float(
                    session.config.model_params.get("temperature", "0")
                )
            },
        )

        started_at = perf_counter()
        try:
            async with jim_agent:
                result = await jim_agent.run(task)
        except Exception:
            await state.attempts.update_status(jim_attempt.attempt_id, "failed")
            raise

        output = result.output
        usage = result.usage() if hasattr(result, "usage") else None
        duration_ms = int((perf_counter() - started_at) * 1000)
        if usage is not None:
            await events.emit(
                LlmCallCompleted(
                    role="builder",
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
        await state.attempts.update_status(jim_attempt.attempt_id, "passed")
        return output

    return summon_jim
