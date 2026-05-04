"""Delegation tools for the manager (Scott) agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jac.config import Settings
    from jac.runtime.events import EventBus
    from jac.runtime.session import SessionState
    from jac.state import StateStore


def make_summon_jim_tool(
    state: StateStore,
    settings: Settings,
    session: SessionState,
    events: EventBus,
):
    """Return a tool function that Scott can call to delegate to Jim."""

    async def summon_jim(task: str) -> str:
        """Delegate a coding task to Jim Halpert (builder).

        Args:
            task: Full description of the coding task for Jim to execute.
        """
        from jac.agents.base import config_loader
        from jac.agents.personas import PERSONAS
        from jac.runtime.events import AgentDelegated, AttemptRecorded

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

        jim_agent = await config_loader(
            state=state,
            settings=settings,
            run_id=session.run_id,
            role="builder",
            events=events,
            model_settings={
                "temperature": float(
                    session.config.model_params.get("temperature", "0")
                )
            },
        )

        try:
            async with jim_agent:
                result = await jim_agent.run(task)
        except Exception:
            await state.attempts.update_status(jim_attempt.attempt_id, "failed")
            raise

        output = result.output
        await state.attempts.update_status(jim_attempt.attempt_id, "passed")
        return output

    return summon_jim
