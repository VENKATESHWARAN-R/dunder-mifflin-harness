"""Default run-level config seeding."""

from __future__ import annotations

from jac.agents.base import AgentConfig
from jac.state import StateStore

DEFAULT_CHAT_PROMPT = (
    "You are a helpful assistant inside JAC, a research CLI. "
    "Answer clearly and keep implementation details "
    "grounded in the user's workspace."
)


async def ensure_default_run_config(
    state: StateStore,
    run_id: str,
    *,
    role: str = "chat",
    system_prompt: str = DEFAULT_CHAT_PROMPT,
    allowed_tools: list[str] | None = None,
    model_tier: str = "worker",
    model_override: str | None = None,
) -> AgentConfig:
    """Idempotently ensure an agent_configs row exists for (run_id, role).

    Returns the existing row if one is already present.
    """
    existing = await state.agent_configs.get_by_run_and_role(run_id, role)
    if existing is not None:
        import json

        allowed_tools_list: list[str] = []
        if existing.allowed_tools:
            try:
                allowed_tools_list = json.loads(existing.allowed_tools)
            except json.JSONDecodeError:
                allowed_tools_list = []
        return AgentConfig(
            config_id=existing.config_id,
            run_id=existing.run_id,
            role=existing.role,
            model_tier=existing.model_tier,
            model_override=existing.model_override,
            system_prompt=existing.system_prompt,
            allowed_tools=allowed_tools_list,
            max_context_tokens=existing.max_context_tokens,
            created_at=existing.created_at,
            updated_at=existing.updated_at,
        )

    row = await state.agent_configs.create(
        run_id=run_id,
        role=role,
        model_tier=model_tier,
        model_override=model_override,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools or [],
    )
    return AgentConfig(
        config_id=row.config_id,
        run_id=row.run_id,
        role=row.role,
        model_tier=row.model_tier,
        model_override=row.model_override,
        system_prompt=row.system_prompt,
        allowed_tools=allowed_tools or [],
        max_context_tokens=row.max_context_tokens,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
