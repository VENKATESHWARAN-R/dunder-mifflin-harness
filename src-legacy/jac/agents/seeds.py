"""Default run-level config seeding."""

from __future__ import annotations

import json
from typing import Any

from jac.agents.base import AgentConfig
from jac.agents.personas import (
    JIM_SYSTEM_PROMPT,
    PAM_SYSTEM_PROMPT,
    PERSONAS,
    SCOTT_SYSTEM_PROMPT,
)
from jac.state import AgentConfigRow, StateStore


def _row_to_agent_config(row: AgentConfigRow) -> AgentConfig:
    allowed_tools: list[str] = []
    if row.allowed_tools:
        try:
            allowed_tools = json.loads(row.allowed_tools)
        except json.JSONDecodeError:
            allowed_tools = []
    return AgentConfig(
        config_id=row.config_id,
        run_id=row.run_id,
        role=row.role,
        persona=row.persona,
        display_name=row.display_name,
        is_minion=row.is_minion,
        parent_role=row.parent_role,
        depth=row.depth,
        model_tier=row.model_tier,
        model_override=row.model_override,
        system_prompt=row.system_prompt,
        allowed_tools=allowed_tools,
        max_context_tokens=row.max_context_tokens,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _ensure_role_config(
    state: StateStore,
    run_id: str,
    *,
    role: str,
    persona: str,
    display_name: str,
    model_tier: str,
    model_override: str | None,
    system_prompt: str,
    allowed_tools: list[str],
) -> AgentConfig:
    existing = await state.agent_configs.get_by_run_and_role(run_id, role)
    if existing is not None:
        return _row_to_agent_config(existing)

    row = await state.agent_configs.create(
        run_id=run_id,
        role=role,
        persona=persona,
        display_name=display_name,
        model_tier=model_tier,
        model_override=model_override,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
    )
    return _row_to_agent_config(row)


async def ensure_manager_config(
    state: StateStore,
    run_id: str,
    *,
    model_tier: str | None = None,
    model_override: str | None = None,
) -> AgentConfig:
    """Idempotently seed the manager (Scott) agent_configs row."""
    p = PERSONAS["manager"]
    tier = model_tier or p.default_tier
    return await _ensure_role_config(
        state,
        run_id,
        role="manager",
        persona=p.persona,
        display_name=p.display_name,
        model_tier=tier,
        model_override=model_override,
        system_prompt=SCOTT_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )


async def ensure_builder_config(
    state: StateStore,
    run_id: str,
    *,
    model_tier: str | None = None,
    model_override: str | None = None,
) -> AgentConfig:
    """Idempotently seed the builder (Jim) agent_configs row."""
    p = PERSONAS["builder"]
    tier = model_tier or p.default_tier
    return await _ensure_role_config(
        state,
        run_id,
        role="builder",
        persona=p.persona,
        display_name=p.display_name,
        model_tier=tier,
        model_override=model_override,
        system_prompt=JIM_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )


async def ensure_planner_config(
    state: StateStore,
    run_id: str,
    *,
    model_tier: str | None = None,
    model_override: str | None = None,
) -> AgentConfig:
    """Idempotently seed the planner (Pam) agent_configs row."""
    p = PERSONAS["planner"]
    tier = model_tier or p.default_tier
    return await _ensure_role_config(
        state,
        run_id,
        role="planner",
        persona=p.persona,
        display_name=p.display_name,
        model_tier=tier,
        model_override=model_override,
        system_prompt=PAM_SYSTEM_PROMPT,
        allowed_tools=["filesystem", "shell"],
    )


async def ensure_default_run_config(
    state: StateStore,
    run_id: str,
    **kwargs: Any,
) -> AgentConfig:
    """Shim — seeds manager config. Preserves old call sites."""
    return await ensure_manager_config(state, run_id, **kwargs)
