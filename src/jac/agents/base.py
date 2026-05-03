"""Agent factory — the only place where pydantic_ai.Agent(...) is called."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from jac.config import Settings
from jac.runtime.events import EventBus, NodeCompleted, NodeStarted
from jac.runtime.models import build_pydantic_model
from jac.state import StateStore
from jac.tools import TOOL_REGISTRY
from jac.tools.types import ToolFn


class AgentConfigNotFound(RuntimeError):
    """Raised when no agent_configs row exists for (run_id, role)."""


class UnknownToolError(RuntimeError):
    """Raised when allowed_tools contains an entry not present in TOOL_REGISTRY."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    config_id: str
    run_id: str
    role: str
    model_tier: str
    model_override: str | None
    system_prompt: str
    allowed_tools: list[str]
    max_context_tokens: int
    created_at: str
    updated_at: str


async def config_loader(
    *,
    state: StateStore,
    settings: Settings,
    run_id: str,
    role: str = "chat",
    output_type: type | None = None,
    events: EventBus | None = None,
    model_settings: Any = None,
) -> Agent:
    """Build a Pydantic AI Agent from persisted config.

    This is the only site that calls ``Agent(...)``.
    """
    if events is not None:
        await events.emit(NodeStarted(node_name="config_loader"))

    cfg = await _load_config(state, run_id, role)
    mcp_toolsets = await _build_mcp_toolsets(state, run_id, role)
    local_tools = _resolve_local_tools(cfg.allowed_tools)
    composed_prompt = await _compose_system_prompt(state, cfg.system_prompt, run_id, role)

    selection = settings.resolve_model_selection(
        model_override=cfg.model_override,
        tier=cfg.model_tier,
    )
    model = build_pydantic_model(selection, settings)

    agent = Agent(
        model,
        instructions=composed_prompt,
        tools=local_tools,
        toolsets=mcp_toolsets,
        output_type=output_type or str,
        model_settings=model_settings,
    )

    if events is not None:
        await events.emit(NodeCompleted(node_name="config_loader"))

    return agent


async def _load_config(state: StateStore, run_id: str, role: str) -> AgentConfig:
    row = await state.agent_configs.get_by_run_and_role(run_id, role)
    if row is None:
        raise AgentConfigNotFound(
            f"No agent_configs row found for run_id={run_id} role={role}"
        )
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
        model_tier=row.model_tier,
        model_override=row.model_override,
        system_prompt=row.system_prompt,
        allowed_tools=allowed_tools,
        max_context_tokens=row.max_context_tokens,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _build_mcp_toolsets(
    state: StateStore, run_id: str, role: str
) -> list[Any]:
    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

    servers = await state.run_mcp_servers.list_active_for_run_with_details(
        run_id, agent_role=role
    )
    toolsets: list[Any] = []
    for server in servers:
        cfg = json.loads(server.config)
        if server.transport == "stdio":
            toolsets.append(
                MCPServerStdio(
                    cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                )
            )
        elif server.transport == "http":
            toolsets.append(MCPServerStreamableHTTP(cfg["url"]))
        elif server.transport == "sse":
            toolsets.append(MCPServerSSE(cfg["url"]))
    return toolsets


def _resolve_local_tools(allowed_tools: list[str]) -> list[ToolFn]:
    local_tools: list[ToolFn] = []
    unknown: list[str] = []
    for name in allowed_tools:
        if name.startswith("mcp:"):
            continue
        tools = TOOL_REGISTRY.get(name)
        if tools is None:
            unknown.append(name)
        else:
            local_tools.extend(tools)
    if unknown:
        known = ", ".join(sorted(TOOL_REGISTRY))
        raise UnknownToolError(
            f"Unknown tool(s): {', '.join(unknown)}. "
            f"Known entries: {known}"
        )
    return local_tools


async def _compose_system_prompt(
    state: StateStore, base_prompt: str, run_id: str, role: str
) -> str:
    skills = await state.run_skills.list_active_for_run_with_details(
        run_id, agent_role=role
    )
    if not skills:
        return base_prompt
    skills_block = "\n\n---\n".join(skill.content for skill in skills)
    return f"{base_prompt}\n\n## Domain Knowledge\n\n{skills_block}"
