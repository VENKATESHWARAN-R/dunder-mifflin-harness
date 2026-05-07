"""Agent factory — the only place where pydantic_ai.Agent(...) is called."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from jac.agents.approval import make_approval_wrapper
from jac.agents.result_filter import make_result_filter_wrapper
from jac.config import Settings
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.runtime.events import EventBus, NodeCompleted, NodeStarted
from jac.runtime.models import build_pydantic_model
from jac.state import StateStore
from jac.tools import TOOL_REGISTRY
from jac.tools.types import ToolFn
from jac.tools.types import ToolApprovalMeta

Summariser = Callable[..., Awaitable[str]]

if TYPE_CHECKING:
    from jac.tools.cache import ToolResultCache


class AgentConfigNotFound(RuntimeError):
    """Raised when no agent_configs row exists for (run_id, role)."""


class UnknownToolError(RuntimeError):
    """Raised when allowed_tools contains an entry not present in TOOL_REGISTRY."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    config_id: str
    run_id: str
    role: str
    persona: str | None
    display_name: str | None
    is_minion: int
    parent_role: str | None
    depth: int
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
    role: str = "manager",
    output_type: type | None = None,
    events: EventBus | None = None,
    approval_policy: ApprovalPolicy | None = None,
    model_settings: Any = None,
    extra_tools: Sequence[ToolFn] | None = None,
    instructions_addendum: str | None = None,
    tool_result_cache: ToolResultCache | None = None,
    summariser: Summariser | None = None,
    tool_timeout_seconds: float = 180.0,
) -> Agent:
    """Build a Pydantic AI Agent from persisted config.

    This is the only site that calls ``Agent(...)``.

    Local tools resolved from `allowed_tools` are wrapped by the approval
    middleware (`agents/approval.py`) so every non-read-only call goes
    through the approval gate before executing. The wrapper requires both
    `events` and `approval_policy`; when either is absent the loader falls
    back to a default INTERACTIVE policy and a fresh EventBus so SDK
    callers without a CLI-side wiring still get a coherent gate.
    """
    if events is not None:
        await events.emit(NodeStarted(node_name="config_loader"))

    effective_events = events if events is not None else EventBus()
    effective_policy = approval_policy or ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)

    cfg = await _load_config(state, run_id, role)
    mcp_toolsets = await _build_mcp_toolsets(state, run_id, role)
    local_tools = list(
        _resolve_local_tools(
            cfg.allowed_tools,
            effective_events,
            effective_policy,
            tool_result_cache=tool_result_cache,
            summariser=summariser,
        )
    )
    if extra_tools:
        local_tools.extend(extra_tools)
    composed_prompt = await _compose_system_prompt(
        state, cfg.system_prompt, run_id, role
    )
    if instructions_addendum:
        composed_prompt = f"{composed_prompt}\n\n---\n\n{instructions_addendum}"

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
        tool_timeout=tool_timeout_seconds,
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


async def _build_mcp_toolsets(state: StateStore, run_id: str, role: str) -> list[Any]:
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


def _resolve_local_tools(
    allowed_tools: list[str],
    events: EventBus,
    policy: ApprovalPolicy,
    *,
    tool_result_cache: ToolResultCache | None = None,
    summariser: Summariser | None = None,
) -> list[ToolFn]:
    local_tools: list[ToolFn] = []
    unknown: list[str] = []
    for name in allowed_tools:
        if name.startswith("mcp:"):
            continue
        tools = TOOL_REGISTRY.get(name)
        if tools is None:
            unknown.append(name)
        else:
            for fn in tools:
                wrapped = make_approval_wrapper(fn, events, policy)
                meta: ToolApprovalMeta | None = getattr(wrapped, "approval", None)
                if (
                    tool_result_cache is not None
                    and summariser is not None
                    and meta is not None
                    and meta.category not in {"cache_passthrough", "agent_spawn"}
                ):
                    wrapped = make_result_filter_wrapper(
                        wrapped, tool_result_cache, summariser
                    )
                local_tools.append(wrapped)
    if unknown:
        known = ", ".join(sorted(TOOL_REGISTRY))
        raise UnknownToolError(
            f"Unknown tool(s): {', '.join(unknown)}. Known entries: {known}"
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
