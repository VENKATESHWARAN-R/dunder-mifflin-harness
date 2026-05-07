"""Universal spawn_minion and tool-result fetch helpers."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from jac.tools.cache import ToolResultCache
from jac.tools.summarize import Summariser
from jac.tools.types import RiskLevel, ToolApprovalMeta, ToolResult, ToolStatus

SPAWN_MINION_DEFAULT_TIMEOUT = 240
SPAWN_MINION_HARD_CAP = 300
DESTRUCTIVE_TOOL_GROUPS = {"shell", "filesystem"}
READONLY_FALLBACKS = {"shell": "shell:read", "filesystem": "filesystem:read"}
MINION_SYSTEM_PROMPT = """\
You are a single-purpose minion in JAC's agentic harness.
Complete one focused task with the provided tools and return a concise result.
Do not call other agents and do not propose plans.
"""


def _safe_default_tools(parent_allowed: list[str]) -> list[str]:
    safe: list[str] = []
    for tool in parent_allowed:
        safe.append(READONLY_FALLBACKS.get(tool, tool))
    return safe


def make_spawn_minion_tool(
    *,
    state,
    settings,
    session,
    events,
    approval_policy,
    cache: ToolResultCache,
    summariser: Summariser,
    parent_role: str,
    parent_depth: int,
    parent_allowed_tools: list[str],
):
    """Return spawn_minion tool bound to the parent agent context."""

    async def spawn_minion(
        ctx: RunContext[None],
        task: str,
        tools: list[str] | None = None,
        tier: Literal["scout", "worker"] = "scout",
        timeout_sec: int = SPAWN_MINION_DEFAULT_TIMEOUT,
    ) -> str:
        from jac.agents.base import config_loader
        from jac.runtime.events import AttemptRecorded, MinionReturned, MinionSpawned

        if parent_depth >= 1:
            return "spawn_minion refused: minions cannot spawn further minions (depth <= 1)."

        timeout_sec = min(max(timeout_sec, 1), SPAWN_MINION_HARD_CAP)
        if tools is None:
            allowed_tools = _safe_default_tools(parent_allowed_tools)
        else:
            invalid = sorted(set(tools) - set(parent_allowed_tools))
            if invalid:
                return (
                    "spawn_minion refused: requested tools "
                    f"{invalid} are outside caller whitelist {parent_allowed_tools}."
                )
            allowed_tools = list(tools)

        minion_role = f"minion:{uuid4().hex[:8]}"
        cfg = await state.agent_configs.create(
            run_id=session.run_id,
            role=minion_role,
            persona=None,
            display_name=None,
            is_minion=1,
            parent_role=parent_role,
            depth=parent_depth + 1,
            model_tier=tier,
            model_override=None,
            system_prompt=MINION_SYSTEM_PROMPT,
            allowed_tools=allowed_tools,
        )
        await events.emit(
            MinionSpawned(
                parent_role=parent_role,
                minion_role=minion_role,
                task_summary=task[:120],
                tools=allowed_tools,
                tier=tier,
                depth=cfg.depth,
            )
        )

        selection = settings.resolve_model_selection(tier=tier)
        attempt = await state.attempts.create(
            run_id=session.run_id,
            role=minion_role,
            model=selection.model_ref,
            tier=tier,
            parent_attempt_id=session.active_attempt_id,
            call_type="minion",
        )
        await events.emit(
            AttemptRecorded(
                attempt_id=attempt.attempt_id,
                role=minion_role,
                call_type="minion",
                parent_attempt_id=attempt.parent_attempt_id,
            )
        )

        minion_agent = await config_loader(
            state=state,
            settings=settings,
            run_id=session.run_id,
            role=minion_role,
            events=events,
            approval_policy=approval_policy,
            tool_result_cache=cache,
            summariser=summariser,
        )

        started = perf_counter()
        success = False
        try:
            async with minion_agent:
                result = await asyncio.wait_for(
                    minion_agent.run(task, usage=ctx.usage, usage_limits=UsageLimits()),
                    timeout=timeout_sec,
                )
            success = True
            return result.output if isinstance(result.output, str) else str(result.output)
        except asyncio.TimeoutError:
            return f"minion timed out after {timeout_sec}s"
        except Exception as exc:  # noqa: BLE001
            return f"minion failed: {exc}"
        finally:
            await state.attempts.update_status(
                attempt.attempt_id, "passed" if success else "failed"
            )
            await events.emit(
                MinionReturned(
                    parent_role=parent_role,
                    minion_role=minion_role,
                    duration_ms=int((perf_counter() - started) * 1000),
                    success=success,
                )
            )

    setattr(
        spawn_minion,
        "approval",
        ToolApprovalMeta(
            category="agent_spawn",
            risk_level=RiskLevel.MEDIUM,
            reversible=False,
            description_fn=lambda task, **_: f"Spawn minion for: {task[:80]}",
        ),
    )
    return spawn_minion


def make_fetch_full_result_tool(cache: ToolResultCache):
    """Return fetch tool for retrieving raw cached output by handle."""

    class FullResult(ToolResult):
        content: str = ""

    async def fetch_full_result(handle: str) -> FullResult:
        content = cache.fetch(handle)
        if content is None:
            return FullResult(
                status=ToolStatus.NOT_FOUND,
                error=(
                    f"no cached result for handle '{handle}' "
                    "(cache is per-run and resets on resume)"
                ),
            )
        return FullResult(content=content)

    setattr(
        fetch_full_result,
        "approval",
        ToolApprovalMeta(
            category="cache_passthrough",
            risk_level=RiskLevel.READ_ONLY,
            reversible=True,
            description_fn=lambda handle, **_: f"Fetch full result `{handle}`",
        ),
    )
    return fetch_full_result


def native_agent_extras(
    *,
    state,
    settings,
    session,
    events,
    approval_policy,
    cache: ToolResultCache,
    summariser: Summariser,
    parent_role: str,
    parent_depth: int,
    parent_allowed_tools: list[str],
) -> list:
    """Build native per-agent extra tools."""

    return [
        make_spawn_minion_tool(
            state=state,
            settings=settings,
            session=session,
            events=events,
            approval_policy=approval_policy,
            cache=cache,
            summariser=summariser,
            parent_role=parent_role,
            parent_depth=parent_depth,
            parent_allowed_tools=parent_allowed_tools,
        ),
        make_fetch_full_result_tool(cache),
    ]
