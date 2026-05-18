"""Repository for the `agent_configs` table (schema v1.5, slim shape).

Per-run override layer for agent configs. The canonical persona shape lives
in YAML under `src/jac/data/personas/<role>.yaml`; this table holds only
what is per-run mutable: tier/model overrides, tool toggles, prompt overlay.

`fetch_per_run_override` is the bridge into the agent factory: the runtime
slice pre-fetches a `PerRunOverride` for `(run_id, role)` and passes it via
the `per_run=` kwarg on `build_agent`, keeping the factory sync.

Slim columns dropped vs. legacy: `persona`, `display_name`, `is_minion`,
`parent_role`, `depth`. Per `docs/contracts/STATE_SCHEMA.md`, these only
return via Phase-2 evidence triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import aiosqlite

from jac.agents.overrides import PerRunOverride
from jac.config import Tier

if TYPE_CHECKING:
    from jac.state.db import StateStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class AgentConfigRow:
    config_id: str
    run_id: str
    role: str
    model_tier: str
    model_override: str | None
    system_prompt: str | None
    allowed_tools: str  # JSON-encoded array
    max_context_tokens: int
    created_at: str
    updated_at: str


class AgentConfigsRepo:
    """CRUD for `agent_configs`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        role: str,
        model_tier: str,
        model_override: str | None = None,
        system_prompt: str | None = None,
        allowed_tools: str = "[]",
        max_context_tokens: int = 8000,
    ) -> AgentConfigRow:
        now = _now()
        config_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO agent_configs (
                config_id, run_id, role, model_tier, model_override,
                system_prompt, allowed_tools, max_context_tokens,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config_id,
                run_id,
                role,
                model_tier,
                model_override,
                system_prompt,
                allowed_tools,
                max_context_tokens,
                now,
                now,
            ),
        )
        await self._connection.commit()
        return AgentConfigRow(
            config_id=config_id,
            run_id=run_id,
            role=role,
            model_tier=model_tier,
            model_override=model_override,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            max_context_tokens=max_context_tokens,
            created_at=now,
            updated_at=now,
        )

    async def get_by_run_and_role(
        self, run_id: str, role: str
    ) -> AgentConfigRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM agent_configs WHERE run_id = ? AND role = ?",
            (run_id, role),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _agent_config_from_row(row) if row is not None else None

    async def update(
        self,
        run_id: str,
        role: str,
        *,
        model_tier: str | None = None,
        model_override: str | None = None,
        system_prompt: str | None = None,
        allowed_tools: str | None = None,
        max_context_tokens: int | None = None,
    ) -> AgentConfigRow | None:
        """Patch the row in place. Any kwarg left as None is preserved."""
        current = await self.get_by_run_and_role(run_id, role)
        if current is None:
            return None
        new_tier = model_tier if model_tier is not None else current.model_tier
        new_override = (
            model_override if model_override is not None else current.model_override
        )
        new_prompt = (
            system_prompt if system_prompt is not None else current.system_prompt
        )
        new_tools = (
            allowed_tools if allowed_tools is not None else current.allowed_tools
        )
        new_max = (
            max_context_tokens
            if max_context_tokens is not None
            else current.max_context_tokens
        )
        now = _now()
        await self._connection.execute(
            """
            UPDATE agent_configs
            SET model_tier = ?, model_override = ?, system_prompt = ?,
                allowed_tools = ?, max_context_tokens = ?, updated_at = ?
            WHERE run_id = ? AND role = ?
            """,
            (new_tier, new_override, new_prompt, new_tools, new_max, now, run_id, role),
        )
        await self._connection.commit()
        return AgentConfigRow(
            config_id=current.config_id,
            run_id=run_id,
            role=role,
            model_tier=new_tier,
            model_override=new_override,
            system_prompt=new_prompt,
            allowed_tools=new_tools,
            max_context_tokens=new_max,
            created_at=current.created_at,
            updated_at=now,
        )


async def fetch_per_run_override(
    state: StateStore, run_id: str, role: str = "manager"
) -> PerRunOverride:
    """Build a `PerRunOverride` from the `agent_configs` row for (run_id, role).

    Returns an empty override when no row matches. Only fields the factory
    consumes are populated — `model_tier` and `model_override`.
    """
    row = await state.agent_configs.get_by_run_and_role(run_id, role)
    if row is None:
        return PerRunOverride()
    tier: Tier | None = (
        cast(Tier, row.model_tier)
        if row.model_tier in ("scout", "worker", "architect")
        else None
    )
    return PerRunOverride(tier=tier, model_override=row.model_override)


def _agent_config_from_row(row: aiosqlite.Row) -> AgentConfigRow:
    return AgentConfigRow(
        config_id=row["config_id"],
        run_id=row["run_id"],
        role=row["role"],
        model_tier=row["model_tier"],
        model_override=row["model_override"],
        system_prompt=row["system_prompt"],
        allowed_tools=row["allowed_tools"],
        max_context_tokens=row["max_context_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
