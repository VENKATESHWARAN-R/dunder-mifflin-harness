"""Repository for the `agent_configs` table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class AgentConfigRow:
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
    allowed_tools: str
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
        model_override: str | None,
        system_prompt: str,
        allowed_tools: list[str],
        max_context_tokens: int = 8000,
        persona: str | None = None,
        display_name: str | None = None,
        is_minion: int = 0,
        parent_role: str | None = None,
        depth: int = 0,
    ) -> AgentConfigRow:
        now = _now()
        config_id = uuid4().hex
        tools_json = json.dumps(allowed_tools)
        await self._connection.execute(
            """
            INSERT INTO agent_configs
                (config_id, run_id, role, persona, display_name, is_minion,
                 parent_role, depth, model_tier, model_override,
                 system_prompt, allowed_tools, max_context_tokens, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config_id,
                run_id,
                role,
                persona,
                display_name,
                is_minion,
                parent_role,
                depth,
                model_tier,
                model_override,
                system_prompt,
                tools_json,
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
            persona=persona,
            display_name=display_name,
            is_minion=is_minion,
            parent_role=parent_role,
            depth=depth,
            model_tier=model_tier,
            model_override=model_override,
            system_prompt=system_prompt,
            allowed_tools=tools_json,
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
        config_id: str,
        *,
        model_tier: str | None = None,
        model_override: str | None = None,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        if model_tier is not None:
            fields.append("model_tier = ?")
            values.append(model_tier)
        if model_override is not None:
            fields.append("model_override = ?")
            values.append(model_override)
        if system_prompt is not None:
            fields.append("system_prompt = ?")
            values.append(system_prompt)
        if allowed_tools is not None:
            fields.append("allowed_tools = ?")
            values.append(json.dumps(allowed_tools))
        if max_context_tokens is not None:
            fields.append("max_context_tokens = ?")
            values.append(max_context_tokens)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(_now())
        values.append(config_id)
        await self._connection.execute(
            f"UPDATE agent_configs SET {', '.join(fields)} WHERE config_id = ?",
            values,
        )
        await self._connection.commit()


def _agent_config_from_row(row: aiosqlite.Row) -> AgentConfigRow:
    keys = row.keys()
    return AgentConfigRow(
        config_id=row["config_id"],
        run_id=row["run_id"],
        role=row["role"],
        persona=row["persona"] if "persona" in keys else None,
        display_name=row["display_name"] if "display_name" in keys else None,
        is_minion=row["is_minion"] if "is_minion" in keys else 0,
        parent_role=row["parent_role"] if "parent_role" in keys else None,
        depth=row["depth"] if "depth" in keys else 0,
        model_tier=row["model_tier"],
        model_override=row["model_override"],
        system_prompt=row["system_prompt"],
        allowed_tools=row["allowed_tools"],
        max_context_tokens=row["max_context_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
