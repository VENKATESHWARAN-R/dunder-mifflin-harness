"""Repository for the `run_skills` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class RunSkillRow:
    id: str
    run_id: str
    skill_id: str
    agent_role: str | None
    enabled: int
    toggled_at: str | None


@dataclass(frozen=True, slots=True)
class ActiveSkillRow:
    run_skill_id: str
    skill_id: str
    name: str
    domain: str
    content: str


class RunSkillsRepo:
    """CRUD for `run_skills`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        skill_id: str,
        agent_role: str | None = None,
        enabled: int = 1,
    ) -> RunSkillRow:
        now = _now()
        row_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO run_skills
                (id, run_id, skill_id, agent_role, enabled, toggled_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row_id, run_id, skill_id, agent_role, enabled, now),
        )
        await self._connection.commit()
        return RunSkillRow(
            id=row_id,
            run_id=run_id,
            skill_id=skill_id,
            agent_role=agent_role,
            enabled=enabled,
            toggled_at=now,
        )

    async def list_active_for_run(
        self, run_id: str, agent_role: str | None = None
    ) -> list[RunSkillRow]:
        """Return enabled run_skills rows joined with enabled skills, ordered by domain, name."""
        params: list[object] = [run_id]
        role_filter = ""
        if agent_role is not None:
            role_filter = "AND (rs.agent_role = ? OR rs.agent_role IS NULL)"
            params.append(agent_role)
        cursor = await self._connection.execute(
            f"""
            SELECT rs.id, rs.run_id, rs.skill_id, rs.agent_role,
                   rs.enabled, rs.toggled_at
            FROM run_skills rs
            JOIN skills s ON s.skill_id = rs.skill_id
            WHERE rs.run_id = ? {role_filter}
              AND rs.enabled = 1
              AND s.is_enabled = 1
            ORDER BY s.domain ASC, s.name ASC
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_run_skill_from_row(r) for r in rows]

    async def list_active_for_run_with_details(
        self, run_id: str, agent_role: str | None = None
    ) -> list[ActiveSkillRow]:
        """Return enabled run_skills joined with skills details, ordered by domain, name."""
        params: list[object] = [run_id]
        role_filter = ""
        if agent_role is not None:
            role_filter = "AND (rs.agent_role = ? OR rs.agent_role IS NULL)"
            params.append(agent_role)
        cursor = await self._connection.execute(
            f"""
            SELECT rs.id AS run_skill_id, rs.skill_id,
                   s.name, s.domain, s.content
            FROM run_skills rs
            JOIN skills s ON s.skill_id = rs.skill_id
            WHERE rs.run_id = ? {role_filter}
              AND rs.enabled = 1
              AND s.is_enabled = 1
            ORDER BY s.domain ASC, s.name ASC
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ActiveSkillRow(
                run_skill_id=r["run_skill_id"],
                skill_id=r["skill_id"],
                name=r["name"],
                domain=r["domain"],
                content=r["content"],
            )
            for r in rows
        ]

    async def toggle(
        self, row_id: str, *, enabled: int
    ) -> RunSkillRow | None:
        now = _now()
        await self._connection.execute(
            """
            UPDATE run_skills
            SET enabled = ?, toggled_at = ?
            WHERE id = ?
            """,
            (enabled, now, row_id),
        )
        await self._connection.commit()
        cursor = await self._connection.execute(
            "SELECT * FROM run_skills WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _run_skill_from_row(row) if row is not None else None


def _run_skill_from_row(row: aiosqlite.Row) -> RunSkillRow:
    return RunSkillRow(
        id=row["id"],
        run_id=row["run_id"],
        skill_id=row["skill_id"],
        agent_role=row["agent_role"],
        enabled=row["enabled"],
        toggled_at=row["toggled_at"],
    )
