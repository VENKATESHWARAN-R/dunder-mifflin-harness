"""Repository for the `run_mcp_servers` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class RunMcpServerRow:
    id: str
    run_id: str
    mcp_server_id: str
    agent_role: str | None
    enabled: int
    toggled_at: str | None


@dataclass(frozen=True, slots=True)
class ActiveMcpServerRow:
    run_mcp_server_id: str
    mcp_server_id: str
    name: str
    transport: str
    config: str


class RunMcpServersRepo:
    """Per-run enablement layer over `mcp_servers`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        mcp_server_id: str,
        agent_role: str | None = None,
        enabled: int = 1,
    ) -> RunMcpServerRow:
        now = _now()
        row_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO run_mcp_servers
                (id, run_id, mcp_server_id, agent_role, enabled, toggled_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row_id, run_id, mcp_server_id, agent_role, enabled, now),
        )
        await self._connection.commit()
        return RunMcpServerRow(
            id=row_id,
            run_id=run_id,
            mcp_server_id=mcp_server_id,
            agent_role=agent_role,
            enabled=enabled,
            toggled_at=now,
        )

    async def list_active_for_run(
        self, run_id: str, agent_role: str | None = None
    ) -> list[RunMcpServerRow]:
        params: list[object] = [run_id]
        role_filter = ""
        if agent_role is not None:
            role_filter = "AND (rm.agent_role = ? OR rm.agent_role IS NULL)"
            params.append(agent_role)
        cursor = await self._connection.execute(
            f"""
            SELECT rm.id, rm.run_id, rm.mcp_server_id, rm.agent_role,
                   rm.enabled, rm.toggled_at
            FROM run_mcp_servers rm
            JOIN mcp_servers ms ON ms.mcp_server_id = rm.mcp_server_id
            WHERE rm.run_id = ? {role_filter}
              AND rm.enabled = 1
              AND ms.is_enabled = 1
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_run_mcp_from_row(r) for r in rows]

    async def list_active_for_run_with_details(
        self, run_id: str, agent_role: str | None = None
    ) -> list[ActiveMcpServerRow]:
        params: list[object] = [run_id]
        role_filter = ""
        if agent_role is not None:
            role_filter = "AND (rm.agent_role = ? OR rm.agent_role IS NULL)"
            params.append(agent_role)
        cursor = await self._connection.execute(
            f"""
            SELECT rm.id AS run_mcp_server_id, rm.mcp_server_id,
                   ms.name, ms.transport, ms.config
            FROM run_mcp_servers rm
            JOIN mcp_servers ms ON ms.mcp_server_id = rm.mcp_server_id
            WHERE rm.run_id = ? {role_filter}
              AND rm.enabled = 1
              AND ms.is_enabled = 1
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ActiveMcpServerRow(
                run_mcp_server_id=r["run_mcp_server_id"],
                mcp_server_id=r["mcp_server_id"],
                name=r["name"],
                transport=r["transport"],
                config=r["config"],
            )
            for r in rows
        ]

    async def toggle(self, row_id: str, *, enabled: int) -> RunMcpServerRow | None:
        now = _now()
        await self._connection.execute(
            "UPDATE run_mcp_servers SET enabled = ?, toggled_at = ? WHERE id = ?",
            (enabled, now, row_id),
        )
        await self._connection.commit()
        cursor = await self._connection.execute(
            "SELECT * FROM run_mcp_servers WHERE id = ?", (row_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _run_mcp_from_row(row) if row is not None else None


def _run_mcp_from_row(row: aiosqlite.Row) -> RunMcpServerRow:
    return RunMcpServerRow(
        id=row["id"],
        run_id=row["run_id"],
        mcp_server_id=row["mcp_server_id"],
        agent_role=row["agent_role"],
        enabled=row["enabled"],
        toggled_at=row["toggled_at"],
    )
