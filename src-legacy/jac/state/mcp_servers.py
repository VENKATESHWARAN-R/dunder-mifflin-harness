"""Repository for the `mcp_servers` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class McpServerRow:
    mcp_server_id: str
    name: str
    description: str
    transport: str
    config: str
    is_enabled: int
    source_scope: str
    source_path: str | None
    created_at: str
    updated_at: str


class McpServersRepo:
    """CRUD for `mcp_servers`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def upsert(
        self,
        *,
        name: str,
        description: str,
        transport: str,
        config: str,
        source_scope: str,
        source_path: str | None,
    ) -> McpServerRow:
        now = _now()
        cursor = await self._connection.execute(
            "SELECT mcp_server_id, created_at FROM mcp_servers WHERE name = ?",
            (name,),
        )
        existing = await cursor.fetchone()
        await cursor.close()
        if existing is not None:
            mcp_server_id = existing["mcp_server_id"]
            created_at = existing["created_at"]
            await self._connection.execute(
                """
                UPDATE mcp_servers
                SET description = ?, transport = ?, config = ?,
                    source_scope = ?, source_path = ?, updated_at = ?
                WHERE mcp_server_id = ?
                """,
                (
                    description,
                    transport,
                    config,
                    source_scope,
                    source_path,
                    now,
                    mcp_server_id,
                ),
            )
        else:
            mcp_server_id = uuid4().hex
            created_at = now
            await self._connection.execute(
                """
                INSERT INTO mcp_servers
                    (mcp_server_id, name, description, transport, config,
                     is_enabled, source_scope, source_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    mcp_server_id,
                    name,
                    description,
                    transport,
                    config,
                    source_scope,
                    source_path,
                    now,
                    now,
                ),
            )
        await self._connection.commit()
        return McpServerRow(
            mcp_server_id=mcp_server_id,
            name=name,
            description=description,
            transport=transport,
            config=config,
            is_enabled=1,
            source_scope=source_scope,
            source_path=source_path,
            created_at=created_at,
            updated_at=now,
        )

    async def get_by_name(self, name: str) -> McpServerRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM mcp_servers WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _mcp_from_row(row) if row is not None else None

    async def list_all(self) -> list[McpServerRow]:
        cursor = await self._connection.execute(
            "SELECT * FROM mcp_servers ORDER BY name"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_mcp_from_row(r) for r in rows]

    async def delete_stale(self, scope: str, known_paths: set[str]) -> list[str]:
        """Delete file-backed rows whose source file is gone and not held by an open run.

        Returns the names of deleted MCP servers.
        """
        cursor = await self._connection.execute(
            "SELECT mcp_server_id, name, source_path FROM mcp_servers "
            "WHERE source_scope = ? AND source_path IS NOT NULL",
            (scope,),
        )
        candidates = await cursor.fetchall()
        await cursor.close()
        deleted: list[str] = []
        for row in candidates:
            if row["source_path"] in known_paths:
                continue
            cursor2 = await self._connection.execute(
                """
                SELECT 1 FROM run_mcp_servers rm
                JOIN runs r ON r.run_id = rm.run_id
                WHERE rm.mcp_server_id = ? AND r.status = 'running'
                LIMIT 1
                """,
                (row["mcp_server_id"],),
            )
            held = await cursor2.fetchone()
            await cursor2.close()
            if held is not None:
                continue
            await self._connection.execute(
                "DELETE FROM mcp_servers WHERE mcp_server_id = ?",
                (row["mcp_server_id"],),
            )
            deleted.append(row["name"])
        if deleted:
            await self._connection.commit()
        return deleted


def _mcp_from_row(row: aiosqlite.Row) -> McpServerRow:
    return McpServerRow(
        mcp_server_id=row["mcp_server_id"],
        name=row["name"],
        description=row["description"],
        transport=row["transport"],
        config=row["config"],
        is_enabled=row["is_enabled"],
        source_scope=row["source_scope"],
        source_path=row["source_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
