"""Repository for the `skills` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class SkillRow:
    skill_id: str
    name: str
    description: str
    domain: str
    content: str
    version: str
    is_enabled: int
    source_scope: str
    source_path: str | None
    created_at: str
    updated_at: str


class SkillsRepo:
    """CRUD for `skills`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def upsert(
        self,
        *,
        name: str,
        description: str,
        domain: str,
        content: str,
        version: str = "1.0",
        source_scope: str,
        source_path: str | None,
    ) -> SkillRow:
        now = _now()
        cursor = await self._connection.execute(
            "SELECT skill_id, created_at FROM skills WHERE name = ?",
            (name,),
        )
        existing = await cursor.fetchone()
        await cursor.close()
        if existing is not None:
            skill_id = existing["skill_id"]
            created_at = existing["created_at"]
            await self._connection.execute(
                """
                UPDATE skills
                SET description = ?, domain = ?, content = ?, version = ?,
                    source_scope = ?, source_path = ?, updated_at = ?
                WHERE skill_id = ?
                """,
                (
                    description,
                    domain,
                    content,
                    version,
                    source_scope,
                    source_path,
                    now,
                    skill_id,
                ),
            )
        else:
            skill_id = uuid4().hex
            created_at = now
            await self._connection.execute(
                """
                INSERT INTO skills
                    (skill_id, name, description, domain, content, version,
                     is_enabled, source_scope, source_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    name,
                    description,
                    domain,
                    content,
                    version,
                    source_scope,
                    source_path,
                    now,
                    now,
                ),
            )
        await self._connection.commit()
        return SkillRow(
            skill_id=skill_id,
            name=name,
            description=description,
            domain=domain,
            content=content,
            version=version,
            is_enabled=1,
            source_scope=source_scope,
            source_path=source_path,
            created_at=created_at,
            updated_at=now,
        )

    async def get_by_name(self, name: str) -> SkillRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM skills WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _skill_from_row(row) if row is not None else None

    async def list_all(self) -> list[SkillRow]:
        cursor = await self._connection.execute("SELECT * FROM skills ORDER BY name")
        rows = await cursor.fetchall()
        await cursor.close()
        return [_skill_from_row(r) for r in rows]

    async def delete_stale(self, scope: str, known_paths: set[str]) -> list[str]:
        """Delete file-backed rows whose source file is gone and not held by an open run.

        Returns the names of deleted skills.
        """
        cursor = await self._connection.execute(
            "SELECT skill_id, name, source_path FROM skills WHERE source_scope = ? AND source_path IS NOT NULL",
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
                SELECT 1 FROM run_skills rs
                JOIN runs r ON r.run_id = rs.run_id
                WHERE rs.skill_id = ? AND r.status = 'running'
                LIMIT 1
                """,
                (row["skill_id"],),
            )
            held = await cursor2.fetchone()
            await cursor2.close()
            if held is not None:
                continue
            await self._connection.execute(
                "DELETE FROM skills WHERE skill_id = ?", (row["skill_id"],)
            )
            deleted.append(row["name"])
        if deleted:
            await self._connection.commit()
        return deleted


def _skill_from_row(row: aiosqlite.Row) -> SkillRow:
    return SkillRow(
        skill_id=row["skill_id"],
        name=row["name"],
        description=row["description"],
        domain=row["domain"],
        content=row["content"],
        version=row["version"],
        is_enabled=row["is_enabled"],
        source_scope=row["source_scope"],
        source_path=row["source_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
