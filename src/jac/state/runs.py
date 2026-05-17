"""Repository for the `runs` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class RunRow:
    run_id: str
    prompt: str
    workflow_mode: str
    status: str
    total_cost: float
    total_tokens: int
    created_at: str
    updated_at: str


class RunsRepo:
    """CRUD for `runs`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        run_id: str,
        prompt: str,
        workflow_mode: str = "feature_by_feature",
        status: str = "running",
    ) -> RunRow:
        now = _now()
        await self._connection.execute(
            """
            INSERT INTO runs (run_id, prompt, workflow_mode, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, prompt, workflow_mode, status, now, now),
        )
        await self._connection.commit()
        return RunRow(
            run_id=run_id,
            prompt=prompt,
            workflow_mode=workflow_mode,
            status=status,
            total_cost=0.0,
            total_tokens=0,
            created_at=now,
            updated_at=now,
        )

    async def get(self, run_id: str) -> RunRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return _run_from_row(row)

    async def update_status(self, run_id: str, status: str) -> None:
        await self._connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, _now(), run_id),
        )
        await self._connection.commit()

    async def list_recent(self, limit: int = 20) -> list[RunRow]:
        cursor = await self._connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_run_from_row(r) for r in rows]


def _run_from_row(row: aiosqlite.Row) -> RunRow:
    return RunRow(
        run_id=row["run_id"],
        prompt=row["prompt"],
        workflow_mode=row["workflow_mode"],
        status=row["status"],
        total_cost=row["total_cost"],
        total_tokens=row["total_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
