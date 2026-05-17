"""Repository for the `messages` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class MessageRow:
    message_id: str
    run_id: str
    task_id: str | None
    role: str
    content: str
    created_at: str


class MessagesRepo:
    """Append-only message log for a run."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def append(
        self,
        run_id: str,
        role: str,
        content: str,
        task_id: str | None = None,
    ) -> MessageRow:
        message_id = uuid4().hex
        created_at = _now()
        await self._connection.execute(
            """
            INSERT INTO messages (message_id, run_id, task_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, run_id, task_id, role, content, created_at),
        )
        await self._connection.commit()
        return MessageRow(
            message_id=message_id,
            run_id=run_id,
            task_id=task_id,
            role=role,
            content=content,
            created_at=created_at,
        )

    async def list_for_run(self, run_id: str) -> list[MessageRow]:
        # Append order is what callers want. Two rows can share a second-precision
        # `created_at`; SQLite's `rowid` is monotonic per insert, so tie-break on
        # it to preserve insertion order across same-second appends.
        cursor = await self._connection.execute(
            "SELECT * FROM messages WHERE run_id = ? ORDER BY rowid ASC",
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            MessageRow(
                message_id=r["message_id"],
                run_id=r["run_id"],
                task_id=r["task_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def count_for_run(self, run_id: str) -> int:
        cursor = await self._connection.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["n"]) if row is not None else 0
