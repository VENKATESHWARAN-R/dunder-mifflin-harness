"""Repository for the `tasks` table (schema v1.5, slim shape).

The task list is JAC's context-resilient memory. Scott maintains it via
`add_task` / `update_task` / `complete_task` / `list_tasks` tools; the list
is injected as a system reminder every turn so it survives compaction and
process restarts.

Slim columns dropped vs. legacy: `acceptance_criteria`, `complexity`,
`tier`, `attempt_count`, `parent_task_id`. They return only via Phase-2
evidence triggers per `docs/contracts/STATE_SCHEMA.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class TaskRow:
    task_id: str
    run_id: str
    title: str
    description: str
    status: str
    order_index: int
    created_at: str
    updated_at: str


class TasksRepo:
    """CRUD for `tasks`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        title: str,
        description: str = "",
        status: str = "pending",
        order_index: int | None = None,
    ) -> TaskRow:
        """Insert a task. `order_index` defaults to `max(order_index)+1` for the run."""
        task_id = uuid4().hex
        now = _now()
        if order_index is None:
            order_index = await self._next_order_index(run_id)
        await self._connection.execute(
            """
            INSERT INTO tasks (
                task_id, run_id, title, description, status,
                order_index, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, run_id, title, description, status, order_index, now, now),
        )
        await self._connection.commit()
        return TaskRow(
            task_id=task_id,
            run_id=run_id,
            title=title,
            description=description,
            status=status,
            order_index=order_index,
            created_at=now,
            updated_at=now,
        )

    async def get(self, task_id: str) -> TaskRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _task_from_row(row) if row is not None else None

    async def list_for_run(
        self, run_id: str, statuses: tuple[str, ...] | None = None
    ) -> list[TaskRow]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            cursor = await self._connection.execute(
                f"SELECT * FROM tasks WHERE run_id = ? AND status IN ({placeholders}) "
                f"ORDER BY order_index ASC, task_id ASC",
                (run_id, *statuses),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM tasks WHERE run_id = ? "
                "ORDER BY order_index ASC, task_id ASC",
                (run_id,),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_task_from_row(r) for r in rows]

    async def update_status(self, task_id: str, status: str) -> None:
        await self._connection.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
            (status, _now(), task_id),
        )
        await self._connection.commit()

    async def update_fields(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> TaskRow | None:
        current = await self.get(task_id)
        if current is None:
            return None
        new_title = title if title is not None else current.title
        new_desc = description if description is not None else current.description
        new_status = status if status is not None else current.status
        now = _now()
        await self._connection.execute(
            "UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? "
            "WHERE task_id = ?",
            (new_title, new_desc, new_status, now, task_id),
        )
        await self._connection.commit()
        return TaskRow(
            task_id=task_id,
            run_id=current.run_id,
            title=new_title,
            description=new_desc,
            status=new_status,
            order_index=current.order_index,
            created_at=current.created_at,
            updated_at=now,
        )

    async def _next_order_index(self, run_id: str) -> int:
        cursor = await self._connection.execute(
            "SELECT COALESCE(MAX(order_index), -1) AS max_idx FROM tasks WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["max_idx"]) + 1 if row is not None else 0


def _task_from_row(row: aiosqlite.Row) -> TaskRow:
    return TaskRow(
        task_id=row["task_id"],
        run_id=row["run_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        order_index=row["order_index"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
