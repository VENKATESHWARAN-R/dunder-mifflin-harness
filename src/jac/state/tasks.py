"""Repository for the `tasks` table."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import aiosqlite


@dataclass(frozen=True, slots=True)
class TaskRow:
    task_id: str
    run_id: str
    title: str
    description: str
    acceptance_criteria: str
    status: str
    complexity: str
    tier: str
    attempt_count: int
    order_index: int
    parent_task_id: str | None


class TasksRepo:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create_many(self, run_id: str, tasks: list[dict]) -> list[TaskRow]:
        rows: list[TaskRow] = []
        for index, task in enumerate(tasks):
            row = await self.create(
                run_id=run_id,
                title=task["title"],
                description=task["description"],
                acceptance_criteria=task["acceptance_criteria"],
                complexity=task.get("complexity", "moderate"),
                order_index=index,
            )
            rows.append(row)
        return rows

    async def create(
        self,
        *,
        run_id: str,
        title: str,
        description: str,
        acceptance_criteria: str,
        complexity: str = "moderate",
        tier: str = "worker",
        order_index: int,
        parent_task_id: str | None = None,
    ) -> TaskRow:
        task_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO tasks
                (task_id, run_id, title, description, acceptance_criteria,
                 status, complexity, tier, attempt_count, order_index, parent_task_id)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, 0, ?, ?)
            """,
            (
                task_id,
                run_id,
                title,
                description,
                acceptance_criteria,
                complexity,
                tier,
                order_index,
                parent_task_id,
            ),
        )
        await self._connection.commit()
        return TaskRow(
            task_id=task_id,
            run_id=run_id,
            title=title,
            description=description,
            acceptance_criteria=acceptance_criteria,
            status="pending",
            complexity=complexity,
            tier=tier,
            attempt_count=0,
            order_index=order_index,
            parent_task_id=parent_task_id,
        )

    async def list_for_run(self, run_id: str) -> list[TaskRow]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM tasks
            WHERE run_id = ?
            ORDER BY order_index ASC, task_id ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            TaskRow(
                task_id=row["task_id"],
                run_id=row["run_id"],
                title=row["title"],
                description=row["description"],
                acceptance_criteria=row["acceptance_criteria"],
                status=row["status"],
                complexity=row["complexity"],
                tier=row["tier"],
                attempt_count=row["attempt_count"],
                order_index=row["order_index"],
                parent_task_id=row["parent_task_id"],
            )
            for row in rows
        ]
