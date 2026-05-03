"""Repository for the `attempts` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class AttemptRow:
    attempt_id: str
    task_id: str | None
    run_id: str
    parent_attempt_id: str | None
    call_type: str
    role: str
    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    cost: float
    duration_ms: int
    eval_score: float | None
    eval_passed: int | None
    eval_feedback: str | None
    status: str
    created_at: str


class AttemptsRepo:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        run_id: str,
        role: str,
        model: str,
        tier: str,
        task_id: str | None = None,
        parent_attempt_id: str | None = None,
        call_type: str = "agent",
    ) -> AttemptRow:
        now = _now()
        attempt_id = uuid4().hex
        await self._connection.execute(
            """
            INSERT INTO attempts
                (attempt_id, task_id, run_id, parent_attempt_id,
                 call_type, role, model, tier, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (
                attempt_id,
                task_id,
                run_id,
                parent_attempt_id,
                call_type,
                role,
                model,
                tier,
                now,
            ),
        )
        await self._connection.commit()
        return AttemptRow(
            attempt_id=attempt_id,
            task_id=task_id,
            run_id=run_id,
            parent_attempt_id=parent_attempt_id,
            call_type=call_type,
            role=role,
            model=model,
            tier=tier,
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            duration_ms=0,
            eval_score=None,
            eval_passed=None,
            eval_feedback=None,
            status="running",
            created_at=now,
        )

    async def update_status(self, attempt_id: str, status: str) -> None:
        await self._connection.execute(
            "UPDATE attempts SET status = ? WHERE attempt_id = ?",
            (status, attempt_id),
        )
        await self._connection.commit()
