"""Repository for the `attempts` table (schema v1.5, slim shape).

One row per agent run within a session. Written when a turn starts; updated
when usage and (later) eval results land. `task_id` is nullable — chat-style
turns that don't progress a task still record an attempt for cost/usage
accounting.

Slim columns dropped vs. legacy: `parent_attempt_id` (no delegation in M1),
`status='escalated'` (no HR escalation in M1). They return only via Phase-2
evidence triggers per `docs/contracts/STATE_SCHEMA.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class AttemptRow:
    attempt_id: str
    run_id: str
    task_id: str | None
    call_type: str
    role: str
    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    requests: int
    tool_calls: int
    cost: float
    duration_ms: int
    eval_score: float | None
    eval_passed: int | None
    eval_feedback: str | None
    status: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RunTotals:
    attempts: int
    tokens_in: int
    tokens_out: int
    requests: int
    tool_calls: int
    cost: float


class AttemptsRepo:
    """CRUD for `attempts`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def create(
        self,
        *,
        attempt_id: str,
        run_id: str,
        model: str,
        tier: str,
        task_id: str | None = None,
        call_type: str = "agent",
        role: str = "manager",
        status: str = "running",
    ) -> AttemptRow:
        now = _now()
        await self._connection.execute(
            """
            INSERT INTO attempts (
                attempt_id, run_id, task_id, call_type, role,
                model, tier, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (attempt_id, run_id, task_id, call_type, role, model, tier, status, now),
        )
        await self._connection.commit()
        return AttemptRow(
            attempt_id=attempt_id,
            run_id=run_id,
            task_id=task_id,
            call_type=call_type,
            role=role,
            model=model,
            tier=tier,
            tokens_in=0,
            tokens_out=0,
            requests=0,
            tool_calls=0,
            cost=0.0,
            duration_ms=0,
            eval_score=None,
            eval_passed=None,
            eval_feedback=None,
            status=status,
            created_at=now,
        )

    async def update_status(self, attempt_id: str, status: str) -> None:
        await self._connection.execute(
            "UPDATE attempts SET status = ? WHERE attempt_id = ?",
            (status, attempt_id),
        )
        await self._connection.commit()

    async def update_usage(
        self,
        attempt_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        requests: int,
        tool_calls: int,
        cost: float = 0.0,
        duration_ms: int = 0,
    ) -> None:
        await self._connection.execute(
            """
            UPDATE attempts
            SET tokens_in = ?, tokens_out = ?, requests = ?,
                tool_calls = ?, cost = ?, duration_ms = ?
            WHERE attempt_id = ?
            """,
            (
                tokens_in,
                tokens_out,
                requests,
                tool_calls,
                cost,
                duration_ms,
                attempt_id,
            ),
        )
        await self._connection.commit()

    async def update_eval(
        self,
        attempt_id: str,
        *,
        score: float,
        passed: bool,
        feedback: str | None = None,
    ) -> None:
        await self._connection.execute(
            """
            UPDATE attempts
            SET eval_score = ?, eval_passed = ?, eval_feedback = ?
            WHERE attempt_id = ?
            """,
            (score, 1 if passed else 0, feedback, attempt_id),
        )
        await self._connection.commit()

    async def get(self, attempt_id: str) -> AttemptRow | None:
        cursor = await self._connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _attempt_from_row(row) if row is not None else None

    async def list_for_run(self, run_id: str) -> list[AttemptRow]:
        cursor = await self._connection.execute(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY created_at ASC, attempt_id ASC",
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_attempt_from_row(r) for r in rows]

    async def totals_for_run(self, run_id: str) -> RunTotals:
        cursor = await self._connection.execute(
            """
            SELECT
                COUNT(*) AS attempts,
                COALESCE(SUM(tokens_in), 0) AS tokens_in,
                COALESCE(SUM(tokens_out), 0) AS tokens_out,
                COALESCE(SUM(requests), 0) AS requests,
                COALESCE(SUM(tool_calls), 0) AS tool_calls,
                COALESCE(SUM(cost), 0.0) AS cost
            FROM attempts
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return RunTotals(0, 0, 0, 0, 0, 0.0)
        return RunTotals(
            attempts=int(row["attempts"]),
            tokens_in=int(row["tokens_in"]),
            tokens_out=int(row["tokens_out"]),
            requests=int(row["requests"]),
            tool_calls=int(row["tool_calls"]),
            cost=float(row["cost"]),
        )


def _attempt_from_row(row: aiosqlite.Row) -> AttemptRow:
    return AttemptRow(
        attempt_id=row["attempt_id"],
        run_id=row["run_id"],
        task_id=row["task_id"],
        call_type=row["call_type"],
        role=row["role"],
        model=row["model"],
        tier=row["tier"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        requests=row["requests"],
        tool_calls=row["tool_calls"],
        cost=row["cost"],
        duration_ms=row["duration_ms"],
        eval_score=row["eval_score"],
        eval_passed=row["eval_passed"],
        eval_feedback=row["eval_feedback"],
        status=row["status"],
        created_at=row["created_at"],
    )
