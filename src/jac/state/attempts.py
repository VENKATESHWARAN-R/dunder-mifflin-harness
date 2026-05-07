"""Repository for the `attempts` table."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class AttemptNode:
    """One attempt in a run tree."""

    row: AttemptRow
    children: tuple["AttemptNode", ...] = ()


@dataclass(frozen=True, slots=True)
class RunTotals:
    """Aggregated usage for a run."""

    tokens_in: int
    tokens_out: int
    requests: int
    tool_calls: int
    by_role: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    by_tier: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)


def _row_from(row: aiosqlite.Row) -> AttemptRow:
    return AttemptRow(
        attempt_id=row["attempt_id"],
        task_id=row["task_id"],
        run_id=row["run_id"],
        parent_attempt_id=row["parent_attempt_id"],
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
                 call_type, role, model, tier, created_at, status,
                 requests, tool_calls)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 0, 0)
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
            requests=0,
            tool_calls=0,
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

    async def update_usage(
        self,
        attempt_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        requests: int,
        tool_calls: int,
        duration_ms: int,
    ) -> None:
        """Persist token usage onto an attempt row."""
        await self._connection.execute(
            """
            UPDATE attempts SET
                tokens_in = ?,
                tokens_out = ?,
                requests = ?,
                tool_calls = ?,
                duration_ms = ?
            WHERE attempt_id = ?
            """,
            (tokens_in, tokens_out, requests, tool_calls, duration_ms, attempt_id),
        )
        await self._connection.commit()

    async def list_for_run(self, run_id: str) -> list[AttemptRow]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM attempts
            WHERE run_id = ?
            ORDER BY created_at ASC, attempt_id ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_from(r) for r in rows]

    async def list_direct_children(self, parent_attempt_id: str) -> list[AttemptRow]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM attempts
            WHERE parent_attempt_id = ?
            ORDER BY created_at ASC, attempt_id ASC
            """,
            (parent_attempt_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_from(r) for r in rows]

    async def tree_for_run(self, run_id: str) -> list[AttemptNode]:
        """Build parent→children trees for attempts in a run."""
        flat = await self.list_for_run(run_id)
        if not flat:
            return []
        by_id = {r.attempt_id: r for r in flat}
        children_map: dict[str, list[AttemptRow]] = {}
        for r in flat:
            pid = r.parent_attempt_id
            if pid is None or pid not in by_id:
                continue
            children_map.setdefault(pid, []).append(r)

        def build_node(row: AttemptRow) -> AttemptNode:
            kids = children_map.get(row.attempt_id, [])
            built = tuple(build_node(c) for c in kids)
            return AttemptNode(row=row, children=built)

        roots: list[AttemptRow] = []
        for r in flat:
            if r.parent_attempt_id is None or r.parent_attempt_id not in by_id:
                roots.append(r)
        return [build_node(r) for r in roots]

    async def totals_for_run(self, run_id: str) -> RunTotals:
        """Aggregate usage; group by role and tier."""
        rows = await self.list_for_run(run_id)
        tin = tout = req = tc = 0
        by_role: dict[str, list[int]] = {}
        by_tier: dict[str, list[int]] = {}

        def bump(
            bucket: dict[str, list[int]],
            key: str,
            a: int,
            b: int,
            c: int,
            d: int,
        ) -> None:
            cur = bucket.setdefault(key, [0, 0, 0, 0])
            cur[0] += a
            cur[1] += b
            cur[2] += c
            cur[3] += d

        for r in rows:
            tin += r.tokens_in
            tout += r.tokens_out
            req += r.requests
            tc += r.tool_calls
            bump(by_role, r.role, r.tokens_in, r.tokens_out, r.requests, r.tool_calls)
            bump(by_tier, r.tier, r.tokens_in, r.tokens_out, r.requests, r.tool_calls)

        return RunTotals(
            tokens_in=tin,
            tokens_out=tout,
            requests=req,
            tool_calls=tc,
            by_role={k: (v[0], v[1], v[2], v[3]) for k, v in by_role.items()},
            by_tier={k: (v[0], v[1], v[2], v[3]) for k, v in by_tier.items()},
        )
