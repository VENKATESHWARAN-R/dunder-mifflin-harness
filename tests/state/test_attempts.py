"""Attempts repo CRUD + usage accounting."""

from __future__ import annotations

from uuid import uuid4

from jac.state import StateStore


async def test_create_attempt_defaults(state: StateStore, run_id: str) -> None:
    attempt_id = uuid4().hex
    row = await state.attempts.create(
        attempt_id=attempt_id,
        run_id=run_id,
        model="anthropic:claude-sonnet-4-6",
        tier="worker",
    )
    assert row.call_type == "agent"
    assert row.role == "manager"
    assert row.status == "running"
    assert row.tokens_in == 0
    assert row.tokens_out == 0
    assert row.eval_score is None


async def test_update_usage_and_status(state: StateStore, run_id: str) -> None:
    attempt_id = uuid4().hex
    await state.attempts.create(
        attempt_id=attempt_id,
        run_id=run_id,
        model="anthropic:claude-haiku-4-5",
        tier="scout",
    )
    await state.attempts.update_usage(
        attempt_id,
        tokens_in=120,
        tokens_out=80,
        requests=1,
        tool_calls=2,
        cost=0.005,
        duration_ms=420,
    )
    await state.attempts.update_status(attempt_id, "passed")
    row = await state.attempts.get(attempt_id)
    assert row is not None
    assert row.tokens_in == 120
    assert row.tokens_out == 80
    assert row.tool_calls == 2
    assert row.duration_ms == 420
    assert row.status == "passed"


async def test_update_eval(state: StateStore, run_id: str) -> None:
    attempt_id = uuid4().hex
    await state.attempts.create(
        attempt_id=attempt_id,
        run_id=run_id,
        model="anthropic:claude-sonnet-4-6",
        tier="worker",
    )
    await state.attempts.update_eval(
        attempt_id, score=0.9, passed=True, feedback="looked good"
    )
    row = await state.attempts.get(attempt_id)
    assert row is not None
    assert row.eval_score == 0.9
    assert row.eval_passed == 1
    assert row.eval_feedback == "looked good"


async def test_totals_for_run(state: StateStore, run_id: str) -> None:
    for tokens in (10, 20, 30):
        attempt_id = uuid4().hex
        await state.attempts.create(
            attempt_id=attempt_id,
            run_id=run_id,
            model="anthropic:claude-sonnet-4-6",
            tier="worker",
        )
        await state.attempts.update_usage(
            attempt_id,
            tokens_in=tokens,
            tokens_out=tokens * 2,
            requests=1,
            tool_calls=0,
            cost=tokens * 0.001,
        )
    totals = await state.attempts.totals_for_run(run_id)
    assert totals.attempts == 3
    assert totals.tokens_in == 60
    assert totals.tokens_out == 120
    assert totals.requests == 3
    assert totals.cost == 60 * 0.001
