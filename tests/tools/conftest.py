"""Shared fixtures for tools-layer tests.

Task-CRUD tools take `RunContext[ScottDeps]`. The `task_ctx` fixture
builds a real `RunContext` against an in-memory SQLite — no mocks — so
the tools exercise the same code path the agent uses at run time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest_asyncio
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from jac.state import StateStore, open_state_store
from jac.tools.types import ScottDeps


@pytest_asyncio.fixture
async def state(tmp_path: Path) -> AsyncIterator[StateStore]:
    """Fresh SQLite state store on tmp_path — same shape as tests/state."""
    store = await open_state_store(tmp_path / "state.db")
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def scott_deps(state: StateStore) -> ScottDeps:
    """A `ScottDeps` bound to a freshly-persisted run row."""
    rid = uuid4().hex
    await state.runs.create(rid, prompt="test run")
    return ScottDeps(run_id=rid, tasks_repo=state.tasks)


@pytest_asyncio.fixture
async def task_ctx(scott_deps: ScottDeps) -> RunContext[ScottDeps]:
    """A real `RunContext` wired to the test state store."""
    return RunContext(deps=scott_deps, model=TestModel(), usage=RunUsage())
