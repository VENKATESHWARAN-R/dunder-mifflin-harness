"""Shared fixtures for state-layer tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest_asyncio

from jac.state import StateStore, open_state_store


@pytest_asyncio.fixture
async def state(tmp_path: Path) -> AsyncIterator[StateStore]:
    """Open a fresh `StateStore` against an empty tmp-path SQLite file."""
    store = await open_state_store(tmp_path / "state.db")
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def run_id(state: StateStore) -> str:
    """A persisted `runs` row, so foreign-key constraints pass."""
    rid = uuid4().hex
    await state.runs.create(rid, prompt="test run")
    return rid
