"""Tests for AgentConfigsRepo."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_create_and_get(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            created = await store.agent_configs.create(
                run_id="run-1",
                role="chat",
                model_tier="worker",
                model_override=None,
                system_prompt="test prompt",
                allowed_tools=["filesystem"],
            )
            fetched = await store.agent_configs.get_by_run_and_role("run-1", "chat")
            return created, fetched
        finally:
            await store.close()

    created, fetched = _run(scenario())
    assert fetched is not None
    assert fetched.config_id == created.config_id
    assert fetched.role == "chat"
    assert fetched.model_tier == "worker"
    assert fetched.system_prompt == "test prompt"
    assert json.loads(fetched.allowed_tools) == ["filesystem"]


def test_get_by_run_and_role_missing(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            return await store.agent_configs.get_by_run_and_role("run-x", "chat")
        finally:
            await store.close()

    assert _run(scenario()) is None


def test_update(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            created = await store.agent_configs.create(
                run_id="run-1",
                role="chat",
                model_tier="worker",
                model_override=None,
                system_prompt="old",
                allowed_tools=[],
            )
            await store.agent_configs.update(
                created.config_id,
                model_tier="architect",
                system_prompt="new",
                allowed_tools=["shell"],
            )
            return await store.agent_configs.get_by_run_and_role("run-1", "chat")
        finally:
            await store.close()

    updated = _run(scenario())
    assert updated is not None
    assert updated.model_tier == "architect"
    assert updated.system_prompt == "new"
    assert json.loads(updated.allowed_tools) == ["shell"]


def test_unique_run_role_constraint(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            await store.agent_configs.create(
                run_id="run-1",
                role="chat",
                model_tier="worker",
                model_override=None,
                system_prompt="first",
                allowed_tools=[],
            )
            # Duplicate (run_id, role) should raise IntegrityError
            await store.agent_configs.create(
                run_id="run-1",
                role="chat",
                model_tier="worker",
                model_override=None,
                system_prompt="second",
                allowed_tools=[],
            )
        finally:
            await store.close()

    with pytest.raises(sqlite3.IntegrityError):
        _run(scenario())
