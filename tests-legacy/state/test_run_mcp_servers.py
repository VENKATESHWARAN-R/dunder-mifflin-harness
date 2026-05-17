"""Tests for RunMcpServersRepo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def test_create_and_list_active(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            server = await store.mcp_servers.upsert(
                name="test-server",
                description="d",
                transport="stdio",
                config=json.dumps({"command": "python", "args": ["-m", "srv"]}),
                source_scope="seeded",
                source_path=None,
            )
            await store.run_mcp_servers.create(
                run_id="run-1", mcp_server_id=server.mcp_server_id
            )
            rows = await store.run_mcp_servers.list_active_for_run("run-1")
            return rows
        finally:
            await store.close()

    rows = _run(scenario())
    assert len(rows) == 1
    assert rows[0].run_id == "run-1"


def test_list_active_with_details(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            server = await store.mcp_servers.upsert(
                name="detail-server",
                description="d",
                transport="http",
                config=json.dumps({"url": "http://localhost:8000"}),
                source_scope="seeded",
                source_path=None,
            )
            await store.run_mcp_servers.create(
                run_id="run-1", mcp_server_id=server.mcp_server_id
            )
            rows = await store.run_mcp_servers.list_active_for_run_with_details("run-1")
            return rows
        finally:
            await store.close()

    rows = _run(scenario())
    assert len(rows) == 1
    assert rows[0].name == "detail-server"
    assert rows[0].transport == "http"
    assert json.loads(rows[0].config)["url"] == "http://localhost:8000"


def test_list_active_respects_agent_role(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            server = await store.mcp_servers.upsert(
                name="scoped-server",
                description="d",
                transport="stdio",
                config=json.dumps({"command": "echo"}),
                source_scope="seeded",
                source_path=None,
            )
            # Scoped to planner only
            await store.run_mcp_servers.create(
                run_id="run-1",
                mcp_server_id=server.mcp_server_id,
                agent_role="planner",
            )
            # Query for chat role should not return it
            chat_rows = await store.run_mcp_servers.list_active_for_run(
                "run-1", agent_role="chat"
            )
            # Query with no role filter should return it (NULL applies to all)
            all_rows = await store.run_mcp_servers.list_active_for_run("run-1")
            # Query for planner should return it
            planner_rows = await store.run_mcp_servers.list_active_for_run(
                "run-1", agent_role="planner"
            )
            return chat_rows, all_rows, planner_rows
        finally:
            await store.close()

    chat_rows, all_rows, planner_rows = _run(scenario())
    assert len(chat_rows) == 0
    assert len(all_rows) == 1
    assert len(planner_rows) == 1


def test_toggle(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="run-1", prompt="p")
            server = await store.mcp_servers.upsert(
                name="toggle-server",
                description="d",
                transport="stdio",
                config=json.dumps({"command": "echo"}),
                source_scope="seeded",
                source_path=None,
            )
            row = await store.run_mcp_servers.create(
                run_id="run-1", mcp_server_id=server.mcp_server_id
            )
            assert row.enabled == 1
            updated = await store.run_mcp_servers.toggle(row.id, enabled=0)
            assert updated is not None
            assert updated.enabled == 0
            # After toggle, should not appear in active list
            active = await store.run_mcp_servers.list_active_for_run("run-1")
            return active
        finally:
            await store.close()

    active = _run(scenario())
    assert len(active) == 0
