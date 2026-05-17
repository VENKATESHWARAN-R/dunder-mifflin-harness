"""Cold-start migration runner + idempotent re-open."""

from __future__ import annotations

from pathlib import Path

from jac.state import open_state_store


_EXPECTED_TABLES = {
    "schema_meta",
    "runs",
    "tasks",
    "attempts",
    "agent_configs",
    "messages",
    "mcp_servers",
    "skills",
    "run_mcp_servers",
    "run_skills",
}


async def test_cold_start_creates_all_v15_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = await open_state_store(db_path)
    cursor = await store.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    rows = await cursor.fetchall()
    await cursor.close()
    found = {r["name"] for r in rows}
    assert _EXPECTED_TABLES.issubset(found)
    await store.close()


async def test_schema_meta_version_is_1_5(tmp_path: Path) -> None:
    store = await open_state_store(tmp_path / "state.db")
    cursor = await store.connection.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    assert row is not None
    assert row["value"] == "1.5"
    await store.close()


async def test_reopen_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    first = await open_state_store(db_path)
    await first.runs.create("r1", prompt="first")
    await first.close()

    second = await open_state_store(db_path)
    row = await second.runs.get("r1")
    assert row is not None
    assert row.prompt == "first"

    cursor = await second.connection.execute(
        "SELECT COUNT(*) AS n FROM schema_meta WHERE key='version'"
    )
    meta_count = await cursor.fetchone()
    await cursor.close()
    assert meta_count is not None
    assert meta_count["n"] == 1
    await second.close()


async def test_foreign_keys_enforced(tmp_path: Path) -> None:
    """A row referencing a non-existent run must fail."""
    import aiosqlite

    store = await open_state_store(tmp_path / "state.db")
    try:
        try:
            await store.messages.append(
                run_id="nonexistent", role="user", content="hi"
            )
        except aiosqlite.IntegrityError:
            return
        raise AssertionError("expected IntegrityError on FK violation")
    finally:
        await store.close()
