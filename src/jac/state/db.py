"""SQLite connection + cold-start migration runner.

`StateStore` owns one `aiosqlite` connection plus the typed repositories.
`open_state_store(path)` creates the parent directory, opens the connection
with the PRAGMAs we want, applies pending migrations, and returns the store.

The migration runner in M1 is intentionally tiny: if `schema_meta` is
missing (or empty), apply every `NNN_*.sql` shipped in `state.migrations` in
filename order. Each migration writes its own `schema_meta.version` value.
A real multi-migration ledger lands when M2 ships its first additive
migration — not before.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import aiosqlite

from jac.state.agent_configs import AgentConfigsRepo
from jac.state.attempts import AttemptsRepo
from jac.state.mcp_servers import McpServersRepo
from jac.state.messages import MessagesRepo
from jac.state.run_mcp_servers import RunMcpServersRepo
from jac.state.run_skills import RunSkillsRepo
from jac.state.runs import RunsRepo
from jac.state.skills import SkillsRepo
from jac.state.tasks import TasksRepo

_MIGRATION_PATTERN = re.compile(r"^(\d{3})_.+\.sql$")
_MIGRATIONS_PACKAGE = "jac.state.migrations"


class StateStore:
    """One SQLite connection + typed repositories sharing it."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.runs = RunsRepo(connection)
        self.messages = MessagesRepo(connection)
        self.attempts = AttemptsRepo(connection)
        self.agent_configs = AgentConfigsRepo(connection)
        self.tasks = TasksRepo(connection)
        self.skills = SkillsRepo(connection)
        self.mcp_servers = McpServersRepo(connection)
        self.run_mcp_servers = RunMcpServersRepo(connection)
        self.run_skills = RunSkillsRepo(connection)

    @property
    def connection(self) -> aiosqlite.Connection:
        return self._connection

    async def close(self) -> None:
        await self._connection.close()


async def open_state_store(db_path: Path) -> StateStore:
    """Open (creating + migrating) the SQLite file at `db_path`."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys = ON")
    await connection.execute("PRAGMA journal_mode = WAL")
    await _apply_pending_migrations(connection)
    return StateStore(connection)


async def _apply_pending_migrations(connection: aiosqlite.Connection) -> None:
    if await _schema_initialised(connection):
        return
    for _ordinal, filename in _list_migrations():
        sql = _read_migration(filename)
        await connection.executescript(sql)
        await connection.commit()


async def _schema_initialised(connection: aiosqlite.Connection) -> bool:
    cursor = await connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        return False
    cursor = await connection.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    return row is not None


def _list_migrations() -> list[tuple[int, str]]:
    package = files(_MIGRATIONS_PACKAGE)
    found: list[tuple[int, str]] = []
    for entry in package.iterdir():
        match = _MIGRATION_PATTERN.match(entry.name)
        if match is None:
            continue
        found.append((int(match.group(1)), entry.name))
    found.sort(key=lambda item: item[0])
    return found


def _read_migration(filename: str) -> str:
    return files(_MIGRATIONS_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
