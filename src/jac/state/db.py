"""SQLite connection management and migration runner."""

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

_MIGRATION_PATTERN = re.compile(r"^(\d{3})_.+\.sql$")


class StateStore:
    """Owns one SQLite connection plus the typed repositories."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.runs = RunsRepo(connection)
        self.messages = MessagesRepo(connection)
        self.skills = SkillsRepo(connection)
        self.mcp_servers = McpServersRepo(connection)
        self.agent_configs = AgentConfigsRepo(connection)
        self.run_mcp_servers = RunMcpServersRepo(connection)
        self.run_skills = RunSkillsRepo(connection)
        self.attempts = AttemptsRepo(connection)

    @property
    def connection(self) -> aiosqlite.Connection:
        return self._connection

    async def close(self) -> None:
        await self._connection.close()


async def open_state_store(db_path: Path) -> StateStore:
    """Open (creating + migrating) the SQLite file at db_path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys = ON")
    await _apply_pending_migrations(connection)
    return StateStore(connection)


def _list_migrations() -> list[tuple[int, str]]:
    """Return sorted (version, filename) pairs for bundled migrations."""
    package = files("jac.state.migrations")
    found: list[tuple[int, str]] = []
    for entry in package.iterdir():
        match = _MIGRATION_PATTERN.match(entry.name)
        if match is None:
            continue
        found.append((int(match.group(1)), entry.name))
    found.sort(key=lambda item: item[0])
    return found


def _read_migration(filename: str) -> str:
    return files("jac.state.migrations").joinpath(filename).read_text(encoding="utf-8")


async def _current_version(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        return 0
    cursor = await connection.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        return 0
    return _version_to_int(row["value"])


def _version_to_int(value: str) -> int:
    # Schema meta stores versions as "1.0", "1.1", etc. Map to migration ordinal.
    major, _, minor = value.partition(".")
    if minor == "":
        minor = "0"
    return int(major) * 100 + int(minor)


def _migration_to_meta_value(version: int) -> str:
    major, minor = divmod(version, 100)
    if major == 0:
        major = 1
    return f"{major}.{minor}"


async def _apply_pending_migrations(connection: aiosqlite.Connection) -> None:
    current = await _current_version(connection)
    for ordinal, filename in _list_migrations():
        target_version = _version_to_int(_ordinal_to_version_string(ordinal))
        if target_version <= current:
            continue
        sql = _read_migration(filename)
        await connection.executescript(sql)
        await connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'version'",
            (_ordinal_to_version_string(ordinal),),
        )
        await connection.commit()
        current = target_version


def _ordinal_to_version_string(ordinal: int) -> str:
    # Migration 001 -> "1.0", 002 -> "1.1", ... keeps schema_meta human-friendly.
    return f"1.{ordinal - 1}"
