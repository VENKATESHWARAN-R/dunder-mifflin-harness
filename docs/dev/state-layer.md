> **Status:** Reference · **Last revised:** 2026-05-04 · **Type:** developer documentation

# State Layer

The state layer is `src/jac/state/`. It provides async SQLite access via `aiosqlite` and a repository per table group. Nothing outside this package touches `aiosqlite` directly.

See also: [`docs/contracts/STATE_SCHEMA.md`](../contracts/STATE_SCHEMA.md) — the authoritative schema. This doc describes the implementation; the contract doc defines what the schema must be.

## Overview

- **Database engine:** SQLite via `aiosqlite`
- **File location:**
  - Project workspace: `<repo>/.agents/state.db`
  - Global workspace: `~/.jac/runs/<cwd-hash>/state.db`
- **One instance per session:** `StateStore` is opened once by `ChatApp.open()` and passed into the coordinator and seeder. There is no connection pool.
- **Auto-migration:** `open_state_store(path)` runs any pending migrations on open.

## StateStore class (`src/jac/state/db.py`)

`StateStore` wraps a single `aiosqlite.Connection` and instantiates all repos as attributes.

```python
state = await open_state_store(path)

state.runs          # RunsRepo
state.messages      # MessagesRepo
state.skills        # SkillsRepo
state.mcp_servers   # McpServersRepo
state.agent_configs # AgentConfigsRepo
state.run_mcp_servers  # RunMcpServersRepo
state.run_skills    # RunSkillsRepo

await state.close()
```

All repos share the same connection object. Writes are wrapped in transactions by the repo methods.

## Tables and activation state

All tables are **created** by migration `001_initial.sql`. Not all are populated yet — activation aligns with the component that owns the table.

| Table | Status | Activated by | Description |
|---|---|---|---|
| `schema_meta` | Active (C1) | C1 | Schema version tracking. Current version: `1.0` |
| `runs` | Active (C1) | C1 | One row per run. Fields: `run_id`, `status`, `prompt`, `created_at`, `completed_at` |
| `messages` | Active (C1) | C1 | Message history. Fields: `id`, `run_id`, `role`, `content`, `created_at` |
| `skills` | Active (C2) | C2 | Skills seeded from disk. Fields: `skill_id`, `name`, `description`, `domain`, `content`, `source_scope`, `source_path`, `is_enabled`, `created_at`, `updated_at` |
| `mcp_servers` | Active (C2) | C2 | MCP server configs seeded from disk. Fields: `mcp_server_id`, `name`, `description`, `transport`, `config`, `source_scope`, `source_path`, `is_enabled`, `created_at`, `updated_at` |
| `agent_configs` | Active (C5) | C5 | Per-run agent configuration. Fields: `config_id`, `run_id`, `role`, `model_tier`, `model_override`, `system_prompt`, `allowed_tools` (JSON array), `max_context_tokens`, `created_at`, `updated_at` |
| `run_mcp_servers` | Active (C5) | C5 | Which MCP servers are enabled for a run and role. Fields: `id`, `run_id`, `mcp_server_id`, `agent_role` (NULL = all roles), `enabled`, `toggled_at` |
| `run_skills` | Active (C5) | C5 | Which skills are enabled for a run and role. Fields: `id`, `run_id`, `skill_id`, `agent_role` (NULL = all roles), `enabled`, `toggled_at` |
| `tasks` | Inactive | C6+ | Task planning graph. Created but not populated. |
| `attempts` | Active (C6) | C7 for usage columns | Per-turn / per-delegation rows (manager + builder); `parent_attempt_id` for Jim under Scott. Token, cost, and duration columns populated at **C7**. |
| `agent_messages` | Inactive | C15 | Agent-to-agent message queue. Created but not populated. |
| `context_chunks` | Inactive | C6+ | Scoped context per agent role. Created but not populated. |

Do not write to inactive tables from outside the component that owns them.

## Repository pattern

Each repo takes an `aiosqlite.Connection` and exposes typed async CRUD methods. Row types are `@dataclass(frozen=True, slots=True)` — immutable value objects returned from queries.

Example (`AgentConfigsRepo`):

```python
# Create
config = await state.agent_configs.create(
    run_id="my-run",
    role="chat",
    model_tier="worker",
    system_prompt="You are a helpful assistant.",
    allowed_tools=["filesystem:read"],
)

# Get (returns None if not found)
config = await state.agent_configs.get_by_run_and_role("my-run", "chat")

# Update (returns updated row)
config = await state.agent_configs.update(
    config.config_id,
    model_override="anthropic:claude-sonnet-4-6",
)
```

The `allowed_tools` column is stored as a JSON array string in SQLite and decoded to `list[str]` by the repo.

## Migration pattern

Migrations are numbered SQL files in `src/jac/state/migrations/`:

```
src/jac/state/migrations/
  001_initial.sql     # shipped with C1 — creates all tables
```

Rules:
- Files are run in filename order.
- `open_state_store` checks `schema_meta` and runs any migration whose sequence number is higher than the recorded version.
- **Never edit a shipped migration.** If you need a schema change, add a new numbered file (e.g. `002_add_column.sql`).
- Migrations are append-only: `ALTER TABLE`, `CREATE INDEX`, `CREATE TABLE IF NOT EXISTS`. No destructive DDL.

To add a migration alongside a code change: create the `.sql` file, bump the version string in `schema_meta` in the migration, and update `docs/contracts/STATE_SCHEMA.md` with `Last revised`.

## Seeder (`src/jac/state/seeder.py`)

`seed_workspace(workspace, state)` is called at app boot (by `ChatApp.open()`). It:

1. Walks `~/.jac/skills/`, then `.agents/skills/` — reads `.md` files, upserts into `skills`
2. Walks `~/.jac/mcp/`, then `.agents/mcp/` — reads `.json` files, upserts into `mcp_servers`
3. Sets `source_scope` to `"global"` or `"project"` accordingly
4. Records `source_path` as the absolute path of the originating file

The seeder does **not** populate `run_skills` or `run_mcp_servers`. Those are wired per-run by `config_loader` (C5) based on which skills/MCP servers are enabled at the time the agent is constructed.
