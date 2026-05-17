# State Schema Contract

> **Status:** Locked · **Last revised:** 2026-05-08 · **Type:** contract
>
> _2026-05-08: Phase-0 reset. Reserved-but-unused tables (`agent_instances`, `agent_teams`, `agent_messages`, `context_store`) dropped. Multi-agent columns on `attempts` and `agent_configs` removed for the M1 single-agent rebuild. `agent_configs.system_prompt` is nullable; canonical persona prompt source is YAML per [`SUBSTRATE.md`](SUBSTRATE.md). Tables return when [`ROADMAP.md`](../ROADMAP.md) Phase-2 evidence triggers them. Source-of-truth brainstorm: [`lab/brainstorm/2026-05-08-jac-reset-from-scratch.md`](../../lab/brainstorm/2026-05-08-jac-reset-from-scratch.md)._

**Schema version:** 1.5
**Storage:** SQLite (single file, local-first, crash-safe)

This document is the authoritative contract for the persistent state store.
Any schema change must be reflected here first — treat it as a migration spec, not an auto-generated artifact.
Future code should derive from this document, not the other way around.

---

## Design Principles

- Every table has a TEXT primary key (`<entity>_id`) using UUIDs, not auto-increment integers.
- Timestamps are stored as ISO-8601 TEXT (`2026-04-30T14:00:00Z`). No DATETIME type.
- JSON payloads are stored as TEXT and validated in application code, not the DB.
- Nullable columns are explicit (`-- nullable`). Non-null columns default to being required.
- Schema changes go through a migration file, never ad-hoc `ALTER TABLE`.
- Cost values are REAL (float). Token counts are INTEGER.
- **Per [`SUBSTRATE.md`](SUBSTRATE.md): SQLite holds per-run state and registries that mirror disk-backed config. Persona prompts live in YAML; user settings in JSON; vendor specs in TOML; skills/instructions in Markdown. The DB is not the canonical source for any value that has a config-file home.**

## Three-tier model resolution

For every agent build, the model used is resolved by walking three substrates — first non-null wins:

1. **Per-run override** — `agent_configs.model_override` or `agent_configs.tier` for `(run_id, role)`. Set by `/model`, `/tier`, escalation, hot-reload.
2. **User preference** — `~/.jac/settings.json` or `<repo>/.agents/settings.json` (`tiers.<tier_name>` map per profile, with `JAC_PROFILE_<SLUG>_*` env overlay).
3. **Shipped default** — `src/jac/data/model_specs.toml` (`tier_defaults_for(provider)`).

This is enforced in `agents/base.py` (the single agent factory site).

---

## Tables

### `runs`

One row per harness invocation.

```sql
CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    workflow_mode   TEXT NOT NULL DEFAULT 'feature_by_feature',
    status          TEXT NOT NULL DEFAULT 'pending',
    -- pending | running | done | failed | cancelled
    total_cost      REAL NOT NULL DEFAULT 0.0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

---

### `tasks`

One row per task tracked within a run. **Active in M1** — Scott maintains the list via `add_task` / `update_task` / `complete_task` / `list_tasks` tools. The list is injected as a system reminder every turn so it survives compaction and resume — this is what makes JAC a long-running harness rather than a chatbot.

```sql
CREATE TABLE tasks (
    task_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    title        TEXT NOT NULL,
    description  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    -- pending | in_progress | completed | failed | cancelled
    order_index  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
```

**Slimmed for M1** — `acceptance_criteria`, `complexity`, `tier`, `attempt_count`, `parent_task_id` removed from the M1 migration. They return as Phase-2 evidence triggers them:

| Column | Returns when |
|---|---|
| `acceptance_criteria` | Pam (planner) ships and writes structured criteria per task |
| `tier` | Per-task tier routing is exercised (typically with multi-role) |
| `complexity` | Task complexity classification influences routing |
| `attempt_count` | Retry / escalation counters are wired |
| `parent_task_id` | Sub-workflow nesting lands |

---

### `attempts`

One row per agent run within a session. Written when a turn starts; updated when usage and (later) eval results are known. `task_id` is nullable — chat-style turns that don't progress a task still record an attempt for cost/usage accounting.

```sql
CREATE TABLE attempts (
    attempt_id    TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    task_id       TEXT REFERENCES tasks(task_id),  -- nullable; null = chat turn or session-level
    call_type     TEXT NOT NULL DEFAULT 'agent',
    -- agent | direct_llm
    -- direct_llm = single model call without full agent loop (e.g., a future Scout summariser)
    role          TEXT NOT NULL DEFAULT 'manager',
    -- which persona made this attempt; M1 has 'manager' only (Scott).
    -- M2+ adds 'planner' | 'builder' | 'evaluator' as personas arrive.
    model         TEXT NOT NULL,   -- exact model id, e.g. anthropic:claude-sonnet-4-6
    tier          TEXT NOT NULL,   -- scout | worker | architect
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    requests      INTEGER NOT NULL DEFAULT 0,
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    cost          REAL NOT NULL DEFAULT 0.0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    eval_score    REAL,            -- nullable until evaluator runs, 0.0–1.0
    eval_passed   INTEGER,         -- nullable until evaluator runs, 0 or 1
    eval_feedback TEXT,            -- nullable
    status        TEXT NOT NULL DEFAULT 'running',
    -- running | passed | failed
    created_at    TEXT NOT NULL
);
```

**Slimmed for M1** — `parent_attempt_id`, `is_minion`-related role prefixes, and `escalated` status removed. They return when multi-agent delegation lands:

| Column / variant | Returns when |
|---|---|
| `parent_attempt_id` | Multi-agent delegation lands (M2+); Jim's attempt rows reference Scott's. |
| `call_type='minion'` | Phase-2 if `spawn_minion` is reintroduced via evidence trigger. |
| `status='escalated'` | HR escalation (Phase 2) wires up. |

**Usage tracking:** `tokens_in`, `tokens_out`, `requests`, `tool_calls`, and `duration_ms` are populated per attempt from Pydantic AI usage. `cost` is reserved for a future pricing component (M5 report computes cost from tokens × pricing-from-`model_specs.toml`).

---

### `agent_configs`

Per-run override layer for agent configs. **The canonical persona shape lives in YAML** under `src/jac/data/personas/<role>.yaml` (with user/project overrides per [`SUBSTRATE.md`](SUBSTRATE.md)). This table holds *only what's per-run-mutable*: tier/model overrides, tool toggles, escalation state.

`system_prompt` is **nullable** — null means "use the persona YAML's `instructions` verbatim." Non-null means "this run overrides the persona prompt" (used by Phase-2 hot-reload, A/B experiments).

```sql
CREATE TABLE agent_configs (
    config_id          TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    role               TEXT NOT NULL,
    -- M1 role: 'manager' (Scott).
    -- M2+ adds 'planner' (Pam) | 'builder' (Jim) | 'evaluator' (Dwight) as personas arrive.
    -- New personas land as YAML files in data/personas/; this column is data, never matched on in code.
    model_tier         TEXT NOT NULL,         -- scout | worker | architect
    model_override     TEXT,                  -- nullable; overrides tier default if set
    system_prompt      TEXT,                  -- nullable; null = use persona YAML; non-null = per-run override
    allowed_tools      TEXT NOT NULL DEFAULT '[]',  -- JSON array of tool/MCP server ids
    max_context_tokens INTEGER NOT NULL DEFAULT 8000,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE(run_id, role)
);
```

**Default tiers per persona (target architecture):**

| Persona | Role | Default tier | Phase-1 milestone |
|---|---|---|---|
| Michael Scott | `manager` | worker | M1 |
| Pam Beesly | `planner` | architect | M2 or M3 (evidence-driven) |
| Jim Halpert | `builder` | worker | M2 or M3 (evidence-driven) |
| Dwight Schrute | `evaluator` | worker | M2 or M3 (evidence-driven) |

The full persona blueprint (instructions, capabilities, model_settings) lives in the YAML file, not here. Code that builds an agent calls the factory at `agents/base.py`, which reads the YAML, applies the three-tier model resolution, and overlays this row's overrides.

**Slimmed for M1** — `persona`, `display_name`, `is_minion`, `parent_role`, `depth` columns removed. The persona name and display name come from the YAML file's `persona` / `display_name` keys (or implicit from filename). Minion-related columns return only via Phase-2 evidence trigger.

**Note on `allowed_tools`:** JSON array of tool ids and MCP server ids. Local tools by name (`filesystem`, `shell`); MCP servers prefixed `mcp:` (`mcp:playwright`). Resolved by the agent factory against `TOOL_REGISTRY` and `mcp_servers`.

---

### `messages`

Full message history for a run. Enables session resume and context reconstruction.

```sql
CREATE TABLE messages (
    message_id  TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    task_id     TEXT REFERENCES tasks(task_id),  -- nullable; null = session-level message
    role        TEXT NOT NULL,  -- user | assistant | tool | system
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
```

---

### `mcp_servers`

Global registry of installed MCP servers available to the harness. Acts as the system-wide
catalogue — what's installed and how to connect to it. Per-run enablement is handled by
`run_mcp_servers`. `is_enabled` is a global kill-switch (disabled = never loaded, regardless
of run config).

```sql
CREATE TABLE mcp_servers (
    mcp_server_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    transport       TEXT NOT NULL,       -- stdio | sse | http
    config          TEXT NOT NULL,       -- JSON: {command, args, env} for stdio, {url} for sse/http
    is_enabled      INTEGER NOT NULL DEFAULT 1,  -- global kill-switch
    source_scope    TEXT NOT NULL DEFAULT 'seeded',
    -- user | project | seeded
    source_path     TEXT,                -- nullable; absolute path for file-seeded rows
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

`source_scope` and `source_path` let the workspace seeding pass reconcile
file-backed rows deterministically. `seeded` is reserved for built-in or
test fixtures that do not come from disk.

---

### `skills`

Global registry of available skills. A skill is a domain knowledge bundle — prompt text injected
into an agent's system prompt to give it domain-specific context (e.g., React patterns, Terraform
best practices, testing conventions). `is_enabled` is a global kill-switch; per-run enablement
is handled by `run_skills`.

```sql
CREATE TABLE skills (
    skill_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    domain      TEXT NOT NULL,       -- frontend | backend | infra | testing | general | etc.
    content     TEXT NOT NULL,       -- injected into agent system prompt when active
    version     TEXT NOT NULL DEFAULT '1.0',
    is_enabled  INTEGER NOT NULL DEFAULT 1,
    source_scope TEXT NOT NULL DEFAULT 'seeded',
    -- user | project | seeded
    source_path  TEXT,               -- nullable; absolute path for file-seeded rows
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
```

`source_scope` and `source_path` follow the same rules as `mcp_servers`.
Project-scope rows override user-scope rows with the same `name` during
seeding; duplicate names within one scope are configuration errors.

---

### `run_mcp_servers`

Controls which MCP servers are active for a specific run, optionally scoped to an agent role.
`enabled` can be toggled mid-run (e.g., via `/disable mcp:playwright`) without deleting the
record — preserving audit history of what was active when.

Note on mid-run disable: toggling `enabled = 0` mid-conversation causes a partial prompt-cache
miss on the next turn because the tool definition block changes. This is intentional and should
be a deliberate user action, not automatic harness behaviour.

```sql
CREATE TABLE run_mcp_servers (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    mcp_server_id   TEXT NOT NULL REFERENCES mcp_servers(mcp_server_id),
    agent_role      TEXT,                        -- nullable = applies to all agents in run
    enabled         INTEGER NOT NULL DEFAULT 1,  -- toggled mid-run; record kept for audit
    toggled_at      TEXT,                        -- nullable; set when enabled is changed
    UNIQUE(run_id, mcp_server_id, agent_role)
);
```

---

### `run_skills`

Controls which skills are active for a specific run, optionally scoped to an agent role.
Same toggle-not-delete pattern as `run_mcp_servers`. Disabling a skill mid-run changes the
system prompt for the next turn — intentional cache tradeoff.

```sql
CREATE TABLE run_skills (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    skill_id    TEXT NOT NULL REFERENCES skills(skill_id),
    agent_role  TEXT,                        -- nullable = applies to all agents in run
    enabled     INTEGER NOT NULL DEFAULT 1,
    toggled_at  TEXT,                        -- nullable
    UNIQUE(run_id, skill_id, agent_role)
);
```

---

## Tables removed in the 2026-05-08 reset (Phase-2 candidates)

The following tables were defined in schema 1.4 but never populated. They are **dropped from the M1 migration** and return only via Phase-2 evidence triggers documented in [`ROADMAP.md`](../ROADMAP.md):

| Removed table | Phase-2 trigger to bring back |
|---|---|
| `agent_instances` | A workflow needs concurrent specialists, not just delegation |
| `agent_teams` | Same trigger |
| `agent_messages` | Same trigger (paired with `agent_instances`) |
| `context_store` | Multi-task / context routing requires shared cross-agent scoped state |

When a trigger fires, re-introduce the table via a new migration **and** add it back to this contract — never re-add a table without ledger evidence motivating it.

---

## Activation Sequence

| Table | First populated by | Notes |
|---|---|---|
| `runs` | M1 | Active from M1; one row per harness invocation. |
| `messages` | M1 | Active from M1; full message history; basis for resume. |
| `attempts` | M1 | Active from M1; one row per agent run within a session, with token usage and (when evaluator lands) eval results. |
| `tasks` | M1 | **Active from M1.** Scott maintains the list via task-CRUD tools; injected as system reminder every turn (context-resilient memory). |
| `agent_configs` | M1 | Per-run override layer; canonical persona shape is YAML at `data/personas/<role>.yaml`. M1 has one row (Scott). |
| `run_mcp_servers` | M1 | Run-start MCP server bindings (kept in M1 even though no MCP servers ship live until Phase 2 evidence trigger). |
| `run_skills` | M1 | Run-start skill bindings (same — registry kept; live skill injection is Phase 2). |
| `mcp_servers` | M1 | Registry seeded from disk by `state/seeder.py`. Live transports remain a Phase-2 evidence-trigger item. |
| `skills` | M1 | Registry seeded from disk by `state/seeder.py`. Dynamic injection is Phase 2. |
| _(removed in 2026-05-08 reset)_ | — | `agent_instances`, `agent_teams`, `agent_messages`, `context_store` — see table above for re-add triggers. |

---

## Migration Policy

1. New column → add with a DEFAULT, document here, write a migration file.
2. Renamed column → write migration, update all query sites, update this doc.
3. New table → define here first, then create the migration.
4. Never DROP a column without an explicit deprecation period documented in this file.
5. Schema version lives in a `schema_meta` table (key/value) and is checked on startup.

```sql
CREATE TABLE schema_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
-- Seed (M1): INSERT INTO schema_meta VALUES ('version', '1.5');
```

The M1 rebuild ships a single migration (`001_m1_initial.sql`) that creates the tables described above at version `1.5`. The historical migrations from C0–C8 (`001_initial.sql` through `003_c7_usage.sql`) are superseded — they describe a schema that's being cut.
