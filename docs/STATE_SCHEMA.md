# State Schema Contract

**Schema version:** 1.1  
**Last updated:** 2026-04-30  
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

One row per planned task within a run. Written by the `plan` node.

```sql
CREATE TABLE tasks (
    task_id             TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    acceptance_criteria TEXT NOT NULL,  -- what evaluate checks against
    status              TEXT NOT NULL DEFAULT 'pending',
    -- pending | in_progress | passed | failed | skipped
    complexity          TEXT NOT NULL DEFAULT 'moderate',
    -- simple | moderate | complex
    tier                TEXT NOT NULL DEFAULT 'worker',
    -- scout | worker | architect
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    order_index         INTEGER NOT NULL,
    parent_task_id      TEXT REFERENCES tasks(task_id)  -- nullable, subtasks v1+
);
```

---

### `attempts`

One row per build attempt on a task. Written by `build` and updated by `evaluate`.

```sql
CREATE TABLE attempts (
    attempt_id      TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    call_type       TEXT NOT NULL DEFAULT 'agent',
    -- agent | direct_llm
    -- direct_llm = single model call without full agent loop (used for simple one-off tasks)
    model           TEXT NOT NULL,   -- exact model id, e.g. anthropic:claude-sonnet-4-6
    tier            TEXT NOT NULL,   -- scout | worker | architect
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    eval_score      REAL,            -- nullable until evaluate runs, 0.0–1.0
    eval_passed     INTEGER,         -- nullable until evaluate runs, 0 or 1
    eval_feedback   TEXT,            -- nullable
    status          TEXT NOT NULL DEFAULT 'running',
    -- running | passed | failed | escalated
    created_at      TEXT NOT NULL
);
```

**Note on `call_type`:** Some nodes make direct model calls without spinning up a full agent loop (e.g., simple routing decisions, quick classification). These still get recorded here for cost and audit purposes, with `call_type = 'direct_llm'`.

---

### `agent_configs`

Stored configuration for each agent role within a run. Loaded at agent instantiation time.
Supports mid-run updates (e.g., tier escalation updates the model/tier fields for that role).

```sql
CREATE TABLE agent_configs (
    config_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    role                TEXT NOT NULL,
    -- planner | builder | evaluator | tester | reviewer (extensible)
    model_tier          TEXT NOT NULL,  -- scout | worker | architect
    model_override      TEXT,           -- nullable; overrides tier default if set
    system_prompt       TEXT NOT NULL,
    allowed_tools       TEXT NOT NULL DEFAULT '[]',  -- JSON array of tool/MCP server ids
    max_context_tokens  INTEGER NOT NULL DEFAULT 8000,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    UNIQUE(run_id, role)
);
```

**Note:** `allowed_tools` stores MCP server IDs and local tool names as a JSON array.
Example: `["filesystem", "shell", "git", "mcp:playwright"]`
MCP server entries are prefixed with `mcp:` to distinguish them from local tools.
The agent instantiation layer resolves these to actual tool/toolset objects at runtime.

---

### `context_store`

Scoped key-value store for inter-agent communication and per-role working context.
Each agent role sees only its own scope. The `shared` scope is readable by all agents in a run.

```sql
CREATE TABLE context_store (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    scope       TEXT NOT NULL,
    -- agent role (planner | builder | evaluator) or 'shared'
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,  -- JSON-serialized
    updated_at  TEXT NOT NULL,
    UNIQUE(run_id, scope, key)
);
```

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
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

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
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
```

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

### `agent_instances` (v1+)

Tracks dynamically spawned agent instances within a run. Used when multiple agents run in
parallel (e.g., a builder agent and a tester agent running concurrently within the same run).
Instances are created from `agent_configs` and can be spawned mid-run.

```sql
CREATE TABLE agent_instances (
    instance_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    config_id       TEXT NOT NULL REFERENCES agent_configs(config_id),
    team_id         TEXT REFERENCES agent_teams(team_id),  -- nullable
    role            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'idle',
    -- idle | working | waiting | done
    current_task_id TEXT REFERENCES tasks(task_id),  -- nullable
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

---

### `agent_teams` (v1+)

Groups agent instances into teams with a coordination strategy. A team is a set of agents
that collaborate on a run — e.g., a developer team (builder + tester) running in parallel
where both agents share a common message queue.

```sql
CREATE TABLE agent_teams (
    team_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    name        TEXT NOT NULL,   -- e.g., 'dev_team', 'review_team'
    strategy    TEXT NOT NULL DEFAULT 'parallel',
    -- parallel | sequential | mixed
    created_at  TEXT NOT NULL
);
```

---

### `agent_messages` (v1+)

Inter-agent coordination queue. Acts as a lightweight shared message board between agent
instances — similar to a Jira board where a tester posts a bug and the builder picks it up
from its queue when it finishes its current task.

```sql
CREATE TABLE agent_messages (
    message_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    from_instance   TEXT NOT NULL,  -- agent_instances.instance_id or 'system'
    to_instance     TEXT NOT NULL,  -- agent_instances.instance_id or 'broadcast'
    message_type    TEXT NOT NULL,
    -- bug_report | task_complete | handoff | question | info | review_request
    payload         TEXT NOT NULL DEFAULT '{}',  -- JSON
    status          TEXT NOT NULL DEFAULT 'queued',
    -- queued | read | processed
    priority        INTEGER NOT NULL DEFAULT 0,  -- higher = more urgent
    created_at      TEXT NOT NULL,
    read_at         TEXT,       -- nullable
    processed_at    TEXT        -- nullable
);
```

**Coordination pattern:**
- Tester agent finds a bug → posts `bug_report` to builder's `to_instance`
- Builder finishes current task → reads its queue → picks up `bug_report` → handles it
- Builder posts `task_complete` back to tester → tester resumes testing that feature
- `broadcast` messages go to all active instances in the run

---

## V0 Scope

For V0, only these tables are active:

| Table | V0 status |
|---|---|
| `runs` | Active |
| `tasks` | Active |
| `attempts` | Active |
| `agent_configs` | Active (single-agent configs) |
| `context_store` | Active |
| `messages` | Active |
| `mcp_servers` | Active (registry only; local tools only in v0) |
| `skills` | Active (registry only; no dynamic injection in v0) |
| `run_mcp_servers` | Active (run-start config; no mid-run toggle UI in v0) |
| `run_skills` | Active (run-start config; no mid-run toggle UI in v0) |
| `agent_instances` | v1+ (schema exists, not used) |
| `agent_teams` | v1+ (schema exists, not used) |
| `agent_messages` | v1+ (schema exists, not used) |

The v1+ tables are defined here to lock down the schema direction but not wired in v0.
The MCP/skills tables are active in v0 for run-start configuration; mid-run toggle via slash
commands is a v1 feature (the DB supports it, the CLI command doesn't exist yet).

---

## Migration Policy

1. New column → add with a DEFAULT, document here, write a migration file.
2. Renamed column → write migration, update all query sites, update this doc.
3. New table → define here first, then create the migration.
4. Never DROP a column in v0/v1 without explicit deprecation period.
5. Schema version lives in a `schema_meta` table (key/value) and is checked on startup.

```sql
CREATE TABLE schema_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
-- Seed: INSERT INTO schema_meta VALUES ('version', '1.1');
```
