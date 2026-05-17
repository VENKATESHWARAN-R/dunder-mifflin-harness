-- 001_m1_initial.sql — schema v1.5 (M1 reset)
--
-- See docs/contracts/STATE_SCHEMA.md. This single migration replaces the
-- legacy 001/002/003 trio that targeted v1.0/1.1/1.5 incrementally.
--
-- Slim columns dropped from M1 (returns only via Phase-2 evidence triggers):
--   attempts:     parent_attempt_id, escalated status
--   tasks:        acceptance_criteria, complexity, tier, attempt_count, parent_task_id
--   agent_configs: persona, display_name, is_minion, parent_role, depth
--
-- Tables removed in the 2026-05-08 reset (Phase-2 candidates):
--   agent_instances, agent_teams, agent_messages, context_store

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE runs (
    run_id        TEXT PRIMARY KEY,
    prompt        TEXT NOT NULL,
    workflow_mode TEXT NOT NULL DEFAULT 'feature_by_feature',
    status        TEXT NOT NULL DEFAULT 'pending',
    total_cost    REAL NOT NULL DEFAULT 0.0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE tasks (
    task_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    order_index INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE attempts (
    attempt_id    TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    task_id       TEXT REFERENCES tasks(task_id),
    call_type     TEXT NOT NULL DEFAULT 'agent',
    role          TEXT NOT NULL DEFAULT 'manager',
    model         TEXT NOT NULL,
    tier          TEXT NOT NULL,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    requests      INTEGER NOT NULL DEFAULT 0,
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    cost          REAL NOT NULL DEFAULT 0.0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    eval_score    REAL,
    eval_passed   INTEGER,
    eval_feedback TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    created_at    TEXT NOT NULL
);

CREATE TABLE agent_configs (
    config_id          TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    role               TEXT NOT NULL,
    model_tier         TEXT NOT NULL,
    model_override     TEXT,
    system_prompt      TEXT,
    allowed_tools      TEXT NOT NULL DEFAULT '[]',
    max_context_tokens INTEGER NOT NULL DEFAULT 8000,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE(run_id, role)
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    task_id    TEXT REFERENCES tasks(task_id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE mcp_servers (
    mcp_server_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL,
    transport     TEXT NOT NULL,
    config        TEXT NOT NULL,
    is_enabled    INTEGER NOT NULL DEFAULT 1,
    source_scope  TEXT NOT NULL DEFAULT 'seeded',
    source_path   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE skills (
    skill_id     TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT NOT NULL,
    domain       TEXT NOT NULL,
    content      TEXT NOT NULL,
    version      TEXT NOT NULL DEFAULT '1.0',
    is_enabled   INTEGER NOT NULL DEFAULT 1,
    source_scope TEXT NOT NULL DEFAULT 'seeded',
    source_path  TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE run_mcp_servers (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    mcp_server_id TEXT NOT NULL REFERENCES mcp_servers(mcp_server_id),
    agent_role    TEXT,
    enabled       INTEGER NOT NULL DEFAULT 1,
    toggled_at    TEXT,
    UNIQUE(run_id, mcp_server_id, agent_role)
);

CREATE TABLE run_skills (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    skill_id   TEXT NOT NULL REFERENCES skills(skill_id),
    agent_role TEXT,
    enabled    INTEGER NOT NULL DEFAULT 1,
    toggled_at TEXT,
    UNIQUE(run_id, skill_id, agent_role)
);

CREATE INDEX idx_messages_run_created ON messages(run_id, created_at);
CREATE INDEX idx_attempts_run         ON attempts(run_id);
CREATE INDEX idx_tasks_run_order      ON tasks(run_id, order_index);

INSERT INTO schema_meta (key, value) VALUES ('version', '1.5');

COMMIT;
PRAGMA foreign_keys = ON;
