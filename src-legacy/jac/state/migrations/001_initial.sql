-- C1 initial migration. Mirrors docs/contracts/STATE_SCHEMA.md (schema v1.0).
-- Tables ordered so foreign-key references resolve.

CREATE TABLE schema_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    workflow_mode   TEXT NOT NULL DEFAULT 'feature_by_feature',
    status          TEXT NOT NULL DEFAULT 'pending',
    total_cost      REAL NOT NULL DEFAULT 0.0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE tasks (
    task_id             TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    acceptance_criteria TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    complexity          TEXT NOT NULL DEFAULT 'moderate',
    tier                TEXT NOT NULL DEFAULT 'worker',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    order_index         INTEGER NOT NULL,
    parent_task_id      TEXT REFERENCES tasks(task_id)
);

CREATE TABLE attempts (
    attempt_id      TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    call_type       TEXT NOT NULL DEFAULT 'agent',
    model           TEXT NOT NULL,
    tier            TEXT NOT NULL,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    eval_score      REAL,
    eval_passed     INTEGER,
    eval_feedback   TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    created_at      TEXT NOT NULL
);

CREATE TABLE agent_configs (
    config_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    role                TEXT NOT NULL,
    model_tier          TEXT NOT NULL,
    model_override      TEXT,
    system_prompt       TEXT NOT NULL,
    allowed_tools       TEXT NOT NULL DEFAULT '[]',
    max_context_tokens  INTEGER NOT NULL DEFAULT 8000,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    UNIQUE(run_id, role)
);

CREATE TABLE context_store (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(run_id, scope, key)
);

CREATE TABLE messages (
    message_id  TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    task_id     TEXT REFERENCES tasks(task_id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE mcp_servers (
    mcp_server_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    transport       TEXT NOT NULL,
    config          TEXT NOT NULL,
    is_enabled      INTEGER NOT NULL DEFAULT 1,
    source_scope    TEXT NOT NULL DEFAULT 'seeded',
    source_path     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE skills (
    skill_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    domain      TEXT NOT NULL,
    content     TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT '1.0',
    is_enabled  INTEGER NOT NULL DEFAULT 1,
    source_scope TEXT NOT NULL DEFAULT 'seeded',
    source_path  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE run_mcp_servers (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    mcp_server_id   TEXT NOT NULL REFERENCES mcp_servers(mcp_server_id),
    agent_role      TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1,
    toggled_at      TEXT,
    UNIQUE(run_id, mcp_server_id, agent_role)
);

CREATE TABLE run_skills (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    skill_id    TEXT NOT NULL REFERENCES skills(skill_id),
    agent_role  TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    toggled_at  TEXT,
    UNIQUE(run_id, skill_id, agent_role)
);

CREATE TABLE agent_teams (
    team_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    name        TEXT NOT NULL,
    strategy    TEXT NOT NULL DEFAULT 'parallel',
    created_at  TEXT NOT NULL
);

CREATE TABLE agent_instances (
    instance_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    config_id       TEXT NOT NULL REFERENCES agent_configs(config_id),
    team_id         TEXT REFERENCES agent_teams(team_id),
    role            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'idle',
    current_task_id TEXT REFERENCES tasks(task_id),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE agent_messages (
    message_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    from_instance   TEXT NOT NULL,
    to_instance     TEXT NOT NULL,
    message_type    TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'queued',
    priority        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    read_at         TEXT,
    processed_at    TEXT
);

INSERT INTO schema_meta (key, value) VALUES ('version', '1.0');
