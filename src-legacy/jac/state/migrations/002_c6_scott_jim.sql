-- C6 migration. Mirrors docs/contracts/STATE_SCHEMA.md (schema v1.3).
-- Extends agent_configs; rebuilds attempts for parent_attempt_id, role, nullable task_id.

ALTER TABLE agent_configs ADD COLUMN persona TEXT;
ALTER TABLE agent_configs ADD COLUMN display_name TEXT;
ALTER TABLE agent_configs ADD COLUMN is_minion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_configs ADD COLUMN parent_role TEXT;
ALTER TABLE agent_configs ADD COLUMN depth INTEGER NOT NULL DEFAULT 0;

PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;

CREATE TABLE attempts_new (
    attempt_id          TEXT PRIMARY KEY,
    task_id             TEXT REFERENCES tasks(task_id),
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    parent_attempt_id   TEXT REFERENCES attempts_new(attempt_id),
    call_type           TEXT NOT NULL DEFAULT 'agent',
    role                TEXT NOT NULL DEFAULT 'builder',
    model               TEXT NOT NULL,
    tier                TEXT NOT NULL,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    cost                REAL NOT NULL DEFAULT 0.0,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    eval_score          REAL,
    eval_passed         INTEGER,
    eval_feedback       TEXT,
    status              TEXT NOT NULL DEFAULT 'running',
    created_at          TEXT NOT NULL
);

INSERT INTO attempts_new (
    attempt_id, task_id, run_id, parent_attempt_id, call_type, role,
    model, tier, tokens_in, tokens_out, cost, duration_ms,
    eval_score, eval_passed, eval_feedback, status, created_at
)
SELECT
    attempt_id,
    task_id,
    run_id,
    NULL,
    call_type,
    'builder',
    model,
    tier,
    tokens_in,
    tokens_out,
    cost,
    duration_ms,
    eval_score,
    eval_passed,
    eval_feedback,
    status,
    created_at
FROM attempts;

DROP TABLE attempts;
ALTER TABLE attempts_new RENAME TO attempts;

COMMIT;
PRAGMA foreign_keys=ON;
