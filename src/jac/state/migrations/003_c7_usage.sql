-- C7: per-attempt request and tool-call counts for usage tracking.

ALTER TABLE attempts ADD COLUMN requests INTEGER NOT NULL DEFAULT 0;
ALTER TABLE attempts ADD COLUMN tool_calls INTEGER NOT NULL DEFAULT 0;
