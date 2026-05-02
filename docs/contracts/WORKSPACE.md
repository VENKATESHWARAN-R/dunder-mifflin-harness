# Workspace Layout

> **Status:** Locked · **Last revised:** 2026-05-02 · **Type:** contract

## Purpose

Defines where JAC reads configuration, credentials, instructions, agent
definitions, skills, and MCP catalogues from disk, and how those files become
rows in the state store.

This contract resolves the disk side of two questions left open by other
contracts:

- `STATE_SCHEMA.md` defines the `skills` and `mcp_servers` tables but not how
  rows get populated.
- `MCP_INTEGRATION.md` describes how `agent_configs.system_prompt` is composed
  from skills but not where the base prompt originates.

JAC adopts the **AGENTS.md** open-standard convention for instruction files
and the **`.agents/`** folder convention for project workspace, mirroring how
Claude Code uses `CLAUDE.md` + `.claude/`. User globals stay under `~/.jac/`
to keep that scope tool-specific.

---

## Locations

JAC reads from three scopes, in increasing precedence:

| Scope | Path | Committed? |
|---|---|---|
| User globals | `~/.jac/` | n/a (per-machine) |
| Project workspace | `<repo>/.agents/` (preferred) or `<repo>/AGENTS.md` (top-level alias) | yes |
| Project local | `<repo>/.agents/settings.local.json`, `<repo>/.agents/.env.local`, `<repo>/.agents/state.db`, `<repo>/.agents/logs/` | no (gitignored) |

### User globals — `~/.jac/`

```
~/.jac/
├── settings.json        # default tier, approval mode, telemetry, model overrides
├── .env                 # optional user-global provider credentials (gitignored)
├── AGENTS.md            # user-global instructions injected into prompts
├── agents/<role>.md     # custom agent role definitions (frontmatter + body)
├── skills/<name>.md     # global skills (frontmatter → row, body → content)
├── mcp/<server>.json    # MCP server catalogue → mcp_servers row
└── history/             # prompt_toolkit history (already in use via config_dir)
```

### Project workspace — `<repo>/.agents/`

```
<repo>/.agents/
├── settings.json        # project tier overrides, default workflow mode (committed)
├── AGENTS.md            # project instructions (committed; alternative to top-level)
├── agents/              # project agent role overrides (committed)
├── skills/              # project skills (committed)
├── mcp/                 # project MCP servers (committed)
├── settings.local.json  # per-developer overrides (gitignored)
├── .env.local           # per-project provider credentials (gitignored)
├── state.db             # SQLite — runs, tasks, attempts, messages (gitignored)
└── logs/<run-id>.jsonl  # optional per-run event log (gitignored)
```

### Top-level alias — `<repo>/AGENTS.md`

If `<repo>/AGENTS.md` exists at the project root, it takes precedence over
`<repo>/.agents/AGENTS.md`. This matches the discovery convention used by
other agentic tools and keeps the file visible to humans skimming the repo
root. Projects that prefer a tidy root keep instructions at
`<repo>/.agents/AGENTS.md` instead.

---

## Layering Precedence

When the same setting, instruction, or registry entry exists in multiple
scopes, the higher-precedence value wins:

1. User globals (`~/.jac/`)
2. Project workspace (`<repo>/.agents/`)
3. Project local overrides (`<repo>/.agents/settings.local.json`)
4. Run-start slash commands (`/model`, `/tier`, `/approval`)
5. Mid-run mutations (e.g., HR escalation rewriting `agent_configs.tier`)

For **instructions**: the resolved system prompt is a **concatenation**, not
a replacement. Prompt composition order is:

1. JAC's shipped base harness instructions.
2. Role instructions (`agents/<role>.md` or the built-in default for that role).
3. User-global instructions (`~/.jac/AGENTS.md`).
4. Project instructions (`<repo>/AGENTS.md` or `<repo>/.agents/AGENTS.md`).
5. Lazy-loaded skills, ordered general first and domain-specific last.

The top-level `<repo>/AGENTS.md` alias replaces `<repo>/.agents/AGENTS.md`
when both exist. It does not replace user-global instructions.

For **skills**: global and project skills are both loaded into `skills`
rows. A project skill with the same `name` as a user skill replaces the
user version (unique constraint on `skills.name`). Duplicate names within
the same scope are a configuration error; `jac doctor` reports them and the
seeding pass refuses to choose arbitrarily.

For **MCP servers**: conflict rules mirror skills. `mcp/<server>.json` rows
are keyed by `name`; project MCP definitions override user-global definitions
with the same name. Duplicate names within the same scope are a configuration
error.

For **agent role files**: rows are keyed by `role`. Project role definitions
override user-global role definitions. Built-in role defaults are used only
when no file exists for the role.

For **settings**: object-level merge. Top-level keys present in a
higher-precedence file replace the same key from a lower-precedence file.
No deep-merge — tools that need partial overrides should use distinct keys.

---

## Credential and Env Resolution

Secrets never live in `settings.json` or `settings.local.json`. Provider
credentials come from environment variables loaded in this order, highest
precedence first:

1. The process environment inherited by the `jac` process.
2. Project-local env file: `<repo>/.agents/.env.local`.
3. User-global env file: `~/.jac/.env`.

The env files use standard dotenv syntax and are never committed. `jac` must
be startable from any directory after `jac init --global` creates `~/.jac/.env`;
project `.agents/.env.local` exists only for per-repo overrides.

Current minimum live-call configuration:

```dotenv
JAC_MODEL=gateway/google-vertex:gemini-3.1-flash-lite-preview
PYDANTIC_AI_GATEWAY_API_KEY=...
```

Future providers use the same resolution path. Examples include
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`, LiteLLM gateway keys, and local model settings such as
`OLLAMA_BASE_URL`.

Only operations that make an LLM call require provider credentials. `jac --help`,
`jac init`, `jac doctor`, `jac config`, project discovery, and
non-model shell helpers must work without API keys. When an LLM call cannot
start, the CLI raises a configuration error that names the selected model and
the missing credential candidates.

---

## File Formats

### `AGENTS.md`

Plain markdown. The full file body is treated as instructional context and
prepended to agent system prompts (after role-specific instructions, before
domain skills). No frontmatter required.

### `agents/<role>.md`

Markdown with YAML frontmatter:

```yaml
---
role: planner
tier: architect           # scout | worker | architect
model_override:           # optional, e.g. anthropic:claude-opus-4-7
allowed_tools:            # mirrors agent_configs.allowed_tools (JSON array)
  - filesystem:read
  - shell
max_context_tokens: 16000
---
```

Body is the role's `system_prompt`. At run start, these populate
`agent_configs` rows for the run.

### `skills/<name>.md`

Markdown with YAML frontmatter:

```yaml
---
name: react-conventions
domain: frontend          # frontend | backend | infra | testing | general | etc.
version: 1.0
description: "React component and hooks conventions used in this repo"
---
```

Frontmatter maps to `skills` table columns; body becomes `skills.content`.

### `mcp/<server>.json`

```json
{
  "name": "playwright",
  "description": "Browser automation via Playwright MCP server",
  "transport": "stdio",
  "config": {
    "command": "npx",
    "args": ["@playwright/mcp"],
    "env": {}
  }
}
```

Each file maps 1:1 to a row in `mcp_servers`. The `config` object is
JSON-serialised into the `config` column (matches `STATE_SCHEMA.md`).

### `settings.json` / `settings.local.json`

```json
{
  "default_tier": "worker",
  "default_approval_mode": "interactive",
  "default_workflow_mode": "feature_by_feature",
  "model_overrides": {
    "scout":     "anthropic:claude-haiku-4-5",
    "worker":    "anthropic:claude-sonnet-4-6",
    "architect": "anthropic:claude-opus-4-7"
  },
  "telemetry": { "enabled": false }
}
```

`settings.local.json` uses the same shape; values present there override
`settings.json`. Both files are optional. Secrets (API keys) never live
here — they come from environment variables.

### `.env` / `.env.local`

Dotenv files contain provider credentials and provider-specific endpoints:

```dotenv
JAC_MODEL=gateway/google-vertex:gemini-3.1-flash-lite-preview
PYDANTIC_AI_GATEWAY_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434/v1
```

`~/.jac/.env` is user-global and created by `jac init --global`.
`<repo>/.agents/.env.local` is project-local and overrides user-global values
for that project. Both files are machine-local and must be gitignored.

---

## Bootstrap and Diagnostics

Workspace setup is owned by an `onboarder` module surfaced through CLI
commands. The module is deterministic: it asks questions, writes files, and
validates configuration; it does not call an LLM.

### `jac init --global`

Creates or updates the user-global workspace:

- `~/.jac/settings.json`
- `~/.jac/.env`
- `~/.jac/AGENTS.md` (optional, user can skip)
- `~/.jac/agents/`, `~/.jac/skills/`, `~/.jac/mcp/`
- `~/.jac/history/`

The flow asks for the default provider/model and the required credential for
that provider. For the initial product shape, the default provider is
Pydantic AI's Logfire gateway. Later providers (LiteLLM, OpenAI, Anthropic,
Gemini, OpenRouter, Ollama) plug into the same prompt flow by declaring
which env vars they need.

### `jac init`

Creates or updates the project workspace:

- `<repo>/.agents/settings.json`
- `<repo>/.agents/AGENTS.md` or top-level `<repo>/AGENTS.md`
- `<repo>/.agents/agents/`, `skills/`, `mcp/`
- optional `<repo>/.agents/.env.local`

It also offers to add local-only entries to `.gitignore`:

```gitignore
.agents/.env.local
.agents/settings.local.json
.agents/state.db
.agents/logs/
```

### `jac doctor`

Reports resolved non-secret configuration and validates setup:

- selected model and provider
- which env source satisfied each required credential (without printing values)
- workspace discovery result
- duplicate skill/MCP/agent names within the same scope
- missing `.gitignore` entries for local-only files

`jac doctor` never makes an LLM call. A separate future smoke check may make
an explicit test call after confirmation.

---

## Seeding: File → DB

Files on disk are the source of truth; the DB is the resolved index.

On harness start (or before any run), JAC performs a **lazy upsert** pass:

1. Walk `~/.jac/skills/`, `~/.jac/mcp/`, `~/.jac/agents/` and collect entries.
2. Walk `<repo>/.agents/skills/`, `<repo>/.agents/mcp/`,
   `<repo>/.agents/agents/`.
3. For each entry, upsert into the matching table by unique key (`name`
   for skills/MCP, `(run_id, role)` for agent configs at run-start time).
   Seeded `skills` and `mcp_servers` rows record `source_scope` and
   `source_path` so later boots can reconcile disk state deterministically.
4. Garbage-collect rows whose source file no longer exists *and* which are
   not referenced by an open run. Closed-run history is preserved for audit.

This means:

- Adding a skill = drop a markdown file. No CLI command.
- Removing a skill = delete the file. Closed runs still see the historical row.
- The DB is regenerable from disk + run history.

The seeding pass is implemented as a deterministic node (no LLM call) so
it can run on every harness boot without cost concerns.

---

## Project Discovery

JAC walks upward from the current working directory looking for, in order:

1. `<dir>/.agents/`
2. `<dir>/AGENTS.md`

The first match anchors the project workspace. If neither is found, the
harness operates in user-globals-only mode (no project instructions, DB
defaults to `~/.jac/runs/<cwd-hash>/state.db`).

`<cwd-hash>` is the first 16 hex characters of
`sha256(Path.cwd().resolve().as_posix())`. The directory and DB file are
created lazily when persistence is first needed. In no-project mode, JAC
still loads `~/.jac/settings.json`, `~/.jac/.env`, user-global skills, MCP
definitions, and user-global role files.

This matches the discovery model of Claude Code (`.claude/`), Git
(`.git/`), and most modern dev tools.

---

## Out of Scope

- **Encrypted credentials** in workspace files. API keys come from
  environment variables, dotenv files, or the user's secret manager, never
  from `settings.json`.
- **Per-task workspace overrides.** Run-time mutations are the only way to
  change config mid-run; on-disk files are read at run-start.
- **Secret storage beyond dotenv.** `jac init` can write local dotenv files,
  but encrypted keychains and hosted secret managers remain external.

---

## Downstream Effects

- `MCP_INTEGRATION.md` grows a "File → DB" section pointing here for MCP
  server seeding.
- `STATE_SCHEMA.md` adds `source_scope` and `source_path` to `skills` and
  `mcp_servers` so seeding and garbage collection are auditable.
- `config.py` `config_dir` default already points at `~/.jac/`; project-dir
  resolution (the upward walk for `.agents/`) and layered dotenv loading are
  new code.
- This contract ships as roadmap component **C2**. It depends on **C1**
  (the SQLite state store) since it needs the DB to upsert into.
  Without it, C1's resume command works but no skills/MCPs are visible
  to agents.
