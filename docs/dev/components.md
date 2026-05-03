> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** developer documentation

# Components (C0–C5)

Each component is a discrete shipping unit. This document records what each one added, which files it touched, and how it integrates with the rest of the system.

---

## C0 — CLI Bootstrap + Runtime Foundation

**Layer:** cli, runtime, config

**Key files:**

- `src/jac/cli/main.py` — Click command group; `jac [PROMPT]` and `jac chat`
- `src/jac/cli/app.py` — `ChatApp`: composition root for interactive sessions
- `src/jac/cli/parser.py` — `parse_input`: plain/slash/shell/file-ref grammar
- `src/jac/cli/commands.py` — slash command registry
- `src/jac/cli/renderer.py` — Rich console renderer, EventBus subscriptions
- `src/jac/cli/input.py` — prompt_toolkit `InputSession` with persistent history
- `src/jac/cli/prompts.py` — approval and question prompt views
- `src/jac/runtime/events.py` — typed event dataclasses, `EventBus`
- `src/jac/runtime/coordinator.py` — `RunCoordinator` (initial: calls pydantic_ai directly, no factory)
- `src/jac/runtime/session.py` — `SessionState`, `SessionConfig`, `ModelTier`, `RunMode`
- `src/jac/config.py` — `Settings` (pydantic-settings, env + settings.json)
- `src/jac/workspace.py` — workspace discovery: project vs. global paths

**What it does:** Delivers the full CLI surface — one-shot (`jac "..."`) and interactive (`jac chat`) modes. The `EventBus` is wired to the `Renderer` so model output streams to the terminal. `RunCoordinator` calls `pydantic_ai` directly in C0 (the factory comes at C5). `SessionState`/`SessionConfig` carry per-session settings (model, tier, mode, approval, cwd).

**Integration point:** C0 is the foundation everything else builds on. `RunCoordinator` is the primary call target from `ChatApp`; subsequent components extend the coordinator without changing its external interface.

---

## C1 — SQLite State + Resume

**Layer:** state

**Key files:**

- `src/jac/state/db.py` — `StateStore`, `open_state_store(path)`
- `src/jac/state/runs.py` — `RunsRepo`
- `src/jac/state/messages.py` — `MessagesRepo`
- `src/jac/state/migrations/001_initial.sql` — creates all tables (including those not yet populated)
- `src/jac/runtime/coordinator.py` — extended to persist `runs` + `messages` on each turn

**What it does:** Wires a SQLite database at `workspace.state_db_path`. Every `submit_message` call now persists a run record and message rows. Adds `jac resume [RUN_ID]` which reads message history from the DB and restores the conversation. Schema version `1.0` is recorded in `schema_meta`.

**Integration point:** `open_state_store` is called by `ChatApp.open()`. The returned `StateStore` is passed into `RunCoordinator`. `jac resume` is a new Click command in `cli/main.py` that calls `resume_run(state, settings, run_id)`.

---

## C2 — Workspace Seeding + Diagnostics

**Layer:** state, cli

**Key files:**

- `src/jac/state/seeder.py` — `seed_workspace(workspace, state)`: upserts skills + MCP servers from disk
- `src/jac/state/skills.py` — `SkillsRepo`
- `src/jac/state/mcp_servers.py` — `McpServersRepo`
- `src/jac/cli/main.py` — `jac doctor` command added
- `src/jac/workspace.py` — extended to expose skills/mcp paths

**What it does:** On boot, `seed_workspace` walks `~/.jac/skills/`, `~/.jac/mcp/`, `.agents/skills/`, `.agents/mcp/` and upserts rows into the `skills` and `mcp_servers` tables. `jac doctor` runs a health check on the workspace: settings validity, API key presence, database accessibility, skill and MCP server counts.

**Integration point:** `seed_workspace` is called by `ChatApp.open()` after `open_state_store`. Doctor diagnostics are standalone — they do not require an agent or coordinator.

---

## C3 — Model Factory + Onboarding

**Layer:** runtime, cli, config

**Key files:**

- `src/jac/runtime/models.py` — `build_pydantic_model(selection, settings)`, `ModelSelection`
- `src/jac/config.py` — extended: multi-provider support, tiers, profiles, profile-scoped env vars
- `src/jac/onboarder.py` — `run_wizard(global_scope)`: interactive `jac init` wizard
- `src/jac/cli/main.py` — `jac init [--global]` command added

**What it does:** Centralises model construction. `build_pydantic_model` takes a `ModelSelection` (provider + model string) and `Settings`, calls `settings.require_model_credentials(...)`, and returns the correct pydantic_ai model object for the provider. `jac init` launches an interactive wizard that writes `settings.json` and `.env` for the chosen provider. Profile-scoped env vars (`JAC_PROFILE_<SLUG>_<VAR>`) are supported from this point.

**Integration point:** `build_pydantic_model` is the sole provider-specific construction site. It is called by `config_loader` (C5) and was used directly by the coordinator before C5. The coordinator delegates to the factory rather than embedding provider logic.

---

## C4 — Shell Tool with Approval Flow

**Layer:** tools, runtime

**Key files:**

- `src/jac/tools/shell.py` — `run_shell`, `run_shell_background`, `list_processes`, `read_process_output`
- `src/jac/tools/filesystem.py` — full CRUD: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep_files`
- `src/jac/runtime/approvals.py` — `ApprovalRequest`, `ApprovalResponse`, `ApprovalPolicy`
- `src/jac/runtime/questions.py` — `QuestionRequest`, `QuestionResponse`
- `src/jac/runtime/events.py` — new events: `ShellCommandStarted`, `ShellCommandCompleted`, `FileEditPreviewed`, `FileEditApplied`, `ApprovalRequested`, `ApprovalResolved`, `QuestionRequested`, `QuestionAnswered`

**What it does:** Delivers the tool implementation layer. Shell tools run subprocesses with configurable timeout (`JAC_SHELL_TIMEOUT_SECONDS`) and output capping (`JAC_SHELL_MAX_OUTPUT_CHARS`). The approval primitive (request/resolve with `asyncio.Future`) blocks the coordinator until the UI responds; `ApprovalPolicy` provides automatic responses for non-interactive modes (auto-edit, yolo). File edit events include a diff preview before the write.

**Integration point:** Tools are registered in `TOOL_REGISTRY` (introduced in C5). The approval flow is invoked by tools when they need consent; the CLI's `PromptViews` subscribes to `ApprovalRequested` and calls `resolve_approval` when the user responds.

---

## C5 — Agent Factory & Config Loader

**Layer:** agents, state

**Key files:**

- `src/jac/agents/base.py` — `config_loader`, `AgentConfig`, `AgentConfigNotFound`, `UnknownToolError`
- `src/jac/agents/seeds.py` — `ensure_default_run_config`
- `src/jac/agents/__init__.py` — exports: `config_loader`, `ensure_default_run_config`, `AgentConfig`
- `src/jac/state/agent_configs.py` — `AgentConfigsRepo`
- `src/jac/state/run_mcp_servers.py` — `RunMcpServersRepo`
- `src/jac/state/run_skills.py` — `RunSkillsRepo`
- `src/jac/state/db.py` — extended: three new repos wired into `StateStore`
- `src/jac/runtime/coordinator.py` — refactored: `_ensure_agent` now delegates to `config_loader`; `reset_agent` added
- `src/jac/runtime/session.py` — `role` field added to `SessionConfig`
- `src/jac/__init__.py` — `build_agent` re-export added (SDK seam)
- `src/jac/tools/__init__.py` — `TOOL_REGISTRY` dict introduced

**What it does:** Introduces `src/jac/agents/` as the single site for `pydantic_ai.Agent(...)` construction. `config_loader` reads `agent_configs` for the `(run_id, role)` pair, resolves `allowed_tools` against `TOOL_REGISTRY`, builds MCP toolsets from `run_mcp_servers`, composes skills into the system prompt under `## Domain Knowledge`, calls `build_pydantic_model`, and returns a configured `Agent`. `ensure_default_run_config` seeds a default chat-role config idempotently. The coordinator now delegates agent construction entirely to the factory. The SDK seam (`from jac import build_agent`) is stable from this point.

**Integration point:** `RunCoordinator._ensure_agent` calls `ensure_default_run_config` then `config_loader` on first use. Nothing outside `src/jac/agents/` calls `pydantic_ai.Agent(...)`. External embedders use `from jac import build_agent` or `from jac.agents import config_loader` without touching the CLI.
