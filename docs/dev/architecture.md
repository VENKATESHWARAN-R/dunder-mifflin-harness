> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** developer documentation

# Architecture

## Layer diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  src/jac/cli/                                                   │
│  terminal adapter                                               │
│  Click commands · prompt_toolkit REPL · Rich renderer           │
│  slash commands · input parser · approval prompts               │
└────────────────────────────┬────────────────────────────────────┘
                             │ calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/jac/runtime/                                               │
│  UI-agnostic runtime                                            │
│  RunCoordinator · EventBus · SessionState · SessionConfig       │
│  approvals · questions · model factory (build_pydantic_model)   │
└──────────┬─────────────────────────────────┬────────────────────┘
           │ delegates agent build            │ reads/writes
           ▼                                  ▼
┌──────────────────────────┐     ┌────────────────────────────────┐
│  src/jac/agents/         │     │  src/jac/state/                │
│  agent factory           │     │  SQLite repos                  │
│  config_loader           │     │  StateStore · RunsRepo         │
│  ensure_default_run_cfg  │◄────│  MessagesRepo · SkillsRepo     │
│  AgentConfig             │     │  McpServersRepo                │
│  (only Agent() site)     │     │  AgentConfigsRepo              │
└──────────┬───────────────┘     │  RunMcpServersRepo             │
           │ resolves tools       │  RunSkillsRepo                 │
           ▼                     └────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  src/jac/tools/                                                 │
│  TOOL_REGISTRY                                                  │
│  filesystem tools · shell tools                                 │
└─────────────────────────────────────────────────────────────────┘

All layers depend on:
┌─────────────────────────────────────────────────────────────────┐
│  src/jac/config.py      src/jac/workspace.py                    │
│  Settings (pydantic-settings)    workspace discovery            │
└─────────────────────────────────────────────────────────────────┘
```

Entry point: `src/jac/cli/main.py` (Click group). All user-facing commands originate here.

## Dependency rules

These rules are critical. Violations break the clean architecture and create circular imports.

| Layer | May import from |
|---|---|
| `cli.*` | `runtime.*`, `config`, `state.*`, `workspace` |
| `runtime.*` | `agents.*`, `tools.*`, `state.*`, `runtime.events`, `runtime.models`, `config` |
| `agents.*` | `tools.*`, `state.*`, `runtime.events`, `runtime.models`, `config` |
| `tools.*` | `jac.runtime.events` only (for tool call events); nothing else from `jac.*` |
| `state.*` | `aiosqlite` only; nothing from `jac.*` |
| `config.py` | nothing from `jac.*` (leaf import) |
| `workspace.py` | nothing from `jac.*` (leaf import) |

**Never:**
- `agents.*` or `runtime.*` importing from `cli.*`
- `state.*` importing from `runtime.*`, `agents.*`, or `tools.*`
- Any layer importing from `lab/specimens/`

## Data flow for a `jac chat` turn

1. User types a message in the prompt_toolkit REPL.
2. `app.py:handle_input` receives the raw string and calls `parse_input(text, cwd, max_attachment_bytes)`.
3. `parse_input` returns a `ParsedInput` — if there are `@file` refs, they are resolved to `FileAttachment` objects.
4. `handle_input` calls `coordinator.submit_message(UserMessage(text=..., attachments=[...]))`.
5. `coordinator.submit_message`:
   a. Calls `_ensure_agent()` on the first turn.
   b. `_ensure_agent` calls `ensure_default_run_config(state, run_id)` (idempotent), then `config_loader(state, settings, run_id)`.
   c. `config_loader` reads `agent_configs`, resolves tools, builds MCP toolsets, composes system prompt with skills, calls `build_pydantic_model`, returns a `pydantic_ai.Agent`.
   d. The agent is cached on the coordinator.
6. `agent.run(prompt)` makes the model API call. Text tokens are streamed via `EventBus` as `AgentTextDelta` events.
7. The `Renderer` in `cli/renderer.py` is subscribed to `AgentTextDelta` and writes tokens to the Rich console as they arrive.
8. When the run completes, `AgentMessageCompleted` fires. The result is persisted to `state.messages`.
9. The coordinator returns the output string to `handle_input`, which returns control to the REPL prompt.

## The EventBus as integration seam

`EventBus` (`src/jac/runtime/events.py`) is the only channel through which the CLI hears from the runtime. Events are typed dataclasses. Handlers register with `events.on(EventType, handler)` where the handler can be sync or async. Events are emitted with `await events.emit(event)`.

Approvals and questions are **not** plain events. They use dedicated request/response primitives with `asyncio.Future` waiters that block the coordinator until the UI responds. This makes the approval flow synchronous from the coordinator's perspective while allowing the CLI to render an interactive prompt.

## StateStore as the single shared mutable object

`StateStore` is opened once per session at `workspace.state_db_path`. The coordinator, the workspace seeder, and the CLI app all hold a reference to the same instance. There is no connection pool — SQLite is single-file and the session is single-process.

## Where things live

| Thing | Location |
|---|---|
| CLI commands | `src/jac/cli/main.py` |
| Interactive app loop | `src/jac/cli/app.py` |
| Input parsing + file refs | `src/jac/cli/parser.py` |
| Slash command registry | `src/jac/cli/commands.py` |
| Rich renderer + event wiring | `src/jac/cli/renderer.py` |
| Approval + question prompts | `src/jac/cli/prompts.py` |
| prompt_toolkit input session | `src/jac/cli/input.py` |
| Run coordinator | `src/jac/runtime/coordinator.py` |
| Typed events | `src/jac/runtime/events.py` |
| Session state | `src/jac/runtime/session.py` |
| Approval primitives | `src/jac/runtime/approvals.py` |
| Question primitives | `src/jac/runtime/questions.py` |
| Model factory | `src/jac/runtime/models.py` |
| Agent factory (only Agent() site) | `src/jac/agents/base.py` |
| Default config seeder | `src/jac/agents/seeds.py` |
| Agent package exports | `src/jac/agents/__init__.py` |
| Tool registry + helpers | `src/jac/tools/__init__.py` |
| Filesystem tools | `src/jac/tools/filesystem.py` |
| Shell tools | `src/jac/tools/shell.py` |
| StateStore + open_state_store | `src/jac/state/db.py` |
| Runs / messages repos | `src/jac/state/runs.py`, `src/jac/state/messages.py` |
| Skills / MCP repos | `src/jac/state/skills.py`, `src/jac/state/mcp_servers.py` |
| Agent config repos | `src/jac/state/agent_configs.py`, `src/jac/state/run_mcp_servers.py`, `src/jac/state/run_skills.py` |
| Workspace file seeder | `src/jac/state/seeder.py` |
| Migrations | `src/jac/state/migrations/` |
| Settings | `src/jac/config.py` |
| Workspace discovery | `src/jac/workspace.py` |
