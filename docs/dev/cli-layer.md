> **Status:** Reference · **Last revised:** 2026-05-04 · **Type:** developer documentation

# CLI Layer

The CLI layer lives in `src/jac/cli/`. It is a **presentation adapter** — it translates user input into runtime calls and subscribes to `EventBus` events to render output. It does not orchestrate the agent or hold any domain logic.

See also: [`docs/contracts/CLI_DESIGN.md`](../contracts/CLI_DESIGN.md) — the binding design contract for the terminal adapter.

## Terminal adapter principle

The CLI knows: how to display things, how to read user input, and which coordinator method to call. It does not know: how the agent is built, which model is used, how tools are resolved, or how state is persisted. Those concerns live in `src/jac/runtime/` and `src/jac/agents/`.

Imports are one-directional: `cli.*` may import `runtime.*`, `state.*`, `config`, and `workspace`. Nothing in `runtime.*` or `agents.*` imports from `cli.*`.

---

## CLI commands (`src/jac/cli/main.py`)

Click command group. Entry point for all user-facing commands.

| Command | Behaviour |
|---|---|
| `jac` | Interactive default: opens `ChatApp`, calls `app.run()` (prompt_toolkit loop) |
| `jac run [PROMPT]` | One-shot: opens coordinator, calls `submit_message(UserMessage(text=PROMPT))`, prints result, exits |
| `jac chat` | Interactive alias for `jac` (kept for compatibility) |
| `jac resume RUN_ID` | Calls `ChatApp.from_resumed(run_id)`, then `app.run()` |
| `jac init [--global]` | Initializes project/global workspace and provider config |
| `jac doctor` / `jac config` | Prints workspace health diagnostics |

---

## ChatApp (`src/jac/cli/app.py`)

Composition root for the interactive session. Owns the lifecycle of all session-scoped objects.

**`ChatApp.open(settings, workspace)`** — factory method for new sessions:
1. Calls `open_state_store(workspace.state_db_path)` to get `StateStore`
2. Calls `seed_workspace(workspace, state)` to upsert skills + MCP servers from disk
3. Creates `SessionState` with a new `run_id`
4. Creates `RunCoordinator(settings, state, session, events)`
5. Wires `EventBus` → `Renderer`
6. Registers all slash commands

**`ChatApp.from_resumed(run_id, settings, workspace)`** — factory method for resumed sessions:
1. Opens `StateStore` as above
2. Calls `resume_run(state, settings, run_id)` to reconstruct the coordinator with message history
3. Wires renderer and slash commands as above

**`app.run()`** — the prompt_toolkit event loop:
1. Shows the `run_id` at session start
2. Reads a line from `InputSession`
3. Calls `handle_input(raw_text)`
4. Loops until `/quit` or EOF

**`handle_input(raw_text)`**:
1. Calls `parse_input(raw_text, cwd, max_attachment_bytes)` → `ParsedInput`
2. Dispatches based on kind:
   - `EMPTY` → ignore
   - `SLASH` → `slash_registry.dispatch(command, args)`
   - `SHELL` → runs subprocess inline via `subprocess.run`, prints output to terminal (not sent to model)
   - `PLAIN` → calls `await coordinator.submit_message(UserMessage(text=..., attachments=...))`

---

## Input parsing (`src/jac/cli/parser.py`)

`parse_input(text: str, cwd: Path, max_attachment_bytes: int) -> ParsedInput`

Parses raw user input into one of four kinds:

| Kind | Trigger | Result |
|---|---|---|
| `EMPTY` | Blank or whitespace-only | `ParsedInput(kind=EMPTY)` |
| `SLASH` | Starts with `/` | `ParsedInput(kind=SLASH, slash=SlashInput(command, args))` |
| `SHELL` | Starts with `!` | `ParsedInput(kind=SHELL, shell_command=str)` |
| `PLAIN` | Everything else | `ParsedInput(kind=PLAIN, text=str, attachments=[...], warnings=[...])` |

For `PLAIN` input, the parser scans for `@path` and `@"path with spaces"` tokens. Each matched path is resolved relative to `cwd` and read. Successful reads produce `FileAttachment(path, content)` objects appended after the user text. Files exceeding `max_attachment_bytes` or that cannot be read produce `AttachmentWarning` objects instead.

---

## Slash command registry (`src/jac/cli/commands.py`)

`SlashCommandRegistry` maps command names to `SlashCommand(name, handler, description)` objects. Handlers are `async def (args: str) -> None`.

All registered commands:

| Command | Effect |
|---|---|
| `/help` | Prints all registered commands and descriptions |
| `/quit` | Sets a stop flag; `app.run()` exits the loop |
| `/model [name]` | With no args: prints current model string. With `name`: sets `session.config.model = name`, calls `coordinator.reset_agent()` |
| `/tier [scout\|worker\|architect]` | With no args: prints current tier. With arg: sets `session.config.tier`, calls `coordinator.reset_agent()` |
| `/mode [autopilot\|hitl]` | With no args: prints current mode. With arg: sets `session.config.mode` |
| `/approval [interactive\|auto-edit\|yolo]` | With no args: prints current approval mode. With arg: sets `session.config.approval_mode` |
| `/params [key value]` | With no args: prints all model params. With `key value`: sets `session.config.model_params[key] = value` (auto-converts to float/int where applicable) |
| `/context` | Prints: run_id, cwd, attached files, message count |
| `/cost` | Prints `session.latest_cost_summary` (tokens, estimated cost) |

All commands mutate `SessionConfig` or the local approval policy. None send data to the model.

---

## Renderer (`src/jac/cli/renderer.py`)

Rich console. Wired to `EventBus` at `ChatApp` construction time. Each subscription maps an event type to a Rich rendering action.

| Event | Rendering |
|---|---|
| `AgentTextDelta` | Writes `e.text` to stdout (streaming, no newline) |
| `AgentMessageCompleted` | Prints trailing newline; optionally formats final message |
| `ToolCallRequested` | Prints tool name + formatted args in a dim panel |
| `ToolCallCompleted` | Prints result summary |
| `ShellCommandStarted` | Prints `$ <command>` in a panel |
| `ShellCommandCompleted` | Prints exit code + output in a bordered panel |
| `FileEditPreviewed` | Renders a syntax-highlighted diff |
| `FileEditApplied` | Prints confirmation with path |
| `WarningRaised` | Prints `e.message` in yellow |

The renderer also exposes direct methods used by slash command handlers:

- `renderer.print_error(msg)` — red text
- `renderer.print_info(msg)` — dim text
- `renderer.print_value(label, value)` — label: value pair

---

## PromptViews (`src/jac/cli/prompts.py`)

Handles approval and question prompts inline in the REPL. Called by `ChatApp`'s event handlers:

- `ApprovalRequested` → `PromptViews.show_approval_prompt(request)` → renders description and preview → reads y/n from the user → calls `await events.resolve_approval(request.id, approved=answer)`
- `QuestionRequested` → `PromptViews.show_question_prompt(request)` → renders question and optional choices → reads answer → calls `await events.answer_question(request.id, answer=answer)`

These methods run synchronously within the prompt_toolkit loop using `PromptSession.prompt()` (a blocking read). The coordinator is suspended waiting on the `asyncio.Future` in `EventBus` until the user responds.

---

## InputSession (`src/jac/cli/input.py`)

prompt_toolkit `PromptSession` wrapper.

- Persistent history file: `~/.jac/input_history`
- Multi-line support: `Meta+Enter` inserts a newline; `Enter` submits
- Keyboard shortcuts: `Ctrl+C` cancels the current input (not the session); `Ctrl+D` on an empty line sends EOF to exit

`InputSession.prompt(session_display)` blocks until the user submits a line or triggers EOF.
