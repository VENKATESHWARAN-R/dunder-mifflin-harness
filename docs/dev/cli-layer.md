> **Status:** Reference · **Last revised:** 2026-05-07 · **Type:** developer documentation

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
| `jac` | Interactive default: opens `ChatApp`, calls `app.run()` |
| `jac run [PROMPT]` | One-shot: opens coordinator, calls `submit_message`, prints result, exits |
| `jac chat` | Interactive alias for `jac` |
| `jac resume [RUN_ID]` | Calls `ChatApp.from_resumed(run_id)`, then `app.run_resumed()`. If `RUN_ID` is omitted, queries `state.runs.list_recent(limit=1)` and resumes the most-recent session. |
| `jac init [--global]` | Initialises project/global workspace and provider config |
| `jac profile [current\|list\|use\|add]` | Manage named provider/model profiles |
| `jac doctor` / `jac config` | Prints workspace health diagnostics |

---

## ChatApp (`src/jac/cli/app.py`)

Composition root for the interactive session. Owns the lifecycle of all session-scoped objects.

**`ChatApp.open(settings)`** — factory for new sessions:
1. Opens `StateStore` and seeds workspace
2. Creates `SessionState` and `RunCoordinator`
3. Creates `InputSession` with `command_source` and `session_config_source` lambdas (feeds completions and toolbar)
4. Wires `EventBus` → `Renderer`
5. Subscribes `FileEditPreviewed` to snapshot files onto `_undo_stack` before each edit
6. Registers all slash commands and aliases

Slash commands that mutate `SessionConfig` (or `ApprovalPolicy` for `/approval`) emit `SessionConfigChanged` after a real value change. The renderer may log a dim debug line when `SessionConfig.debug` is on. `/tier` prints the manager’s resolved model; `/model` can emit a `WarningRaised` if credentials appear missing (non-blocking).

**`ChatApp.from_resumed(run_id, settings)`** — factory for resumed sessions:
1. Opens `StateStore`
2. Runs `seed_workspace(...)` for deterministic file→DB refresh
3. Calls `resume_run(state, settings, run_id)` to reconstruct the coordinator with message history
4. Wires renderer and commands as above

**`app.run()`**:
1. Calls `renderer.render_welcome(model, tier, mode)` — shows config inline
2. Reads from `InputSession` in a loop until `_should_exit`

**`app.run_resumed()`**:
1. Loads last 6 messages from DB and calls `renderer.render_resume_context(messages)` — shows context preview
2. Calls `app.run()`

**`handle_input(raw_text)`**:
1. Parses via `parse_input()`
2. `EMPTY` → ignore
3. Multiple attachment warnings → consolidated into one `WarningRaised` event
4. `SLASH` → `commands.dispatch(command, args)`. Unknown command shows `(try /help)` hint
5. `SHELL` → destructive pattern check, optional confirm, then `_handle_shell()`
6. `PLAIN` → `_submit_message(UserMessage(...))` with retry-on-failure

**`_submit_message(message)`**: wraps `coordinator.submit_message()` in try/except. If it raises (i.e. `RunFailed` fired), offers `ask_yn("Retry?")` and re-submits once.

**`_undo_stack`**: `list[tuple[Path, bytes]]` capped at 20 entries. Populated by the `FileEditPreviewed` handler reading the file before the edit lands.

---

## Input parsing (`src/jac/cli/parser.py`)

`parse_input(text: str, cwd: Path, max_attachment_bytes: int) -> ParsedInput`

| Kind | Trigger | Result |
|---|---|---|
| `EMPTY` | Blank or whitespace-only | `ParsedInput(kind=EMPTY)` |
| `SLASH` | Starts with `/` | `ParsedInput(kind=SLASH, slash=SlashInput(command, args))` |
| `SHELL` | Starts with `!` | `ParsedInput(kind=SHELL, shell_command=str)` |
| `PLAIN` | Everything else | `ParsedInput(kind=PLAIN, text=str, attachments=[...], warnings=[...])` |

`@path` and `@"path with spaces"` tokens are resolved relative to `cwd`. Failures produce `AttachmentWarning` objects. `ChatApp` consolidates multiple warnings into a single `WarningRaised` emission.

---

## Slash command registry (`src/jac/cli/commands.py`)

`SlashCommandRegistry` maps command names → `SlashCommand(name, handler, description, example)`.

Aliases are stored in `_aliases: dict[str, str]`. `dispatch(name, args)` resolves aliases before returning `False` for unknown commands.

`descriptions() -> dict[str, str]` returns `{name: description}` for the tab completer.

`help_text()` returns the full formatted help including examples, alias table, keyboard shortcuts, and input prefix reference.

All registered commands:

| Command | Aliases | Effect |
|---|---|---|
| `/help` | `/h`, `/?` | Print full help text |
| `/quit` | `/q` | Set `_should_exit = True` |
| `/model [id]` | `/m` | Show or set `session.config.model`; resets agent |
| `/tier [value]` | `/t` | Show or set `session.config.tier` (scout/worker/architect); resets agent |
| `/mode [value]` | — | Show or set `session.config.mode` (autopilot/hitl) |
| `/approval [value]` | — | Show or set `session.config.approval_mode`; updates `ApprovalPolicy.mode` |
| `/params [key value]` | — | Show or set `session.config.model_params`; resets agent |
| `/context` | `/x` | Show run ID, cwd, message count, attached files |
| `/usage` (alias `/cost`) | — | Rich tree + per-role totals from `attempts` |
| `/history [n]` | — | Load last `n` messages from DB and call `renderer.render_message_history()` |
| `/save [file]` | — | Export all messages from DB as Markdown |
| `/undo` | — | Pop `_undo_stack`, restore original file bytes |
| `/clear` | — | `console.clear()` |
| `/capabilities` | — | Show model/tier/mode/approval + allowed_tools/MCP/skills from agent config |

All validation errors show the invalid value, list valid options, and give an example.

---

## Renderer (`src/jac/cli/renderer.py`)

Rich console. Stateless — no business logic, only presentation. Wired to `EventBus` at construction time.

| Event | Rendering |
|---|---|
| `AgentTextDelta` | Buffers text in `_stream_buffer` |
| `AgentMessageCompleted` | Flushes buffer as Markdown; prints newline |
| `ToolCallRequested` | Flushes buffer; prints tool name + params in cyan panel |
| `ToolCallCompleted` | Prints result summary (red border on error) |
| `NodeStarted/Completed/Failed` | Dim/red one-liners |
| `FileEditPreviewed` | Syntax-highlighted unified diff in yellow panel |
| `FileEditApplied` | `[green]file edited:[/green] <path>` |
| `ShellCommandStarted` | Panel with command, cwd, timeout |
| `ShellCommandCompleted` | Panel with exit code, stdout, stderr |
| `SessionUsageUpdated` | Compact one-liner: cumulative tokens + ctx headroom in dim |
| `WarningRaised` | `[yellow]warning:[/yellow] <message>` |
| `RunFailed` | `[red]run failed:[/red] <message>` |
| `AgentDelegated` | `→ Handing to <name>…` in dim |

Additional methods:

- `render_welcome(model, tier, mode)` — banner with version and current config
- `render_resume_context(messages)` — compact preview of recent messages on session resume
- `render_message_history(messages, n)` — table of last `n` messages (for `/history`)
- `print_error / print_info / print_warning / print_value` — direct print helpers

---

## PromptViews (`src/jac/cli/prompts.py`)

Handles approval and yes/no prompts. All methods are **async** and use `prompt_toolkit.PromptSession` internally.

### `ask_approval(request) → ApprovalResponse`

Interactive approval gate:

1. Renders a Rich panel with the action summary and details.
2. Prints a static option list (`[a] approve once`, `[d] deny`, `[r] redirect`, etc.).
3. Opens a `PromptSession` with:
   - A dynamic message showing the currently highlighted option (`▶ <label>`).
   - ↑/↓ (or Ctrl+P/N) to navigate.
   - Enter to confirm the highlighted option.
   - Single-letter shortcuts that exit immediately without Enter.
   - Ctrl+C/D defaults to deny.
4. For `redirect`: opens a second `PromptSession` collecting the feedback text.

Returns an `ApprovalResponse` with one of: `APPROVE_ONCE`, `DENY`, `ALLOW_TOOL_FOR_SESSION`, `ALLOW_EXACT_FOR_SESSION`, or `REDIRECT` (with `redirect_message`).

**REDIRECT in the approval wrapper** (`agents/approval.py`): returns `ToolResult(status=PERMISSION_DENIED, error="[User feedback] <message>")`. The model sees this as the tool's output and can issue a revised tool call without any additional user prompt.

### `ask_yn(prompt) → bool`

Async yes/no prompt via `PromptSession`. `y`/`Y` returns `True`; `n`, Enter, Ctrl+C, Ctrl+D return `False`.

### `ask_question(request) → QuestionResponse`

Synchronous Rich-based question prompt (free-text, single-choice, multi-choice). Uses `console.input()`. Returns a structured `QuestionResponse`.

---

## InputSession (`src/jac/cli/input.py`)

prompt_toolkit `PromptSession` wrapper.

**History**: `DedupFileHistory` at `~/.jac/input_history`. Skips consecutive duplicate entries.

**Completions**: `JacCompleter` (custom `Completer` subclass):
- On `/`: completes command names with descriptions as `display_meta`.
- On `/tier `, `/mode `, `/approval `, `/params `: completes valid argument values.
- On `@`: delegates to `PathCompleter` for file path completion.
- `complete_while_typing=False` (Tab only, not on every keystroke).

**Toolbar**: `bottom_toolbar` lambda reads from `session_config_source()` and renders `model · tier · mode · approval` on every render cycle.

**Key bindings**: custom `KeyBindings` adds Esc+Enter for multiline. Ctrl+R history search uses prompt_toolkit's default emacs bindings.

**Placeholder**: `(esc+enter for newline)` shown when the buffer is empty.

`InputSession` accepts two optional callables at construction:
- `command_source: () → dict[str, str]` — fed by `lambda: self.commands.descriptions()` in `ChatApp`.
- `session_config_source: () → SessionConfig` — fed by `lambda: self.session.config` in `ChatApp`.
