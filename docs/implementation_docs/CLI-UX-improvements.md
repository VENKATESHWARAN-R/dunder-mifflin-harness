# CLI UX Improvements — Implementation Plan

> **Status:** Implemented · **Last revised:** 2026-05-04 · **Type:** Implementation Plan
>
> All batches shipped 2026-05-04. See ROADMAP.md Done → CLI-UX for the full checklist.
> Arrow-key approval navigation and REDIRECT decision added beyond original scope.

---

## Goals

Improve JAC's terminal experience across four axes without violating any locked contracts:

1. **Speed** — fewer keystrokes to accomplish common actions
2. **Clarity** — output that communicates intent, not just data
3. **Confidence** — safe defaults for destructive operations; visible state
4. **Discoverability** — users find capabilities without reading docs

## Non-goals

- Textual/TUI (explicitly out of scope; CLI_DESIGN.md)
- `--dry-run` (blocked on runtime support; deferred to C21 HITL)
- Browser adapter changes (deferred to C28)
- Merging approvals and questions (locked as separate primitives)

## Key files

| File | What changes |
|---|---|
| `src/jac/cli/input.py` | Completions, history dedup, Ctrl+R, bottom toolbar, dynamic prompt |
| `src/jac/cli/commands.py` | New slash commands, aliases, enriched help |
| `src/jac/cli/app.py` | Wire new commands, undo stack, retry prompt, session resume |
| `src/jac/cli/renderer.py` | Per-message cost inline, collapsible tool calls, richer welcome |
| `src/jac/cli/prompts.py` | Destructive shell detection, confirm prompt |
| `src/jac/cli/main.py` | `jac resume` with no ID, `--json` flag stub |
| `tests/test_cli_*.py` | Unit tests for each new behavior |

---

## Batch 1 — Input Layer (Speed + Discoverability)

These live entirely in `src/jac/cli/input.py` and `commands.py`. No runtime changes.

### 1a. Tab completions

prompt_toolkit already reserves this slot (comment in `input.py`). Wire a `NestedCompleter`:

```
/           → shows all slash commands with one-line hint
/model      → completes known model IDs from Settings.available_models()
/tier       → completes scout | worker | architect
/approval   → completes interactive | auto-edit | yolo
/mode       → completes autopilot | hitl
@           → PathCompleter(expanduser=True, only_directories=False)
```

Implementation:
- Build `_build_completer(registry, settings) -> NestedCompleter` in `input.py`
- Pass it into `PromptSession(..., completer=_build_completer(...), complete_while_typing=True)`
- PathCompleter for `@` requires a custom `Completer` subclass that activates after `@`
- Feed slash command descriptions as `{"/model": "<current> | set model"}` display text

### 1b. Ctrl+R reverse history search

The current `KeyBindings()` only adds Esc+Enter. It doesn't disable the default `ctrl+r` handler from prompt_toolkit — so this may already work. Verify; if not, remove the `key_bindings=bindings` and rely on defaults, or use `merge_key_bindings([load_emacs_bindings(), bindings])`.

### 1c. History deduplication

Wrap prompt_toolkit's `FileHistory` to skip consecutive identical lines:

```python
class DedupFileHistory(FileHistory):
    def append_string(self, string: str) -> None:
        strings = list(self.load_history_strings())
        if strings and strings[0] == string:
            return
        super().append_string(string)
```

### 1d. Bottom toolbar

Add a `bottom_toolbar` lambda to `PromptSession` showing live session config:

```python
def _toolbar(session_state: SessionState) -> str:
    return (
        f" model: {session_state.model or 'default'}"
        f" | tier: {session_state.tier or 'worker'}"
        f" | mode: {session_state.mode}"
        f" | approval: {session_state.approval_mode}"
    )
```

Pass `bottom_toolbar=lambda: _toolbar(self._session)` into `PromptSession`. Requires `InputSession` to receive a reference to `SessionState` (or a lazy callable).

### 1e. Dynamic prompt

Replace the static `"jac › "` prompt with a callable that shows run count or cost:

```python
def _prompt(session: SessionState) -> list[tuple[str, str]]:
    turns = session.turn_count or 0
    cost = session.cumulative_cost_usd
    cost_str = f"${cost:.4f} " if cost else ""
    return [("class:prompt", f"{cost_str}jac [{turns}] › ")]
```

`turn_count` and `cumulative_cost_usd` are already tracked in `SessionState` (or trivially added).

### 1f. Multiline hint

Add `placeholder=FormattedText([("class:placeholder", "  (esc+enter for newline)")])` to `PromptSession`. This shows as dim hint when input is empty.

**Acceptance:** Tab on `/` shows command list. Tab on `@` shows path list. Ctrl+R cycles history. Toolbar is visible. Placeholder shows on empty input.

---

## Batch 2 — Help & Discoverability

All in `commands.py`, `app.py`, `renderer.py`. No runtime changes.

### 2a. Enrich `/help`

Current: just lists command names. New format:

```
Slash commands:
  /help              — show this message
  /model [id]        — show or set active model  (e.g. /model claude-sonnet-4-6)
  /tier [name]       — set model tier            (scout | worker | architect)
  /mode [name]       — set run mode              (autopilot | hitl)
  /approval [name]   — set approval policy       (interactive | auto-edit | yolo)
  /params [k v]      — set model parameters      (e.g. /params temperature 0.2)
  /context           — show session context
  /history [n]       — show last n messages      (default: 10)
  /cost              — show cost breakdown
  /undo              — revert last file edit
  /save [file]       — save session transcript
  /clear             — clear screen
  /capabilities      — show available tools and MCP servers
  /quit              — exit

Input grammar:
  @path              — attach a file to next message
  !cmd               — run a shell command locally
  esc+enter          — insert newline
  ctrl+r             — search history
  ctrl+c             — cancel input
  ctrl+d             — exit
```

Store this as a structured `HELP_TEXT` dict in `commands.py` keyed by command name so completions can also pull the one-liner.

### 2b. Short aliases

Register aliases in `SlashCommandRegistry`:

```python
ALIASES = {
    "h": "help",
    "q": "quit",
    "m": "model",
    "t": "tier",
    "x": "context",
    "?": "help",
}
```

Dispatch: if command not found in registry, check ALIASES and re-dispatch.

### 2c. `/clear` command

```python
async def _cmd_clear(args: str) -> None:
    console.clear()
```

Trivial. Clears terminal, session state unchanged.

### 2d. `/history [n]` command

Show last `n` messages from `SessionState.messages` (already persisted):

```python
async def _cmd_history(args: str) -> None:
    n = int(args.strip()) if args.strip().isdigit() else 10
    msgs = await session.get_recent_messages(n)
    for msg in msgs:
        role_style = "bold cyan" if msg.role == "user" else "bold green"
        console.print(f"[{role_style}]{msg.role}[/]: {msg.text[:200]}{'…' if len(msg.text) > 200 else ''}")
```

### 2e. `/save [file]` command

Export session transcript as Markdown:

```python
async def _cmd_save(args: str) -> None:
    path = args.strip() or f"jac-session-{session.run_id[:8]}.md"
    msgs = await session.get_all_messages()
    content = "\n\n".join(f"**{m.role}:** {m.text}" for m in msgs)
    Path(path).write_text(content)
    renderer.print_info(f"Saved to {path}")
```

### 2f. `/capabilities` command

Show what tools, MCP servers, and skills are active for this run:

```python
async def _cmd_capabilities(args: str) -> None:
    config = await coordinator.get_agent_config()
    # Print: allowed_tools, mcp_servers, skills
```

Requires `RunCoordinator.get_agent_config()` to expose the resolved config — add a thin accessor that returns the already-loaded `AgentConfig`.

### 2g. Welcome banner improvements

In `renderer.render_welcome()`, add:

```
JAC v0.3.x  ·  model: sonnet | tier: worker | mode: autopilot
Type a message to start, /help for commands, ctrl+d to exit.
```

Replace the existing static banner with one that pulls current session config.

**Acceptance:** `/help` shows full command table with examples. `/h`, `/q` work. `/clear` works. `/history` shows messages. `/save` writes file. Welcome banner shows config.

---

## Batch 3 — Error Quality & Resilience

### 3a. Actionable validation errors

In each slash command handler, replace terse errors with structured messages:

```python
# Before
raise ValueError("Approval mode must be interactive, auto-edit, or yolo.")

# After
VALID = {"interactive", "auto-edit", "yolo"}
if value not in VALID:
    renderer.print_error(
        f"Unknown approval mode: '{value}'\n"
        f"Valid options: {', '.join(sorted(VALID))}\n"
        f"Example: /approval auto-edit"
    )
    return
```

Apply same pattern to `/model`, `/tier`, `/mode`, `/params`.

For `/model <id>` with unknown ID: show currently configured model and suggest checking provider docs. Don't fail the session — just print the error and continue.

For `@path` attachment failures: after parsing, if `parsed.warnings`, print a consolidated warning block:

```
Attachment warnings:
  @bigfile.log — file too large (4.2 MB, limit 1 MB)
  @missing.txt — file not found
```

### 3b. Retry prompt on `RunFailed`

In `ChatApp.handle_input`, after coordinator run completes with failure:

```python
if run_result.failed:
    renderer.print_error(f"Run failed: {run_result.error}")
    answer = await prompts.ask_yn("Retry with the same input?")
    if answer:
        return await self._execute_message(user_message)  # re-submit
```

Use a simple `ask_yn(prompt) -> bool` helper in `prompts.py` (reads 'y'/'n', defaults False on EOF).

### 3c. Per-message cost inline

After `RunCompleted` event fires, if a `CostUpdated` was received during the run, print a compact one-liner below the response:

```python
# In renderer, on RunCompleted:
if self._last_cost:
    cost = self._last_cost
    console.print(
        f"[dim]  ↳ {cost.total_tokens:,} tok · ${cost.total_usd:.4f}[/dim]"
    )
    self._last_cost = None
```

Store `_last_cost` in renderer state; update it on each `CostUpdated` event (already subscribed).

This requires `CostUpdated` to fire per-run, not just on demand — verify `RunCoordinator` already emits it at run end (it does, at C7, but for now can be stubbed until C7 ships).

### 3d. Rate-limit / context-limit messaging

In `ChatApp._handle_run_error(err)`, detect specific error types:

```python
if "rate_limit" in str(err).lower() or "429" in str(err):
    renderer.print_warning("Rate limited by provider. Wait a moment and retry.")
elif "context_length" in str(err).lower() or "maximum context" in str(err).lower():
    renderer.print_warning(
        "Context limit reached. Consider /history to review, then start a new session."
    )
else:
    renderer.print_error(f"Run failed: {err}")
```

**Acceptance:** Bad `/model` shows valid options. Bad `/tier` shows valid options. Attachment warnings consolidated. Rate-limit errors show human message. Per-message cost shows after run.

---

## Batch 4 — Safety & Confidence

### 4a. Destructive shell command detection

In `ChatApp._handle_shell(command)`, before execution:

```python
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f?\b",   # rm -rf, rm -fr
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+push\s+.*--force\b",
    r"\bgit\s+clean\s+-[a-z]*f\b",
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bdd\s+if=",                    # disk write
    r"\bmkfs\b",
]

if any(re.search(p, command, re.IGNORECASE) for p in DESTRUCTIVE_PATTERNS):
    renderer.print_warning(f"Potentially destructive: {command}")
    confirmed = await prompts.ask_yn("Run anyway?")
    if not confirmed:
        renderer.print_info("Cancelled.")
        return
```

Note: this applies only to **user-typed** `!` shell commands, not agent-requested shell commands (those already go through the approval gate at C5a).

### 4b. `/undo` — revert last file edit

Track an undo stack in `ChatApp`:

```python
# In app.py
self._undo_stack: list[tuple[Path, bytes]] = []  # (path, original_bytes)
```

Subscribe to `FileEditPreviewed` event: before the edit applies, snapshot the original bytes. On `/undo`:

```python
async def _cmd_undo(args: str) -> None:
    if not self._undo_stack:
        renderer.print_info("Nothing to undo.")
        return
    path, original = self._undo_stack.pop()
    path.write_bytes(original)
    renderer.print_info(f"Reverted: {path}")
```

Limit stack depth to 20 entries. Store `bytes` (not `str`) to handle binary files correctly.

**Important:** this is a pure CLI-layer undo (in-memory during session). It does not interact with the state DB or the agent. It reverts the filesystem only.

### 4c. `jac resume` with no run-id

In `main.py`, the `resume` command currently requires `run_id`. Make it optional:

```python
@jac.command()
@click.argument("run_id", required=False, default=None)
def resume(run_id: str | None) -> None:
    if run_id is None:
        run_id = state.get_most_recent_run_id()
        if run_id is None:
            click.echo("No prior sessions found.", err=True)
            raise SystemExit(1)
        click.echo(f"Resuming most recent session: {run_id[:8]}…")
    ...
```

`StateStore.get_most_recent_run_id()` → `SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1` — trivial.

### 4d. Resume context preview

When resuming a session (both `jac resume` and `jac resume <id>`), before entering the loop print the last 3 user messages and their assistant responses:

```python
renderer.render_resume_context(recent_messages[-6:])  # 3 turns = 6 messages
```

**Acceptance:** `!rm -rf .` prompts for confirmation. `/undo` reverts last file edit. `jac resume` without ID finds most recent. Resume shows context preview.

---

## Batch 5 — Machine-Readable Output (Lower priority)

### 5a. `--json` flag on `jac run`

When `--json` is passed, replace `Renderer` with `JsonRenderer` that writes newline-delimited JSON to stdout:

```python
{"type": "text_delta", "text": "Hello"}
{"type": "tool_call", "name": "read_file", "status": "completed"}
{"type": "run_completed", "cost_usd": 0.0023, "tokens": 1247}
```

Implementation:
- `JsonRenderer(events: EventBus)` subscribes to the same events as `Renderer`
- `ChatApp` or `main.py` selects which renderer to instantiate based on `--json` flag
- Interactive features (approvals, questions) auto-approve/skip in `--json` mode (or use `--approval auto-edit` to handle)

This is more involved because it requires the renderer to be swappable at construction. `ChatApp` already receives `renderer` as a dependency, so this is a clean extension.

---

## Ordering and shipping

These batches are independent and can be shipped in any order. Recommended sequence by impact/effort ratio:

| Batch | Effort | Impact | Ship order |
|---|---|---|---|
| Batch 1: Input layer | M | High | 1st |
| Batch 2: Help & discoverability | S | High | 2nd |
| Batch 3: Error quality | S | High | 3rd |
| Batch 4: Safety & confidence | M | High | 4th |
| Batch 5: JSON output | M | Medium | 5th |

Batch 1 and 2 can run in parallel since they touch different files. Batch 3 is fast (mostly error message wording). Batch 4 requires a bit more coordination (undo stack event subscription). Batch 5 is lower priority and can wait until scripting is actually needed.

## Acceptance checks (full)

- [ ] Tab on `/` shows all slash commands with descriptions
- [ ] Tab on `@` completes file paths relative to cwd
- [ ] `/model ` tab shows known model IDs; `/tier ` tab shows scout/worker/architect
- [ ] Ctrl+R cycles through history with incremental search
- [ ] Consecutive identical history entries are deduplicated
- [ ] Bottom toolbar shows current model/tier/mode/approval
- [ ] Empty input shows `(esc+enter for newline)` placeholder
- [ ] `/h` is an alias for `/help`; `/q` for `/quit`
- [ ] `/help` shows full table with descriptions and examples
- [ ] `/clear` clears the terminal screen, session continues
- [ ] `/history 5` shows last 5 messages
- [ ] `/save transcript.md` writes session to file
- [ ] `/capabilities` lists active tools and MCP servers
- [ ] Welcome banner shows model/tier/mode
- [ ] `/model nonexistent` shows error with valid options listed
- [ ] `@nonexistent.txt` shows consolidated warning, not inline crash
- [ ] RunFailed prompts retry; answering 'y' re-submits
- [ ] Per-message cost shown inline after response (when C7 ships)
- [ ] Rate-limit error shows human-readable message
- [ ] `!rm -rf .` prompts for confirmation before executing
- [ ] `/undo` reverts last file edit made during session
- [ ] `jac resume` with no ID resumes most recent session
- [ ] Resume shows last 3 message pairs as context preview
- [ ] `jac run "..." --json` outputs newline-delimited JSON (Batch 5)

## Contracts touched

None of these changes require updates to locked contracts. The CLI_DESIGN.md already reserves prompt_toolkit for completions (explicitly noted as "future") and permits new slash commands as long as they improve control/transparency/debugging (not cosmetic). All new features are terminal-adapter concerns and do not touch `runtime/`, `agents/`, or `tools/`.

If `get_agent_config()` is added to `RunCoordinator` for `/capabilities`, that accessor should be documented in the runtime layer doc (`docs/dev/runtime-layer.md`) as a read-only inspector, not a mutation surface.
