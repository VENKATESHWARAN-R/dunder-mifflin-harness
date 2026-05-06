# CLI Design

> **Status:** Locked · **Last revised:** 2026-05-06 · **Type:** contract

## Purpose

The JAC CLI is a terminal adapter over the agent runtime. It should make local development runs observable and controllable without letting terminal concerns leak into the backend.

The backend owns agent behavior, workflow routing, tool execution, model selection, state persistence, and future A2A/browser integrations. The CLI owns process arguments, interactive input, rendering, and collecting human responses.

## Library Roles

- Click owns process-level entrypoints, options, exit codes, and script integration.
- prompt_toolkit owns interactive input, history (with deduplication), multiline behavior, tab completions, bottom toolbar, and key bindings.
- Rich owns terminal rendering for agent text, tool activity, file diffs, shell output, warnings, and summaries.

Textual is intentionally out of scope for now. A full TUI would add surface area before the runtime is mature enough to benefit from it.

## Boundaries

CLI modules may depend on runtime abstractions. Runtime, tools, nodes, workflows, and modes must not import CLI modules.

The communication boundary is an event bus. Runtime code emits typed events such as text deltas, tool calls, shell activity, file edit previews, approvals, questions, and run completion. UI surfaces subscribe to those events and respond through explicit request/response handles.

Approvals and questions are separate concepts:

- An approval asks whether a concrete action may proceed.
- A question asks the user for information the agent needs to continue.

These should not share a yes/no primitive.

## Entrypoints

Command surface:

- `jac "say hi"` / `jac run "say hi"` — one-shot prompt.
- `jac` / `jac chat` — interactive REPL.
- `jac resume [run-id]` — resume a prior session. When `run-id` is omitted, the most-recent session is selected automatically.
- `jac init [--global]` — initialise project or global workspace.
- `jac profile list|current|use|add` — manage named provider/model profiles.
- `jac doctor` / `jac config` — resolved settings diagnostics.

Until graph workflows exist, one-shot and chat mode use the same runtime coordinator around the current Pydantic AI agent.

## Input Grammar

Plain text is sent to the runtime as a user message.

Slash commands are local commands by default. They configure the session or inspect
state and are not sent to the model unless explicitly defined as model-routed
workflow entrypoints in the roadmap/contracts (for example, `/plan` and `/init`
at C6b).

```text
/help
/model gateway/google-vertex:gemini-3.1-flash-lite-preview
/approval interactive
/params temperature 0.2
```

File references attach files to the message without rewriting the user's text.

```text
summarize @README.md and @"docs/file with spaces.md"
```

Rules:

- `@path` and `@"path with spaces"` are supported.
- Email-style `name@example.com` is not treated as a file reference.
- Relative paths resolve from the session cwd.
- Directories, binary files, unreadable files, and oversized files become warnings.
- Multiple attachment warnings are consolidated into a single block rather than scattered inline.

Shell commands run only when the trimmed input starts with `!`.

```text
!pytest tests/test_cli.py
```

Rules:

- Inline `!` inside a normal prompt is not expanded.
- User-typed shell commands do not invoke the model automatically.
- Agent-requested shell commands must go through the approval layer unless the current policy auto-approves them.
- User-typed shell commands that match a list of destructive patterns (e.g. `rm -rf`, `git reset --hard`, `git push --force`, `DROP TABLE`) require an explicit confirmation before running.

## Slash Commands

Full command set:

| Command | Effect |
|---|---|
| `/help` (alias `/h`, `/?`) | List all commands with descriptions, examples, and keyboard shortcuts |
| `/quit` (alias `/q`) | Exit the chat loop |
| `/model [id]` (alias `/m`) | Show or set the session model override |
| `/tier [scout\|worker\|architect]` (alias `/t`) | Show or set preferred model tier |
| `/mode [autopilot\|hitl]` | Show or set session run mode |
| `/approval [interactive\|auto-edit\|yolo]` | Show or set approval policy |
| `/params [key value]` | Show or set model parameters (`temperature`, `max_tokens`) |
| `/context` (alias `/x`) | Show run ID, cwd, message count, and attached files |
| `/cost` | Show current run cost summary |
| `/history [n]` | Show last `n` messages from the session (default: 10) |
| `/save [file]` | Export session transcript as Markdown |
| `/undo` | Revert the last file edit applied during this session |
| `/clear` | Clear the terminal screen (session state unchanged) |
| `/capabilities` | Show active model, tier, mode, approval policy, tools, MCP servers, and skills |

Aliases are single-letter shortcuts for the most common commands. Most slash
commands mutate `SessionConfig` or local session state. Model-routed slash commands
must be explicitly listed in this contract and handled as first-class runtime flows.

Avoid cosmetic commands until the backend has enough behavior to justify them.

## Input Completion and Navigation

The prompt_toolkit session provides:

- **Tab completion** for slash command names (with one-line description), slash command arguments (`/tier`, `/mode`, `/approval`, `/params`), and `@`-prefixed file paths.
- **Reverse history search** via Ctrl+R (prompt_toolkit default emacs binding).
- **Multiline input** via Esc+Enter.
- **Bottom toolbar** showing the current `model · tier · mode · approval` at all times.
- **Placeholder hint** `(esc+enter for newline)` when the input buffer is empty.
- **History deduplication**: consecutive identical entries are not written to disk.

## Approval Semantics

Approval requests include:

- request id
- action kind
- summary
- risk level
- structured details
- allowed response options
- optional tool name
- optional exact action key

Responses:

- `approve_once` — allow this action once.
- `deny` — deny this action.
- `allow_tool_for_session` — allow this tool for the remainder of the session.
- `allow_exact_for_session` — allow this exact action for the remainder of the session.
- `redirect` — deny execution but return the user's feedback message as the tool result so the model can adjust its approach and retry without prompting again.

The approval prompt uses a prompt_toolkit `PromptSession` with:

- Arrow keys (↑/↓) or Ctrl+P/N to navigate the option list.
- Enter to confirm the highlighted option.
- Single-letter shortcuts (`a`, `d`, `r`, `t`, `e`) to select directly without Enter.
- Ctrl+C or Ctrl+D defaults to deny.

When `redirect` is selected, a second prompt collects the feedback text before returning.

Interrupted or invalid approval prompts default to deny.

`yolo` mode can be used for benchmarking, but it must be explicit and visible.

## Question Semantics

Question requests are used for HITL checkpoints and agent clarifications.

Supported forms:

- free-text question
- single-choice question
- multi-choice question

Answers are returned as structured values so a future browser UI or A2A server can implement the same contract without terminal-specific parsing.

## Rendering Rules

Use Rich for clarity, not decoration.

- Buffer streamed assistant text and render it as Markdown at phase boundaries.
- Show tool starts and completions compactly.
- Show file edits as syntax-highlighted unified diffs.
- Show shell commands with cwd, timeout, exit code, stdout, and stderr.
- Truncate large outputs with head and tail preserved.
- Show warnings without stopping the run unless the runtime says the run failed.
- Show cost summaries inline as a compact one-liner (`↳ N tok · $X.XXXX`).
- Machine-readable output should stay separate from interactive rendering.

## Safety Behaviours

- **Destructive shell detection**: user-typed `!` commands matching known destructive patterns prompt for confirmation before execution.
- **File edit undo**: `ChatApp` maintains an in-memory undo stack (up to 20 entries). `FileEditPreviewed` events snapshot the original file bytes before any write. `/undo` restores from the stack.
- **Retry on failure**: when `RunFailed` fires, the CLI offers "Retry? [y/N]" before returning to the prompt.
- **Resume context preview**: `jac resume` shows the last three turn pairs from the prior session before entering the loop.

## Extension Checklist

When adding a new CLI feature:

1. Decide whether it is a runtime behavior or a presentation behavior.
2. Add runtime events or request types first if the backend needs to communicate it.
3. Keep slash commands as session/runtime mutations by default. If a slash command
   is model-routed, document it explicitly in this contract and roadmap entry.
4. Add focused parser, policy, or renderer tests before wiring the REPL.
5. Avoid adding UI features that do not improve control, transparency, or debugging.
