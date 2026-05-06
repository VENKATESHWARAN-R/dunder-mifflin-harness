> **Status:** Reference · **Last revised:** 2026-05-06 · **Type:** user guide

# Usage

This document covers all the ways to interact with JAC: one-shot mode, interactive chat, session resume, file attachments, inline shell execution, slash commands, and approval modes.

## One-shot mode

Pass a prompt directly on the command line. JAC runs a single turn, prints the response, and exits.

```bash
jac "explain what this repo does"
jac "write a one-liner to count lines in all .py files"
```

Useful for scripting or quick questions where you do not need a conversation.

## Chat mode

Start an interactive REPL with persistent conversation history:

```bash
jac chat
```

- Type a message and press **Enter** to send it.
- Press **Ctrl+C** to cancel an in-flight request without exiting.
- Press **Ctrl+D** or type `/quit` to exit the session.
- Use `/context` to view the run ID if you want to resume later.

The REPL supports multi-line input. The prompt_toolkit input session maintains persistent history across launches at `~/.jac/input_history`.

## Resuming a run

Every session is persisted in the state database. Resume the most recent run:

```bash
jac resume
```

Resume a specific run by ID:

```bash
jac resume <run_id>
```

Resuming restores the full message history from the database and continues the conversation where you left off.

## File attachments

Attach a file to your next message using the `@` prefix:

```
@src/jac/config.py explain the settings class
```

For paths with spaces, use quoted syntax:

```
@"my notes/design thoughts.md" summarise this
```

JAC reads the file, appends its contents as context, and sends everything to the model in one turn. Multiple attachments are supported in a single message:

```
@src/jac/runtime/coordinator.py @src/jac/agents/base.py how does the agent factory integrate with the coordinator?
```

**Size limit:** files above `JAC_MAX_ATTACHMENT_BYTES` (default 200,000 bytes / ~200 KB) are rejected with a warning. Adjust the limit via environment variable or `settings.json`.

## Inline shell

Run a shell command inside the REPL using the `!` prefix:

```
!ls -la
!git status
!uv run pytest tests/test_cli.py -x
```

The command runs in the terminal and its output is displayed inline. Inline shell output is **not** sent to the model — it is for your own reference. To share shell output with the model, copy it into a regular message or use the shell tool (if enabled by the agent config).

## Slash commands

Slash commands mutate your local session configuration. In current shipped
behavior they are dispatched locally (model-routed slash flows are planned for
later components).

| Command | Description |
|---|---|
| `/help` | Show all available slash commands |
| `/quit` | Exit the chat loop |
| `/model [name]` | Show the current model string, or set it to `name` |
| `/tier [scout\|worker\|architect]` | Show or set the model tier |
| `/mode [autopilot\|hitl]` | Show or set the run mode |
| `/approval [interactive\|auto-edit\|yolo]` | Show or set the approval mode |
| `/params [key value]` | Show all model params, or set a specific param (e.g. `/params temperature 0.2`) |
| `/context` | Show session context: run ID, working directory, attached files, message count |
| `/cost` | Show cost summary for the current session |
| `/history [n]` | Show the most recent `n` messages (default: 10) |
| `/save [file]` | Save the current session transcript as Markdown |
| `/undo` | Revert the last file edit applied in this session |
| `/clear` | Clear the terminal screen (session state unchanged) |
| `/capabilities` | Show active model/tier/mode/approval, tools, MCP servers, and skills |

### Examples

```
/tier architect
/model anthropic:claude-opus-4-5
/params temperature 0.1
/approval auto-edit
/context
```

Changes made with slash commands take effect on the next message. Changing the model or tier clears the cached agent so the new settings are picked up immediately.

## Approval modes

JAC agents can call tools (filesystem, shell) on your behalf. The approval mode controls how much you are prompted before each tool use.

| Mode | Behaviour |
|---|---|
| `interactive` | Prompts before every tool use. You approve or deny each action. Default mode. |
| `auto-edit` | Automatically approves file edits; still prompts for shell commands. |
| `yolo` | Automatically approves everything — no prompts. Use with care. |

Approval mode governs agent-requested tool calls. User-typed inline shell
commands (`!cmd`) are handled directly by the CLI and use destructive-command
confirmation instead.

Switch approval mode mid-session:

```
/approval auto-edit
```

Or set it as a default in your session config. The approval mode applies to the current session only; it is not persisted to the database.
