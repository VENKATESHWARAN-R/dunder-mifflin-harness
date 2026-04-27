# CLI Design

## Purpose

The harness CLI is a terminal adapter over the agent runtime. It should make local development runs observable and controllable without letting terminal concerns leak into the backend.

The backend owns agent behavior, workflow routing, tool execution, model selection, state persistence, and future A2A/browser integrations. The CLI owns process arguments, interactive input, rendering, and collecting human responses.

## Library Roles

- Click owns process-level entrypoints, options, exit codes, and script integration.
- prompt_toolkit owns interactive input, history, multiline behavior, and future completions.
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

Initial command surface:

- `harness "say hi"` runs a one-shot prompt.
- `harness chat` starts the interactive REPL.
- `harness run --mode autopilot|hitl "task"` is the future workflow-oriented shape.
- `harness resume <run-id>` is reserved for persistent runs.
- `harness config` is reserved for resolved non-secret settings.

Until graph workflows exist, one-shot and chat mode use the same runtime coordinator around the current Pydantic AI agent.

## Input Grammar

Plain text is sent to the runtime as a user message.

Slash commands are local commands. They configure the session or inspect state and are not sent to the model.

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
- Directories, binary files, unreadable files, and oversized files become visible warnings.

Shell commands run only when the trimmed input starts with `!`.

```text
!pytest tests/test_cli.py
```

Rules:

- Inline `!` inside a normal prompt is not expanded.
- User-typed shell commands do not invoke the model automatically.
- Agent-requested shell commands must go through the approval layer unless the current policy auto-approves them.

## Slash Commands

The first practical command set is intentionally small:

- `/help`: list available commands.
- `/quit`: exit the chat loop.
- `/model [model-id]`: show or set the session model override.
- `/tier [scout|worker|architect]`: show or set the preferred model tier.
- `/mode [autopilot|hitl]`: show or set session mode before a run.
- `/approval [interactive|auto-edit|yolo]`: show or set approval policy.
- `/params [key value]`: show or set safe model parameters.
- `/context`: show current cwd and attached context summary.
- `/cost`: show current run cost summary when the runtime reports one.

Avoid cosmetic commands until the backend has enough behavior to justify them.

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

Initial responses:

- `approve_once`: allow this action once.
- `deny`: deny this action.
- `allow_tool_for_session`: allow this tool for the session.
- `allow_exact_for_session`: allow this exact action for the session.

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
- Show cost summaries at run end, `/cost`, or model escalation.

Machine-readable output should stay separate from interactive rendering.

## Extension Checklist

When adding a new CLI feature:

1. Decide whether it is a runtime behavior or a presentation behavior.
2. Add runtime events or request types first if the backend needs to communicate it.
3. Keep slash commands as session/runtime mutations, not hidden backend shortcuts.
4. Add focused parser, policy, or renderer tests before wiring the REPL.
5. Avoid adding UI features that do not improve control, transparency, or debugging.
