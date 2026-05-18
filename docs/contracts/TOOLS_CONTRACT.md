# Tools Contract

> **Status:** Locked · **Last revised:** 2026-05-18 · **Type:** contract

This document is the authoritative reference for writing agent tools in JAC.
Read it before adding a new tool. Any tool that doesn't follow this contract will be
rejected at code review.

---

## What Is a Tool?

A tool is a plain async Python function that gives an agent a capability (read a file,
run a shell command, search code, mutate the run's task list). Tools live in `src/jac/tools/`.

Tools are **not** agents. They do not call LLMs, loop, or make decisions. They perform
one well-defined operation and return a structured result.

There are two flavors:

- **Stateless tools** — file / shell. Plain async functions whose only inputs are the
  LLM-supplied kwargs. The factory passes them to `Agent.from_file(tools=[...])` as-is.
- **Stateful tools** — task-CRUD. They take Pydantic AI's
  `RunContext[ScottDeps]` as the first positional argument so the runtime can
  inject the per-run `run_id` and a repo handle. See **Stateful tools
  (`RunContext[Deps]`)** below.

Both styles carry the same `.approval: ToolApprovalMeta` attribute and are wrapped by
the same factory-side approval middleware.

---

## File Layout

```
tools/
  types.py          ← shared result types, approval metadata, ScottDeps
  filesystem.py     ← file read / write / search tools
  shell.py          ← shell execution + background process registry
  tasks.py          ← task-CRUD tools (RunContext[ScottDeps])
  __init__.py       ← TOOL_REGISTRY — the only import surface for the factory
```

Future tools go in a new file under `tools/`. Register them in `TOOL_REGISTRY` in
`__init__.py`. Do **not** add them directly to existing files unless they are closely
related to that file's concern.

---

## Return Type Contract

Every tool must return a subtype of `ToolResult` (from `tools/types.py`).

```python
class ToolResult(BaseModel):
    status: ToolStatus = ToolStatus.OK
    warnings: list[str] = []
    error: str | None = None  # populated only when status != OK
```

### Status codes

| Status | When to use |
|---|---|
| `OK` | Operation completed successfully |
| `ERROR` | Operation failed for a recoverable reason (bad input, write failed) |
| `NOT_FOUND` | Target path or resource does not exist |
| `PERMISSION_DENIED` | OS rejected the operation |
| `TIMEOUT` | Shell command exceeded its timeout |
| `TRUNCATED` | Partial success — output was cut to fit size limits |

### Result subtypes

Define a concrete subtype per tool group in `tools/types.py`. Add the payload fields
that the LLM and the runtime need to interpret the result. Avoid giant flat structs —
if a tool has multiple sub-operations, model them as nested types.

```python
class MyToolResult(ToolResult):
    output_field: str = ""
    count: int = 0
```

Pydantic serializes this to JSON for the agent. Design the fields so the JSON is
readable — the LLM sees it directly.

### Error handling rules

- Never raise exceptions out of a tool function. Catch `OSError`, `PermissionError`,
  and any other expected failures and return a result with the appropriate status.
- Set `error` to a short, human-readable message. The agent and the approval UI both
  display it.
- `warnings` is for non-fatal observations (truncation, encoding fallbacks). A result
  with warnings still has `status = OK`.

---

## Approval Metadata Contract

Every tool function **must** have a `.approval` attribute of type `ToolApprovalMeta`.
This is how the approval wrapper knows what to show the user and what to auto-approve.

```python
from jac.tools.types import ToolApprovalMeta, RiskLevel

async def my_tool(path: str, content: str) -> MyToolResult:
    ...

setattr(my_tool, "approval", ToolApprovalMeta(
    category="file_write",          # file_read | file_write | shell
    risk_level=RiskLevel.LOW,
    reversible=True,
    description_fn=lambda path, content="", **_: f"Write {len(content):,} bytes → `{path}`",
))
```

### Risk levels

| Level | Meaning | Auto-approved in |
|---|---|---|
| `READ_ONLY` | No side effects | All modes |
| `LOW` | Reversible local change (edit a file) | `auto-edit`, `yolo` |
| `MEDIUM` | Harder to reverse (create new file) | `yolo` |
| `HIGH` | Significant side effects (shell command) | `yolo` only |

The `ApprovalPolicy` in `runtime/approvals.py` owns the mode-to-risk-level mapping.
Tools declare their risk level; the policy decides the gate.

### `description_fn`

A callable that receives the tool's call-time kwargs and returns a human-readable
action string for the approval prompt.

```python
# Pattern: lambda with the tool's known params + **_ to absorb extras
description_fn=lambda command, cwd=None, **_: (
    f"Run `{command}`" + (f" in `{cwd}`" if cwd else "")
)
```

- Keep it short — one line, under ~80 chars
- Prefer showing the key argument (path, command) over showing all arguments
- Use f-string formatting — it is rendered in a Rich panel in the terminal

### Timeout policy

`ToolApprovalMeta.timeout_seconds` can override the agent default tool timeout.

| Tool | Timeout |
|---|---|
| Agent default (`Agent(tool_timeout=...)`) | 180s |
| `read_file` | 30s |

`None` means "use the agent default". The approval wrapper enforces the timeout via
`asyncio.wait_for` and returns `ToolResult(status=TIMEOUT, error=...)` on overrun —
the tool itself never sees the cancellation.

---

## Tool Registry

`tools/__init__.py` exports `TOOL_REGISTRY`, a dict that maps group names to lists of
tool functions. `build_agent` in `agents/base.py` uses this to resolve the
`allowed_tools` JSON array from `agent_configs` (or the M1 default
`["filesystem", "shell", "tasks"]` until the Runtime slice writes the agent_configs
row at run start).

```python
TOOL_REGISTRY: dict[str, list[ToolFn]] = {
    "filesystem":      [read_file, write_file, edit_file, list_directory, search_files, grep_files],
    "filesystem:read": [read_file, list_directory, search_files, grep_files],
    "shell":           [run_shell, run_shell_background, read_process_output],
    "shell:read":      [read_process_output],
    "tasks":           [add_task, update_task, complete_task, list_tasks],
}
```

### Naming conventions

| Entry name | Meaning |
|---|---|
| `"filesystem"` | Full read + write access |
| `"filesystem:read"` | Read-only subset |
| `"shell"` | Shell execution |
| `"shell:read"` | Read-only subset (output polling only) |
| `"tasks"` | Task-list CRUD (requires `deps_type=ScottDeps`) |
| `"mcp:<name>"` | Resolved via `mcp_servers` table — not in this registry |

To add a new tool group: add it to `TOOL_REGISTRY` in `__init__.py`. The key becomes
a valid value in `agent_configs.allowed_tools`.

---

## Checklist for a New Tool

Before merging, verify all of the following:

- [ ] Returns a subtype of `ToolResult` defined in `tools/types.py`
- [ ] Never raises exceptions — all failure paths return `ToolResult(status=ERROR, error=...)`
- [ ] Has a `.approval` attribute of type `ToolApprovalMeta` with correct `risk_level`
- [ ] `description_fn` uses `**_` to absorb unexpected kwargs
- [ ] Added to `TOOL_REGISTRY` in `tools/__init__.py`
- [ ] Type-checks cleanly (`just typecheck`)
- [ ] Has at least one test covering the happy path and one failure path

---

## Complete Example

```python
# tools/example.py

from pathlib import Path

from jac.tools.types import (
    RiskLevel,
    ToolApprovalMeta,
    ToolResult,
    ToolStatus,
)
from pydantic import BaseModel


class CountLinesResult(ToolResult):
    path: str = ""
    line_count: int = 0


async def count_lines(path: str) -> CountLinesResult:
    """Count the number of lines in a text file."""
    p = Path(path)
    try:
        count = len(p.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return CountLinesResult(status=ToolStatus.NOT_FOUND, error=f"file not found: {path}", path=path)
    except OSError as exc:
        return CountLinesResult(status=ToolStatus.ERROR, error=str(exc), path=path)
    return CountLinesResult(path=path, line_count=count)


setattr(count_lines, "approval", ToolApprovalMeta(
    category="file_read",
    risk_level=RiskLevel.READ_ONLY,
    reversible=True,
    description_fn=lambda path, **_: f"Count lines in `{path}`",
))
```

Then register it:

```python
# tools/__init__.py
from jac.tools.example import count_lines

TOOL_REGISTRY = {
    ...,
    "count_lines": [count_lines],
}
```

---

## Background Process Pattern

Some tasks require starting a long-running process (a dev server, a test watcher) and
then interacting with it. Use the three-step pattern:

```python
# 1. Start the process
proc = await run_shell_background("python -m http.server 8080", cwd="/tmp/project")
# → BackgroundProcessResult(process_id="a3f9c12b", ...)

# 2. Poll until ready
output = await read_process_output(proc.process_id)
# → ProcessOutputResult(stdout="Serving HTTP on ...", running=True, exit_code=None)

# 3. Do work against the running process...
result = await run_shell("curl -s http://localhost:8080/")

# 4. Inspect all background processes if needed
procs = await list_processes()
```

Background processes live only as long as the harness session. Their stdout/stderr are
written to temp files so polling `read_process_output` always returns the full accumulated
output, not just new bytes since the last read.

---

## Stateful tools (`RunContext[Deps]`)

Some tools need per-run state that the LLM cannot supply — the `run_id` to scope
their writes against, a database handle, or other runtime-only dependencies. JAC
uses Pydantic AI's `RunContext[DepsT]` for this; the runtime coordinator builds
a deps instance once per run and passes it to `agent.run(..., deps=...)`.

For Scott, deps live in `jac.tools.types.ScottDeps`:

```python
@dataclass(frozen=True, slots=True)
class ScottDeps:
    run_id: str
    tasks_repo: TasksRepo  # typed as Any in source to avoid state→tools cycle
```

A stateful tool takes `ctx: RunContext[ScottDeps]` as its first positional argument,
then its normal kwargs. The `.approval` metadata attaches the same way as a stateless
tool — the wrapper accepts `*args, **kwargs` and forwards `ctx` through `fn(*args, **kwargs)`.

```python
async def add_task(
    ctx: RunContext[ScottDeps],
    title: str,
    description: str = "",
) -> TaskResult:
    deps = ctx.deps
    try:
        row = await deps.tasks_repo.create(
            run_id=deps.run_id,
            title=title,
            description=description,
            status="pending",
        )
    except Exception as exc:  # noqa: BLE001 — tools must not raise
        return TaskResult(status=ToolStatus.ERROR, error=str(exc))
    return TaskResult(task=_row_to_info(row))


setattr(add_task, "approval", ToolApprovalMeta(
    category="task",
    risk_level=RiskLevel.LOW,
    reversible=True,
    description_fn=lambda title="", **_: f"Add task: `{title}`",
))
```

Important rules:

- **Tools never construct deps themselves.** `agent.run(deps=...)` is the only
  injection point. Unit tests build a real `RunContext` (no mocks) via the
  `task_ctx` fixture in `tests/tools/conftest.py`.
- **`description_fn` receives only kwargs**, not `ctx`. The wrapper strips
  the positional `ctx` before calling the description function.
- **The dependency direction stays inward.** `tools/` may not import from
  `runtime/`, `agents/`, or `cli/`. `ScottDeps.tasks_repo` is typed as `Any`
  in `tools/types.py` so the tools module doesn't import `jac.state`.

---

## What Tools Must NOT Do

- Call LLMs or make network requests (use MCP tools for that)
- Import from `cli/`, `runtime/`, or `agents/` — tools are at the bottom of the
  dependency chain
- Print to stdout or log directly — emit side effects through the return value only
- Maintain state across calls — tools are stateless; state lives in the `state/` layer
