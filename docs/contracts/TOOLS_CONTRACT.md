# Tools Contract

> **Status:** Locked · **Last revised:** 2026-04-30 · **Type:** contract

This document is the authoritative reference for writing agent tools in JAC.
Read it before adding a new tool. Any tool that doesn't follow this contract will be
rejected at code review.

---

## What Is a Tool?

A tool is a plain async Python function that gives an agent a capability (read a file,
run a shell command, search code). Tools live in `src/jac/tools/`.

Tools are **not** agents. They do not call LLMs, loop, or make decisions. They perform
one well-defined operation and return a structured result.

---

## File Layout

```
tools/
  types.py          ← all shared result types and approval metadata type
  filesystem.py     ← file read/write/search tools  (also has CLI attachment helpers)
  shell.py          ← shell execution tool           (also has CLI shell executor)
  __init__.py       ← TOOL_REGISTRY — the only import surface for config_loader
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

---

## Tool Registry

`tools/__init__.py` exports `TOOL_REGISTRY`, a dict that maps group names to lists of
tool functions. `config_loader` in `agents/base.py` uses this to resolve the
`allowed_tools` JSON array from `agent_configs`.

```python
TOOL_REGISTRY: dict[str, list[ToolFn]] = {
    "filesystem":       [read_file, write_file, edit_file, list_directory, search_files, grep_files],
    "filesystem:read":  [read_file, list_directory, search_files, grep_files],
    "shell":            [run_shell],
}
```

### Naming conventions

| Entry name | Meaning |
|---|---|
| `"filesystem"` | Full read + write access |
| `"filesystem:read"` | Read-only subset |
| `"shell"` | Shell execution |
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

## Future: Sub-Agent Toolgroup

The harness will eventually support spawning custom sub-agents from within an agent tool
call. This is distinct from workflow-level delegation (which goes through `pydantic_graph`
edges). Sub-agents are used for:

- **Context isolation**: a sub-agent starts with a clean message history, preventing the
  parent's long history from bleeding into a focused sub-task.
- **Parallel sub-tasks**: multiple sub-agents run concurrently on independent tasks and
  return structured results to the parent.

When implemented, this will be a future `agents` toolgroup in `TOOL_REGISTRY`:

```python
"agents": [spawn_agent],  # spawn_agent(role, task, context_scope) -> AgentResult
```

The `spawn_agent` tool will go through the same `config_loader` factory as workflow agents
and will be tracked in the `attempts` table. Do **not** add ad-hoc agent instantiation
anywhere else in anticipation of this.

---

## What Tools Must NOT Do

- Call LLMs or make network requests (use MCP tools for that)
- Import from `cli/`, `runtime/`, or `agents/` — tools are at the bottom of the
  dependency chain
- Print to stdout or log directly — emit side effects through the return value only
- Maintain state across calls — tools are stateless; state lives in the `state/` layer
