# MCP Integration

> **Status:** Locked · **Last revised:** 2026-05-02 · **Type:** contract

## Purpose

This document defines how MCP servers and skills are loaded from the state store and
wired into Pydantic AI agents at instantiation time. It is the contract for the
`config_loader` node — the only place where database configuration translates to
live agent instances.

---

## Principles

- Agent code never imports MCP server configs directly. It receives a fully-built agent.
- MCP connections are opened inside `async with agent:` — the agent lifecycle owns connect/disconnect.
- Skills are injected into the system prompt as a composed block, not as separate instructions.
- Changing which MCP servers or skills are active means rebuilding the agent for the next turn.
  Message history carries over; the tool definition block and system prompt change.
- The registry and wiring code exist from day one; remote MCP transports come online at C17
  by registering rows in `mcp_servers` without touching agent code. Until then, only local
  tools are resolved through `allowed_tools`.

---

## Transport Mapping

The `mcp_servers.transport` field maps to a Pydantic AI class:

| `transport` value | Pydantic AI class | Config fields used |
|---|---|---|
| `stdio` | `MCPServerStdio` | `command`, `args`, `env` (optional) |
| `http` | `MCPServerStreamableHTTP` | `url` |
| `sse` | `MCPServerSSE` | `url` (legacy — prefer `http`) |

`MCPServerStdio` launches a subprocess. `MCPServerStreamableHTTP` connects to a running server.
Both are passed to `Agent(toolsets=[...])` and started/stopped via the agent's context manager.

### Example — config rows to toolsets

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
import json

def build_mcp_toolsets(server_rows: list[dict]) -> list:
    toolsets = []
    for row in server_rows:
        cfg = json.loads(row["config"])
        if row["transport"] == "stdio":
            toolsets.append(
                MCPServerStdio(cfg["command"], args=cfg.get("args", []), env=cfg.get("env"))
            )
        elif row["transport"] in ("http", "sse"):
            cls = MCPServerStreamableHTTP if row["transport"] == "http" else MCPServerSSE
            toolsets.append(cls(cfg["url"]))
    return toolsets
```

---

## `allowed_tools` Resolution

`agent_configs.allowed_tools` is a JSON array that can contain two kinds of entries:

| Entry format | Meaning |
|---|---|
| `"filesystem"` | Local Python tool by name — resolved to a `Tool(fn)` object |
| `"mcp:playwright"` | MCP server by name — looked up in `mcp_servers` table |

The `config_loader` node resolves this array in two passes:
1. Strip `mcp:` prefix entries → query `mcp_servers WHERE name = ?` → build MCP toolsets
2. Plain name entries → look up in the local tool registry → collect `Tool(fn)` objects
3. Both lists are merged into `toolsets=[...]` and `tools=[...]` on the agent

This means adding a new MCP server is: register it in `mcp_servers`, add `"mcp:server-name"`
to the relevant `agent_configs.allowed_tools`, and it's live on the next run.

---

## Skills Injection

Skills are injected into the agent's system prompt as a composed block appended after the
role-specific instructions.

```python
def compose_system_prompt(base_prompt: str, skill_rows: list[dict]) -> str:
    if not skill_rows:
        return base_prompt
    skills_block = "\n\n---\n".join(row["content"] for row in skill_rows)
    return f"{base_prompt}\n\n## Domain Knowledge\n\n{skills_block}"
```

Skills are ordered by `skills.domain` (general skills first, domain-specific skills last)
so general conventions do not override domain-specific ones.

---

## Agent Instantiation Flow (config_loader node)

```
config_loader receives: run_id, agent_role
    │
    ├── 1. Query agent_configs WHERE run_id + role → base config
    │
    ├── 2. Query run_mcp_servers JOIN mcp_servers
    │       WHERE run_id AND (agent_role = role OR agent_role IS NULL) AND enabled = 1
    │       → build MCP toolsets via transport mapping
    │
    ├── 3. Resolve allowed_tools local names → Tool(fn) list from local tool registry
    │
    ├── 4. Query run_skills JOIN skills
    │       WHERE run_id AND (agent_role = role OR agent_role IS NULL) AND enabled = 1
    │       ORDER BY domain (general first)
    │       → compose system prompt with skills block
    │
    └── 5. Instantiate Agent(
                model=resolved_model,        # tier default or model_override
                instructions=composed_prompt,
                tools=[...local tools...],
                toolsets=[...mcp toolsets...]
            )
```

The returned agent is used inside `async with agent:` so MCP connections open/close correctly.

---

## Model Tier Resolution

`agent_configs.model_tier` is the default; `model_override` takes precedence if set.
Tier → model mapping is defined in `config.py` and can be changed without touching agent code.

```python
TIER_DEFAULTS = {
    "scout":     "anthropic:claude-haiku-4-5",
    "worker":    "anthropic:claude-sonnet-4-6",
    "architect": "anthropic:claude-opus-4-7",
}

def resolve_model(tier: str, override: str | None) -> str:
    return override if override else TIER_DEFAULTS[tier]
```

---

## Mid-Run Toggle (C18)

When a user runs `/disable mcp:playwright` mid-conversation:

1. Set `run_mcp_servers.enabled = 0`, `toggled_at = now()` in DB
2. Emit `MCPServerToggled` event so the UI can acknowledge it
3. On the next agent turn, `config_loader` re-runs → builds agent without that toolset
4. Pass `previous_result.all_messages()` as message history → context is preserved
5. The tool definition block changes → partial prompt-cache miss on this turn

This is a deliberate user action, not automatic. The harness never auto-disables MCP servers
based on token pressure — that policy choice is intentional and not on the roadmap.

---

## Source of Truth: File → DB

MCP server rows in the `mcp_servers` table are populated from JSON files in
`~/.jac/mcp/` (user globals) and `<repo>/.agents/mcp/` (project scope).
Files are the source of truth; the DB is the resolved index. See
`docs/contracts/WORKSPACE.md` for the file format and seeding policy.

Adding an MCP server is therefore: write `~/.jac/mcp/<name>.json` (or the
project equivalent), restart the harness, reference it as `mcp:<name>` in
`agent_configs.allowed_tools`. No code change required.

---

## Initial Tool Set (through C11)

Until remote MCP integration lands at C17, only local tools are resolved through
`allowed_tools`:

| Tool name | First shipped by | Description |
|---|---|---|
| `filesystem` | C3 | Read, write, edit files within the project sandbox |
| `shell` | C4 | Run shell commands (subject to approval policy) |
| `git` | C4 | Git operations: commit, status, diff, log |

These are registered as Python `Tool(fn)` objects in a local tool registry. They satisfy
`allowed_tools` entries without any MCP overhead.

Remote MCP servers (Playwright, browser DevTools, etc.) come online at C17 by registering
rows in `mcp_servers` — no change to agent logic.
