> **Status:** Reference · **Last revised:** 2026-05-03 · **Type:** developer documentation

# Agents Layer

The agents layer lives in `src/jac/agents/`. It is the **single site** where `pydantic_ai.Agent(...)` is called. No other module in the codebase constructs a Pydantic AI agent.

See also: [`docs/contracts/MCP_INTEGRATION.md`](../contracts/MCP_INTEGRATION.md) — binding spec for the config_loader contract. [`docs/contracts/TOOLS_CONTRACT.md`](../contracts/TOOLS_CONTRACT.md) — required reading before adding a tool.

## The single Agent() site principle

All agent construction, tool wiring, MCP toolset building, and model selection converge in `src/jac/agents/base.py`. This means:

- Adding a new provider only requires editing `src/jac/runtime/models.py` — not every caller.
- Adding a new tool entry only requires editing `src/jac/tools/` and `TOOL_REGISTRY` — not `config_loader`.
- A caller can unit-test agent configuration by inspecting the `agent_configs` row and `TOOL_REGISTRY` — no need to construct a live agent.

## `config_loader` (`src/jac/agents/base.py`)

The factory function. Signature:

```python
async def config_loader(
    *,
    state: StateStore,
    settings: Settings,
    run_id: str,
    role: str = "chat",
    output_type: type | None = None,
    events: EventBus | None = None,
    model_settings: dict | None = None,
) -> Agent: ...
```

Five-step construction flow:

**Step 1 — Load agent config**

Calls `state.agent_configs.get_by_run_and_role(run_id, role)`. Raises `AgentConfigNotFound` if no row exists. The row contains `model_tier`, `model_override`, `system_prompt`, `allowed_tools` (decoded from JSON to `list[str]`), and `max_context_tokens`.

**Step 2 — Build MCP toolsets**

Queries `state.run_mcp_servers` for active rows matching `run_id` and `role` (or `agent_role IS NULL` for role-agnostic entries). For each server, reads the `mcp_servers` row and constructs the appropriate toolset:

- `transport = "stdio"` → `MCPServerStdio`
- `transport = "streamable-http"` → `MCPServerStreamableHTTP`
- `transport = "sse"` → `MCPServerSSE`

**Step 3 — Resolve allowed tools**

Iterates `allowed_tools` and looks up each entry in `TOOL_REGISTRY`. Entries prefixed with `mcp:` are skipped (handled in step 2). An entry not found in `TOOL_REGISTRY` raises `UnknownToolError`. The result is a flat list of tool functions to pass to the agent.

**Step 4 — Compose system prompt with skills**

Queries `state.run_skills` for active rows matching `run_id` and `role` (or `agent_role IS NULL`), ordered by `domain ASC, name ASC`. The skill contents are appended to the system prompt under a `## Domain Knowledge` header, each skill as a named subsection. If no skills are active, the system prompt is used unchanged.

**Step 5 — Build model and return Agent**

Calls `settings.resolve_model_selection(model_override=..., tier=...)` to get a `ModelSelection`, then `build_pydantic_model(selection, settings)` to get a pydantic_ai `Model`. Finally constructs and returns:

```python
Agent(
    model=model,
    instructions=composed_system_prompt,
    tools=resolved_tools,
    toolsets=mcp_toolsets,
    output_type=output_type,
    model_settings=model_settings,
)
```

If an `EventBus` is provided, `config_loader` emits `NodeStarted(node_name="config_loader")` before step 1 and `NodeCompleted` after step 5.

## `ensure_default_run_config` (`src/jac/agents/seeds.py`)

Signature:

```python
async def ensure_default_run_config(
    state: StateStore,
    run_id: str,
    *,
    role: str = "chat",
    system_prompt: str = DEFAULT_CHAT_PROMPT,
    allowed_tools: list[str] | None = None,
    model_tier: str = "worker",
    model_override: str | None = None,
) -> AgentConfig: ...
```

Idempotent: returns the existing row if `(run_id, role)` already exists in `agent_configs`, otherwise creates one with the provided defaults. Called by `RunCoordinator._ensure_agent` before the first turn of every session. Safe to call multiple times.

## `AgentConfig` dataclass (`src/jac/agents/base.py`)

The in-memory representation decoded from an `AgentConfigRow`:

```python
@dataclass(frozen=True, slots=True)
class AgentConfig:
    config_id: str
    run_id: str
    role: str
    model_tier: str
    model_override: str | None
    system_prompt: str
    allowed_tools: list[str]   # decoded from JSON in the DB row
    max_context_tokens: int | None
    created_at: datetime
    updated_at: datetime
```

## `TOOL_REGISTRY` (`src/jac/tools/__init__.py`)

Maps `allowed_tools` entry names to lists of tool functions. Used exclusively by `config_loader` step 3.

| Entry | Included tools |
|---|---|
| `"filesystem"` | `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep_files` |
| `"filesystem:read"` | `read_file`, `list_directory`, `search_files`, `grep_files` |
| `"shell"` | `run_shell`, `run_shell_background`, `list_processes`, `read_process_output` |
| `"shell:read"` | `list_processes`, `read_process_output` |

Colon-separated entries (`filesystem:read`, `shell:read`) are read-only subsets of the full tool groups. To add a new tool group: implement the tool functions in `src/jac/tools/`, add an entry to `TOOL_REGISTRY`, and add a row to [`docs/contracts/TOOLS_CONTRACT.md`](../contracts/TOOLS_CONTRACT.md).

## SDK independence

`src/jac/agents/` has no dependency on `jac.cli.*` or `jac.runtime.coordinator`. Its allowed imports are:

- `jac.config` — `Settings`
- `jac.state` — `StateStore` and repo types
- `jac.tools` — `TOOL_REGISTRY`
- `jac.runtime.events` — `EventBus`, `NodeStarted`, `NodeCompleted`
- `jac.runtime.models` — `build_pydantic_model`, `ModelSelection`

This means the agent factory can be used from a server, a test, or a notebook with no TTY or CLI context.

## Events

When an `EventBus` is passed to `config_loader`:

| Event | When |
|---|---|
| `NodeStarted(node_name="config_loader")` | Before step 1 |
| `NodeCompleted(node_name="config_loader")` | After step 5, agent returned |

The `events` parameter is optional. If `None`, no events are emitted and the factory works identically.
