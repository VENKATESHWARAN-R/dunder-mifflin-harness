> **Status:** Reference · **Last revised:** 2026-05-07 · **Type:** developer/sdk guide

# SDK Guide

JAC exposes a Python SDK so you can embed the agent factory, run coordinator, and state layer in your own code — without a terminal or Click context.

## Installing as a library

```bash
pip install jac
# or
uv add jac
```

## The SDK seam

The stable integration point is `from jac import build_agent`. This is an alias for `config_loader` in `src/jac/agents/`. The factory is the only place that calls `pydantic_ai.Agent(...)`, so all model selection, tool wiring, MCP toolset construction, and system prompt composition flow through one function. Your code stays decoupled from internal implementation details.

## Minimal example

Build an agent for a run and call it:

```python
import asyncio
from jac.config import Settings
from jac.state import open_state_store
from jac.agents import config_loader, ensure_default_run_config

async def main():
    settings = Settings()
    state = await open_state_store("/tmp/my-state.db")

    # Create a run record
    await state.runs.create(run_id="my-run", prompt="initial")

    # Seed the default agent config for this run (idempotent)
    await ensure_default_run_config(state, "my-run")

    # Build the agent
    agent = await config_loader(state=state, settings=settings, run_id="my-run")

    async with agent:
        result = await agent.run("Say hello")

    print(result.output)
    await state.close()

asyncio.run(main())
```

`open_state_store` opens (or creates) the SQLite database and auto-runs any pending migrations. `ensure_default_run_config` seeds a default `agent_configs` row if one does not exist for `(run_id, role)` — it is safe to call repeatedly.

## `build_agent` shorthand

`from jac import build_agent` is the same function as `config_loader`:

```python
from jac import build_agent
from jac.config import Settings
from jac.state import open_state_store
from jac.agents import ensure_default_run_config

async def main():
    settings = Settings()
    state = await open_state_store("/tmp/my-state.db")
    await state.runs.create(run_id="my-run", prompt="initial")
    await ensure_default_run_config(state, "my-run")
    agent = await build_agent(state=state, settings=settings, run_id="my-run")
    async with agent:
        result = await agent.run("Say hello")
    print(result.output)
    await state.close()
```

Use whichever import form reads more clearly in your code.

## Configuring tools

Pass `allowed_tools` to `ensure_default_run_config` to control which tools the agent has access to:

```python
await ensure_default_run_config(
    state,
    "my-run",
    allowed_tools=["filesystem:read"],   # read-only file access
)
```

Available tool entries:

| Entry | Tools included |
|---|---|
| `"filesystem"` | `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep_files` |
| `"filesystem:read"` | `read_file`, `list_directory`, `search_files`, `grep_files` |
| `"shell"` | `run_shell`, `run_shell_background`, `list_processes`, `read_process_output` |
| `"shell:read"` | `list_processes`, `read_process_output` |

Combine entries:

```python
allowed_tools=["filesystem:read", "shell:read"]
```

## Using a model override

Force a specific model regardless of tier settings:

```python
await ensure_default_run_config(
    state,
    "my-run",
    model_override="anthropic:claude-sonnet-4-6",
)
```

Or use a specific tier with the default tier-to-model mapping from `settings.json`:

```python
await ensure_default_run_config(
    state,
    "my-run",
    model_tier="architect",
)
```

## Embedding `RunCoordinator` for event-driven usage

`RunCoordinator` wraps agent construction, message persistence, and event emission in one object. Use it when you want streaming output or need to react to tool calls and approvals:

```python
import asyncio
from jac.config import Settings
from jac.state import open_state_store
from jac.runtime.coordinator import RunCoordinator, UserMessage
from jac.runtime.events import EventBus, AgentTextDelta
from jac.runtime.session import SessionState, SessionConfig

async def main():
    settings = Settings()
    state = await open_state_store("/tmp/my-state.db")

    events = EventBus()
    events.on(AgentTextDelta, lambda e: print(e.text, end="", flush=True))

    session = SessionState(config=SessionConfig(cwd=Path(".")))
    coordinator = RunCoordinator(
        settings=settings,
        state=state,
        session=session,
        events=events,
    )

    output = await coordinator.submit_message(UserMessage(text="hello"))
    print()  # newline after streamed output
    await state.close()

asyncio.run(main())
```

Key events you can subscribe to:

| Event | When it fires |
|---|---|
| `AgentTextDelta` | Each token of streaming text output |
| `AgentMessageCompleted` | Model turn finished |
| `ToolCallRequested` | Agent is about to call a tool |
| `ToolCallCompleted` | Tool call returned |
| `ShellCommandStarted` | Shell tool started executing |
| `ShellCommandCompleted` | Shell tool finished |
| `FileEditPreviewed` | File edit is about to be applied |
| `FileEditApplied` | File edit was applied |
| `ApprovalRequested` | Agent is waiting for user approval |
| `SessionUsageUpdated` | Cumulative token usage + context headroom updated |
| `WarningRaised` | Non-fatal warning from runtime |

## No-CLI note

All SDK classes import cleanly without a TTY, a Click context, or any terminal dependency. The `src/jac/agents/`, `src/jac/runtime/`, `src/jac/state/`, and `src/jac/config.py` modules have no dependency on `src/jac/cli/`. You can run the agent factory in a server, a notebook, or a test suite with no side effects.
