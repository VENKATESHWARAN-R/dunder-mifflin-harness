# Project Philosophy

> **Status:** Reference · **Last revised:** 2026-05-02 · **Type:** principles

## Purpose

JAC is a harness for long-running agentic software development. The core product is not the terminal UI, a single agent, or a specific provider. The core product is the runtime system that can plan, route, execute, evaluate, persist state, and explain what happened.

Everything else is an adapter, tool, or workflow around that runtime.

## First Principle

Put code where its reason to change lives.

- If it changes because the terminal experience changes, it belongs in `cli/`.
- If it changes because a browser, A2A server, or CLI all need the same behavior, it belongs behind `runtime/`.
- If it changes because an agent role or workflow strategy changes, it belongs in future `agents/`, `nodes/`, `workflows/`, or `modes/`.
- If it changes because local execution changes, it belongs in `tools/` or a future execution/sandbox package.
- If it changes because persisted state changes, it belongs in a future state/store package, not in the CLI.

## Dependency Direction

Dependencies should flow inward:

```text
cli/ or servers/
  -> runtime/
    -> workflows/, modes/
      -> nodes/
        -> agents/
          -> tools/, state/, config
```

Outer layers may depend on inner layers. Inner layers must not import outer layers.

Examples:

- `cli/` may import `runtime.events`.
- `runtime/` must not import `cli.renderer`.
- `nodes/` may import `agents/` and `state/`.
- `agents/` must not import `nodes/` or `workflows/`.
- A future A2A server may import `runtime.RunCoordinator`.
- A future workflow runner must not know whether the user is in a terminal, browser, or remote agent session.

## Communication Model

Use typed events and explicit request/response objects across boundaries.

- Runtime emits events for progress, text, tools, files, shell commands, cost, warnings, and failures.
- Runtime requests approvals when a concrete action may have side effects.
- Runtime asks questions when it needs information to continue.
- UI or server adapters render those requests and return structured responses.

Do not use terminal prompts, Rich objects, Click contexts, or prompt_toolkit sessions outside `cli/`.

## Where New Code Goes

### New CLI Feature

Put terminal-only behavior in `src/jac/cli/`.

Use this for Click commands, slash commands, prompt_toolkit input, Rich rendering, and human prompt views. If the feature needs backend state or behavior, define that in `runtime/` first and let the CLI call it.

### New Runtime Behavior

Put UI-agnostic session behavior in `src/jac/runtime/`.

Use this for event types, request/response contracts, session configuration, approval policy, human questions, and the facade that coordinates runs.

### New Custom Agent Or Sub-Agent

Prefer a future `agents/` package for role definitions and agent construction.

Agent code should describe role, model tier, prompt/instructions, allowed tools, and structured outputs. It should not render UI, read terminal input, or decide how sessions are displayed.

If an agent is one step in a workflow, expose it through a node rather than calling it directly from the CLI.

**Critical rule:** `agents/base.py` (the config_loader) is the only place in the codebase that
instantiates live Pydantic AI `Agent` objects. It reads from `agent_configs`, resolves MCP
toolsets and skills from the state store, and returns a fully-built agent. Nothing else should
call `Agent(...)` directly. This keeps model tier resolution, MCP wiring, and skills injection
in one auditable location.

### New Node

Use a future `nodes/` package.

A node should be a small unit of work with a uniform state-in/state-out contract. It can call an LLM, inspect state, route tasks, execute tools, or evaluate results. It should report progress through runtime events instead of printing.

### New Workflow Or Mode

Use future `workflows/` and `modes/` packages.

Workflows wire nodes into directed graphs using `pydantic_graph`. Modes choose a workflow
composition and policy defaults. Autopilot and HITL should be separate compositions that share
nodes.

`RunCoordinator` (in `runtime/coordinator.py`) is the only entry point into the workflow layer.
It delegates to a `pydantic_graph` `Graph` runner. The CLI never imports workflow or graph code
directly.

### New Tool

Use `tools/` for local reusable capabilities and future tool wrappers.

Tools should expose clear inputs, outputs, side-effect descriptions, and approval metadata. File writes, shell commands, network calls, and sandbox actions should be approval-aware.

### MCP Server

There are two distinct MCP concerns — keep them separate:

- **Consuming MCP servers** (harness uses external MCP tools): handled in `agents/base.py` via
  `MCPServerStdio` / `MCPServerStreamableHTTP` toolsets. Configuration lives in the `mcp_servers`
  and `run_mcp_servers` state tables. See `docs/MCP_INTEGRATION.md`.

- **Exposing the harness as an MCP server** (external tools call the harness): put this adapter
  code in a future `servers/mcp/` package. It should expose runtime or tool capabilities through
  MCP, not reimplement agent logic. If it needs a capability that only exists in CLI code, move
  that capability inward first.

### A2A Server

Put A2A adapter code in a future `servers/a2a/` package.

It should communicate with the same runtime event/request contract as the CLI. Cross-agent session communication should go through runtime/session/state abstractions, not direct CLI hooks.

### Browser UI

Put browser-serving code in a future `ui/` or `servers/web/` package.

The browser UI should subscribe to runtime events and answer runtime requests. It should not duplicate workflow or agent code.

### Sandbox Or Execution Environment

Put sandbox abstractions in a future `execution/` package.

The tool layer should call execution interfaces instead of hardcoding local, Docker, or cloud behavior. Changing execution environment should not change agent logic.

### Persistent State

Put durable storage in a future `state/` package.

State should track runs, tasks, attempts, costs, configs, messages, and scoped context. CLI commands such as `/context`, `/cost`, or `resume` should read from this layer through runtime APIs.

## Design Rules

- Runtime contracts come before UI rendering.
- Side effects go through tools or execution services.
- Human interaction goes through approval or question requests.
- Cost, model choice, and escalation should be observable events, not hidden logs.
- Reference projects in `specimens/` are inspiration only.
- Prefer replacing in-progress abstractions over layering compatibility shims around unshipped code.

## Quick Placement Guide

```text
Terminal command?                        cli/
Slash command?                           cli/commands.py
Prompt parsing?                          cli/parser.py
Rich output?                             cli/renderer.py
Approval/question contract?              runtime/
Session config/state?                    runtime/
Coordinating a run?                      runtime/coordinator.py
Agent role definition?                   future agents/
Agent instantiation (config_loader)?     future agents/base.py  ← only place Agent() is called
Planner/build/evaluate step?             future nodes/
Graph wiring?                            future workflows/  (uses pydantic_graph)
Autopilot vs HITL selection?             future modes/
File/shell/local capability?             tools/
Consuming an MCP server (tool access)?   agents/base.py + state/mcp_servers
Exposing harness as MCP server?          future servers/mcp/
A2A protocol surface?                    future servers/a2a/
Sandbox backend?                         future execution/
Run/task/attempt persistence?            future state/
MCP server registry / skills registry?  future state/ (mcp_servers, skills tables)
```
