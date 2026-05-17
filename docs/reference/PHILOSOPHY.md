# Project Philosophy

> **Status:** Reference · **Last revised:** 2026-05-08 · **Type:** principles
>
> Source-of-truth brainstorm: [`lab/brainstorm/2026-05-08-jac-reset-from-scratch.md`](../../lab/brainstorm/2026-05-08-jac-reset-from-scratch.md). When this file disagrees with the brainstorm, the brainstorm wins until this file is revised.

## What JAC is

> **JAC is a multi-agent CLI coding harness with two coupled goals.** As a *research harness*, it measures whether four fixed roles — **Michael Scott (manager)**, **Pam Beesly (planner)**, **Jim Halpert (builder)**, **Dwight Schrute (evaluator)** — with tiered model routing can build small applications autonomously at materially lower cost than a single-model baseline; the hypothesis is proven or disproven on the **Notes CLI** benchmark with a **single-function bug-fix** secondary. As a *daily tool*, it is a terminal coding assistant — chat, plan, build, evaluate, resume — backed by SQLite persistence, multi-provider tier routing with in-flight model swap, and a typed event/runtime contract that keeps the runtime independent of the terminal surface.

The four-persona cast is the **target architecture**. Phase 1 introduces personas incrementally, driven by run-log evidence — Scott alone in M1, others added when the data shows they're needed.

## The five principles

Each principle rules something *out*. Read them as decision rules, not aspirations.

### 1. The runtime is the contract; the UI is one surface.

The CLI is product, but anything that should outlive the terminal surface (browser, A2A, headless) lives in `runtime/`. UI may import runtime; runtime must not import UI.

**Rules out:** Rich rendering inside `runtime/`, slash-command logic that touches business state, `prompt_toolkit` references outside `cli/`, terminal prompts inside agent or tool code.

**Test:** if you can describe a non-CLI surface (browser, A2A peer, headless run) consuming this code, it belongs in `runtime/` or below; if it only makes sense at a terminal, it belongs in `cli/`.

### 2. Keep the spine thin.

JAC's spine is: **prompt → plan → route → build → evaluate → ledger → report**. New structure (modules, abstractions, hooks, middleware) is justified only by a logged run that demonstrates the spine breaking under load.

**Rules out:** speculative components, premature abstraction, scaling-problem solutions before scaling problems exist, "future-proof" interfaces with one implementation.

**Test:** before adding a new module, name the existing run/log/error that motivated it. If you can't, don't add it.

### 3. Measure before you abstract.

No new persona, tool, contract, or dependency without a recorded ledger entry (attempts row, run trace, cost log) that motivates it. The ledger is the gate.

**Rules out:** new tools added "because the agent might need them," new personas added because they'd be cool, new contracts written before code that uses them.

**Test:** can you point at a specific row in `attempts` or a specific run trace that the new thing addresses? If not, it's not ready.

### 4. State is durable; agents are disposable.

Runs, messages, attempts, tasks, configs live in SQLite. Agent instances, sessions, in-memory caches are reconstructible from state at any time. Resume from disk must produce the same behaviour as the live session.

**Rules out:** in-memory caches that aren't re-derivable from state, agent state that can't survive process restart, "soft" caches that mutate state without ledgering.

**Test:** kill the process. Restart. Resume. Does behaviour match what would have happened without the kill? If not, you have hidden state.

### 5. Config-as-data, not config-as-code.

Declarative values live in YAML / TOML / JSON / Markdown by the [substrate rule](../contracts/SUBSTRATE.md). Python files contain code, never tunable values. Code patterns matching on role names, model names, persona names is forbidden.

**Rules out:** Python files holding system prompts, model lists hardcoded in seed functions, `if role == "manager":` branches, persona-as-Python-class, tunable thresholds buried as integer literals.

**Test:** could a non-developer change this value safely without editing Python? If not, and the value is genuinely tunable, it's misplaced.

## Hygiene (working rules, not principles)

These are practices, not north stars — they belong here so AI agents don't dilute the five principles above:

- **Testable:** every new behaviour gets a test. Bug fixes get a test that fails before the fix.
- **Documented:** when a layer's structure changes meaningfully, update the relevant `docs/dev/<layer>.md`. When a contract changes, bump `Last revised`. When a top-level concept enters or leaves, update `CLAUDE.md`.

## Dependency direction

```
cli/ or surfaces/
  -> runtime/
    -> agents/, nodes/, workflows/    (nodes/, workflows/ arrive in M2/M3)
      -> tools/, state/, config
```

Outer layers may depend on inner layers. Inner layers must not import outer layers.

| Layer | May import from |
|---|---|
| `cli/` or `surfaces/*/` | `runtime`, `config`, `state`, `workspace` |
| `runtime/` | `agents`, `tools`, `state`, `runtime.events`, `runtime.models`, `config` |
| `agents/` | `tools`, `state`, `runtime.events`, `runtime.models`, `config` |
| `nodes/` (M2+) | `agents`, `tools`, `state`, `runtime.events`, `runtime.models`, `config` |
| `workflows/` (M3+) | `nodes`, `state`, `runtime.events`, `runtime.models`, `config` |
| `tools/` | `runtime.events` only (for tool-call events); nothing else from `jac.*` |
| `state/` | `aiosqlite` only; nothing from `jac.*` |
| `config.py`, `workspace.py` | nothing from `jac.*` (leaf imports) |

**Never:**
- `agents.*` or `runtime.*` importing from `cli.*` or `surfaces.*`
- `state.*` importing from `runtime.*`, `agents.*`, or `tools.*`
- Any layer importing from `lab/specimens/`

## Where new code goes

### Terminal-only behaviour
`cli/` — Click commands, slash commands, prompt_toolkit input, Rich rendering, prompt views.

### A2A or future browser surface
`surfaces/<name>/` — adapters that subscribe to runtime events and answer runtime requests. They translate between protocol/UI and the runtime event contract; they do not duplicate workflow or agent logic.

### Runtime behaviour
`runtime/` — event types, request/response contracts, session configuration, approval policy, human questions, the `RunCoordinator` facade.

### Agent role definitions
`agents/` for the factory and tool wrappers. Persona prompts and shapes live as YAML in `src/jac/data/personas/` (shipped) ← `~/.jac/personas/` (user) ← `<repo>/.agents/personas/` (project).

`agents/base.py` (the factory) is **the only place** in the codebase that calls Pydantic AI's `Agent.from_file()` / `Agent.from_spec()` to instantiate live agents. Tier resolution, MCP wiring, approval-middleware wrapping happen there.

### Workflow nodes (M2 and beyond, not yet)
`nodes/` — small `BaseNode` subclasses with state-in / state-out. One unit of work per node. Nodes report progress through runtime events; they don't print.

### Workflows (M3 and beyond, not yet)
`workflows/` — small Python files (~50–80 LOC) wiring nodes via `pydantic_graph`. Adding a new strategy is a new file in this folder, not a new code path inside an existing one.

### Local tools
`tools/` — local capabilities. Tools expose clear inputs/outputs, side-effect descriptions, and `ToolApprovalMeta`. File writes, shell commands, network calls are approval-aware via factory-side middleware.

### Persistent state
`state/` — durable storage. Runs, messages, attempts, tasks, agent_configs, MCP/skill registries. Migrations are append-only and ordered by filename.

### Sandbox / execution backend (Phase 2 only)
`execution/` — pluggable executor backends behind the shell tool. Default is local subprocess; container backend lands only if Phase-1 data motivates it.

## Quick placement guide

```text
Terminal command?                        cli/main.py
Slash command?                           cli/commands.py
Prompt parsing?                          cli/parser.py
Rich output?                             cli/renderer.py
Approval/question contract?              runtime/
Session config/state?                    runtime/
Coordinating a run?                      runtime/coordinator.py
A2A adapter?                             surfaces/a2a/
Persona definition?                      data/personas/<role>.yaml
Persona override?                        ~/.jac/personas/ or <repo>/.agents/personas/
Agent instantiation?                     agents/base.py  ← only place Agent.from_file/from_spec is called
Approval-middleware wrapping?            agents/base.py (factory side)
Workflow node?                           nodes/  (M2+)
Graph wiring?                            workflows/  (M3+)
File/shell/local capability?             tools/
Consuming an MCP server?                 agents/base.py + state/mcp_servers
A2A protocol surface?                    surfaces/a2a/
Run/task/attempt persistence?            state/
Skills + MCP registries?                 state/ (markdown→DB seeder)
Tunable value?                           YAML / TOML / JSON / Markdown by SUBSTRATE.md — NEVER Python literal
```

## Design rules

- Runtime contracts come before UI rendering.
- Side effects go through tools or future execution services.
- Human interaction goes through approval or question requests, never raw terminal prompts.
- Cost, model choice, and escalation are observable events on the bus and rows in the ledger — not hidden logs.
- Reference projects in `lab/specimens/` are inspiration only; never imported.
- Prefer replacing in-progress abstractions over layering compatibility shims around unshipped code.

## Phase-1 reminder

Phase 1's M1 ships **Scott alone**. There are no `nodes/` or `workflows/` packages yet — they enter when M2/M3 evidence requires them. Phase-2 components (compaction, hooks, skills, HITL, alt strategies, sandboxing, browser UI, multi-repo A2A, agent teams) are catalogued in [`docs/ROADMAP.md`](../ROADMAP.md) with explicit data triggers; none ship until M5's data motivates them.
