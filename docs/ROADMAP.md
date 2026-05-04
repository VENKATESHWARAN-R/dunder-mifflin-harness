# Roadmap — JAC

> **Status:** Living · **Last revised:** 2026-05-04 · **Type:** component plan, in dependency order
>
> _2026-05-04: C5a (tool approval middleware) shipped — see Done section._
> _2026-05-04: CLI-UX batch shipped — tab completions, toolbar, aliases, new slash commands, arrow-key approvals, REDIRECT decision, undo stack, destructive-shell guard, retry prompt — see Done section._

JAC is built component by component, not slice by slice. Each entry below is a self-contained module with a stable ID (`C0`..`Cn`). Order reflects **dependency**, not calendar — `Cn+1` assumes `Cn` is in place.

## How to read this

- **IDs are stable.** Once a component has an ID, that ID never moves. New components get appended; if order has to change, dependencies are updated and we live with non-monotonic IDs.
- **Order is dependency-driven.** A component cannot ship before its prerequisites. Within a layer, ordering reflects what unblocks the most downstream work.
- **Checkpoints, not gates.** Some IDs are tagged as **acceptance checkpoints** (e.g. C11 = Notes CLI benchmark). They mark moments where end-to-end behaviour can be validated. They do not block thinking, designing, or brainstorming about anything later in the list.
- **Brainstorming is welcome at every layer.** A `lab/brainstorm/` note that talks about C20 is not "premature" — it just sits ahead of the current build front. Promote it to a Draft contract whenever the design is clear enough.
- **One component, one entry.** If a component grows into something with multiple shippable sub-pieces, split it (`C13a`, `C13b`) rather than redefining `C13`.

## System overview

Components grouped by the layer they live in. This is the target shape, not what's shipped today — see the **Status** field on each entry below for the current state.

```mermaid
flowchart TB
  subgraph surface[surface — UI / external adapters]
    C0v[C0 CLI]
    C28v[C28 Browser UI]
    C29v[C29 A2A Server]
    C30v[C30 Multi-repo A2A]
  end
  subgraph runtime[runtime — events, sessions, coordinator]
    C0r[C0 Runtime core]
    C7v[C7 Cost tracking]
    C18v[C18 Mid-run toggles]
  end
  subgraph workflow[workflow — graphs, modes]
    C10v[C10 Graph orchestration]
    C11v[C11 Multi-task routing]
    C21v[C21 HITL mode]
    C22v[C22 Alt strategies]
    C23v[C23 Sub-workflow nesting]
    C24v[C24 Strategy auto-select]
  end
  subgraph agents[agents — roles, factory, teams]
    C5v[C5 Agent factory]
    C6v[C6 Scott + Jim]
    C6bv[C6b Pam + Date Mike]
    C9v[C9 Dwight + retry]
    C14v[C14 Holly / Temp Agency]
    C15v[C15 Agent teams]
    C19v[C19 HR escalation]
    C20v[C20 Hot-reload]
  end
  subgraph tools[tools — capabilities exposed to agents]
    C3v[C3 File tools]
    C4v[C4 Shell tool]
    C17v[C17 Remote MCP]
    C25v[C25 Browser eval]
  end
  subgraph infra[infra — cross-cutting modules]
    C8v[C8 Model tiers]
    C12v[C12 Context mgmt]
    C13v[C13 Hooks]
    C16v[C16 Skills]
  end
  subgraph state[state — workspace + DB]
    C1v[C1 SQLite store]
    C2v[C2 Workspace + seeding]
  end
  subgraph execution[execution — where shells run]
    C26v[C26 Sandbox]
    C27v[C27 Cloud headless]
  end

  surface --> runtime
  runtime --> workflow
  workflow --> agents
  agents --> tools
  agents --> infra
  agents --> state
  tools --> execution
```

## Acceptance checkpoints

| Checkpoint | Reached after | What it proves |
|---|---|---|
| **Persistence checkpoint** | C2 | Sessions resume; workspace files seed into the DB. |
| **Single-task autonomy** | C9 | Scott routes a build prompt → Pam analyses → Date Mike plans → Jim builds → Dwight evaluates → retry on fail → done. |
| **Notes CLI v0 benchmark** | C11 | Full feature-by-feature cycle on `docs/reference/V0_BENCHMARK.md`. Cost target: <$15. |
| **Multi-agent autonomy** | C15 | Holly recruits minions; specialists run as a team (parallel/sequential) with deterministic post-processing via hooks. |
| **Multi-strategy harness** | C22 | The same prompt produces measurably different runs under POC-swarm vs feature-by-feature. |
| **External surfaces** | C29 | Browser UI and A2A server consume the same event contract as the CLI. |

---

## Components

### C0 — CLI/Runtime Foundation

**Layer:** surface + runtime
**Status:** shipped (2026-04-27)
**Depends on:** —
**Brainstorm/contract:** [`CLI_DESIGN.md`](contracts/CLI_DESIGN.md), [`EVENT_CONTRACT.md`](contracts/EVENT_CONTRACT.md)

The presentation adapter and UI-agnostic runtime boundary. Click entrypoint, prompt_toolkit chat input, Rich rendering, typed events, approvals, questions, session state, `RunCoordinator`. Wraps a single Pydantic AI agent today; replaced internally by later components without changing the CLI/event contract.

**Ships:**
- `harness "prompt"` and `harness chat` entrypoints
- prompt_toolkit input with `@`-file references and `!`-shell shortcuts
- Rich renderer subscribed to runtime events
- `runtime/` package: events, approvals, questions, session, coordinator
- Tests covering CLI entrypoints, parser, renderer, runtime contracts, config, shell helpers

**Evaluation:**
- `uv run jac --help`, `uv run jac "say hello"`, `uv run jac chat` smoke
- `just test` green

```mermaid
flowchart LR
  subgraph surface[surface]
    cli[Click + prompt_toolkit + Rich]
  end
  subgraph runtime[runtime]
    bus[event bus]
    coord[RunCoordinator]
    sess[SessionState]
  end
  cli -->|commands| coord
  coord -->|events / requests| bus
  bus -->|render| cli
```

---

### C1 — Persistent State Store (SQLite)

**Layer:** state
**Status:** shipped (2026-05-03)
**Depends on:** C0
**Brainstorm/contract:** [`STATE_SCHEMA.md`](contracts/STATE_SCHEMA.md)

Move session/message state from in-memory dicts into a SQLite file. Establishes the `state/` package and the migration discipline. Active tables at this point: `runs`, `messages`. Other tables in `STATE_SCHEMA.md` are reserved — defined now so the schema direction is locked, populated as later components arrive.

**Ships:**
- `state/` package wrapping a SQLite file with a migration runner
- `runs` and `messages` rows persisted by `RunCoordinator`
- `harness resume <run-id>` rehydrates a prior session
- `/context` reads from the DB, not just in-memory attachments

**Evaluation:**
- Start a session, exit, `resume`; chat history visible
- `state.db` exists on disk; schema version row matches `STATE_SCHEMA.md`

```mermaid
flowchart LR
  subgraph runtime[runtime]
    coord[RunCoordinator]
  end
  subgraph state[state]
    sqlite[(SQLite state.db)]
    runs[runs / messages]
  end
  coord -->|persist + resume| sqlite
  sqlite --> runs
```

---

### C2 — Workspace Layout & File→DB Seeding

**Layer:** state
**Status:** shipped (2026-05-03)
**Depends on:** C1
**Brainstorm/contract:** [`WORKSPACE.md`](contracts/WORKSPACE.md)

Anchors JAC to a project workspace. Adds `~/.jac/` for user globals and `<repo>/.agents/` for project state, with dotenv onboarding, `AGENTS.md` instruction files, and no-project fallback. A deterministic seeding pass walks both scopes on harness boot and upserts skills/MCP/agent rows so they're visible to later components.

**Ships:**
- Project discovery (upward walk for `.agents/` or `AGENTS.md`)
- Layered settings, dotenv, and instruction resolution
- `jac init`, `jac init --global`, and `jac doctor` onboarding/diagnostic commands
- Seeding node: file → `mcp_servers` / `skills` upsert on every boot
- Source metadata for seeded `skills` and `mcp_servers`
- GC: stale file-backed rows removed on re-seed if not held by an open run
- `jac doctor` reports duplicate skill/MCP names per scope and missing `.gitignore` entries

**Evaluation:**
- Add `<repo>/.agents/skills/foo.md` → row appears in `skills`
- Delete the file, reboot → row garbage-collected (no open run referencing it)
- Installed `jac` launched outside this checkout reads `~/.jac/.env` and reports missing credentials clearly

```mermaid
flowchart LR
  subgraph fs[disk]
    user[~/.jac/]
    proj[.agents/]
  end
  subgraph infra[infra]
    seed[seeding node]
  end
  subgraph state[state]
    db[(state.db)]
  end
  user --> seed
  proj --> seed
  seed -->|upsert| db
```

---

### C3 — File Tools with Approval

**Layer:** tools
**Status:** shipped (2026-05-03)
**Depends on:** C0
**Brainstorm/contract:** [`TOOLS_CONTRACT.md`](contracts/TOOLS_CONTRACT.md)

First agent-facing tool group: `read_file`, `write_file`, `edit_file`. Establishes the `ToolResult` / `ToolApprovalMeta` pattern. Writes show a Rich diff before approval; approval responses flow back as structured decisions.

**Ships:**
- `tools/types.py`: `ToolResult`, `ToolStatus`, `RiskLevel`, `ToolApprovalMeta`
- `tools/filesystem.py`: read/write/edit + list/search/grep
- `TOOL_REGISTRY` entries for `filesystem` and `filesystem:read`
- Diff preview wired to `FileEditPreviewed` event

**Evaluation:**
- Agent asked to create a file → diff renders, approval prompt fires, file lands
- Edit a file → diff shows, edit applies on approve

```mermaid
flowchart LR
  subgraph agents[agents]
    a[Agent]
  end
  subgraph tools[tools]
    fs[filesystem tools]
  end
  subgraph runtime[runtime]
    appr[ApprovalPolicy]
  end
  a -->|tool call| fs
  fs -->|approval meta| appr
  appr -->|approve / deny| fs
```

---

### C4 — Shell Tool with Approval

**Layer:** tools
**Status:** shipped (2026-05-03)
**Depends on:** C3
**Brainstorm/contract:** [`TOOLS_CONTRACT.md`](contracts/TOOLS_CONTRACT.md)

Agent shell execution sharing the same executor that powers user `!` shortcuts. HIGH-risk approvals by default. Output truncated head/tail to fit context. Background-process pattern (`run_shell_background`, `read_process_output`, `list_processes`) included from day one.

**Ships:**
- `tools/shell.py`: `run_shell`, `run_shell_background`, `read_process_output`, `list_processes`
- `TOOL_REGISTRY["shell"]`
- Shared logging/rendering with user `!` commands

**Evaluation:**
- Agent runs `ls` → approval prompt, output rendered
- Long output → truncated head+tail with byte-count warning

```mermaid
flowchart LR
  subgraph agents[agents]
    a[Agent]
  end
  subgraph tools[tools]
    sh[shell tool]
  end
  subgraph execution[execution]
    local[local subprocess]
  end
  a -->|run_shell| sh
  sh --> local
  local -->|stdout/stderr/exit| sh
```

---

### C5 — Agent Factory & Config Loader

**Layer:** agents
**Status:** done (2026-05-03)
**Depends on:** C1, C2, C3, C4
**Brainstorm/contract:** [`MCP_INTEGRATION.md`](contracts/MCP_INTEGRATION.md)

The single `Agent(...)` construction site. Reads `agent_configs` and `run_mcp_servers` / `run_skills`, resolves `allowed_tools`, composes the system prompt with skills, instantiates a Pydantic AI `Agent`. Nothing else in the codebase calls `Agent(...)` directly. This is the seam every later agent-shaped component plugs into.

**Ships:**
- `agents/base.py`: `config_loader` factory
- Tier → model mapping (`TIER_DEFAULTS`)
- `allowed_tools` resolution: local registry + MCP servers (registered, none active yet)
- Tests: factory builds agents from DB rows; tool/toolset wiring verified

**Evaluation:**
- Insert an `agent_configs` row, build agent, run a no-op prompt
- Switch `allowed_tools`, rebuild, verify the agent's tools change

```mermaid
flowchart LR
  subgraph state[state]
    cfg[agent_configs / run_skills]
  end
  subgraph agents[agents]
    fac[config_loader]
    agt[Agent instance]
  end
  subgraph tools[tools]
    reg[TOOL_REGISTRY]
  end
  cfg --> fac
  reg --> fac
  fac -->|builds| agt
```

---

### C5a — Tool Approval Middleware

**Layer:** agents + tools
**Status:** done (2026-05-04)
**Depends on:** C5
**Implementation doc:** [`C5a-tool-approval-middleware.md`](implementation_docs/C5a-tool-approval-middleware.md)

The missing link between local tool execution and the approval system. C3/C4 shipped
tool implementations with `.approval` metadata and the CLI wired to handle
`ApprovalRequested` events — but nothing bridges them. Agents currently call tools
without any gate. This component wraps every non-read-only local tool call with an
approval check before execution, and adds the `FileEditPreviewed` → approve → apply
flow for file writes.

**Ships:**
- `agents/approval.py`: `make_approval_wrapper(fn, events, policy)` — approval-gating
  decorator that reads `.approval` metadata and calls `events.request_approval()` before
  each write/shell tool call
- `edit_file` refactored: compute-diff and apply-write split so preview fires before write
- `config_loader` passes `ApprovalPolicy` into `_resolve_local_tools`; `RunCoordinator`
  threads session approval mode through
- `FileEditPreviewed` emitted before the approval gate for file write/edit tools
- `FileEditApplied` emitted on successful write
- Denial returns a `ToolResult(status=ERROR)` so the model receives coherent feedback
- Tests: interactive approve, interactive deny (file not written), auto-edit shell bypass,
  read-only gate skip, yolo no-prompt, preview-before-write ordering

**Evaluation:**
- `jac "write a file called test.py"` → diff preview renders → approval prompt fires → file lands on approve
- `jac "run ls"` → HIGH-risk approval prompt fires before shell executes
- `jac --approval yolo "run ls"` → no prompt, immediate execution
- Denial: file does not exist on disk after deny

---

### C6 — Scott (Manager) + Jim (Builder)

**Layer:** agents
**Status:** planned
**Depends on:** C5a
**Brainstorm/contract:** [`lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md`](../lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md), [`IDEA.md`](reference/IDEA.md) §5

The shift away from "every prompt goes through a planner+builder loop." JAC's default agent is **Michael Scott (manager)**, persistent across turns, who tool-routes everything: trivial chat replies stay in Scott; build-shaped prompts get delegated. C6 wires Scott + **Jim Halpert (builder)** through Pydantic AI agent-as-tool delegation. One specialist is enough to prove the manager-specialist primitive end-to-end. Planner, analyst, and evaluator come at C6b/C9.

**Ships:**
- Seed `agent_configs` rows for `manager` (Scott, Worker) and `builder` (Jim, Worker)
- Scott's tool surface: `summon_jim(task)` (agent-as-tool delegation, `usage=ctx.usage` rolls cost up)
- `attempts` table activated; `parent_attempt_id` recorded for the call tree
- Persona/role fields wired (semantic `role` in code, `persona` displayed in events)

**Evaluation:**
- "What's 2+2?" → Scott replies directly; no Jim call
- "Write a Python script that prints fibonacci(10)" → Scott calls `summon_jim`; Jim writes the file and runs it
- `attempts` rows show the parent → child structure with correct cost rollup

```mermaid
flowchart LR
  subgraph agents[agents]
    sc[Scott / manager]
    jm[Jim / builder]
  end
  subgraph state[state]
    att[attempts]
  end
  sc -->|tool: summon_jim| jm
  sc --> att
  jm --> att
```

---

### C6b — Pam (Analyst) + Date Mike (Planner)

**Layer:** agents
**Status:** planned
**Depends on:** C6
**Brainstorm/contract:** [`lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md`](../lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md)

Adds the **autopilot intake + planning** layer in front of Jim. Pam Beesly (analyst, Worker) probes the environment (what runtimes, package managers, etc. are available) and synthesises a high-level requirements brief. Date Mike (planner, Architect) takes that brief and produces a structured task list with pseudo-code per task. Jim follows the plan; the plan does the thinking, which is why planner is on the highest tier and builder stays on Worker.

In **HITL mode** Pam is bypassed — Scott asks the user clarifying questions directly via runtime question events. Pam is autopilot-only.

`context_store` activates at this component (was originally slated for C11). Pam writes the requirements brief to the `shared` scope; Date Mike reads it.

**Ships:**
- Seed `agent_configs` rows for `analyst` (Pam, Worker) and `planner` (Date Mike, Architect)
- Scott's `kick_off_build(brief)` tool: orchestrates Pam → Date Mike → Jim sequentially (no graph yet — that's C10)
- Mode-aware analyst skip: in HITL, Scott emits question events instead
- `tasks` table populated by Date Mike's structured output (one row per task with `description`, `acceptance_criteria`, `complexity`, `order_index`)
- `context_store` write/read APIs wired for the `shared` scope

**Evaluation:**
- Autopilot: `jac "build a basic calculator"` → Scott routes to `kick_off_build` → Pam probes env → Date Mike plans 3-5 tasks → Jim executes → working calculator
- HITL: `jac chat`, then "build a basic calculator" → Scott asks "Python or Node? CLI or GUI?" → user answers → Date Mike plans → Jim executes (Pam never summoned)
- `tasks` rows match Date Mike's output; `context_store` shows the requirements brief in `shared`

```mermaid
flowchart LR
  subgraph agents[agents]
    sc[Scott / manager]
    pm[Pam / analyst]
    dm[Date Mike / planner]
    jm[Jim / builder]
  end
  subgraph state[state]
    cs[context_store · shared]
    tk[tasks]
  end
  sc -->|kick_off_build| pm
  pm --> cs
  cs --> dm
  dm --> tk
  tk --> jm
```

---

### C7 — Cost Tracking & `/cost`

**Layer:** runtime
**Status:** planned
**Depends on:** C6, C6b
**Brainstorm/contract:** [`STATE_SCHEMA.md`](contracts/STATE_SCHEMA.md) (`attempts`)

Per-call model, token, duration, and cost records captured into `attempts` (with `call_type='agent'` or `'direct_llm'`, plus `parent_attempt_id` for the call tree). `CostUpdated` events emitted at run end and on model changes. `/cost` renders a tree-shaped breakdown (Scott → Pam → Date Mike → Jim) with per-tier and per-persona totals.

**Ships:**
- Cost-tracking wrapper around model calls
- Structured `CostUpdated` event (replacing the summary string)
- `/cost` slash command + Rich panel renderer

**Evaluation:**
- Run a small task; `/cost` shows tokens, cost per role, total
- `attempts` rows match the rendered totals

```mermaid
flowchart LR
  subgraph agents[agents]
    a[Agent / direct LLM call]
  end
  subgraph runtime[runtime]
    ct[cost wrapper]
    bus[event bus]
  end
  subgraph state[state]
    att[attempts]
  end
  a --> ct
  ct --> att
  ct -->|CostUpdated| bus
```

---

### C8 — Model Tier Routing

**Layer:** infra
**Status:** planned
**Depends on:** C5, C7
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §2 (Model Buckets)

Provider-agnostic tier map (Scout / Worker / Architect) wired into `config_loader`. At least two providers represented. `/model` and `/tier` slash commands update session config safely. Defaults per persona: Scott (manager) → Worker, Pam (analyst) → Worker, Date Mike (planner) → **Architect**, Jim (builder) → Worker, Dwight (evaluator) → Worker, Holly (recruiter) → Scout.

**Ships:**
- `TIER_DEFAULTS` covering 2+ providers
- `/model <id>` and `/tier <scout|worker|architect>` commands
- `model_override` honoured on `agent_configs`
- `SessionConfigChanged` events on tier/model changes

**Evaluation:**
- Switch tier mid-session; next turn uses new model (visible in `attempts`)
- `/cost` shows ≥2 tiers exercised in a single run

```mermaid
flowchart LR
  subgraph runtime[runtime]
    sess[SessionState]
  end
  subgraph infra[infra]
    tier[tier resolver]
  end
  subgraph agents[agents]
    fac[config_loader]
  end
  sess -->|tier override| tier
  tier --> fac
```

---

### C9 — Dwight (Evaluator) + Retry Loop

**Layer:** workflow (pre-graph)
**Status:** planned
**Depends on:** C6b, C8
**Brainstorm/contract:** [`EVENT_CONTRACT.md`](contracts/EVENT_CONTRACT.md) (Evaluation events), [`lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md`](../lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md)

Adds **Dwight Schrute (evaluator, Worker)** to grade pass/fail against the acceptance criteria Date Mike wrote. Dwight also classifies failure type: **code-fault** (implementation is wrong, plan was fine → retry Jim) vs **spec-fault** (plan was premised on something wrong, e.g., wrong runtime → bounce back to Pam to re-analyse). One retry per task. Surfaces `EvaluationStarted` / `EvaluationCompleted` events.

**Ships:**
- `evaluator` agent config seeded with Dwight's persona + by-the-book grading prompt
- Retry loop wrapping build → evaluate inside Scott's `kick_off_build` tool
- Failure-type classifier in Dwight's structured output (`fault_type: code | spec`)
- Spec-fault routing back to Pam (in autopilot) or surfacing a question to user (in HITL)
- `attempts.eval_score` / `eval_passed` / `eval_feedback` populated
- `pydantic_evals` integration for structured scoring

**Evaluation:**
- Code-fault path: ambiguous task; first attempt fails eval, retries, second passes
- Spec-fault path: brief assumed Node was available but env has only Python; Dwight flags spec-fault → Pam re-analyses → Date Mike replans → Jim succeeds
- **Acceptance checkpoint: single-task autonomy.** Scott routes a build prompt → Pam → Date Mike → Jim → Dwight → done (with at most one retry).

```mermaid
flowchart LR
  subgraph agents[agents]
    pm[Pam / analyst]
    dm[Date Mike / planner]
    jm[Jim / builder]
    dw[Dwight / evaluator]
  end
  subgraph workflow[workflow]
    retry[retry controller]
  end
  jm --> dw
  dw -->|pass| done([done])
  dw -->|code-fault| retry
  retry -->|attempt+1| jm
  dw -->|spec-fault| pm
  pm --> dm --> jm
```

---

### C10 — Graph Orchestration (pydantic_graph)

**Layer:** workflow
**Status:** planned
**Depends on:** C9
**Brainstorm/contract:** [`PHILOSOPHY.md`](reference/PHILOSOPHY.md), [`IDEA.md`](reference/IDEA.md) §5

Behaviour-preserving refactor onto `pydantic_graph`. A `nodes/` package with a uniform `BaseNode` (state-in / state-out). A `workflows/feature_by_feature.py` wires Pam → Date Mike → Jim → Dwight as graph nodes. The graph lives **inside** Scott's `kick_off_build` tool — Scott still drives via Pydantic AI tool delegation; the user-facing surface is unchanged. `RunCoordinator` delegates to the graph runner without changing the event contract.

**Ships:**
- `nodes/` with `BaseNode` interface and the existing nodes wrapped
- `workflows/feature_by_feature.py`
- `RunCoordinator` graph-driven, single-task only

**Evaluation:**
- Same task as C9 produces identical events and outputs
- Graph traversal visible in events (`NodeStarted` carries `run_id`/`task_id`)

```mermaid
flowchart LR
  subgraph workflow[workflow]
    g[feature_by_feature graph]
    n1[plan node] --> n2[build node] --> n3[evaluate node] --> n4[pass_check]
  end
  g --> n1
```

---

### C11 — Multi-Task Routing & Notes CLI Checkpoint

**Layer:** workflow
**Status:** planned
**Depends on:** C10, C2
**Brainstorm/contract:** [`V0_BENCHMARK.md`](reference/V0_BENCHMARK.md)

Adds `task_router` (reads pending tasks from the DB) and `context_loader` (builds scoped context per agent role). A multi-task plan now executes end-to-end. **Acceptance checkpoint:** the Notes CLI benchmark in `V0_BENCHMARK.md` runs autonomously, hits ≥5/7 features, and stays under $15.

**Ships:**
- `task_router` and `context_loader` nodes
- Multi-task `feature_by_feature` workflow
- Notes CLI benchmark harness (run + score reporting)

**Evaluation:**
- **Acceptance checkpoint: Notes CLI v0 benchmark.** All commands exit 0, ≥5/7 features pass, total cost <$15, ≥2 tiers exercised.
- Run trace shows: plan → router → loader → build → evaluate → router → … → done

```mermaid
flowchart LR
  subgraph workflow[workflow]
    rt[task_router]
    cl[context_loader]
    bd[build]
    ev[evaluate]
    pc[pass_check]
  end
  subgraph state[state]
    tk[tasks / context_store]
  end
  rt --> cl --> bd --> ev --> pc
  pc -->|next| rt
  tk <--> rt
  tk <--> cl
```

---

### C12 — Context Management Module

**Layer:** infra
**Status:** planned
**Depends on:** C11
**Brainstorm/contract:** [`lab/brainstorm/2026-05-02-context-management-module.md`](../lab/brainstorm/2026-05-02-context-management-module.md)

A `ContextManager` interface behind Pydantic AI `history_processors`, called per-role. Supports token/message-count triggers and head/tail/segment-rollup strategies. Direct-LLM summarisation routed through the cost tracker (`call_type='direct_llm'`). Lands after the Notes CLI checkpoint because that benchmark is too small to trigger compaction.

**Ships:**
- `ContextManager` module + `history_processor` adapters
- Per-role config (`compact_at_tokens`, `strategy`, `summary_tier`)
- `ContextCompacted` event + `attempts` audit trail

**Evaluation:**
- Synthetic long conversation triggers compaction; task progress preserved
- Summarisation cost recorded; total tokens drop on the next turn

```mermaid
flowchart LR
  subgraph agents[agents]
    a[Agent]
  end
  subgraph infra[infra]
    cm[ContextManager]
  end
  subgraph state[state]
    msg[messages / attempts]
  end
  a -->|history| cm
  cm -->|reshape| a
  cm -->|summary call| msg
```

---

### C13 — Hooks/Callbacks Module

**Layer:** infra
**Status:** planned
**Depends on:** C10
**Brainstorm/contract:** [`lab/brainstorm/2026-05-02-hooks-and-callbacks-module.md`](../lab/brainstorm/2026-05-02-hooks-and-callbacks-module.md)

A `HookManager` with named lifecycle points (`pre_run`, `post_run`, `pre_model`, `post_model`, `post_tool` — `pre_tool` is already the approval system). Hooks run deterministic code (type checks, linters, telemetry), composing in registration order with a deny/retry signal. Conceptually separate from approvals.

**Ships:**
- `HookManager` with registration API
- Five lifecycle seams wired through `Agent.iter()`
- Built-in hooks (telemetry, audit) plus user-loadable hooks via `<repo>/.agents/hooks/`
- `HOOKS.md` contract drafted and promoted to Locked

**Evaluation:**
- Register a `post_run` hook that runs `uv run ty check`; failure triggers retry without LLM call
- Hook ordering is deterministic; deny short-circuits later hooks at the same seam

```mermaid
flowchart LR
  subgraph workflow[workflow]
    iter[agent.iter loop]
  end
  subgraph infra[infra]
    hm[HookManager]
  end
  subgraph agents[agents]
    a[Agent]
  end
  iter -->|pre/post seams| hm
  hm -->|deterministic code| a
```

---

### C14 — Holly Flax / Temp Agency (Dynamic Minion Spawning)

**Layer:** agents + tools
**Status:** planned
**Depends on:** C5, C13
**Brainstorm/contract:** [`lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md`](../lab/brainstorm/2026-05-04-manager-specialist-minion-pattern.md), [`TOOLS_CONTRACT.md`](contracts/TOOLS_CONTRACT.md) §"Sub-Agent Toolgroup (C14)"

Adds **Holly Flax (recruiter, Scout)** as the centralized minion factory. Other agents don't spawn minions directly — they call Holly via a `consult_holly(need_description, suggested_tools)` tool. Holly crafts the persona/prompt, picks tools from the caller's whitelist, and instantiates the minion through `agents/factory.py` (preserving the rule that only the factory calls `Agent(...)`). The minion runs once with isolated message history, returns a structured `AgentResult`, and is discarded.

**Guardrails:**
- **Depth ≤ 1** — minions cannot spawn minions (cannot call Holly themselves).
- **Tool whitelist** — minion's available tools are bounded by the caller's. No privilege escalation.
- **Budget inheritance** — minion runs against the caller's remaining run budget; over-cap refuses.
- **Factory-mediated** — `is_minion=1`, `parent_role` set on the `agent_configs` row for audit.

**Ships:**
- `agent_configs` row for `recruiter` (Holly, Scout) seeded
- `tools/agents.py` with `consult_holly(need, suggested_tools)` (replaces the earlier `spawn_agent` framing)
- `TOOL_REGISTRY["agents"]`
- Hook attachment at spawn time
- `MinionRecruited` / `MinionReturned` events (rename of `AgentSpawned` / `AgentReturned`)

**Evaluation:**
- Scott (or any specialist) calls `consult_holly("Look up the latest pydantic_graph API for cycles", ["web_search"])` → Holly spawns a research minion → returns a summary
- Minion attempts to call Holly itself → refused (depth limit)
- `post_run` hook (`ty check`) on the minion runs; failure re-dispatches deterministically

```mermaid
flowchart LR
  subgraph agents[agents]
    caller[caller agent]
    holly[Holly / recruiter]
    fac[config_loader]
    minion[temp / minion]
  end
  subgraph tools[tools]
    ch[consult_holly]
  end
  subgraph infra[infra]
    hm[HookManager]
  end
  caller --> ch --> holly --> fac --> minion
  minion -.post hooks.-> hm
  minion -->|result| caller
```

---

### C15 — Agent Teams & Inter-Agent Messaging

**Layer:** agents + state
**Status:** planned
**Depends on:** C14
**Brainstorm/contract:** [`STATE_SCHEMA.md`](contracts/STATE_SCHEMA.md) (`agent_instances`, `agent_teams`, `agent_messages`)

Activates the reserved tables. Multiple agent instances run within one `run_id` and coordinate via the `agent_messages` queue (bug reports, handoffs, broadcasts). Strategies: `parallel`, `sequential`, `mixed`. **Acceptance checkpoint:** multi-agent autonomy — a builder/tester pair runs concurrently, the tester posts a bug report, the builder picks it up.

**Ships:**
- `agent_instances` / `agent_teams` / `agent_messages` wired live
- Team strategy executor (parallel / sequential / mixed)
- Inter-agent message types: `bug_report`, `task_complete`, `handoff`, `question`, `info`, `review_request`

**Evaluation:**
- **Acceptance checkpoint: multi-agent autonomy.**
- Builder + Tester run on a 5-feature task; tester posts a bug → builder fixes → tester resumes
- All instance/message activity visible in DB and events

```mermaid
flowchart LR
  subgraph agents[agents]
    bd[Builder instance]
    ts[Tester instance]
  end
  subgraph state[state]
    msg[agent_messages]
    inst[agent_instances]
  end
  bd <-->|read/post| msg
  ts <-->|read/post| msg
  inst --> bd
  inst --> ts
```

---

### C16 — Skills System (Dynamic Injection)

**Layer:** infra
**Status:** planned
**Depends on:** C5, C2
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.6, [`MCP_INTEGRATION.md`](contracts/MCP_INTEGRATION.md) §Skills Injection

A `skill_injector` node looks up domain skills (per task classification) and prepends them to agent system prompts at instantiation time. Sourced from the `skills` table (seeded by C2). Examples: React conventions for frontend tasks, Terraform patterns for infra tasks.

**Ships:**
- `skill_injector` node
- Domain classification on task plan
- `run_skills` populated per task domain
- `/skills` slash command (read-only listing)

**Evaluation:**
- Same task with/without a relevant skill: output measurably more idiomatic with skill
- Audit shows which skills were active per attempt

```mermaid
flowchart LR
  subgraph state[state]
    sk[skills / run_skills]
  end
  subgraph infra[infra]
    si[skill_injector]
  end
  subgraph agents[agents]
    fac[config_loader]
  end
  sk --> si --> fac
```

---

### C17 — Remote MCP Server Integration

**Layer:** tools
**Status:** planned
**Depends on:** C5
**Brainstorm/contract:** [`MCP_INTEGRATION.md`](contracts/MCP_INTEGRATION.md)

Wires the `mcp:<name>` half of `allowed_tools` resolution. Up to C16 the MCP infrastructure is registry-only; this component adds live transports (`stdio`, `http`, `sse`) and the `MCPServerConnected` / `Disconnected` lifecycle events. First real MCP server: Playwright (used by C25).

**Ships:**
- `MCPServerStdio`, `MCPServerStreamableHTTP`, `MCPServerSSE` wiring in `config_loader`
- `async with agent:` lifecycle owned by the agent run
- `MCPServerConnected` / `MCPServerDisconnected` events

**Evaluation:**
- Register a Playwright MCP row, reference `mcp:playwright` in `allowed_tools`, run an agent that calls a Playwright tool
- Transport disconnect surfaces as an event; agent retries gracefully

```mermaid
flowchart LR
  subgraph state[state]
    mcp[mcp_servers]
  end
  subgraph agents[agents]
    fac[config_loader]
    a[Agent]
  end
  subgraph tools[tools]
    ts[MCP toolset]
  end
  mcp --> fac --> ts --> a
```

---

### C18 — Mid-Run Config Toggles

**Layer:** runtime
**Status:** planned
**Depends on:** C16, C17
**Brainstorm/contract:** [`EVENT_CONTRACT.md`](contracts/EVENT_CONTRACT.md), [`MCP_INTEGRATION.md`](contracts/MCP_INTEGRATION.md) §Mid-Run Toggle

`/disable mcp:<name>` and `/disable skill:<name>` (and their `/enable` siblings). Sets `enabled=0` on `run_mcp_servers` / `run_skills`, emits a toggle event, rebuilds the agent on the next turn. Deliberate user action only — never automatic.

**Ships:**
- `toggle_mcp` and `toggle_skill` commands
- `MCPServerToggled` / `SkillToggled` events
- Slash-command UI; rebuild path in `config_loader`

**Evaluation:**
- `/disable mcp:playwright` mid-conversation → next turn agent has no Playwright tool
- `/enable` restores it; message history preserved across both transitions

```mermaid
flowchart LR
  subgraph surface[surface]
    cli[/disable mcp:.../]
  end
  subgraph runtime[runtime]
    cmd[command dispatch]
  end
  subgraph state[state]
    rm[run_mcp_servers / run_skills]
  end
  subgraph agents[agents]
    fac[config_loader]
  end
  cli --> cmd --> rm --> fac
```

---

### C19 — Dynamic Model Escalation (HR agent)

**Layer:** agents
**Status:** planned
**Depends on:** C9, C15
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.1

The deterministic `hr_escalation` node tracks per-task pass/fail counts per tier; after N failures, rewrites `agent_configs.tier` for that task and re-dispatches. Every escalation logged for the research dataset (`task_type → minimum tier`).

**Ships:**
- `hr_escalation` node + `TaskEscalated` event
- Per-task attempt counter in `tasks.attempt_count`
- Escalation audit: `task_type`, `from_tier`, `to_tier`, `reason`

**Evaluation:**
- Force-fail a Scout-tier task → escalates to Worker → succeeds
- Audit query returns the escalation row with full provenance

```mermaid
flowchart LR
  subgraph workflow[workflow]
    pc[pass_check]
  end
  subgraph agents[agents]
    hr[hr_escalation]
    fac[config_loader]
  end
  subgraph state[state]
    tk[tasks / agent_configs]
  end
  pc -->|N fails| hr --> tk
  tk --> fac
```

---

### C20 — Agent Config Hot-Reloading Mid-Run

**Layer:** agents
**Status:** planned
**Depends on:** C19
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.3

Generalises the escalation rebuild path. Any change to `agent_configs` (slash command, hook, escalation, A/B experiment) takes effect on the next turn for that role. Message history is preserved; the tool block and system prompt may change. The cache-miss tradeoff is intentional and visible.

**Ships:**
- `config_loader` always rebuilds from DB at turn start (no per-run caching)
- `AgentConfigChanged` event
- Documented cache-miss semantics

**Evaluation:**
- Change `system_prompt` mid-run via the DB; next turn picks it up
- Cache-miss recorded in cost log on the rebuild turn

```mermaid
flowchart LR
  subgraph state[state]
    cfg[agent_configs]
  end
  subgraph agents[agents]
    fac[config_loader]
    a[Agent (turn N)]
    a2[Agent (turn N+1)]
  end
  cfg --> fac
  fac --> a
  cfg -->|mutated| fac
  fac --> a2
```

---

### C21 — HITL Mode

**Layer:** workflow
**Status:** planned
**Depends on:** C13
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.4

A second top-level workflow composition that reuses every node in `feature_by_feature` and inserts `human_checkpoint` nodes at key decision points. Uses the existing question primitive in `runtime/questions.py`. Selected via `harness run --mode hitl` or `/mode hitl`.

**Mode-aware analyst handling.** In HITL, **Pam (analyst) is bypassed** — Scott himself emits clarification questions to the user via runtime question events instead of summoning Pam to probe the environment. Pam is autopilot-only. Date Mike, Jim, and Dwight are unchanged across modes; the difference is purely in how the requirements brief is sourced.

**Ships:**
- `workflows/feature_by_feature_hitl.py`
- `human_checkpoint` node + `feedback_injector` node
- `--mode autopilot|hitl` flag and `/mode` slash command

**Evaluation:**
- Run Notes CLI under HITL: pauses at planning, waits for human input, integrates feedback
- Same run under autopilot: no pauses, identical output otherwise

```mermaid
flowchart LR
  subgraph workflow[workflow]
    pl[plan] --> hc[human_checkpoint] --> fi[feedback_injector] --> rt[task_router] --> bd[build] --> ev[evaluate]
  end
```

---

### C22 — Alternative Workflow Strategies

**Layer:** workflow
**Status:** planned
**Depends on:** C10, C13
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.2

Add POC-then-swarm, TDD, agile/sprint, and spec-driven as separate workflow compositions over the shared node library. Strategy nodes (`sprint_planner`, `poc_builder`, `test_generator`, `swarm_dispatcher`, `retrospective`) added per workflow. **Acceptance checkpoint:** the same prompt produces measurably different runs under POC-swarm vs feature-by-feature.

**Ships:**
- `workflows/poc_swarm.py`, `workflows/tdd.py`, `workflows/agile.py`, `workflows/spec_driven.py`
- Strategy nodes registered alongside core nodes
- `harness run --strategy <name>` + `/strategy` command

**Evaluation:**
- **Acceptance checkpoint: multi-strategy harness.**
- Same Notes CLI prompt under each strategy; compare cost / quality / duration

```mermaid
flowchart LR
  subgraph workflow[workflow]
    fbyf[feature_by_feature]
    poc[poc_swarm]
    tdd[tdd]
    ag[agile]
    sp[spec_driven]
  end
  subgraph agents[agents]
    nodes[shared node library]
  end
  fbyf --> nodes
  poc --> nodes
  tdd --> nodes
  ag --> nodes
  sp --> nodes
```

---

### C23 — Sub-Workflow Nesting

**Layer:** workflow
**Status:** planned
**Depends on:** C22
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §5

Workflows can invoke other workflows as sub-routines. Examples: feature-by-feature dispatching a TDD sub-workflow for a critical feature; POC-swarm using feature-by-feature as the swarm-worker strategy. Implemented via `pydantic_graph` graph composition + `run_subgraph` helper.

**Ships:**
- `run_subgraph(workflow_name, task_id)` node
- Sub-run lifecycle in events (`SubRunStarted` / `SubRunCompleted`)
- `tasks.parent_task_id` populated for sub-workflow tasks

**Evaluation:**
- Critical task inside Notes CLI gets routed through TDD sub-workflow
- Cost / token / scoring rolls up to the parent run

```mermaid
flowchart LR
  subgraph workflow[workflow]
    parent[parent workflow]
    sub[sub-workflow]
  end
  parent -->|run_subgraph| sub
  sub -->|result| parent
```

---

### C24 — Strategy Auto-Selection

**Layer:** workflow
**Status:** planned
**Depends on:** C22
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §3.2

A `strategy_selector` node (deterministic or LLM-classified) chooses which workflow to run for a given prompt — simple apps → POC-swarm, complex apps → feature-by-feature, etc. Closes the loop on the research questions about strategy fit.

**Ships:**
- `strategy_selector` node
- Classification rubric (heuristic + LLM fallback)
- `/strategy auto|<name>` command

**Evaluation:**
- Diverse prompts produce diverse strategy choices
- Override (`--strategy <name>`) bypasses the selector cleanly

```mermaid
flowchart LR
  subgraph workflow[workflow]
    sel[strategy_selector]
    fbyf[feature_by_feature]
    poc[poc_swarm]
    tdd[tdd]
  end
  sel --> fbyf
  sel --> poc
  sel --> tdd
```

---

### C25 — Browser-Based Evaluation (Playwright)

**Layer:** tools
**Status:** planned
**Depends on:** C17
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §4

The evaluator gains a Playwright MCP toolset for browser-driven acceptance tests on apps with a UI. CLI-shaped benchmarks (Notes CLI) keep their shell-based evaluation; web app benchmarks use Playwright. Same `evaluate` node, different toolset.

**Ships:**
- Playwright MCP server entry seeded into `~/.jac/mcp/`
- Evaluator config gains `mcp:playwright` in `allowed_tools` for web tasks
- Browser test artefacts (screenshots) attached to `attempts.eval_feedback`

**Evaluation:**
- A small web-app benchmark passes evaluator + Playwright checks
- Screenshots stored, retrievable from the run record

```mermaid
flowchart LR
  subgraph agents[agents]
    ev[Evaluator]
  end
  subgraph tools[tools]
    pw[Playwright MCP]
  end
  subgraph execution[execution]
    br[browser]
  end
  ev --> pw --> br
```

---

### C26 — Sandboxed/Container Shell Execution

**Layer:** execution
**Status:** planned
**Depends on:** C4
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §2 (Execution: Local First)

Pluggable execution backend behind the shell tool. Default backend stays local; container backend (Docker / Podman) added without changing the agent code. Workspace bind-mount, network policy, env scrubbing live here.

**Ships:**
- `execution/` package with `LocalExecutor` and `ContainerExecutor`
- Backend selected via `settings.json` / `--executor` flag
- Tool-level approval semantics unchanged

**Evaluation:**
- Notes CLI benchmark runs identically under local vs container executor
- Container blocks network egress when policy says so

```mermaid
flowchart LR
  subgraph tools[tools]
    sh[shell tool]
  end
  subgraph execution[execution]
    sel[executor selector]
    loc[LocalExecutor]
    con[ContainerExecutor]
  end
  sh --> sel --> loc
  sel --> con
```

---

### C27 — Cloud Headless Execution

**Layer:** execution + surface
**Status:** planned
**Depends on:** C26
**Brainstorm/contract:** [`IDEA.md`](reference/IDEA.md) §2

Run JAC headlessly against a managed container backend so overnight benchmarks no longer pin a workstation. Reuses the same runtime + execution interface; differs only in the surface (no terminal, structured stdout/stderr) and the executor (remote container).

**Ships:**
- Headless surface (machine-readable event stream)
- Cloud executor adapter (container runtime API)
- `harness run --headless --remote` invocation path

**Evaluation:**
- Notes CLI benchmark scheduled remotely; result + cost report retrieved on completion
- Same event types as local run; consumers don't notice the difference

```mermaid
flowchart LR
  subgraph surface[surface]
    hl[headless adapter]
  end
  subgraph runtime[runtime]
    coord[RunCoordinator]
  end
  subgraph execution[execution]
    rem[remote executor]
  end
  hl --> coord --> rem
```

---

### C28 — Browser UI Adapter

**Layer:** surface
**Status:** planned
**Depends on:** C0 (event contract)
**Brainstorm/contract:** [`PHILOSOPHY.md`](reference/PHILOSOPHY.md), [`EVENT_CONTRACT.md`](contracts/EVENT_CONTRACT.md)

A browser UI subscribed to the same events the CLI consumes. No runtime changes — only a new adapter under `surfaces/browser/`. Implements approval and question handling so HITL works across the wire.

**Ships:**
- `surfaces/browser/` package (web server + websocket bridge)
- Renderers for all event types in `EVENT_CONTRACT.md`
- Approval and question response forwarded to `EventBus.resolve_*`

**Evaluation:**
- Same Notes CLI run viewable in CLI and browser concurrently
- Approving a tool call from the browser unblocks the runtime

```mermaid
flowchart LR
  subgraph surface[surface]
    cli[CLI]
    web[Browser UI]
  end
  subgraph runtime[runtime]
    bus[event bus]
  end
  bus --> cli
  bus --> web
  cli --> bus
  web --> bus
```

---

### C29 — A2A Server Adapter

**Layer:** surface
**Status:** planned
**Depends on:** C28, C15
**Brainstorm/contract:** [`PHILOSOPHY.md`](reference/PHILOSOPHY.md)

JAC exposes an A2A endpoint so other agents can dispatch tasks to it as if it were one of their own roles. Reuses the runtime event contract; the surface translates between A2A protocol messages and runtime events.

**Ships:**
- `surfaces/a2a/` adapter
- Authentication + capability discovery
- Run-as-peer mode: incoming task creates a `runs` row, returns result

**Evaluation:**
- Peer agent dispatches a Notes CLI task; receives result + cost
- Failure semantics covered (peer death, timeout)

```mermaid
flowchart LR
  subgraph surface[surface]
    a2a[A2A server]
  end
  subgraph runtime[runtime]
    coord[RunCoordinator]
  end
  peer[peer agent] --> a2a --> coord
  coord --> a2a --> peer
```

---

### C30 — Multi-Repo A2A Runs

**Layer:** surface + agents
**Status:** planned
**Depends on:** C29, C15
**Brainstorm/contract:** [`lab/brainstorm/2026-05-02-multi-repo-a2a-runs.md`](../lab/brainstorm/2026-05-02-multi-repo-a2a-runs.md)

Per-repo agents, each rooted in its own `<repo>/.agents/` workspace, communicating via A2A. A coordinator addresses peers by name and dispatches subtasks across the repo boundary. Tier routing per repo becomes natural: docs-repo agents skew Scout, infra-repo agents skew Architect.

**Ships:**
- Peer discovery (config or `~/.jac/peers.json`)
- Cross-repo super-run lifecycle (parent run links cross-repo runs)
- Cross-boundary failure semantics (retry / escalate / fail-fast)

**Evaluation:**
- Two-repo benchmark: app + infra. Coordinator dispatches infra task to infra-repo peer, app task to app-repo peer; both complete; super-run rolls up cost.

```mermaid
flowchart LR
  subgraph repoA[repo A]
    aA[A2A peer]
  end
  subgraph repoB[repo B]
    aB[A2A peer]
  end
  subgraph coord[coordinator]
    sup[super-run]
  end
  sup <--> aA
  sup <--> aB
```

---

## Notes

Worth remembering — decisions, dead ends, tricks that worked.

- CLI is an adapter over `runtime/`, not the orchestrator.
- Approvals and questions are separate primitives; do not reuse approval prompts for clarification questions.
- Future browser UI and A2A surfaces consume the runtime event/request contract, not CLI modules.
- Orchestration library locked: Pydantic AI throughout. `pydantic_graph` enters at C10.
- State store locked: SQLite. Schema in `docs/contracts/STATE_SCHEMA.md` — update that doc before touching DB code.
- Benchmark for the Notes CLI checkpoint: `docs/reference/V0_BENCHMARK.md`.
- HITL is its own workflow composition (C21), not a checkpoint inside the autopilot workflow.
- `agents/base.py` (config_loader) is the sole factory for live Pydantic AI agents — nothing else calls `Agent(...)` directly.

---

## Done

Move components here when shipped, with the date.

### CLI-UX — Terminal UX improvements (2026-05-04)

Cross-cutting UX pass on `src/jac/cli/`. Not a numbered component — improves the existing C0 foundation.

- [x] **Input layer** (`input.py`): `DedupFileHistory` skips consecutive duplicates; `JacCompleter` completes slash command names (with descriptions), command arguments (`/tier`, `/mode`, `/approval`, `/params`), and `@`-prefixed file paths; `bottom_toolbar` shows `model · tier · mode · approval` live; `(esc+enter for newline)` placeholder; Ctrl+R history search via prompt_toolkit emacs defaults
- [x] **Slash commands** (`commands.py`): `SlashCommand` gains optional `example` field; `SlashCommandRegistry` gains `alias()` and `descriptions()`; `help_text()` shows examples, aliases, keyboard shortcuts, and input prefix reference
- [x] **New slash commands** (`app.py`): `/clear`, `/history [n]`, `/save [file]`, `/undo`, `/capabilities`
- [x] **Aliases**: `/h`→`/help`, `/q`→`/quit`, `/m`→`/model`, `/t`→`/tier`, `/x`→`/context`, `/?`→`/help`
- [x] **Actionable validation errors**: all slash command handlers show the invalid value, list valid options, and provide an example
- [x] **Undo stack** (`app.py`): `FileEditPreviewed` handler snapshots files before edits; `/undo` restores from stack (depth 20)
- [x] **Destructive shell guard** (`app.py`): user `!` commands matching `rm -rf`, `git reset --hard`, `git push --force`, `DROP TABLE`, etc. prompt for confirmation
- [x] **Retry on failure** (`app.py`): `RunFailed` is caught; `ask_yn("Retry?")` re-submits once
- [x] **Consolidated attachment warnings** (`app.py`): multiple `@path` failures emitted as a single `WarningRaised` block
- [x] **Auto-resume** (`main.py`): `jac resume` with no ID selects most-recent run from DB; `run_resumed()` shows a context preview of last 3 turns
- [x] **Richer welcome banner** (`renderer.py`): shows `model · tier · mode` at session start; `CostUpdated` rendered as compact one-liner instead of full panel; `render_resume_context` and `render_message_history` methods added
- [x] **REDIRECT approval decision** (`runtime/approvals.py`, `agents/approval.py`, `cli/prompts.py`): new `ApprovalDecision.REDIRECT` with `redirect_message` field on `ApprovalResponse`; approval wrapper returns feedback as tool result so model can adjust and retry; approval prompt is now async with arrow-key navigation; `ask_yn` also async
- [x] 3 new tests for REDIRECT in `tests/test_agent_approval.py`; all existing tests passing

---

### C5a — Tool Approval Middleware (2026-05-04)

- [x] `src/jac/agents/approval.py`: `make_approval_wrapper(fn, events, policy)` — preserves `__name__` / `__annotations__` / `__wrapped__` so pydantic_ai schema introspection still works
- [x] `_resolve_local_tools` in `agents/base.py` wraps every entry from `TOOL_REGISTRY`; `config_loader` accepts a new `approval_policy` kwarg with INTERACTIVE fallback for SDK callers
- [x] `RunCoordinator` constructs `ApprovalPolicy(mode=session.config.approval_mode)` and threads it through; `ChatApp` shares the same policy instance so `/approval yolo` updates flow through to the wrapper
- [x] `edit_file` split into `compute_edit` (returns `PreparedEdit | FileEditResult`) and `apply_edit`; `write_file` gains `preview_write` so the wrapper can emit `FileEditPreviewed` before any byte hits disk
- [x] Denial returns a typed `ToolResult(status=PERMISSION_DENIED)` of the wrapped tool's return type — never raises
- [x] `FileEditApplied` emitted post-write for file_write tools; READ_ONLY tools skip the gate entirely
- [x] 13 new tests in `tests/test_agent_approval.py` covering all approval modes, denial short-circuit, preview-before-write ordering, session-scoped allowances, and compute-error short-circuit
- [x] Full test suite green (159 passing)

---

### C0 — CLI/Runtime Foundation (2026-04-27)

- [x] `docs/contracts/CLI_DESIGN.md` captures CLI philosophy, input grammar, events, approvals, questions, extension rules
- [x] Click entrypoint supports `harness "prompt"` and `harness chat`
- [x] prompt_toolkit chat input
- [x] Rich renderer subscribes to runtime events
- [x] Runtime has typed events, approvals, questions, session state, `RunCoordinator`
- [x] `@` file references become structured attachments
- [x] User `!` shell commands use a shared shell executor
- [x] Focused tests cover CLI entrypoints, parser, renderer, runtime contracts, config, shell helpers

---

### C3 — File Tools with Approval (2026-05-03)

- [x] `tools/types.py`: `ToolResult`, `ToolStatus`, `RiskLevel`, `ToolApprovalMeta`; all filesystem and shell result types
- [x] `tools/filesystem.py`: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep_files` — each with `ToolApprovalMeta` attached
- [x] `TOOL_REGISTRY` entries for `filesystem` (full) and `filesystem:read` (read-only subset)
- [x] `FileEditResult.diff` carries a unified diff so the approval middleware can emit `FileEditPreviewed` without recomputing
- [x] `FileEditPreviewed` / `FileEditApplied` event types defined; renderer subscribed
- [x] Tests: 28 cases across all 6 tools including approval metadata, error paths, and edge cases
- [x] Approval gate wired in C5a (2026-05-04)

---

### C5 — Agent Factory & Config Loader (2026-05-03)

- [x] `src/jac/agents/base.py`: `config_loader` — the only site that calls `pydantic_ai.Agent(...)`. Reads `agent_configs`, resolves `allowed_tools` against `TOOL_REGISTRY`, builds MCP toolsets from `run_mcp_servers`, composes system prompt with skills from `run_skills`
- [x] `src/jac/agents/seeds.py`: `ensure_default_run_config` — idempotently seeds the default `chat` role row so C0 behaviour stays green without callers pre-populating the DB
- [x] `src/jac/state/agent_configs.py`, `run_mcp_servers.py`, `run_skills.py`: three new repos activating the C5 tables
- [x] `StateStore` wired with `agent_configs`, `run_mcp_servers`, `run_skills`
- [x] `RunCoordinator.build_agent` now delegates to `config_loader`; fallback path retained when `state=None`
- [x] `SessionConfig.role: str = "chat"` added
- [x] `from jac import build_agent` SDK re-export (`__getattr__` lazy import)
- [x] 136 tests pass; `AgentConfigNotFound`, `UnknownToolError` error types
- [x] `docs/guide/` (5 files) and `docs/dev/` (7 files) created
- [x] Version bumped to 0.3.0 (new public SDK seam)

---

### C4 — Shell Tool with Approval (2026-05-03)

- [x] `tools/shell.py`: `run_shell`, `run_shell_background`, `list_processes`, `read_process_output` — each with `ToolApprovalMeta` attached
- [x] `TOOL_REGISTRY["shell"]` (full) and `TOOL_REGISTRY["shell:read"]` (inspection only)
- [x] `ChatApp._handle_shell` emits `ShellCommandStarted` / `ShellCommandCompleted` around `run_shell` — shared rendering path for user `!` commands and future agent tool calls
- [x] `ShellCommandStarted` / `ShellCommandCompleted` event types defined; renderer subscribed
- [x] Tests: background process lifecycle (start, list, read output), approval metadata for all 4 functions
- [x] Approval gate wired in C5a (2026-05-04)

---

### C2 — Workspace Layout & File→DB Seeding (2026-05-03)

- [x] `SkillsRepo` and `McpServersRepo` added to `src/jac/state/`; wired into `StateStore`
- [x] `src/jac/state/seeder.py`: walks `~/.jac/skills/`, `~/.jac/mcp/`, `.agents/skills/`, `.agents/mcp/` on each boot; upserts rows with `source_scope`/`source_path`; project scope overrides user scope for same name
- [x] Garbage collection: stale file-backed rows deleted on re-seed when not held by an open run
- [x] Duplicate detection within scope: error reported, entry skipped
- [x] `seed_workspace()` called at boot in `ChatApp.open()` and `run_prompt()`
- [x] `jac doctor` extended with `check_workspace_files()`: reports duplicates and missing `.gitignore` entries
- [x] 13 seeder tests covering upsert, override, duplicate error, idempotency, GC, and doctor checks
- [x] **Persistence checkpoint reached**: sessions resume (`jac resume`), workspace files seed into DB

---

### C1 — Persistent State Store (SQLite) (2026-05-03)

- [x] `src/jac/state/` package with `StateStore`, `RunsRepo`, `MessagesRepo`
- [x] `001_initial.sql` migration creates all 12 tables from `STATE_SCHEMA.md` v1.0; `schema_meta` records the applied version
- [x] `aiosqlite` connection with `PRAGMA foreign_keys = ON`; idempotent `open_state_store(db_path)` runs pending migrations
- [x] `run_id` semantics changed: one run per harness invocation, generated on `SessionState`, persisted at first turn, tags every message
- [x] `RunCoordinator` accepts an optional `StateStore`; persists `runs` row and user/assistant `messages` rows; updates run status to `done`/`failed`
- [x] `resume_run(state, settings, run_id)` rebuilds a coordinator with prior message history fed into the next `agent.run(...)` call
- [x] `jac resume <run-id>` Click subcommand drops into the chat loop with restored context
- [x] `/context` reads run id, message count from the DB; cwd and attachments still shown
- [x] State DB lives at `Workspace.state_db_path` (project `.agents/state.db` or `~/.jac/runs/<cwd-hash>/state.db`) — wired through `discover_workspace`
- [x] Tests: 7 cases for the state package (migration, repos, idempotency, FK enforcement) + 5 integration cases (coordinator persistence, run-id stability across turns, resume hydration, missing-run error, no-state fallback) + 2 CLI cases for the `resume` command
- [x] Version bumped to 0.2.0 (intentional break in `run_id` semantics)
