# Roadmap — dunder-mifflin-harness

One slice per weekend. Each slice ends with something runnable and demoable. Sunday night = ship + post a 30-second demo somewhere (even one friend counts).

**Rules:**
- Don't redesign earlier slices. W6 is the refactor weekend.
- Each slice is allowed to be embarrassing.
- Finish early → ship and stop. Don't roll into the next one.
- Found something to fix/improve along the way? Add it under "Backlog" below.

---

## Slices

### W1 — "It remembers"
- [ ] Create `state/` package with SQLite backend (schema from `docs/STATE_SCHEMA.md`)
- [ ] Persist chat sessions and message history (`runs` + `messages` tables)
- [ ] `harness resume <run-id>` restores a prior session
- [ ] `/context` shows persisted session context, not just in-memory attachments
- [ ] Headless and chat mode share the same session store
- [ ] Demo posted

### W2 — "It can safely act"
- [ ] Define the first agent tool contract on top of `runtime.events`
- [ ] Wire `read_file`, `write_file`, and `edit_file` tools
- [ ] Agent-requested file writes show Rich diff previews before approval
- [ ] Approval responses flow back to the runtime as structured decisions
- [ ] It can create or edit a file on request
- [ ] Demo posted

### W3 — "It can use the shell"
- [ ] Add an agent shell tool using the shared shell executor
- [ ] Agent-requested shell commands require approval by default
- [ ] Shell output is truncated with head/tail preservation
- [ ] User `!` shell shortcuts and agent shell tools share logging/rendering behavior
- [ ] Demo posted

### W4 — "It plans before it acts"
- [ ] Create `agents/` package with `base.py` (config_loader — the only Agent() factory)
- [ ] Planner agent returns a small structured task plan (Pydantic AI agent delegation, no graph yet)
- [ ] Builder agent executes one planned task using tools
- [ ] Small task runs end-to-end (e.g. fibonacci script)
- [ ] Demo posted

### W5 — "It has state and cost"
- [ ] Persistent run/task/attempt store exists
- [ ] Per-call model, token, duration, and cost records are captured
- [ ] `/cost` prints a useful run/session breakdown
- [ ] `CostUpdated` events are emitted at run end or model changes
- [ ] Demo posted

### W6 — "It has tiers"
- [ ] Tier-3 model for planner
- [ ] Tier-2 model for builder
- [ ] Tier-1 model for at least one cheap/read-only node
- [ ] Two providers minimum
- [ ] Model/tier slash commands update session config safely
- [ ] Demo posted

### W7 — "It evaluates itself"
- [ ] `evaluate` node runs after build
- [ ] LLM grades pass/fail against task criteria
- [ ] One retry loop on failure
- [ ] Ambiguous task fails eval, retries, passes
- [ ] Demo posted

### W8 — "It's a graph"
- [ ] Create `nodes/` package with uniform `BaseNode` state-in/state-out interface (via `pydantic_graph`)
- [ ] Create `workflows/feature_by_feature.py` wiring existing planner/builder/evaluator onto the graph
- [ ] W7 functionality refactored onto the graph — no behaviour change, just new wiring
- [ ] `RunCoordinator` delegates to the `pydantic_graph` Graph runner without changing CLI event contracts
- [ ] Demo posted

### W9+ — "It picks up tasks"
- [ ] `task_router` node reads pending tasks from the SQLite `tasks` table
- [ ] `context_loader` node builds scoped context per agent role from `context_store`
- [ ] Multi-task runs work end-to-end
- [ ] Full V0 cycle: plan → router → loader → build → evaluate (V0 benchmark: Notes CLI)
- [ ] See `docs/V0_BENCHMARK.md` for acceptance criteria
- [ ] Demo posted

---

## Backlog

Stuff to fix/improve found along the way. Pull from here when a slice finishes early, or schedule into a future weekend.

- [ ] Add machine-readable headless output once workflows produce structured run data
- [ ] Add browser UI adapter over the same runtime events after terminal flows stabilize
- [ ] Add A2A server adapter after sessions and persistent state are real
- [ ] Add richer prompt_toolkit completions for slash commands and file paths
- [ ] Consider sandboxed shell execution after local shell behavior is proven
- [ ] Remove `cli.py` root compat shim (clean up at W6 refactor)
- [ ] Fix `just typecheck` justfile — `main.py` does not exist, check `src tests` only
- [ ] Add `/disable mcp:<name>` and `/disable skill:<name>` slash commands (v1+ mid-run toggle)

---

## Done

Move slices here once shipped, with the date.

### W0 — "CLI/runtime foundation" (2026-04-27)
- [x] `docs/CLI_DESIGN.md` captures CLI philosophy, input grammar, events, approvals, questions, and extension rules
- [x] Click entrypoint supports `harness "prompt"` and `harness chat`
- [x] prompt_toolkit chat input exists
- [x] Rich renderer subscribes to runtime events
- [x] Runtime has typed events, approvals, questions, session state, and `RunCoordinator`
- [x] `@` file references become structured attachments
- [x] User `!` shell commands use a shared shell executor
- [x] Focused tests cover CLI entrypoints, parser, renderer, runtime contracts, config, and shell helpers

---

## Notes

Anything worth remembering — a decision, a dead end, a trick that worked. Keep it short.

- CLI is now an adapter over `runtime/`, not the orchestrator.
- Keep approvals and questions separate; do not reuse approval prompts for clarification questions.
- Future browser UI and A2A server should consume the runtime event/request contract, not CLI modules.
- Orchestration library locked: Pydantic AI throughout. `pydantic_graph` for graph layer (W8+).
- State store locked: SQLite. Schema contract in `docs/STATE_SCHEMA.md` — update that doc before touching DB code.
- V0 benchmark locked: Notes CLI. Full spec in `docs/V0_BENCHMARK.md`.
- HITL mode is v1+. Do not add HITL checkpoints to any v0 slice.
- `agents/base.py` (config_loader) is the sole factory for live Pydantic AI agents. Nothing else calls `Agent(...)` directly.
