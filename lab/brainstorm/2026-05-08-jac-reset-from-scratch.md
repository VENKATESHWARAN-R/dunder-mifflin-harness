# 2026-05-08 — JAC reset from scratch (Phase 0 brainstorm)

> **Status:** Locked brainstorm · **Date:** 2026-05-08 · **Role:** source of truth for the Phase-0 doc rewrites.
>
> Outputs derived from this note: `docs/reference/PHILOSOPHY.md` (rewrite), `docs/contracts/SUBSTRATE.md` (new), `docs/ROADMAP.md` (rewrite), `CLAUDE.md` (edit), `docs/reference/IDEA.md` (light edit), `docs/contracts/STATE_SCHEMA.md` (slim), `docs/README.md` (index edit). Where docs and this note disagree, this note wins until docs are revised.

## Why this exists

User branched fresh and asked to challenge the project from the ground up. State at C8: ~9.6k LOC, ~30% of it CLI/onboarding polish, the actual research spine (build → evaluate → route loop) unbuilt, schema with 4 reserved-but-unused tables, 31-component roadmap for a system that needs ~5 to prove its hypothesis, and persona system prompts living as Python literals in violation of the substrate boundary the project hadn't yet articulated.

The reset doesn't throw away ideas — Scott / Pam / Jim / Dwight, tier routing, SQLite ledger, MCP-first tools, agent-as-tool delegation all stay. It throws away the *order* (multi-agent before single-agent depth), the *over-spec* (workflows for strategies that haven't been needed yet), and the *config-as-code* mistakes.

## Round A — Idea

### A1. Locked product definition

> **JAC is a multi-agent CLI coding harness with two coupled goals.** As a *research harness*, it measures whether four fixed roles — **Michael Scott (manager)**, **Pam Beesly (planner)**, **Jim Halpert (builder)**, **Dwight Schrute (evaluator)** — with tiered model routing can build small applications autonomously at materially lower cost than a single-model baseline; the hypothesis is proven or disproven on the **Notes CLI** benchmark with a **single-function bug-fix** secondary. As a *daily tool*, it is a terminal coding assistant — chat, plan, build, evaluate, resume — backed by SQLite persistence, multi-provider tier routing with in-flight model swap, and a typed event/runtime contract that keeps the runtime independent of the terminal surface.

### A2. Cast

Four fixed personas as **target architecture**:

| Persona | Role | Default tier |
|---|---|---|
| Michael Scott | `manager` | Worker |
| Pam Beesly | `planner` | Architect |
| Jim Halpert | `builder` | Worker |
| Dwight Schrute | `evaluator` | Worker |

Personas live as YAML files loaded by Pydantic AI's `Agent.from_file()` / `Agent.from_spec()`:

```
src/jac/data/personas/    (shipped defaults)
~/.jac/personas/          (user override)
<repo>/.agents/personas/  (project override; project wins)
```

**Critical reframe:** in Phase 1 the cast is **introduced incrementally**, not pre-committed. M1 ships **only Scott**. Subsequent personas are added when M1 logs show what Scott alone struggles with. The 4-persona target stands for Phase 2; what the data actually requires is a Phase-1 outcome.

Code never pattern-matches on persona names. Adding a fifth persona in Phase 2 = drop a YAML file, no Python edits.

### A3. Benchmarks

- **Primary:** Notes CLI (`docs/reference/V0_BENCHMARK.md`).
- **Secondary:** single-function bug-fix in a known repo. Catches "harness only works on greenfield." Spec drafted before M4.

### A4. Kill conditions (any one ends Phase 1)

1. **Cost ratio fails:** JAC ≥ 80% of single-Opus baseline cost for equivalent quality. The 3–5× target is the headline; <1.25× is "interesting research, not a result."
2. **Quality collapses:** can't pass ≥5/7 features on Notes CLI in 5 trials.
3. **Orchestration overhead eats the savings:** planner/evaluator/escalation cost ≥ what was saved by routing builders to lower tiers.

Any of these → Phase 1 ends with a writeup, not more components.

### A5. Substrate boundary (5-substrate rule)

| Substrate | Holds | Why |
|---|---|---|
| **YAML** | persona blueprints, agent specs (consumable by `Agent.from_file`) | Native to Pydantic AI; supports templating, capabilities, model_settings |
| **TOML** | vendor-fact data JAC ships (model context windows, tier defaults), `pyproject.toml` | Vendor knowledge; rarely changes; Python-native |
| **JSON** | user settings/profiles, MCP server stubs | Already used by `pydantic-settings`; MCP ecosystem standard |
| **Markdown** | skills (with frontmatter), `AGENTS.md`, `JAC.md`, archived plans | Free-form prose for human + agent consumption |
| **SQLite** | per-run state — runs, messages, attempts, tasks, per-run agent_configs overrides, MCP/skill bindings | Queryable, FK-enforced, transactional |
| **Python** | code only — runtime, tools, factory, nodes, workflows, schema migrations | If a value changes without code review, it's config |

**The rule:** if it's declarative and stable → TOML. If it's runtime-context settings → JSON. If it's prose meant to be read → Markdown. If it's structured state with relationships → SQLite. If it's an agent-shape spec → YAML. Code never holds config; config never holds code.

### A6. Workflows = small Python files (not YAML)

`pydantic_graph` has no YAML loader. Nodes are Python `@dataclass` subclasses of `BaseNode` whose `run()` method returns the next node — conditional edges live in code. JAC will **not** build a YAML-graph DSL.

- `src/jac/nodes/` — library of `BaseNode` subclasses (when M2/M3 introduces them).
- `src/jac/workflows/feature_by_feature.py` — small file (~50–80 LOC) wiring nodes; ships only when needed (M2 or M3).
- Adding TDD/spec-driven/agile in Phase 2 = drop another small Python file.

Phase 1 ships **at most one** workflow, and only when manual delegation strains.

### A7. Schema redesign for the cut

- `agent_configs.system_prompt` becomes nullable. Canonical prompt source is the persona YAML. DB row is per-run override only.
- `tasks` table active in M1 with a slimmer shape: `task_id`, `run_id`, `title`, `description`, `status`, `order_index`, `created_at`, `updated_at`. Drop `acceptance_criteria`, `tier`, `parent_task_id`, `complexity` until planner/evaluator land.
- Reserved-but-unused tables (`agent_instances`, `agent_teams`, `agent_messages`, `context_store`) **dropped** from the M1 migration. They re-enter the schema only when an evidence-gated Phase-2 component needs them.
- `attempts` slimmed: drop `parent_attempt_id`, `is_minion`, `parent_role`, `depth`. Single-row attempts in M1; multi-agent columns return when needed.
- Schema version bumps to 1.5 when the slim ships.

### A8. Multi-provider + in-flight model swap = M1 capability

`jac init` lets users pick provider and per-tier models. Tier resolution at every turn is a 3-step lookup, first match wins:

1. **Per-run override** (SQLite `agent_configs.model_override` or `agent_configs.tier`) — slash command, escalation, hot-reload.
2. **User preference** (JSON `settings.json` or `JAC_PROFILE_*` env).
3. **Shipped default** (TOML `data/model_specs.toml`).

This collapses the original C8 (tier routing), C18 (mid-run toggles), C20 (hot-reload) into a single rule built into the agent factory from M1 day one.

## Round B — Principles

The user's starting list (simple, testable, documented, maintainable, extensible, scalable, secure, reliable, performant) was retired by name: too many overlap, too many premature ("scalable"/"performant" for a single-process local harness), too few have teeth ("simple" gives an agent no decision rule). Kept as **working rules** in `CLAUDE.md`: testable and documented as hygiene practices.

Five principles, each rules something *out*:

### B1. The runtime is the contract; the UI is one surface.

The CLI is product, but anything that should outlive the terminal surface (browser, A2A, headless) lives in `runtime/`. UI may import runtime; runtime must not import UI.

**Rules out:** Rich rendering inside `runtime/`, slash-command logic that touches business state, `prompt_toolkit` references outside `cli/`.

### B2. Keep the spine thin.

JAC's spine: prompt → plan → route → build → evaluate → ledger → report. New structure (modules, abstractions, hooks, middleware) is justified only by a logged run that demonstrates the spine breaking under load.

**Rules out:** speculative components, premature abstraction, scaling-problem solutions before scaling problems exist.

### B3. Measure before you abstract.

No new persona, tool, contract, or dependency without a recorded ledger entry (attempts row, run trace, cost log) that motivates it. The ledger is the gate.

**Rules out:** new tools added "because the agent might need them," new personas added because they'd be cool, contracts written before code that uses them.

### B4. State is durable; agents are disposable.

Runs, messages, attempts, tasks, configs live in SQLite. Agent instances, sessions, in-memory caches are reconstructible from state at any time. Resume from disk must produce the same behaviour as the live session.

**Rules out:** in-memory caches that aren't re-derivable, agent state that can't survive restart, "soft" caches that mutate state without ledgering.

### B5. Config-as-data, not config-as-code.

Declarative values live in YAML/TOML/JSON/Markdown by the substrate rule. Python files contain code, never tunable values. Code patterns matching on role names, model names, persona names is forbidden.

**Rules out:** `personas.py` Python strings, `modes.py` slash-mode prompts, `if role == "manager":` branches, model lists hardcoded in seed functions.

## The real harness — long-running task orchestration

LLM context windows are finite. A harness that loses the plot at compaction is just a chatbot. **The `tasks` table is what makes JAC a harness** — it's the only context-resilient memory:

- Scott maintains the task list via tools (`add_task`, `update_task`, `complete_task`, `list_tasks`).
- The current task list is injected as a system reminder every turn — it travels outside the message history that compaction reshapes.
- On resume after process restart, the list reorients Scott against work-in-flight without re-deriving from chat history.
- After compaction, Scott reads the list to know what's done, what's next, and what blocked.

This is part of M1, not a deferred feature. Without it, "deeply tuned single agent" is just a polished chatbot. With it, Scott is a harness even before any other persona arrives.

## Round C — Roadmap

### Phase 1 — five milestones

#### M1. Scott as a deeply-tuned single agent + A2A + long-running task orchestration

**Ships:**
- Module layout: `cli/`, `runtime/`, `agents/`, `tools/`, `state/`, `surfaces/a2a/`.
- YAML persona system loaded from `data/personas/scott.yaml` only (one file).
- Multi-provider model resolution + in-flight tier/model swap (slash commands write to `agent_configs`; next turn picks up).
- Slim CLI (~500–600 LOC max): chat, `/model`, `/tier`, `/context`, `/history`, `/clear`, attachments, `!` shell, `@` file refs, `jac chat`, `jac init`, `jac doctor`, `jac resume`.
- A2A peer surface via `agent.to_a2a()` (~80 LOC adapter under `surfaces/a2a/`).
- Full Scott tool surface: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep_files`, `run_shell`, `run_shell_background`, `read_process_output` — all approval-aware.
- **Task-list CRUD tools:** `add_task`, `update_task`, `complete_task`, `list_tasks`. Scott uses these; system prompt teaches the discipline.
- **Context engineering:** history processor (drop tool-result noise from past turns; keep user prompts and decisions), scoped tool-output handling, token-budget awareness, `AGENTS.md` (project) + `JAC.md` (user) instruction injection.
- State: `runs`, `messages`, `attempts` (single-row), `agent_configs` (single row), `tasks` active, `mcp_servers` / `skills` registries (markdown→DB seeder kept).
- **Scott's YAML iterated 10+ times against real run logs** — prompt engineering is the deliverable.

**Acceptance:**
- A2A round-trip: external peer dispatches a task via A2A, gets a result.
- 3 internal autonomous tasks with task-list lifecycle visible in the `tasks` table (created → in_progress → completed) across compactions / resumes.

#### M2. Decompose by evidence

Read M1 logs. Identify the single biggest failure mode of Scott alone (likely candidates: planning depth, evaluation honesty, multi-step tracking, context bloat). Add **the one persona** that addresses it. Coordination via Pydantic AI agent-as-tool delegation — no `pydantic_graph` yet.

**Acceptance:** the added role measurably improves the class of tasks Scott struggled with. The M5 report can write "Scott alone failed at X; adding Y closed the gap."

#### M3. Add what's needed for autonomy + Notes CLI

Continue evidence-driven additions. By end of M3 the cast is whatever the data says (likely 2–4 personas). `pydantic_graph` is introduced **only when** manual delegation strains under retry/route logic. Notes CLI benchmark runs end-to-end.

**Acceptance:** Notes CLI ≥5/7 features pass autonomously, total cost <$15, ≥2 tiers exercised. Cast composition justified by M1+M2 logs.

#### M4. Tier discipline + secondary benchmark

Per-role `/tier` (if multi-role by now). In-flight tier swap exercised under load. Single-function bug-fix benchmark wired and run.

**Acceptance:** both benchmarks pass; tier mix in ledger reflects deliberate decisions, not defaults.

#### M5. Comparison report — THE DELIVERABLE

Read SQLite ledger. Generate cost ratio, tier mix, attempts breakdown, eval pass rates. Compare:
- Scott-alone (M1 logs) vs single-Opus baseline.
- JAC-final-cast (M3+M4) vs single-Opus baseline.
- Both benchmarks: Notes CLI + bug-fix.

Three kill-conditions evaluated against data.

**Acceptance:** markdown report committed to repo. Phase 1 ends with a published finding or a documented pivot. The deliverable is the *data*, not a polished tool.

### Cut list (M1 rebuild)

| Cut | Why | LOC saved (rough) |
|---|---|---|
| `agents/spawn.py`, `agents/result_filter.py`, `tools/summarize.py`, `tools/cache.py`, `agents/tools.py:make_summon_jim_tool` | No second agent in M1; multi-agent infra returns when needed | ~600 |
| `agents/personas.py`, `agents/modes.py`, `agents/plans.py` (Python persona/mode literals) | Replaced by `data/personas/scott.yaml`; B5 violation | ~200 |
| Slash-mode addendum machinery (`MODE_PROMPTS`, `submit_slash_run`), `/plan`, `/init` | No mode dispatch needed for solo Scott; reintroduce when planner arrives | ~230 |
| `agents/seeds.py` shrinks to `ensure_scott_config` | Single role | ~120 |
| `runtime/coordinator.py` `if role == "manager"` branches, `summon_jim` injection | B5 violation; replaced by config-driven dispatch | ~80 |
| Reserved tables: `agent_instances`, `agent_teams`, `agent_messages`, `context_store` | Dropped from M1 migration; return only via Phase-2 evidence | drop 4 tables |
| `attempts.parent_attempt_id`, `is_minion`, `parent_role`, `depth`; `tasks.acceptance_criteria`/`tier`/`parent_task_id`/`complexity` | Single-agent M1 doesn't need them | column drops |
| `cli/app.py` undo stack, destructive guard, retry handler, `/save`, `/capabilities`, `/history` file write | Daily-driver polish; reintroduce in Phase 2 if data warrants | ~250 |
| Persona-display rendering theater in `cli/renderer.py` | One persona, one shape | ~80 |
| `onboarder.py` slimming | Multi-provider wizard kept; persona/skill seeding moves to file templating | ~200 |

**Estimated cut: ~1,800 LOC** out of 9,611 today. **Estimated additions in M1:** ~80 LOC A2A adapter + ~150 LOC context/history processor + ~50 LOC instruction injection = ~280 LOC. **Net target M1:** ~5,200–5,500 LOC.

### Phase 2 — evidence-gated catalog (no order, no commitment)

Each entry has a **data trigger** — what M5 must show for the item to advance to a real component:

| Item | Original C ID | Data trigger to promote |
|---|---|---|
| Compaction module | C12 | M1/M3 runs hit context limits |
| Hooks / callbacks | C13 | Deterministic post-step (typecheck/lint) measurably improves quality |
| Skills (dynamic injection) | C16 | Skill-attached run beats skill-less on bug-fix benchmark |
| Mid-run toggles slash UI | C18 | Users hit a friction point that mid-run toggling solves |
| HR escalation | C19 | M2's deterministic retry shows escalation patterns worth automating |
| Hot-reload UX polish | C20 | Multi-iteration sessions show frequent config changes |
| HITL mode | C21 | Cost-quality changes meaningfully when a human reviews plans |
| Alt strategies (TDD/spec/agile) | C22 | Strategy variation moves cost-quality on Notes CLI |
| Sub-workflow nesting | C23 | Alt strategies need composition |
| Strategy auto-selection | C24 | 2+ strategies exist and selection isn't trivial |
| Browser eval (Playwright) | C25 | Web-app benchmark added |
| Container execution | C26 | Local execution causes a real safety incident |
| Cloud headless | C27 | Overnight benchmarks become a thing |
| Browser UI | C28 | Post-product; another user wants it |
| A2A server polish | C29 | M1 minimal A2A surface needs production hardening |
| Multi-repo A2A | C30 | Speculative; very low priority |
| Agent teams + `agent_messages` | C15 | A workflow needs concurrent specialists, not just delegation |
| Remote MCP transports | C17 | A benchmark needs an MCP server JAC doesn't ship locally |

### Doc-rewrite plan (this session)

| # | File | Action | Status |
|---|---|---|---|
| 1 | `lab/brainstorm/2026-05-08-jac-reset-from-scratch.md` | **new** (this file) | Locked brainstorm |
| 2 | `docs/reference/PHILOSOPHY.md` | rewrite — A1 paragraph + B1–B5 + dependency-direction layer guide | Reference |
| 3 | `docs/contracts/SUBSTRATE.md` | new — locks the 5-substrate rule with concrete examples + violations | Locked |
| 4 | `docs/ROADMAP.md` | rewrite — replace 31 components with M1–M5 + Phase-2 catalog; preserve Done section | Living |
| 5 | `CLAUDE.md` | edit — align "Where to Put a New X", drop multi-agent guidance, note Scott-only M1, add testable+documented hygiene rules | n/a |
| 6 | `docs/reference/IDEA.md` | light edit — add "Locked product definition (2026-05-08)" section at top | Reference |
| 7 | `docs/contracts/STATE_SCHEMA.md` | slim — drop reserved tables, slim attempts/tasks, mark system_prompt nullable, document 3-tier model resolution; bump to 1.5 | Locked |
| 8 | `docs/README.md` | edit — add SUBSTRATE.md row, note implementation_docs as historical | n/a |

### Sequence after this brainstorm

1. **This session (locked):** the 8 doc artifacts.
2. **Next session — M1 code rebuild:** start fresh on this branch. Module layout, persona YAML, A2A adapter, task-list tools, context engineering. Iterate Scott's YAML against real runs.
3. **Subsequent sessions — M2 onward:** evidence-driven decomposition.

## What was retired

For the record, so future sessions don't accidentally re-introduce these without evidence:

- The 9-principle list (simple/testable/documented/maintainable/extensible/scalable/secure/reliable/performant) — replaced by 5 with teeth. Hygiene moves to `CLAUDE.md` working rules.
- The 31-component roadmap — replaced by M1–M5 + Phase-2 catalog. C0–C8 stay in the Done section as historical record.
- "All four personas pre-committed in M1" framing — replaced by Scott-solo M1 + evidence-driven decomposition.
- "Workflows-as-YAML" — pydantic_graph isn't YAML-loadable; workflows stay small Python files.
- `personas.py` / `modes.py` Python literals as the canonical persona source — replaced by YAML files under `data/personas/`.
- `spawn_minion` + `result_filter` + `read_file_smart` + `summarize` + `cache` — solving a scaling problem before the spine works. Returns only if M1+ data shows it's needed.
- Slash-mode addendums (`/plan`, `/init`) as prompt-injection — replaced by either no special mode (Scott handles it) or, when planners arrive, a workflow selection.

## Open questions deferred to follow-up sessions

These don't block the doc rewrites or M1 code, but flag them so they aren't lost:

- **Bug-fix benchmark spec** — defer drafting `docs/reference/V0_BENCHMARK_SECONDARY.md` until M3.
- **Slash command final list for M1** — current proposal: `/help`, `/quit`, `/clear`, `/history`, `/context`, `/model`, `/tier`, `/approval`, `/resume`. Confirm before M1 code.
- **`AGENTS.md` / `JAC.md` frontmatter rules** — what fields are allowed, how they merge with persona YAML. Locked for M1 when context-engineering work begins.
- **Compaction strategy in M1** — minimum viable is "drop tool noise from history, keep user prompts + decisions + task list (which is external)." Whether that's sufficient is an M1 finding.
- **Single-agent A2A surface scope** — what does Scott-as-A2A-peer expose? Probably "submit a task, stream events, return final result." Detailed spec when A2A code lands.
- **CLI / event / tools / workspace / MCP contract slimming** — deferred to follow-up sessions; do as M1 lands and we know the exact shipped surface.

## Decision log (in case anything is challenged)

| Round | Decision | Rationale |
|---|---|---|
| A1 | Dual-goal product definition | Single-goal "research only" was too narrow; user explicitly wants daily-tool usability |
| A2 | 4 fixed personas, incremental Phase-1 introduction | Pre-committing the cast is a B3 violation; cast is a Phase-1 outcome |
| A3 | Notes CLI primary + bug-fix secondary | Catches "harness only works on greenfield" |
| A4 | Three kill conditions | Hypothesis must be falsifiable; without explicit exits, the project can't end |
| A5 | 5-substrate rule | Codifies the boundary the codebase already implicitly wanted; agents need an explicit rule |
| A6 | Workflows = Python, not YAML | Verified `pydantic_graph` has no YAML loader; building a DSL on top would violate B2 |
| A7 | Schema slim + system_prompt nullable | Locked but unused tables are speculation; YAML being canonical means DB shouldn't duplicate |
| A8 | Multi-provider + in-flight swap in M1 | Collapses C8 + C18 + C20 into one factory rule; user explicitly requires it |
| B | 5 principles with teeth | Original 9 had no decision-rule shape; AI agents need rules they can apply |
| C-M1 | Scott-solo M1 with task list as the real harness | Inverts depth-vs-breadth; long-running task orchestration is what makes JAC a harness, not a chatbot |
| C-cuts | ~1,800 LOC cut, ~280 LOC added | Net target ~5,200–5,500 LOC; cuts target multi-agent infra and CLI polish that don't earn keep |
