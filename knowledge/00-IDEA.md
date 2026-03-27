# dunder-mifflin-harness — IDEA.md

> A research & development project exploring whether a multi-agent system with smart model routing
> can match or exceed the performance of Anthropic's long-running agentic coding harness,
> at significantly lower cost.

---

## 1. Project Genesis

### Inspiration
Three Anthropic engineering blog posts form the benchmark:
1. **Context Engineering** (Sep 2025) — Context is finite; treat it as a depletable resource. Key techniques: compaction, structured note-taking, sub-agent architectures.
2. **Effective Harnesses for Long-Running Agents** (Nov 2025) — Two-agent pattern (initializer + coder) with file-based handoff. Solved: premature completion, broken state between sessions, untested features.
3. **Harness Design for Long-Running App Development** (Mar 2026) — GAN-inspired three-agent architecture (Planner → Generator → Evaluator). Benchmarks: DAW app in ~4hr at $124 (Opus 4.6). Solo baseline: 20min, $9, broken app.

### Existing Work
- **dunder-mifflin-play** (github.com/VENKATESHWARAN-R/dunder-mifflin-play): Multi-agent system built with Google ADK simulating The Office characters as an IT team. Features role-based specialization, sub-agents, Conference Room orchestration, Temp Agency (A2A protocol), Docker-based isolation.

### Core Hypothesis
> A multi-agent harness with intelligent model routing (model buckets) and structured context engineering
> can achieve comparable quality to Anthropic's harness at 3-5x lower cost, by routing the majority of
> token spend through cheaper models while reserving expensive models for high-leverage tasks.

---

## 2. Key Architectural Decisions

### Framework: Google ADK 2.0 (Alpha)
- Use graph-based orchestration primitives out of the box (nodes, edges, routers)
- Don't reinvent the wheel — leverage ADK's native graph capabilities
- Mix of: agent nodes, direct LLM call nodes, and plain Python function nodes
- Not everything needs to be an agent — use direct LLM calls for simple checks, Python functions for deterministic logic
- Keep graph complexity manageable; add structure only where it earns its keep

### Model Routing: LiteLLM + OpenRouter
- Provider-agnostic via LiteLLM abstraction
- OpenRouter for broader model access and routing
- **Model Buckets** (tiered classification):

| Bucket | Role | Example Models | Typical Tasks |
|--------|------|---------------|---------------|
| **Tier 1 — Scout** | Cheap, fast, focused | Gemini Flash Lite, Claude Haiku 4.5, GPT-5.4 Nano | File reading, boilerplate, simple transforms, status checks, formatting |
| **Tier 2 — Worker** | Capable, balanced | Claude Sonnet 4.6, GPT-5.4 Mini, Gemini Flash/Pro | Feature implementation, code generation, testing, bug fixes, evaluation |
| **Tier 3 — Architect** | Most capable, expensive | Claude Opus 4.6, GPT-5.4, Gemini Ultra | Planning, architecture, complex debugging, high-stakes evaluation |

### Context Storage: SQLite (start simple)
- Single-file database, zero infrastructure
- Start with minimal schema, let it grow based on real usage patterns
- **Core design principle**: Engineer the RIGHT context at the RIGHT moment for the RIGHT agent
  - An implementation agent doesn't need the full PRD — it needs: where to edit, what's failing, what to do
  - An evaluator doesn't need build history — it needs: what was built, acceptance criteria, how to test
- Scoped reads per agent role (column filters, task-level isolation)
- Can migrate to PostgreSQL later if needed

### Tool Strategy: MCP-First
- All tools exposed via MCP servers
- **Key benefit**: Agent code stays identical across environments (local, sandbox, cloud)
  - Changing execution environment = changing MCP server implementation, not agent code
- Curated, minimal tool sets per agent — avoid tool bloat and context confusion
- Skills can be layered on top later once patterns emerge (ADK supports skills natively)
- For v0: direct tool calls via MCP + well-crafted prompts are sufficient

### Execution: Local First
- Start with local execution (like Anthropic did)
- Container-based sandbox later (considering OpenShell or similar)
- Cloud Run / K8s for production headless runs (future)
- MCP abstraction ensures smooth transition between environments

---

## 3. Novel Concepts

### 3.1 Dynamic Model Escalation ("HR Agent")
> **Most original idea in the project.**

When a task assigned to a lower-tier model fails repeatedly:
1. Track pass/fail count per task per model tier
2. After N failures (configurable, e.g., 2-3), escalate to next tier
3. Escalation logic is a **simple Python function** (not an LLM call) checking counters
4. Could also be a lightweight graph node ("HR node") in the workflow
5. Updates the model config in the database for that task type
6. On next loop iteration, the agent/LLM call automatically uses the upgraded model

**Audit & Research Value:**
- Every escalation is logged: task_type, original_model, escalated_model, attempt_count, reason
- Over time, builds a dataset: "task type X needs at minimum Tier Y"
- Enables data-driven model selection defaults for future runs
- This is not just a harness feature — it's a research contribution

**Important constraints:**
- Escalation does NOT mean always starting with the cheapest model
- Initial model assignment should be intelligent (based on task complexity classification)
- Escalation is a safety net, not the primary routing strategy

### 3.2 Configurable Workflow Strategies (Dev Modes)
The harness doesn't lock into one development approach. Instead, development strategies are **configurable workflow modes** — different graph compositions built from shared, reusable nodes. This turns research questions into A/B tests: run the same prompt through different modes, compare cost and quality.

**Candidate Workflow Modes:**

| Mode | Strategy | Flow Pattern |
|------|----------|-------------|
| **Feature-by-feature** | Anthropic's proven approach (TDD-ish) | Plan → decompose → (pick feature → build → test → evaluate) loop |
| **POC-then-swarm** | Build skeleton fast, dispatch workers to harden | Plan → POC build (Tier 3) → evaluate POC → spawn workers (Tier 1-2, parallel) → evaluate |
| **Spec-driven** | Heavy upfront spec, lighter execution | Plan → detailed spec → (implement per spec section → evaluate) loop |
| **Agile/Sprint** | Sprint planning + incremental delivery | Plan → (sprint plan → build sprint → sprint review → retrospective) loop |
| **TDD** | Tests first, then implementation | Plan → generate tests → (build to pass tests → run tests → evaluate) loop |

**Key design principle**: Workflows are compositions of shared nodes. A node library provides the building blocks; each workflow mode is a different wiring diagram.

**V0**: Ship ONE workflow only (feature-by-feature — proven by Anthropic). But design node interfaces for reusability from day one.

**Future**: The `plan` node (or a dedicated strategy-selector node) can intelligently choose which workflow mode to use based on task characteristics — simple apps get POC-swarm (faster), complex apps get feature-by-feature (safer). Workflows can also nest: a feature-by-feature workflow might dispatch a TDD sub-workflow for a particularly critical feature.

**POC-then-swarm specifics (for when this mode is implemented):**
- Phase 1: Tier 3 model builds a functional prototype/skeleton
- Phase 2: Focused Tier 1-2 workers harden, extend, and polish against concrete code
- Risk: bad architectural choices in POC get inherited — needs quality gate between phases
- Advantage: smaller models work against concrete code, not abstract specs; parallel work possible

### 3.3 DB-Stored Agent Configurations
- Agent configs (model tier, prompt templates, tool access, retry limits) stored in database
- Configs are loaded at agent instantiation time
- **Mid-run updates**: Configs can be changed during execution (e.g., by the HR escalation node)
- Next time that agent/LLM call fires, it picks up the updated config
- Enables: dynamic model swapping, prompt tuning during runs, A/B testing

### 3.4 Operation Modes (Autopilot vs HITL)
Implemented as **separate workflow graphs** that share node definitions from a common node library. This is cleaner than one graph with conditional checkpoint nodes — avoids polluting the autopilot flow with HITL conditional logic, and each graph is independently testable.

| Mode | Description | Graph Difference | Use Case |
|------|-------------|-----------------|----------|
| **Autopilot** | Single prompt → autonomous execution → result | Core nodes only, no pause points | Benchmarking, batch processing, overnight runs |
| **HITL** | AI plans & works, pauses at checkpoints for human input | Adds `human_checkpoint` nodes at key decision points | Complex decisions, learning the system, high-stakes tasks |

**Shared nodes**: Both modes reuse the same `plan`, `build`, `evaluate`, `task_router`, `state_writer` nodes.
**HITL-specific nodes**: `human_checkpoint` (pauses graph, presents state, waits for input), `human_feedback_injector` (incorporates human input into task context).

**Design pattern**: Nodes are the building blocks (node library). Workflows are compositions. Operation modes select which workflow graph to execute.

### 3.5 Domain-Agnostic Design
The harness is NOT specific to fullstack web apps. The universal workflow pattern:
```
Plan → Decompose → Implement → Evaluate → Iterate
```
What changes per domain:
- **Tool set** (MCP servers): web dev tools vs. infra tools vs. migration tools
- **Evaluation criteria**: UI testing vs. infra validation vs. test suite passing
- **File/artifact patterns**: source code vs. Terraform vs. config files

Example domains (future):
- Full-stack web app development (v0 benchmark target)
- Infrastructure design and provisioning
- Codebase migration (e.g., Java version upgrades)
- Git repo refactoring / modernization
- Documentation generation
- Vulnerability remediation

### 3.6 Dynamic Skills Attachment (v1+)
Skills are domain-specific knowledge bundles (prompt templates, tool configurations, best practices) that can be dynamically attached to agents based on task domain. ADK natively supports skills.

**Mechanism** (builds on DB-stored agent configs):
1. Task comes in → HR node / strategy selector classifies the domain
2. Looks up relevant skills in DB (e.g., `frontend_skill`, `infra_skill`, `migration_skill`)
3. Injects skill into agent config before instantiation
4. Agent runs with domain-appropriate knowledge and tool access

**Examples:**
- Infra task → attach Terraform/K8s skills to planner and builder agents
- Frontend task → attach React/CSS/accessibility skills
- Migration task → attach version-specific migration guides and patterns

**Not in v0**: Direct tool calls via MCP + well-crafted prompts are sufficient for the initial implementation. Skills add value once we see patterns emerge from real runs across multiple domains.

---

## 4. Anthropic Benchmark Comparison

### What Anthropic Built (Latest — March 2026)

**Architecture**: Planner → Generator → Evaluator (GAN-inspired)
- Single model throughout (Opus 4.5, then Opus 4.6)
- File-based communication (progress files, feature lists, git commits)
- Sprint contracts between generator and evaluator (later removed with Opus 4.6)
- Playwright MCP for end-to-end testing
- Claude Agent SDK for orchestration

**Benchmarks:**

| Setup | Task | Duration | Cost |
|-------|------|----------|------|
| Solo (Opus 4.5) | Retro Game Maker | 20 min | $9 |
| Full harness v1 (Opus 4.5) | Retro Game Maker | 6 hr | $200 |
| Full harness v2 (Opus 4.6) | Browser DAW | ~4 hr | $124 |

**Key Findings:**
- With Opus 4.6, sprint construct was removed — better models need less scaffolding
- Evaluator value depends on task difficulty relative to model capability
- File-based handoff enables clean context resets
- Self-evaluation is unreliable — separate evaluator agent is significantly better

### Where Our Approach Differs

| Aspect | Anthropic | dunder-mifflin-harness |
|--------|-----------|----------------------|
| **Orchestration** | Linear pipeline (Agent SDK) | Graph-based (ADK 2.0) with conditional branching |
| **Models** | Single model (Opus) for everything | Tiered model buckets via LiteLLM |
| **Communication** | Flat files (progress.txt, features.json) | SQLite with scoped reads |
| **Model selection** | Static | Dynamic escalation (HR agent) |
| **Build strategy** | Feature-by-feature (linear) | Configurable: feature-by-feature, POC-swarm, TDD, agile (v0: feature-by-feature) |
| **Tool management** | Direct tool access | MCP abstraction layer |
| **Evaluation** | Playwright MCP | TBD (Playwright MCP likely, plus custom) |
| **Cost target** | $124 (DAW) | $30-50 (similar quality) |

### What We Borrow from Anthropic
- Separate planner/builder/evaluator concerns (proven effective)
- File/artifact-based handoff for clean context boundaries
- Feature list with pass/fail tracking (structured progress)
- Git for version control and rollback
- End-to-end testing via browser automation
- Incremental progress over one-shot attempts

---

## 5. Workflow & Node Architecture

### Design Philosophy
The system is built on three layers of composition:
1. **Nodes** — atomic units of work (LLM calls, Python functions, agent instances)
2. **Workflows** — directed graphs wiring nodes together for a specific development strategy
3. **Modes** — top-level configurations selecting which workflow to run and how (autopilot vs HITL)

This separation means: nodes are reusable across workflows, workflows are reusable across modes, and new strategies can be added by composing existing nodes in new patterns.

### Node Taxonomy

Nodes fall into three categories based on their role:

#### Core Nodes (used by every workflow)
| Node | Type | Description |
|------|------|-------------|
| `plan` | LLM call (Tier 3) | Expands user prompt into spec, decomposes into tasks |
| `build` | LLM call (Tier 1-3) | Implements a task using MCP tools, writes code |
| `evaluate` | LLM call (Tier 2) | Tests implementation, scores against criteria |
| `task_router` | Python function | Picks next task, classifies complexity, selects model |
| `state_writer` | Python function | Persists results to SQLite, commits to git |

#### Strategy Nodes (specific to workflow modes, added as needed)
| Node | Type | Used By | Description |
|------|------|---------|-------------|
| `sprint_planner` | LLM call | Agile mode | Breaks spec into sprint-sized chunks |
| `poc_builder` | LLM call (Tier 3) | POC-swarm mode | Builds functional skeleton/prototype |
| `test_generator` | LLM call | TDD mode | Writes tests before implementation |
| `test_runner` | Python function | TDD mode | Executes test suite, reports pass/fail |
| `swarm_dispatcher` | Python function | POC-swarm mode | Spawns parallel worker tasks |
| `retrospective` | LLM call | Agile mode | Reviews sprint, adjusts approach |
| `strategy_selector` | LLM call / Python fn | Meta | Chooses which workflow mode to use for a given task |

#### Infrastructure Nodes (system concerns, reusable everywhere)
| Node | Type | Description |
|------|------|-------------|
| `context_loader` | Python function | Queries SQLite, builds scoped context for the next LLM call |
| `pass_check` | Python function | Checks eval score, decides: next task / retry / escalate |
| `hr_escalation` | Python function | Model escalation logic — checks fail count, upgrades tier |
| `config_loader` | Python function | Loads agent config from DB (model, prompt, tools, skills) |
| `cost_tracker` | Python function | Accumulates token/cost metrics per run |
| `human_checkpoint` | Python function | HITL only — pauses graph, waits for human input |
| `skill_injector` | Python function | Loads domain skills from DB, attaches to agent config |

### Workflow Composition (V0: Feature-by-Feature)

```
Entry
  │
  ▼
[plan] ──writes──▶ SQLite (tasks table)
  │
  ▼
[task_router] ──reads──▶ SQLite (next pending task)
  │                       │
  │                       ▼
  │               [context_loader] ──reads──▶ SQLite (scoped context)
  │                       │
  ▼                       ▼
[config_loader] ──reads──▶ SQLite (agent config, model tier)
  │
  ▼
[build] ──uses──▶ MCP tools (file write, bash, git)
  │        └──writes──▶ SQLite (task_attempts)
  │
  ▼
[evaluate] ──tests──▶ running app / code checks
  │           └──writes──▶ SQLite (eval_score, feedback)
  │
  ▼
[pass_check]
  │
  ├── Pass ──▶ [state_writer] ──▶ [task_router] (next task, or DONE)
  │
  └── Fail ──▶ [hr_escalation] ──▶ [config_loader] ──▶ [build] (retry)
```

### Sub-Workflow Nesting
Workflows can invoke other workflows as sub-routines. This enables:
- A feature-by-feature workflow dispatching a TDD sub-workflow for a critical feature
- A POC-swarm workflow using feature-by-feature as the "swarm worker" strategy
- Future: AI-selected strategy per task (meta-workflow that picks the best approach)

### Node Interface Contract
Every node, regardless of type, follows the same interface:
- **Input**: Receives state dict (run_id, task_id, context, config)
- **Output**: Returns updated state dict + status (success/failure/needs_human)
- **Side effects**: Reads/writes SQLite, calls MCP tools, writes files
- **Logging**: Every node logs to `task_attempts` with model, tokens, cost, duration

This uniform interface is what makes nodes composable across workflows.

---

## 6. V0 Scope — Solo Baseline

### Goal
Build the minimal harness that can autonomously produce a working application from a single prompt.
Beat the solo baseline: better than $9 / 20 min / broken output.

### What V0 Includes
- [ ] ADK 2.0 graph with feature-by-feature workflow: Plan → Build → Evaluate → (loop or done)
- [ ] Node interfaces designed for reuse (uniform input/output contract) even though only one workflow ships
- [ ] LiteLLM integration with at least 2 model tiers
- [ ] SQLite for run state, task tracking, and agent context
- [ ] MCP server for basic tools (file read/write, bash execution, git)
- [ ] Simple evaluation (does it run? does the basic feature work?)
- [ ] Single mode: Autopilot
- [ ] Local execution only
- [ ] Basic logging and cost tracking

### What V0 Does NOT Include
- HITL mode (v1)
- Dynamic model escalation / HR agent (v1)
- Alternative workflow strategies: POC-swarm, TDD, agile (v1)
- Sub-workflow nesting (v1)
- Dynamic skills attachment (v1+)
- Cloud/sandbox execution (v1+)
- Multi-domain support (v1+)
- Advanced evaluation (Playwright, visual testing) (v1)
- Agent config hot-reloading from DB (v1)
- Strategy auto-selection (v1+)

### V0 Test Case
Same type of task as Anthropic's benchmark — a fullstack web app from a single prompt.
Suggested: something simpler than a DAW, to validate the harness mechanics first.
Example: "Build a task management app with boards, cards, and drag-and-drop."

### Success Criteria for V0
1. App runs without crashing
2. Core feature works end-to-end
3. Total cost < $15
4. Harness completes autonomously (no human intervention)
5. Clear logs showing which model handled which task

---

## 7. Architecture Sketch (Conceptual)

```
                    ┌─────────────────────────────┐
                    │    User Prompt (1-4 lines)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   ADK 2.0 Graph Orchestrator │
                    │   (entry node)               │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   PLAN (Tier 3 LLM call)     │
                    │   Expand prompt → full spec   │
                    │   Decompose into tasks        │
                    │   Write to SQLite             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   TASK ROUTER (Python fn)     │
                    │   Pick next task from DB      │
                    │   Classify complexity          │
                    │   Select model tier            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   BUILD (Tier 1-2 LLM call)  │
                    │   Scoped context from DB      │
                    │   Tools via MCP               │
                    │   Write code, commit to git   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   EVALUATE (Tier 2 LLM call) │
                    │   Test: does it work?         │
                    │   Score against criteria       │
                    │   Write results to DB          │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   Pass?            │
                         │   (Python fn)      │
                         └────┬──────────┬────┘
                              │          │
                         Yes  │          │  No
                              │          │
                    ┌─────────▼─┐  ┌─────▼──────────┐
                    │  Next task │  │  Retry / Escal. │
                    │  or DONE   │  │  (HR logic)     │
                    └────────────┘  └─────────────────┘


    ┌─────────────────────────────────────────────────┐
    │              SQLite Database                      │
    │  ┌──────────┬──────────┬──────────┬───────────┐ │
    │  │  runs    │  tasks   │  outputs │  configs  │ │
    │  └──────────┴──────────┴──────────┴───────────┘ │
    │  ┌──────────┬──────────┐                         │
    │  │  logs    │  models  │                         │
    │  └──────────┴──────────┘                         │
    └─────────────────────────────────────────────────┘
```

### Key Graph Nodes (V0)

| Node | Type | Model Tier | Purpose |
|------|------|-----------|---------|
| `plan` | LLM call | Tier 3 | Expand prompt, generate spec and task list |
| `task_router` | Python function | None | Pick next task, classify, select model |
| `context_loader` | Python function | None | Query SQLite, build scoped context for builder |
| `build` | LLM call | Tier 1-2 | Implement the task using MCP tools |
| `evaluate` | LLM call | Tier 2 | Test and score the implementation |
| `pass_check` | Python function | None | Check eval score, decide: next task / retry / escalate |
| `state_writer` | Python function | None | Update SQLite with results, git commit |

### SQLite Schema (Minimal V0)

```sql
-- Run-level tracking
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    status TEXT DEFAULT 'active',  -- active, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_cost REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0
);

-- Task decomposition
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, passed, failed
    priority INTEGER DEFAULT 0,
    complexity TEXT DEFAULT 'medium',  -- low, medium, high
    assigned_model_tier INTEGER DEFAULT 2,
    current_model TEXT,
    attempt_count INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    parent_task_id TEXT REFERENCES tasks(id),  -- for subtask decomposition
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Agent/LLM outputs per attempt
-- NOTE: model_performance metrics can be derived from this table via queries.
-- A dedicated model_performance table may be added in v1 if query patterns demand it.
CREATE TABLE task_attempts (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id),
    attempt_number INTEGER,
    model_used TEXT NOT NULL,
    model_tier INTEGER NOT NULL,
    input_context TEXT,           -- what context was fed to the model
    output TEXT,                  -- model's response
    eval_score REAL,             -- 0.0 - 1.0
    eval_feedback TEXT,          -- evaluator's critique
    status TEXT,                 -- success, failure, error
    tokens_used INTEGER,
    cost REAL,
    duration_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent configurations (hot-reloadable)
CREATE TABLE agent_configs (
    id TEXT PRIMARY KEY,
    agent_role TEXT NOT NULL,    -- 'planner', 'builder', 'evaluator'
    model_tier INTEGER DEFAULT 2,
    model_override TEXT,         -- NULL = use tier default, or specific model name
    prompt_template TEXT,
    tools_allowed TEXT,          -- JSON array of MCP tool names
    max_context_tokens INTEGER DEFAULT 4000,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simple key-value context store for inter-agent communication
CREATE TABLE context_store (
    key TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id),
    scope TEXT DEFAULT 'global',  -- 'global', 'planner', 'builder', 'evaluator'
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (key, run_id, scope)
);
```

---

## 8. Open Design Questions

### Resolved
- [x] Framework: ADK 2.0 with graph primitives
- [x] Model routing: LiteLLM + OpenRouter
- [x] Context storage: SQLite (start simple)
- [x] Tools: MCP-first
- [x] Execution: Local first
- [x] V0 scope: Solo baseline
- [x] Workflow architecture: Composable nodes → workflows → modes
- [x] Operation modes: Separate graphs sharing node library (not conditional checkpoints)
- [x] Dev strategies: Configurable workflow modes (v0: feature-by-feature only)
- [x] Skills: Out of v0, dynamic attachment via DB configs in v1+
- [x] Node design: Uniform interface contract for reusability

### Open
- [ ] Exact ADK 2.0 graph primitive mapping (which nodes use which ADK types)
- [ ] MCP server design: single server with all tools, or multiple specialized servers?
- [ ] Evaluation strategy for v0: simple "does it run" check, or structured scoring?
- [ ] Prompt engineering: how much do we invest in system prompts for v0?
- [ ] Git strategy: one commit per task? Per attempt? How granular?
- [ ] Compaction strategy: when context grows too large within a single task, how to handle?
- [ ] Task complexity classification: rule-based (Python fn) or LLM-assisted?
- [ ] What's the "right" initial model tier assignment per task type?
- [ ] How to feed evaluation results back as context without polluting the build context?
- [ ] Node state dict schema: what fields are mandatory vs optional?
- [ ] How does ADK 2.0 handle sub-graph invocation? Native or custom wrapper needed?
- [ ] Strategy selector: should the planner choose the workflow mode, or a dedicated node?

---

## 9. Research Questions

These are the things we want to learn from running the harness:

1. **Cost efficiency**: What's the actual cost distribution across model tiers for a typical app build?
2. **Model-task fit**: Which task types genuinely need Tier 3 vs. which are fine with Tier 1?
3. **Escalation patterns**: How often does escalation happen? Does it converge (fewer escalations over time)?
4. **Context scoping impact**: Does giving agents less context actually improve output quality?
5. **Evaluation reliability**: How well do cheaper models evaluate work done by other cheap models?
6. **Graph overhead**: Does multi-agent orchestration overhead negate the model cost savings?
7. **Diminishing returns**: At what point does adding more agent structure stop improving quality?
8. **Workflow strategy comparison**: Same prompt through different dev modes — which produces best cost/quality?
9. **POC vs. feature-by-feature**: Which build strategy produces better results for different app complexities?
10. **Skills impact**: Does attaching domain skills measurably improve output quality vs. prompt-only?
11. **Node reuse effectiveness**: Do shared nodes actually work across different workflow compositions without modification?

---

## 10. References

- [Context Engineering Blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Effective Harnesses Blog](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Harness Design Blog](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [dunder-mifflin-play Repo](https://github.com/VENKATESHWARAN-R/dunder-mifflin-play)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Google ADK](https://google.github.io/adk-docs/)
- [LiteLLM](https://docs.litellm.ai/)
- [OpenRouter](https://openrouter.ai/)
- [Anthropic Quickstart — Autonomous Coding](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding)
- [ADK 2.0 Docs — Graph Workflows](https://google.github.io/adk-docs/workflows/)
- [ADK 2.0 Docs — Dynamic Workflows](https://google.github.io/adk-docs/workflows/dynamic/)
- [ADK 2.0 Docs — Collaborative Agents](https://google.github.io/adk-docs/workflows/collaboration/)
- **IMPLEMENTATION_PLAN.md** — Step-by-step build guide (companion doc)

---

## 11. Session Log

### Session 1 — Initial Brainstorm (March 26, 2026)
- Reviewed all three Anthropic blog posts
- Analyzed dunder-mifflin-play repo structure and capabilities
- Established core hypothesis: multi-agent + model routing = lower cost, comparable quality
- Decided on: ADK 2.0, LiteLLM/OpenRouter, SQLite, MCP, local-first
- Key novel concepts identified:
  - Dynamic model escalation (HR agent)
  - POC-first then swarm build strategy
  - DB-stored agent configs with mid-run updates
  - Dual mode: Autopilot + HITL
- V0 scope defined: solo baseline, minimal harness, autopilot only
- Created IDEA.md for persistent reference

**Refinement round:**
- Evolved "POC-first" into **configurable workflow strategies** — multiple dev modes (feature-by-feature, POC-swarm, TDD, agile, spec-driven) as composable graph workflows
- Redesigned Autopilot/HITL as **separate workflow graphs sharing a node library** (cleaner than conditional checkpoints in one graph)
- Defined **three-layer node taxonomy**: core nodes, strategy nodes, infrastructure nodes
- Added **dynamic skills attachment** concept (v1+): domain skills loaded from DB and injected into agent config based on task classification (same mechanism as HR escalation but for capabilities)
- Established **node interface contract**: uniform input/output (state dict + status) for all nodes regardless of type
- Key design decision: v0 ships ONE workflow but designs nodes for reuse from day one
- Added **sub-workflow nesting** concept: workflows can invoke other workflows (e.g., feature-by-feature dispatching TDD for critical features)
- Updated IDEA.md with all refinements
- **Next steps**: TBD

**Implementation planning round:**
- Reviewed ADK 2.0 alpha docs: graph-based workflows, graph routes, dynamic workflows, collaborative agents, LiteLLM integration
- Key ADK 2.0 primitives confirmed: `Workflow` with `edges`, `Agent` nodes, Python function nodes, route branching, parallel fan-out/join, nested workflows, human input nodes, MCP toolset, Pydantic data contracts
- Created IMPLEMENTATION_PLAN.md with:
  - Full repo structure (lab/, src/harness/, mcp_server/, db/, workspace/)
  - 5-phase implementation approach: scaffolding → lab experiments → core infra → node library → first workflow
  - Detailed lab experiments (numbered scripts for incremental validation)
  - Node-by-node build order
  - ADK 2.0 primitive mapping table
- **Next steps**: Start Phase 0 scaffolding + Phase 1 lab experiments
