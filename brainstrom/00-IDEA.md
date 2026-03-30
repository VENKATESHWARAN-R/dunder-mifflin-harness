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
- **dunder-mifflin-play** (github.com/VENKATESHWARAN-R/dunder-mifflin-play): Multi-agent system simulating The Office characters as an IT team. Features role-based specialization, sub-agents, orchestrator-level routing, an A2A protocol for inter-team handoffs, and Docker-based isolation.

### Core Hypothesis
> A multi-agent harness with intelligent model routing (model buckets) and structured context engineering
> can achieve comparable quality to Anthropic's harness at 3-5x lower cost, by routing the majority of
> token spend through cheaper models while reserving expensive models for high-leverage tasks.

---

## 2. Core Design Principles

### Orchestration: Graph-Based Workflows
- The harness is structured as a directed graph of nodes connected by conditional edges
- Not everything needs to be an agent — LLM calls handle reasoning, deterministic logic handles routing and state management
- Keep graph complexity manageable; add structure only where it earns its keep
- Nodes are the building blocks; workflows are compositions of nodes

### Model Routing: Tiered Model Buckets
- Provider-agnostic routing through a unified LLM interface
- Models are classified into tiers based on capability and cost, not by vendor
- **Model Buckets** (tiered classification):

| Bucket                 | Role                    | Typical Tasks                                                           |
| ---------------------- | ----------------------- | ----------------------------------------------------------------------- |
| **Tier 1 — Scout**     | Cheap, fast, focused    | File reading, boilerplate, simple transforms, status checks, formatting |
| **Tier 2 — Worker**    | Capable, balanced       | Feature implementation, code generation, testing, bug fixes, evaluation |
| **Tier 3 — Architect** | Most capable, expensive | Planning, architecture, complex debugging, high-stakes evaluation       |

### Context Storage: Structured & Scoped
- Run state, task progress, and agent configurations are stored in a persistent structured store
- **Core design principle**: Engineer the RIGHT context at the RIGHT moment for the RIGHT agent
  - An implementation agent doesn't need the full PRD — it needs: where to edit, what's failing, what to do
  - An evaluator doesn't need build history — it needs: what was built, acceptance criteria, how to test
- Context reads are scoped per agent role — agents see only what's relevant to their task
- Start minimal; let the schema grow based on real usage patterns

### Tool Abstraction Layer
- Tools are exposed through a standardized interface, keeping agent logic identical across environments
- **Key benefit**: Changing execution environment means changing the tool implementation, not the agent code
- Curated, minimal tool sets per agent — avoid tool bloat and context confusion
- For v0: direct tool calls with well-crafted prompts are sufficient

### Execution: Local First
- Start with local execution (like Anthropic did)
- Container-based sandbox isolation later
- Cloud headless execution in future
- The tool abstraction layer ensures smooth transition between environments

---

## 3. Novel Concepts

### 3.1 Dynamic Model Escalation ("HR Agent")
> **Most original idea in the project.**

When a task assigned to a lower-tier model fails repeatedly:
1. Track pass/fail count per task per model tier
2. After N failures (configurable, e.g., 2-3), escalate to the next tier
3. Escalation logic is deterministic (not an LLM call) — just checking counters
4. A dedicated escalation step in the workflow updates the model assignment for that task
5. On the next loop iteration, the agent automatically uses the upgraded model

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

| Mode                   | Strategy                                        | Flow Pattern                                                                              |
| ---------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Feature-by-feature** | Anthropic's proven approach (TDD-ish)           | Plan → decompose → (pick feature → build → test → evaluate) loop                          |
| **POC-then-swarm**     | Build skeleton fast, dispatch workers to harden | Plan → POC build (Tier 3) → evaluate POC → spawn workers (Tier 1-2, parallel) → evaluate  |
| **Spec-driven**        | Heavy upfront spec, lighter execution           | Plan → detailed spec → (implement per spec section → evaluate) loop                       |
| **Agile/Sprint**       | Sprint planning + incremental delivery          | Plan → (sprint plan → build sprint → sprint review → retrospective) loop                  |
| **TDD**                | Tests first, then implementation                | Plan → generate tests → (build to pass tests → run tests → evaluate) loop                 |

**Key design principle**: Workflows are compositions of shared nodes. A node library provides the building blocks; each workflow mode is a different wiring diagram.

**V0**: Ship ONE workflow only (feature-by-feature — proven by Anthropic). But design node interfaces for reusability from day one.

**Future**: A strategy-selector step can intelligently choose which workflow mode to use based on task characteristics — simple apps get POC-swarm (faster), complex apps get feature-by-feature (safer). Workflows can also nest: a feature-by-feature workflow might dispatch a TDD sub-workflow for a particularly critical feature.

**POC-then-swarm specifics (for when this mode is implemented):**
- Phase 1: Tier 3 model builds a functional prototype/skeleton
- Phase 2: Focused Tier 1-2 workers harden, extend, and polish against concrete code
- Risk: bad architectural choices in POC get inherited — needs quality gate between phases
- Advantage: smaller models work against concrete code, not abstract specs; parallel work possible

### 3.3 Stored Agent Configurations (Hot-Reloadable)
- Agent configs (model tier, prompt templates, tool access, retry limits) are stored in persistent state
- Configs are loaded at agent instantiation time, not hardcoded
- **Mid-run updates**: Configs can be changed during execution (e.g., by the escalation step)
- Next time that agent fires, it picks up the updated config
- Enables: dynamic model swapping, prompt tuning during runs, A/B testing across runs

### 3.4 Operation Modes (Autopilot vs HITL)
Implemented as **separate workflow compositions** that share the same node library. This is cleaner than one workflow with conditional checkpoint logic — each mode is independently testable.

| Mode          | Description                                             | Workflow Difference                                | Use Case                                                  |
| ------------- | ------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| **Autopilot** | Single prompt → autonomous execution → result           | Core nodes only, no pause points                   | Benchmarking, batch processing, overnight runs            |
| **HITL**      | AI plans & works, pauses at checkpoints for human input | Adds human checkpoint steps at key decision points | Complex decisions, learning the system, high-stakes tasks |

**Shared steps**: Both modes reuse the same plan, build, evaluate, route, and persist steps.
**HITL-specific steps**: Human checkpoint (pauses workflow, presents state, waits for input), feedback injector (incorporates human input into task context).

**Design pattern**: Steps are the building blocks (step library). Workflows are compositions. Operation modes select which workflow graph to execute.

### 3.5 Domain-Agnostic Design
The harness is NOT specific to fullstack web apps. The universal workflow pattern:
```
Plan → Decompose → Implement → Evaluate → Iterate
```
What changes per domain:
- **Tool set**: web dev tools vs. infra tools vs. migration tools
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
Skills are domain-specific knowledge bundles (prompt templates, tool sets, best practices) that can be dynamically attached to agents based on the task domain.

**Mechanism** (builds on stored agent configs):
1. Task comes in → strategy selector classifies the domain
2. Looks up relevant skills in the config store (e.g., `frontend_skill`, `infra_skill`, `migration_skill`)
3. Injects skill into agent config before the agent runs
4. Agent executes with domain-appropriate knowledge and tool access

**Examples:**
- Infra task → attach Terraform/K8s skills to planner and builder agents
- Frontend task → attach React/CSS/accessibility skills
- Migration task → attach version-specific migration guides and patterns

**Not in v0**: Direct tool calls with well-crafted prompts are sufficient for the initial implementation. Skills add value once patterns emerge from real runs across multiple domains.

---

## 4. Anthropic Benchmark Comparison

### What Anthropic Built (Latest — March 2026)

**Architecture**: Planner → Generator → Evaluator (GAN-inspired)
- Single model throughout (Opus 4.5, then Opus 4.6)
- File-based communication (progress files, feature lists, git commits)
- Sprint contracts between generator and evaluator (later removed with Opus 4.6)
- End-to-end browser testing via Playwright
- Claude Agent SDK for orchestration

**Benchmarks:**

| Setup                      | Task             | Duration | Cost |
| -------------------------- | ---------------- | -------- | ---- |
| Solo (Opus 4.5)            | Retro Game Maker | 20 min   | $9   |
| Full harness v1 (Opus 4.5) | Retro Game Maker | 6 hr     | $200 |
| Full harness v2 (Opus 4.6) | Browser DAW      | ~4 hr    | $124 |

**Key Findings:**
- With Opus 4.6, sprint construct was removed — better models need less scaffolding
- Evaluator value depends on task difficulty relative to model capability
- File-based handoff enables clean context resets
- Self-evaluation is unreliable — separate evaluator agent is significantly better

### Where Our Approach Differs

| Aspect              | Anthropic                                | dunder-mifflin-harness                                                            |
| ------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| **Orchestration**   | Linear pipeline                          | Graph-based with conditional branching                                            |
| **Models**          | Single model (Opus) for everything       | Tiered model buckets with intelligent routing                                     |
| **Communication**   | Flat files (progress.txt, features.json) | Structured persistent store with scoped reads                                     |
| **Model selection** | Static                                   | Dynamic escalation (HR agent)                                                     |
| **Build strategy**  | Feature-by-feature (linear)              | Configurable: feature-by-feature, POC-swarm, TDD, agile (v0: feature-by-feature)  |
| **Tool management** | Direct tool access                       | Abstracted tool interface (environment-independent)                               |
| **Evaluation**      | Browser automation (Playwright)          | TBD — browser automation likely, plus structured scoring                          |
| **Cost target**     | $124 (DAW)                               | $30-50 (similar quality)                                                          |

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
1. **Nodes** — atomic units of work (LLM calls, deterministic logic, agent instances)
2. **Workflows** — directed graphs wiring nodes together for a specific development strategy
3. **Modes** — top-level configurations selecting which workflow to run and how (autopilot vs HITL)

This separation means: nodes are reusable across workflows, workflows are reusable across modes, and new strategies can be added by composing existing nodes in new patterns.

### Node Taxonomy

Nodes fall into three categories based on their role:

#### Core Nodes (used by every workflow)
| Node           | Type                | Description                                              |
| -------------- | ------------------- | -------------------------------------------------------- |
| `plan`         | LLM call (Tier 3)   | Expands user prompt into spec, decomposes into tasks     |
| `build`        | LLM call (Tier 1-3) | Implements a task using available tools, writes code     |
| `evaluate`     | LLM call (Tier 2)   | Tests implementation, scores against acceptance criteria |
| `task_router`  | Deterministic       | Picks next task, classifies complexity, selects model    |
| `state_writer` | Deterministic       | Persists results to state store, commits to git          |

#### Strategy Nodes (specific to workflow modes, added as needed)
| Node                | Type                     | Used By        | Description                                          |
| ------------------- | ------------------------ | -------------- | ---------------------------------------------------- |
| `sprint_planner`    | LLM call                 | Agile mode     | Breaks spec into sprint-sized chunks                 |
| `poc_builder`       | LLM call (Tier 3)        | POC-swarm mode | Builds functional skeleton/prototype                 |
| `test_generator`    | LLM call                 | TDD mode       | Writes tests before implementation                   |
| `test_runner`       | Deterministic            | TDD mode       | Executes test suite, reports pass/fail               |
| `swarm_dispatcher`  | Deterministic            | POC-swarm mode | Spawns parallel worker tasks                         |
| `retrospective`     | LLM call                 | Agile mode     | Reviews sprint, adjusts approach                     |
| `strategy_selector` | LLM call / Deterministic | Meta           | Chooses which workflow mode to use for a given task  |

#### Infrastructure Nodes (system concerns, reusable everywhere)
| Node               | Type          | Description                                                      |
| ------------------ | ------------- | ---------------------------------------------------------------- |
| `context_loader`   | Deterministic | Queries state store, builds scoped context for the next LLM call |
| `pass_check`       | Deterministic | Checks eval score, decides: next task / retry / escalate         |
| `hr_escalation`    | Deterministic | Model escalation logic — checks fail count, upgrades tier        |
| `config_loader`    | Deterministic | Loads agent config (model, prompt, tools, skills) from store     |
| `cost_tracker`     | Deterministic | Accumulates token/cost metrics per run                           |
| `human_checkpoint` | Deterministic | HITL only — pauses workflow, waits for human input               |
| `skill_injector`   | Deterministic | Loads domain skills, attaches to agent config before execution   |

### Workflow Composition (V0: Feature-by-Feature)

```
Entry
  │
  ▼
[plan] ──writes──▶ State Store (tasks)
  │
  ▼
[task_router] ──reads──▶ State Store (next pending task)
  │                       │
  │                       ▼
  │               [context_loader] ──reads──▶ State Store (scoped context)
  │                       │
  ▼                       ▼
[config_loader] ──reads──▶ State Store (agent config, model tier)
  │
  ▼
[build] ──uses──▶ tool interface (file write, bash, git)
  │        └──writes──▶ State Store (attempt record)
  │
  ▼
[evaluate] ──tests──▶ running app / code checks
  │           └──writes──▶ State Store (eval score, feedback)
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
- **Input**: Receives a state object (run_id, task_id, context, config)
- **Output**: Returns updated state + status (success / failure / needs_human)
- **Side effects**: Reads/writes state store, calls tools, writes files
- **Logging**: Every node records model used, tokens, cost, and duration

This uniform interface is what makes nodes composable across workflows.

---

## 6. V0 Scope — Solo Baseline

### Goal
Build the minimal harness that can autonomously produce a working application from a single prompt.
Beat the solo baseline: better than $9 / 20 min / broken output.

### What V0 Includes
- [ ] Feature-by-feature workflow: Plan → Build → Evaluate → (loop or done)
- [ ] Node interfaces designed for reuse (uniform input/output contract) even though only one workflow ships
- [ ] At least 2 model tiers wired up
- [ ] Structured persistent state for run tracking, tasks, and agent context
- [ ] Tool interface for basic operations (file read/write, bash execution, git)
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
- Advanced evaluation using browser automation (v1)
- Agent config hot-reloading mid-run (v1)
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
                    │   Graph Orchestrator         │
                    │   (entry node)               │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   PLAN (Tier 3 LLM call)     │
                    │   Expand prompt → full spec  │
                    │   Decompose into tasks       │
                    │   Write to state store       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   TASK ROUTER (deterministic)│
                    │   Pick next task from store  │
                    │   Classify complexity        │
                    │   Select model tier          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   BUILD (Tier 1-2 LLM call)  │
                    │   Scoped context from store  │
                    │   Tools via abstraction layer│
                    │   Write code, commit to git  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   EVALUATE (Tier 2 LLM call) │
                    │   Test: does it work?        │
                    │   Score against criteria     │
                    │   Write results to store     │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   Pass?            │
                         │   (deterministic)  │
                         └────┬──────────┬────┘
                              │          │
                         Yes  │          │  No
                              │          │
                    ┌─────────▼──┐  ┌────▼───────────┐
                    │  Next task  │  │ Retry / Escal. │
                    │  or DONE   │  │  (HR logic)    │
                    └────────────┘  └────────────────┘


    ┌─────────────────────────────────────────────────┐
    │              Persistent State Store              │
    │  ┌──────────┬──────────┬──────────┬───────────┐ │
    │  │  runs    │  tasks   │  outputs │  configs  │ │
    │  └──────────┴──────────┴──────────┴───────────┘ │
    │  ┌──────────┬──────────┐                         │
    │  │  logs    │  context │                         │
    │  └──────────┴──────────┘                         │
    └─────────────────────────────────────────────────┘
```

### Key Nodes (V0)

| Node             | Type          | Model Tier | Purpose                                                |
| ---------------- | ------------- | ---------- | ------------------------------------------------------ |
| `plan`           | LLM call      | Tier 3     | Expand prompt, generate spec and task list             |
| `task_router`    | Deterministic | —          | Pick next task, classify, select model                 |
| `context_loader` | Deterministic | —          | Query state store, build scoped context for builder    |
| `build`          | LLM call      | Tier 1-2   | Implement the task using available tools               |
| `evaluate`       | LLM call      | Tier 2     | Test and score the implementation                      |
| `pass_check`     | Deterministic | —          | Check eval score, decide: next task / retry / escalate |
| `state_writer`   | Deterministic | —          | Update state store with results, git commit            |

### What Needs to Be Stored

The state store needs to capture at minimum:
- **Run-level**: prompt, status, total cost, total tokens, timestamps
- **Task-level**: title, description, status, complexity, assigned tier, attempt count, parent task (for subtask decomposition)
- **Attempt-level**: model used, tier, input context, output, eval score, eval feedback, status, tokens, cost, duration
- **Agent configs**: role, model tier, model override, prompt template, allowed tools, max context size
- **Context store**: key-value scoped context for inter-agent communication, scoped by run and agent role

---

## 8. Open Design Questions

### Resolved
- [x] Orchestration: graph-based with conditional branching
- [x] Model routing: tiered model buckets with a unified LLM interface
- [x] Context storage: structured persistent store with scoped reads
- [x] Tools: abstracted interface (environment-independent)
- [x] Execution: local first
- [x] V0 scope: solo baseline
- [x] Workflow architecture: composable nodes → workflows → modes
- [x] Operation modes: separate workflow compositions sharing a node library (not conditional checkpoints in one graph)
- [x] Dev strategies: configurable workflow modes (v0: feature-by-feature only)
- [x] Skills: out of v0, dynamic attachment via stored configs in v1+
- [x] Node design: uniform interface contract for reusability

### Open
- [ ] Orchestration framework: which graph orchestration library fits best without over-constraining?
- [ ] Tool interface design: single server with all tools, or multiple specialized servers?
- [ ] Evaluation strategy for v0: simple "does it run" check, or structured 0–1 scoring?
- [ ] Prompt engineering: how much do we invest in system prompts for v0?
- [ ] Git strategy: one commit per task? Per attempt? How granular?
- [ ] Compaction strategy: when context grows too large within a single task, how to handle?
- [ ] Task complexity classification: rule-based or LLM-assisted?
- [ ] What's the "right" initial model tier assignment per task type?
- [ ] How to feed evaluation results back as context without polluting the build context?
- [ ] Node state schema: what fields are mandatory vs optional?
- [ ] Sub-workflow invocation: native support in orchestration layer, or custom wrapper?
- [ ] Strategy selector: should the planner choose the workflow mode, or a dedicated routing node?

---

## 9. Research Questions

These are the things we want to learn from running the harness:

1. **Cost efficiency**: What's the actual cost distribution across model tiers for a typical app build?
2. **Model-task fit**: Which task types genuinely need Tier 3 vs. which are fine with Tier 1?
3. **Escalation patterns**: How often does escalation happen? Does it converge (fewer escalations over time)?
4. **Context scoping impact**: Does giving agents less context actually improve output quality?
5. **Evaluation reliability**: How well do cheaper models evaluate work done by other cheap models?
6. **Orchestration overhead**: Does multi-agent orchestration overhead negate the model cost savings?
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
- [Anthropic Quickstart — Autonomous Coding](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding)

---

## 11. Session Log

### Session 1 — Initial Brainstorm (March 26, 2026)
- Reviewed all three Anthropic blog posts
- Analyzed dunder-mifflin-play repo structure and capabilities
- Established core hypothesis: multi-agent + model routing = lower cost, comparable quality
- Key novel concepts identified:
  - Dynamic model escalation (HR agent)
  - POC-first then swarm build strategy
  - Stored agent configs with mid-run updates
  - Dual mode: Autopilot + HITL
- V0 scope defined: solo baseline, minimal harness, autopilot only
- Created IDEA.md for persistent reference

**Refinement round:**
- Evolved "POC-first" into **configurable workflow strategies** — multiple dev modes (feature-by-feature, POC-swarm, TDD, agile, spec-driven) as composable graph workflows
- Redesigned Autopilot/HITL as **separate workflow compositions sharing a node library** (cleaner than conditional checkpoints in one graph)
- Defined **three-layer node taxonomy**: core nodes, strategy nodes, infrastructure nodes
- Added **dynamic skills attachment** concept (v1+): domain skills stored in config store and injected into agent config based on task classification
- Established **node interface contract**: uniform input/output (state object + status) for all nodes regardless of type
- Key design decision: v0 ships ONE workflow but designs nodes for reuse from day one
- Added **sub-workflow nesting** concept: workflows can invoke other workflows (e.g., feature-by-feature dispatching TDD for critical features)
- Updated IDEA.md with all refinements

**Cleanup round (March 30, 2026):**
- Removed framework-specific references (orchestration library, model routing library) — implementation TBD
- Made architecture section technology-agnostic: described what to store rather than schema, removed vendor-specific tool names
- Focused document on the *what* and *why*, not the *how*
- **Next steps**: Framework selection + technology design decisions (separate design doc)
