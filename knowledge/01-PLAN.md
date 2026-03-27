# dunder-mifflin-harness — Implementation Plan

> Step-by-step guide to go from zero to a working v0 harness.
> Start simple, validate early, expand incrementally.

---

## Phase 0: Project Scaffolding

### Repo Structure

```
dunder-mifflin-harness/
│
├── pyproject.toml                  # uv project config
├── .python-version                 # 3.12+
├── .env.example                    # required env vars template
├── .gitignore
├── README.md
├── IDEA.md                         # brainstorming & design decisions
│
├── lab/                            # 🧪 Local experiments & scratch work
│   ├── README.md                   # what this folder is for
│   ├── 01_litellm_routing.py       # test LiteLLM with different models
│   ├── 02_adk_basic_workflow.py    # first ADK 2.0 Workflow graph
│   ├── 03_adk_with_litellm.py     # ADK agent using LiteLLM model
│   ├── 04_mcp_tool_test.py         # test MCP tool connection
│   ├── 05_sqlite_context.py        # test DB read/write patterns
│   └── notebooks/                  # jupyter notebooks for exploration
│       └── exploration.ipynb
│
├── src/
│   └── harness/
│       ├── __init__.py
│       │
│       ├── core/                   # 🧱 Core infrastructure
│       │   ├── __init__.py
│       │   ├── db.py               # SQLite connection, migrations, helpers
│       │   ├── models.py           # Pydantic models (shared data contracts)
│       │   ├── config.py           # Settings, env vars, model bucket definitions
│       │   └── cost_tracker.py     # Token/cost accumulation utilities
│       │
│       ├── nodes/                  # 🔲 Node library (reusable across workflows)
│       │   ├── __init__.py
│       │   ├── plan.py             # Planning node (Tier 3 LLM call)
│       │   ├── build.py            # Build/implementation node (Tier 1-3)
│       │   ├── evaluate.py         # Evaluation node (Tier 2)
│       │   ├── task_router.py      # Task selection + model routing (Python fn)
│       │   ├── context_loader.py   # Scoped context builder from DB (Python fn)
│       │   ├── state_writer.py     # Persist results + git commit (Python fn)
│       │   └── pass_check.py       # Pass/fail decision logic (Python fn)
│       │
│       ├── workflows/              # 🔀 Workflow compositions (graphs)
│       │   ├── __init__.py
│       │   └── feature_by_feature.py   # V0: the main workflow
│       │   # future: tdd.py, poc_swarm.py, agile.py, etc.
│       │
│       ├── prompts/                # 📝 Prompt templates (separate from code)
│       │   ├── __init__.py
│       │   ├── planner.py          # System prompts for planning
│       │   ├── builder.py          # System prompts for implementation
│       │   └── evaluator.py        # System prompts for evaluation/QA
│       │
│       └── tools/                  # 🔧 Tool definitions (for MCP or direct)
│           ├── __init__.py
│           └── file_tools.py       # File read/write, bash, git wrappers
│
├── mcp_server/                     # 🌐 MCP server (tool provider)
│   ├── __init__.py
│   ├── server.py                   # MCP server entry point
│   └── tools/                      # Tool implementations
│       ├── __init__.py
│       ├── filesystem.py           # File read, write, list
│       ├── shell.py                # Bash command execution
│       └── git.py                  # Git operations
│
├── db/                             # 💾 Database
│   ├── schema.sql                  # SQLite schema (source of truth)
│   └── migrations/                 # Future: migration scripts
│       └── 001_initial.sql
│
├── workspace/                      # 📂 Working directory for harness output
│   └── .gitkeep                    # (harness creates project files here)
│
└── tests/                          # ✅ Tests
    ├── __init__.py
    ├── test_db.py
    ├── test_nodes.py
    └── test_workflows.py
```

### Key Design Decisions in Structure

1. **`lab/` is first-class** — Numbered scripts for incremental learning.
   You run these to validate each piece before wiring them together.
   This is where you figure out ADK 2.0 quirks, LiteLLM routing, MCP patterns.

2. **`nodes/` is the node library** — Each file exports functions or Agent
   instances that can be wired into any workflow. The uniform interface
   (input → output + side effects) makes them composable.

3. **`workflows/` composes nodes into graphs** — Each file defines a
   `Workflow` using ADK 2.0's graph syntax. V0 has one; future versions add more.

4. **`prompts/` separated from logic** — Prompts are the most-iterated part
   of any agent system. Keeping them in their own module means you can tune
   prompts without touching orchestration code.

5. **`mcp_server/` is independent** — Can run as a separate process.
   Agent code never imports from mcp_server directly — they communicate
   via the MCP protocol. This is what gives us environment portability.

6. **`workspace/` is ephemeral** — The harness writes generated code here.
   Each run gets a subdirectory. Git init happens inside.

---

## Phase 1: Lab Experiments (Get Each Piece Working)

The goal is to validate each building block in isolation before composing.
Work through these in order — each builds on the previous.

### Step 1.1 — LiteLLM Model Routing
**File**: `lab/01_litellm_routing.py`

What to validate:
- LiteLLM can call models from multiple providers (Gemini, Claude, OpenAI)
- OpenRouter routing works
- Token counting and cost tracking is accessible from responses
- Model bucket abstraction works (pick model by tier)

Minimal experiment:
```python
# Pseudocode — test that tier-based routing works
from litellm import completion

BUCKETS = {
    1: "gemini/gemini-2.0-flash-lite",
    2: "anthropic/claude-sonnet-4-6",
    3: "anthropic/claude-opus-4-6",
}

for tier, model in BUCKETS.items():
    resp = completion(model=model, messages=[{"role": "user", "content": "Say hello"}])
    print(f"Tier {tier} ({model}): {resp.choices[0].message.content}")
    print(f"  Tokens: {resp.usage.total_tokens}, Cost: ${resp._hidden_params.get('response_cost', '?')}")
```

### Step 1.2 — ADK 2.0 Basic Workflow
**File**: `lab/02_adk_basic_workflow.py`

What to validate:
- ADK 2.0 `Workflow` class works with edges
- Mixing Agent nodes and Python function nodes in same graph
- Data flows correctly between nodes via Pydantic models
- Route branching works (conditional next node)

Minimal experiment: A tiny graph that plans a task (Agent) → classifies it (Python fn) → routes to one of two handlers.

### Step 1.3 — ADK + LiteLLM Integration
**File**: `lab/03_adk_with_litellm.py`

What to validate:
- ADK Agent can use LiteLLM as its model backend
- Different agents in the same workflow can use different models
- ADK's native LiteLLM connector works as documented

This is where we confirm the model bucket concept works inside ADK graphs.

### Step 1.4 — MCP Tool Connection
**File**: `lab/04_mcp_tool_test.py`

What to validate:
- MCP server starts and exposes tools
- ADK agent can discover and call MCP tools
- File write, bash exec, git operations work through MCP
- Tool results flow back into agent context correctly

### Step 1.5 — SQLite Context Patterns
**File**: `lab/05_sqlite_context.py`

What to validate:
- Schema creates correctly
- Write a run + tasks + attempts
- Read scoped context (e.g., "give me only what the builder needs for task X")
- Context size stays within token budgets
- Concurrent read/write works (for future parallel nodes)

---

## Phase 2: Core Infrastructure

Once lab experiments pass, build the real modules.

### Step 2.1 — Database Layer (`src/harness/core/db.py`)
- SQLite connection with context manager
- Schema initialization from `db/schema.sql`
- Helper functions: `create_run()`, `create_task()`, `record_attempt()`, `get_next_task()`, `get_scoped_context()`
- Keep it simple — raw SQL with parameterized queries, no ORM

### Step 2.2 — Pydantic Models (`src/harness/core/models.py`)
- Data contracts that flow between nodes:
  - `PlanOutput`: spec text, list of tasks with priorities
  - `TaskContext`: scoped info for builder (task description, relevant files, what to change)
  - `BuildOutput`: files changed, git diff summary, self-assessment
  - `EvalResult`: score (0-1), pass/fail, feedback text, specific issues found
  - `RunState`: current run status, completed tasks, remaining tasks, total cost

### Step 2.3 — Config & Model Buckets (`src/harness/core/config.py`)
- Load from env vars + optional config file
- Model bucket definitions (tier → model mapping)
- Default prompt templates path
- Workspace directory settings
- MCP server URL

### Step 2.4 — Cost Tracker (`src/harness/core/cost_tracker.py`)
- Accumulate tokens and cost per node invocation
- Per-run totals
- Simple: just wraps the data from LiteLLM responses

---

## Phase 3: Node Library

Build nodes one at a time. Test each in isolation before composing.

### Step 3.1 — Plan Node (`src/harness/nodes/plan.py`)
- Input: raw user prompt (string)
- LLM call: Tier 3 model
- Output: `PlanOutput` (spec + decomposed task list)
- Side effect: writes tasks to SQLite
- This is the most important prompt to get right — spend time here

### Step 3.2 — Task Router (`src/harness/nodes/task_router.py`)
- Input: current `RunState`
- Pure Python: queries DB for next pending task, classifies complexity
- Output: `TaskContext` with selected model tier
- No LLM call — deterministic logic

### Step 3.3 — Context Loader (`src/harness/nodes/context_loader.py`)
- Input: task_id
- Pure Python: queries DB for only what the builder needs
- Output: enriched `TaskContext` with scoped information
- This is where context engineering happens — be selective

### Step 3.4 — Build Node (`src/harness/nodes/build.py`)
- Input: `TaskContext`
- LLM call: Tier selected by router (1, 2, or 3)
- Tools: file write, bash, git (via MCP)
- Output: `BuildOutput`
- Side effect: writes files to workspace, records attempt in DB

### Step 3.5 — Evaluate Node (`src/harness/nodes/evaluate.py`)
- Input: `BuildOutput` + evaluation criteria
- LLM call: Tier 2 model
- Output: `EvalResult` (score, pass/fail, feedback)
- Side effect: records eval in DB
- V0: simple "does it run?" check. V1: Playwright-based testing

### Step 3.6 — Pass Check (`src/harness/nodes/pass_check.py`)
- Input: `EvalResult`
- Pure Python: checks score against threshold
- Output: routing decision — "next_task" | "retry" | "done" | "escalate"
- This is where HR escalation logic lives (v1)

### Step 3.7 — State Writer (`src/harness/nodes/state_writer.py`)
- Input: current state + build/eval results
- Pure Python: updates DB, commits to git, updates run totals
- Output: updated `RunState`

---

## Phase 4: First Workflow

### Step 4.1 — Feature-by-Feature Workflow (`src/harness/workflows/feature_by_feature.py`)
Wire the nodes into an ADK 2.0 `Workflow`:

```python
# Pseudocode — actual ADK 2.0 syntax
from google.adk import Workflow

root_workflow = Workflow(
    name="feature_by_feature",
    edges=[
        # Planning phase (runs once)
        ("START", plan_node, init_state_writer),

        # Build loop entry
        (init_state_writer, task_router_node),

        # Route: has tasks remaining?
        (task_router_node, route_has_tasks),  # branch function

        # Build path
        (context_loader, build_node, evaluate_node, pass_check_node),

        # Pass check branches
        (pass_check_node, route_pass_fail),  # branch function
        # pass → state_writer → task_router (loop)
        # fail → retry (back to build_node with feedback)
        # done → END
    ],
)
```

### Step 4.2 — Entry Point / CLI
Simple script to run the harness:

```python
# run.py or src/harness/__main__.py
"""
Usage: uv run python -m harness "Build a todo app with drag-and-drop kanban boards"
"""
```

### Step 4.3 — First End-to-End Test
Run the workflow against a simple prompt and observe:
- Does planning decompose the prompt into reasonable tasks?
- Does the builder produce code that compiles/runs?
- Does the evaluator catch obvious issues?
- Does the loop terminate?
- What's the total cost?

---

## Phase 5: Iterate & Improve

Based on Phase 4 results:

### 5.1 — Prompt Tuning
- Review traces from real runs
- Tune planner prompts for better task decomposition
- Tune builder prompts for cleaner code output
- Tune evaluator prompts for more useful feedback

### 5.2 — Model Bucket Optimization
- Try different models per tier
- Compare cost vs. quality across tiers
- Build the escalation dataset

### 5.3 — Evaluation Improvement
- Move from "does it run?" to structured scoring
- Add Playwright MCP for browser testing (for web apps)
- Evaluator prompt calibration with few-shot examples

### 5.4 — Additional Workflows (V1)
- HITL mode (add human checkpoint nodes)
- HR escalation node
- POC-swarm workflow
- TDD workflow

---

## Implementation Order (What to Build When)

```
Week 1: Phase 0 + Phase 1 (scaffolding + lab experiments)
         ↓ Validate: "I can call LLMs, wire ADK graphs, connect MCP, use SQLite"

Week 2: Phase 2 (core infrastructure)
         ↓ Validate: "DB works, models defined, config loads, costs tracked"

Week 3: Phase 3 (node library — plan + build + evaluate)
         ↓ Validate: "Each node works in isolation"

Week 4: Phase 4 (first workflow + end-to-end test)
         ↓ Validate: "Harness runs autonomously, produces output, tracks cost"

Week 5+: Phase 5 (iterate based on real results)
```

This is aggressive but realistic given that you're building this as R&D,
not production software. The lab experiments in Phase 1 are the
highest-leverage work — they'll surface ADK 2.0 alpha gotchas early
before you've built on top of them.

---

## Getting Started (Right Now)

```bash
# 1. Create the repo
mkdir dunder-mifflin-harness && cd dunder-mifflin-harness
git init

# 2. Init uv project
uv init

# 3. Add core dependencies
uv add google-adk litellm pydantic

# 4. Copy IDEA.md into the repo

# 5. Create the folder structure
# (see repo structure above)

# 6. Start with lab/01_litellm_routing.py
```

---

## ADK 2.0 Primitives Mapping

How our conceptual design maps to ADK 2.0:

| Our Concept | ADK 2.0 Primitive | Notes |
|-------------|-------------------|-------|
| LLM call node | `Agent(model=..., instruction=..., output_schema=...)` | Use LiteLLM model string |
| Python function node | Plain `def fn(input) -> output` | Just a function in the edges |
| Conditional routing | Branch function returning next node | `def route(input): return node_a if ... else node_b` |
| Workflow composition | `Workflow(edges=[...])` | Nested workflows are nodes |
| Sub-workflow | `Workflow` as a node inside another `Workflow` | Native support |
| Parallel execution | Fan-out/join in edges | ADK 2.0 supports this |
| HITL checkpoint | Human input node | ADK 2.0 has built-in pattern |
| Data passing | Pydantic `input_schema` / `output_schema` on agents | Type-safe between nodes |
| MCP tools | `MCPToolset` attached to agents | Native ADK integration |
| Model routing | LiteLLM model string per Agent | Different agents = different models |