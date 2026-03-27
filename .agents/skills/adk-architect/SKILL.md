---
name: google-adk
description: >
  Expert Google ADK (Agent Development Kit) architect for Python. Use this skill whenever
  the user wants to design, build, or debug AI agents using Google ADK — including
  brainstorming which agent type fits a use case, choosing between LlmAgent, Sequential,
  Loop, Parallel, Custom, or ADK 2.0 graph-based workflows, and generating production-ready
  code. Trigger on: "ADK agent", "google-adk", "LlmAgent", "SequentialAgent", "LoopAgent",
  "ParallelAgent", "ADK workflow", "ADK graph", "ADK 2.0", "A2A protocol", "ADK MCP tools",
  "ADK session/state/memory", "ADK function tool", "adk run", "adk web", "ADK callbacks",
  "ADK artifacts", "ADK plugins", "ADK skills", "ADK evaluation", or any request to
  build/design an agent with Google ADK. Also trigger when the user describes an agent use
  case and wants help deciding the architecture. Always read the relevant reference file(s)
  before generating code or giving detailed implementation advice.
---

# Google ADK — Architect Knowledge Base

**Role:** You are an expert ADK Python architect. Before writing any code, run the Design
Framework below to commit to a single recommended architecture. When the user needs
implementation details, load the relevant reference file.

---

## Reference Files (load when writing code or giving detailed guidance)

| File | Covers |
|------|--------|
| `references/ref_agents.md` | LlmAgent config, all workflow agents, multi-agent, custom BaseAgent, AgentConfig YAML |
| `references/ref_runtime.md` | Runner, Sessions, State, Memory, Artifacts, Events, Context Compression |
| `references/ref_tools.md` | Function tools, MCP tools, OpenAPI tools, ADK Skills (experimental), Auth |
| `references/ref_cross_cutting.md` | Callbacks (all 6 types), Plugins, A2A protocol, Evaluation |
| `references/ref_adk2.md` | ADK 2.0 graph workflows, graph routes, human input, data handling, dynamic loops, collaborative agents |

---

## Design Framework

### Step 1 — Understand the Use Case

Ask these before choosing anything:

1. **Flow shape**: Is the task sequence fixed, or does the agent decide what to do next?
2. **Parallelism**: Are there independent sub-tasks that can run concurrently?
3. **Iteration**: Does the agent need to loop/retry until a quality condition is met?
4. **Human-in-the-loop**: Must a human approve, validate, or provide input mid-run?
5. **State lifetime**: Turn-only, session-only, or across sessions (long-term memory)?
6. **Model**: Gemini, Ollama (local), LiteLLM proxy, or direct Anthropic/OpenAI API?
7. **Interoperability**: Does this agent need to expose itself or call other services via A2A?
8. **Production readiness**: Is ADK 2.0 alpha acceptable, or must it be stable 1.x?

### Step 2 — Pick the Agent Type

```
Fixed control flow?
├── YES → Workflow Agent family
│    ├── A → B → C (linear)                    → SequentialAgent
│    ├── Repeat until condition                 → LoopAgent
│    ├── Independent tasks, fan-out             → ParallelAgent
│    ├── Complex branching + human checkpoints  → ADK 2.0 WorkflowGraph ⚠️ alpha
│    └── Arbitrary Python logic as a node       → Custom BaseAgent
│
└── NO → LLM decides (non-deterministic)
     ├── Single agent, open-ended tasks         → LlmAgent (root_agent)
     ├── Multiple specialists, LLM routes       → Multi-agent (orchestrator + sub_agents)
     └── Specialists with coordinator + graph   → ADK 2.0 Collaborative Agents ⚠️ alpha
```

### Step 3 — Agent Type Cheat Sheet

| Type | Class | Best For | Key Param |
|------|-------|----------|-----------|
| Conversational LLM | `LlmAgent` | Q&A, tool use, dynamic decisions | `tools`, `instruction` |
| Structured pipeline | `SequentialAgent` | ETL, step-by-step processing | `sub_agents` |
| Retry/eval loop | `LoopAgent` | Generate → validate → retry | `max_iterations` |
| Fan-out | `ParallelAgent` | Independent concurrent sub-tasks | `sub_agents` |
| LLM orchestrator | `LlmAgent` w/ `sub_agents` | Dynamic routing to specialists | `sub_agents`, `description` |
| Deterministic graph | `WorkflowGraph` (2.0) | State machines, complex branching | nodes + edges |
| Full custom logic | `BaseAgent` subclass | Pure Python control flow as node | `_run_async_impl` |

### Step 4 — Pick Supporting Components

Use the feature catalog below to identify which ADK components the design needs.

---

## Feature Catalog

Everything ADK offers, organized by concern. Read this to understand what's available
and when to reach for each. Load the reference file for implementation details.

---

### AGENTS

#### LlmAgent (aka `Agent`)
The workhorse. Wraps an LLM with tools, instructions, and structured I/O. Every ADK app
has at least one. Non-deterministic — the LLM decides what to do.

**Key design decisions:**
- `instruction` — the system prompt. Can be a string or a callable receiving the
  invocation context (use the callable form for dynamic, state-aware instructions).
- `output_schema` + `output_key` — force structured Pydantic output and auto-save it to
  `session.state[output_key]`, making it available to downstream agents.
- `include_contents` — set to `"none"` for stateless processing agents that shouldn't see
  prior conversation history (e.g., a classifier that only needs the current input).
- `generate_content_config` — tune temperature, max tokens, top-p, etc.
- `sub_agents` — turns this into an orchestrator; the LLM decides when to delegate.

**When NOT to use:** When you want deterministic, code-controlled execution order — use a
workflow agent instead and nest LlmAgents inside it.

→ See `ref_agents.md`

#### Workflow Agents
Deterministic orchestrators — they don't use an LLM themselves; they just control when
their sub-agents run. All inherit from `BaseAgent`, so they support `before_agent_callback`
and `after_agent_callback`.

**SequentialAgent** — runs sub-agents one by one in order. Data passes between steps via
`output_key` → `session.state`. Think of it like a Unix pipe.

**LoopAgent** — keeps cycling through its sub-agents until one sets
`tool_context.actions.escalate = True` or `max_iterations` is reached. Classic pattern:
generator agent + critic/validator agent in the loop.

**ParallelAgent** — runs all sub-agents concurrently. Each must write to a unique
`output_key` to avoid collisions. A follow-up SequentialAgent typically aggregates results.

→ See `ref_agents.md`

#### Multi-Agent Systems
An `LlmAgent` with `sub_agents` becomes an orchestrator. The LLM reads each sub-agent's
`description` to decide who to call. Sub-agents can also have their own sub-agents,
creating hierarchies. Control transfer happens via explicit tool calls or via the
`transfer_to_agent` action.

**Key design rules:**
- Every sub-agent needs a precise, differentiated `description` — this is what the
  orchestrator uses to route.
- Sub-agents should be focused and narrow; the orchestrator handles breadth.

→ See `ref_agents.md`

#### Custom BaseAgent
Subclass `BaseAgent` and implement `_run_async_impl`. Use this when you need arbitrary
Python logic as an agent node — e.g., a validation step, a DB lookup, a rule engine —
that doesn't need an LLM but must participate in an ADK workflow.

→ See `ref_agents.md`

#### AgentConfig (YAML-driven)
Define agents declaratively in YAML rather than Python. Good for config-driven or
externally managed deployments. Supports tools, sub-agents, model, instruction.

→ See `ref_agents.md`

---

### TOOLS

#### Function Tools
Regular Python functions registered on an agent. ADK auto-generates JSON schema from type
hints + docstrings and sends it to the LLM. The docstring IS the tool description.

**Types:**
- Plain function — sync or async, returns a `dict`.
- Function with `ToolContext` — access `session.state`, trigger actions (escalate,
  transfer, confirmation request).
- Long-running tool — subclass `LongRunningTool` for background jobs.

**Rules of thumb:** Return `dict`, always annotate params, keep docstrings accurate
(the LLM uses them), handle errors gracefully inside the function (return error info in the dict,
don't raise — let the LLM decide how to handle it).

→ See `ref_tools.md`

#### MCP Tools
Wrap any MCP server as a tool provider. Two connection modes:
- **Stdio** — spawn a subprocess (local MCP servers, `uvx` tools, custom Python servers).
- **SSE** — connect to an HTTP MCP server.

Can filter which tools from the server are exposed to the agent. Also: you can build your
own MCP server with `FastMCP` to expose Python functions as MCP tools, then connect remote
agents to it.

→ See `ref_tools.md`

#### OpenAPI Tools
Point at an OpenAPI spec (URL or file) and ADK auto-generates tools from every endpoint.
Good for integrating REST APIs without writing function tools manually.

→ See `ref_tools.md`

#### ADK Skills (Experimental)
Reusable, packaged agent capabilities — think of them as modules you attach to agents.
Different from MCP tools: Skills are ADK-native and can include state, memory, and
multi-turn logic. Currently experimental.

→ See `ref_tools.md`

#### Authentication
For tools that call external APIs requiring credentials. Supported schemes: API key,
OAuth2, service account. `ToolContext` provides token retrieval helpers. OAuth2 flows
can be paused mid-agent-run to complete the auth handshake.

→ See `ref_tools.md`

---

### STATE, SESSIONS & MEMORY

#### Sessions
A session groups a conversation (multiple turns) for a user in an app. Created and
managed by a `SessionService`. Always tied to `app_name` + `user_id` + `session_id`.

**Backends:** `InMemorySessionService` (dev/testing), or bring your own persistent
backend (`DatabaseSessionService` pattern).

**Rewind:** You can roll back a session to a previous event — useful for debugging or
letting users undo actions.

→ See `ref_runtime.md`

#### State (`session.state`)
The session's key-value scratchpad. Persists across turns within a session.
Accessed in tools via `ToolContext.state`, in instructions via `{key}` templating.

**Scope prefixes:**
- `"key"` — session-scoped (default, clears when session ends)
- `"user:key"` — persists across sessions for a user
- `"app:key"` — shared across all users in the app
- `"temp:key"` — cleared after each turn

**Best practice:** update state through `output_key` (agent) or `ToolContext.state`
(tools), not by directly mutating `session.state` from outside — ADK tracks changes
for persistence.

→ See `ref_runtime.md`

#### Memory (long-term)
Cross-session memory. State is per-session; memory is per-user and spans sessions.
`load_memory` / `save_memory` tools let agents recall past interactions.

**Backends:** `InMemoryMemoryService` (dev), or implement `BaseMemoryService`
with your own vector store (e.g., Qdrant, ChromaDB).

→ See `ref_runtime.md`

---

### RUNTIME

#### Runner
The execution engine. Takes an agent + session service (+ optional memory/artifact
services) and drives the event loop. Use `runner.run_async()` to programmatically
run agents. Call it with `user_id`, `session_id`, and the new user message.

**Modes:**
- `InMemoryRunner` — convenience wrapper for quick tests (bundles its own session service).
- `Runner` — production use, takes external services.

→ See `ref_runtime.md`

#### Event Loop
Every agent run is a stream of `Event` objects. The runner yields them as they happen.
Consume them with `async for event in runner.run_async(...)`.

**Event types you care about:**
- Model response (text from LLM)
- Tool call / tool result
- Agent transfer
- Final response (`event.is_final_response()`)
- Human input request (ADK 2.0)

Events are also persisted to the session history and are the basis for callbacks.

→ See `ref_runtime.md`

#### Context & Context Compression
Each agent turn gets an `InvocationContext` carrying the session, agent config, and
conversation history. As history grows, it can overflow the model's context window.

**Context Compression:** Configure a compactor (e.g., `LlmSummarizingCompactor`) with a
trigger threshold (e.g., token count). ADK automatically summarizes old history when the
threshold is hit, keeping the agent functional in long-running tasks.

→ See `ref_runtime.md`

#### Artifacts
Named, versioned binary storage for files produced or consumed during an agent run.
Not for structured data (use state for that) — for actual files: generated PDFs, images,
audio, exports. Stored via `ArtifactService`. Tools save/load artifacts via `ToolContext`.
Artifacts are scoped to `app/user/session` or just `app/user` (shared across sessions).

**When to use:** Agent generates a report, image, or export file that needs to be
retrieved later or passed to another system.

→ See `ref_runtime.md`

---

### CALLBACKS

Six interception points on every agent. Agent-scoped (defined per agent, not globally —
use Plugins for global behavior).

| Callback | When it fires | Can short-circuit? | Common uses |
|----------|--------------|-------------------|-------------|
| `before_agent_callback` | Before agent's `_run_async_impl` | ✅ return `Content` to skip agent | Auth checks, state setup, logging entry |
| `after_agent_callback` | After agent produces its final output | ✅ return `Content` to replace output | Post-process output, audit logging |
| `before_model_callback` | Before LLM API call | ✅ return `LlmResponse` to skip LLM | Input guardrails, caching, prompt injection |
| `after_model_callback` | After LLM response received | ✅ return `LlmResponse` to replace | Output filtering, token tracking, PII scrubbing |
| `before_tool_callback` | Before tool function executes | ✅ return `dict` to skip tool | Tool-level auth, argument validation, dry-run mode |
| `after_tool_callback` | After tool function returns | ✅ return `dict` to replace result | Result transformation, logging, error enrichment |

**Short-circuiting:** If a callback returns a non-None value, it replaces the normal
output and the skipped step never runs. Return `None` to proceed normally.

**Note:** For security guardrails meant to apply across all agents, prefer **Plugins**
over callbacks (ADK's own guidance). Use callbacks for agent-specific logic.

→ See `ref_cross_cutting.md`

---

### PLUGINS

Global hooks that intercept every event in every agent run — not agent-specific.
Implement `BasePlugin` and override `on_event`. Must return the event (can modify it).

**When to use over callbacks:**
- Cross-cutting concerns: observability, tracing, rate limiting, security policies
- Behavior that must apply uniformly to all agents without configuring each one

**Built-in plugins:** `ReflectAndRetryPlugin` — makes an agent reflect on why a tool
call failed and retry with better arguments. Especially useful for complex tool use.

→ See `ref_cross_cutting.md`

---

### A2A PROTOCOL (Agent-to-Agent)

Standard protocol for agents to call other agents across process/service boundaries —
language and framework agnostic.

**Two roles:**
- **Exposing (server):** Wrap your ADK runner in an `A2AServer`. Exposes
  `/.well-known/agent.json` (capability discovery) and a `/tasks` endpoint.
- **Consuming (client):** Use `A2ATool.from_url()` to load a remote agent's capabilities
  and register it as a tool on your local agent. The orchestrator calls it like any
  other tool.

**When to use:** When you want to decompose a system into independently deployable
agents that communicate over HTTP, or when integrating with non-ADK agents.

**A2A Extension:** ADK also supports the A2A streaming extension for long-running tasks.

→ See `ref_cross_cutting.md`

---

### ADK SKILLS (AGENT SKILLS — EXPERIMENTAL)

Packaged, reusable capability bundles for agents. Think of them as installable
agent modules — richer than MCP tools because they can carry state, memory, and
multi-step logic. Currently experimental in ADK.

Distinct from Claude Code skills (this skill file you're reading now). ADK Skills are
a framework feature for composing agent capabilities.

→ See `ref_tools.md`

---

### EVALUATION

ADK has a built-in evaluation framework for testing agent behavior.

**Two dimensions:**
- **Response quality** — did the agent say the right thing? (evaluators: accuracy,
  coherence, groundedness, custom rubrics)
- **Tool use quality** — did the agent call the right tools with the right arguments?

**Modes:**
- `adk eval` CLI — run eval sets from YAML/JSON files
- `User Simulation` — let another LLM simulate the user for automated multi-turn evals
- Custom metrics — implement `BaseEvalMetric` for domain-specific scoring

**Eval set format:** JSON files with `query`, `expected_response`, `expected_tool_calls`.

→ See `ref_cross_cutting.md`

---

### ADK 2.0 — GRAPH WORKFLOWS (ALPHA)

> ⚠️ Alpha. Python 3.11+. `pip install google-adk --pre`. Do NOT share storage with 1.x.

Replaces/extends the workflow agent family with explicit graph semantics:
nodes + directed edges. Gives you precise control over routing logic.

**Node types:**
- `FunctionNode` — pure Python function, reads/writes the shared state dict
- `AgentNode` — wraps an LlmAgent
- `HumanInputNode` — pauses the graph and waits for human input before continuing
- `CoordinatorNode` — an LLM that orchestrates sub-agents within the graph

**Edge types:**
- Static edge — always go from A to B
- `ConditionalEdge` — route to different nodes based on a Python function that reads state

**Key advantages over 1.x workflow agents:**
- Mixed node types in one graph (Python + LLM + human)
- Explicit, inspectable routing logic
- Human-in-the-loop as a first-class construct
- Dynamic loops without LoopAgent's limitations
- Collaborative agents with coordinator pattern

→ See `ref_adk2.md`

---

## Model Strings (Open-source / non-Google)

| Backend | Install extra | Model string example |
|---------|--------------|---------------------|
| Gemini | (included) | `"gemini-2.5-flash"` |
| Ollama | `google-adk[ollama]` | `"ollama/llama3.2"` |
| LiteLLM | `google-adk[litellm]` | `"litellm/openai/gpt-4o"` |
| vLLM | `google-adk[vllm]` | `"vllm/mistralai/Mistral-7B-v0.1"` |
| Anthropic | `google-adk[anthropic]` | `"claude-sonnet-4-5"` |

For Ollama: set `OLLAMA_HOST`. For LiteLLM: set the provider's API key. All models
are swappable — the agent interface is the same regardless.

---

## CLI Reference

```bash
# Scaffold & run
adk create my_agent                   # scaffold project with agent.py + .env
adk run my_agent                      # interactive CLI chat
adk web --port 8000                   # dev UI (run from PARENT dir of my_agent/)
adk api_server my_agent               # REST API server
adk eval my_agent eval_set.json       # run evaluation set
```

---

## Project Layout Convention

```
my_project/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py          # root_agent defined here (required by ADK CLI)
│   ├── tools.py          # function tool definitions
│   ├── subagents.py      # sub-agent definitions
│   ├── callbacks.py      # callback functions
│   ├── plugins.py        # plugin classes
│   └── .env              # model API keys
└── tests/
    └── eval_set.json     # evaluation dataset
```
