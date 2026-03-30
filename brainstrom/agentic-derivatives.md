# The definitive guide to AI agent building blocks across frameworks

**Every production AI agent, regardless of framework, is built from the same ~17 universal components.** Understanding these building blocks framework-agnostically transforms you from a single-framework developer into an agent architect who can pick the right tool for each job—or combine them. This guide maps PydanticAI, LangChain/LangGraph, Deep Agents, and Google ADK against a unified taxonomy of agent primitives, then demonstrates everything through two concrete use cases: a simple one implementable in all frameworks, and a complex multi-framework ecosystem where agents from PydanticAI, LangGraph, and ADK collaborate via A2A and MCP.

The audience is an experienced Python developer fluent in Google ADK who wants to rapidly internalize PydanticAI and LangChain/LangGraph/Deep Agents. ADK concepts serve as the anchor throughout—every new concept is mapped back to its ADK equivalent.

---

## The universal anatomy of an AI agent

Every agentic system, from a simple ReAct loop to a multi-agent research swarm, is composed of these **17 building blocks**. The table below shows the concrete API for each across all four framework groups:

| #   | Building Block           | PydanticAI                                                                     | LangChain/LangGraph                                                       | Deep Agents                                            | Google ADK                                                                            |
| --- | ------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| 1   | **Agent definition**     | `Agent('model', deps_type=T, output_type=R)`                                   | `create_agent(model, tools, system_prompt)` / `StateGraph`                | `create_deep_agent(model, tools)`                      | `LlmAgent(name, model, instruction)`                                                  |
| 2   | **State management**     | Message history + deps (stateless between runs)                                | `TypedDict` state with reducers on `StateGraph`                           | LangGraph state + virtual filesystem                   | `session.state` with `temp:`, `user:`, `app:` prefixes                                |
| 3   | **Session management**   | Manual via `message_history` param                                             | `thread_id` in config → checkpointer                                      | Inherits LangGraph threads                             | `Session` + `SessionService` (InMemory/DB/Vertex/Firestore)                           |
| 4   | **Memory**               | No built-in; user-implemented via DB                                           | `BaseStore` (cross-thread), checkpointer (within-thread)                  | Auto-summarization + filesystem persistence            | `MemoryService` + `LoadMemoryTool`/`PreloadMemoryTool`                                |
| 5   | **Tools**                | `@agent.tool`, `Tool()`, `AbstractToolset`                                     | `@tool`, `BaseTool`, `StructuredTool`                                     | Inherits LangChain + built-in plan/file/subagent tools | `FunctionTool`, `AgentTool`, auto-wrapping                                            |
| 6   | **MCP tools**            | `MCPServerStdio`, `MCPServerStreamableHTTP` (native)                           | `langchain-mcp-adapters` (adapter)                                        | Via `langchain-mcp-adapters`                           | `McpToolset` (native)                                                                 |
| 7   | **Instructions**         | `instructions=` param + `@agent.instructions` decorator                        | `system_prompt=` on `create_agent`, `ChatPromptTemplate`                  | `system_prompt=` + auto-injected planning prompts      | `instruction=` (string or callable with `ReadonlyContext`)                            |
| 8   | **Dependency injection** | `deps_type=T`, `RunContext[T]` in tools/prompts                                | `RunnableConfig`, `ToolRuntime` (1.0)                                     | Inherits LangChain                                     | `ToolContext`, `CallbackContext`, `InvocationContext`                                 |
| 9   | **Execution loop**       | Internal graph: `UserPromptNode→ModelRequestNode→CallToolsNode`                | LangGraph superstep model (Pregel-inspired)                               | LangGraph + planning loop + context windowing          | Runner event loop: yield/pause/process/resume                                         |
| 10  | **Multi-agent**          | Agent-in-tool, programmatic handoff, `pydantic_graph`                          | Supervisor (`langgraph-supervisor`), Swarm (`langgraph-swarm`), subgraphs | Built-in subagent spawning via `task` tool             | `sub_agents=[]`, transfer, `AgentTool`, `SequentialAgent`/`ParallelAgent`/`LoopAgent` |
| 11  | **Human-in-the-loop**    | `requires_approval=True`, `DeferredToolRequests`                               | `interrupt()` + `Command(resume=)`, breakpoints                           | Inherits LangGraph interrupt                           | `require_confirmation=True`, `request_confirmation()`                                 |
| 12  | **Structured output**    | `output_type=PydanticModel` (Tool/Native/Prompted modes)                       | `with_structured_output()`, `response_format=`                            | Inherits LangChain                                     | `output_schema=PydanticModel`, `output_key`                                           |
| 13  | **Guardrails**           | `@agent.output_validator`, `ModelRetry`, third-party `pydantic-ai-guardrails`  | `Middleware` system (1.0), `PIIMiddleware`, graph-based validation nodes  | Inherits LangChain middleware                          | `before_model_callback`/`after_model_callback`, plugins, `SafetySettings`             |
| 14  | **Callbacks**            | `Capabilities` system: `before_/after_/wrap_/on_error` for run/model/tool/node | `BaseCallbackHandler`: `on_llm_start/end`, `on_tool_start/end`, etc.      | Inherits LangChain                                     | `before_agent/after_agent/before_model/after_model/before_tool/after_tool` callbacks  |
| 15  | **Error handling**       | `ModelRetry` exception, `retries=` param, tool timeout                         | `.with_retry()`, `.with_fallbacks()`, `ToolNode(handle_tool_errors=True)` | Auto context repair, dangling tool call repair         | Event error codes, `ReflectAndRetryToolPlugin`                                        |
| 16  | **Checkpointing**        | External: Temporal (`TemporalAgent`), DBOS (`DBOSAgent`)                       | Native: `MemorySaver`, `SqliteSaver`, `PostgresSaver`                     | Inherits LangGraph checkpointing                       | `DatabaseSessionService`, `VertexAiSessionService`, `rewind_async()`                  |
| 17  | **Streaming**            | `run_stream()`, `run_stream_events()`, delta/accumulated modes                 | `.stream()`, `.astream()`, `astream_events()`, 5 stream modes             | Inherits LangGraph streaming                           | `run_async()` event generator, `run_live()` for bidirectional audio/video             |

---

## How state flows through each framework

State management is where these frameworks diverge most sharply. Understanding the state model is the key to understanding each framework's philosophy.

**Google ADK** uses the model you already know: a `Session` object holds `state` (a dict with prefix-based scoping) and `events` (chronological history). The `Runner` creates an `InvocationContext` per call, and state changes are tracked atomically through `EventActions.state_delta`. The prefix system (`temp:`, `user:`, `app:`) elegantly controls persistence scope. This is the most explicit state model of the four frameworks.

**PydanticAI** takes a deliberately stateless approach. There is **no built-in session or state store**. Each `agent.run()` call is independent. To maintain conversation continuity, you pass `message_history=result.all_messages()` between calls. Dependencies (`deps`) are injected per-run and provide typed context (like ADK's `ToolContext`). For persistence, PydanticAI delegates to external durable execution systems—Temporal wraps agent runs as activities, DBOS checkpoints to Postgres/SQLite. This "bring your own persistence" philosophy keeps the core library focused but means more wiring for stateful apps.

**LangGraph** treats state as a first-class graph citizen. You define a `TypedDict` state schema with **reducer annotations** (e.g., `Annotated[list, add_messages]` for append semantics, `operator.add` for accumulation). State flows through nodes, and each node returns partial updates merged via reducers—critical for parallel execution where multiple nodes update state simultaneously. Checkpointers (`MemorySaver`, `PostgresSaver`) save state at every superstep, enabling time-travel debugging and resumability. The `Store` API adds cross-thread long-term memory with namespace-scoped key-value storage and optional vector search. Thread-based sessions (`thread_id`) provide conversation continuity.

**Deep Agents** layers on top of LangGraph's state with three additions: a **virtual filesystem** (or pluggable backend) for storing intermediate work products, **automatic conversation summarization** when context windows fill up, and **large tool result eviction** that dumps oversized results to the filesystem. The `Backend` abstraction (v0.2) generalizes this—you can plug in LangGraph State, local filesystem, LangSmith Sandbox, or cloud storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STATE ARCHITECTURE SPECTRUM                   │
│                                                                 │
│  Stateless ◄──────────────────────────────────────► Stateful    │
│                                                                 │
│  PydanticAI        LangGraph          Deep Agents    Google ADK │
│  (bring your       (built-in          (auto-managed  (explicit  │
│   own state)        graph state +      state + FS +   session + │
│                     checkpointers)     summarization) event      │
│                                                       sourcing) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tools, MCP, and agent-as-tool across frameworks

Tools are the most portable building block. All four frameworks follow the same pattern: a Python function with type hints and a docstring becomes a tool the LLM can call. The differences emerge in MCP support, tool context, and the agent-as-tool pattern.

**Tool definition** is nearly identical everywhere. PydanticAI uses `@agent.tool` (with `RunContext` for deps access) or `@agent.tool_plain`. LangChain uses `@tool` from `langchain.tools`. ADK auto-wraps plain functions passed to `tools=[]`. All extract schemas from type hints and descriptions from docstrings.

**MCP integration maturity** varies significantly. PydanticAI and ADK have **native MCP support**: PydanticAI's `MCPServerStdio` and `MCPServerStreamableHTTP` connect directly as toolsets, while ADK's `McpToolset.from_server()` discovers and wraps MCP tools. LangChain requires the `langchain-mcp-adapters` package, which converts MCP tools to `BaseTool` instances. LangGraph can also **expose agents as MCP servers** via a built-in `/mcp` endpoint, making any deployed LangGraph agent consumable by any MCP client.

**Agent-as-tool** is where multi-agent patterns begin. In PydanticAI, you call `inner_agent.run()` inside a tool function—straightforward but manual. In LangGraph, agents are graphs that can be embedded as subgraph nodes, or you use `create_handoff_tool()` for peer-to-peer delegation. In ADK, `AgentTool` formally wraps any agent as a callable tool, and `sub_agents=[]` enables LLM-driven delegation where the parent agent decides which child to invoke based on their descriptions.

**Tool context richness** differs. PydanticAI's `RunContext` gives access to deps, retry count, usage stats, and model settings. LangChain 1.0's new `ToolRuntime` provides `state`, `context`, `store`, `config`, and `stream_writer`. ADK's `ToolContext` offers `state`, `save_artifact()`, `load_artifact()`, `search_memory()`, `request_credential()`, and `request_confirmation()`. ADK's `ToolContext` remains the richest, with direct access to artifacts, memory, and auth—capabilities that require additional wiring in other frameworks.

---

## Instructions and dependency injection compared to ADK

If you're coming from ADK, PydanticAI's dependency injection system will feel like a more type-safe version of what you already know. ADK's `ReadonlyContext` in dynamic instruction callables maps directly to PydanticAI's `RunContext[DepsType]` in `@agent.instructions` decorators. The key difference: PydanticAI enforces the dependency type at the generic level (`Agent[MyDeps, MyOutput]`), giving you IDE autocompletion and static type checking throughout.

ADK's state templating (`instruction="Process data from {data}"` with auto-substitution from `session.state`) has a PydanticAI equivalent in `TemplateStr('Hello {{name}}')` used in Agent Specs (YAML/JSON config). LangChain uses `ChatPromptTemplate` with `{variable}` syntax and `MessagesPlaceholder` for dynamic message injection.

**LangChain 1.0** introduced `ToolRuntime` as its answer to ADK's `ToolContext`—a single object injected into tools that carries `state`, `context` (immutable user data), `store` (long-term memory), and `stream_writer`. This is a significant improvement over LangChain's previous pattern of passing config dicts.

PydanticAI's **Capabilities system** is unique: an `AbstractCapability` bundles tools, lifecycle hooks, instructions, and model settings into a reusable unit. Think of it as a plugin that can modify every aspect of agent behavior. Built-in capabilities include `Thinking`, `WebSearch`, `WebFetch`, and `Hooks`. This has no direct ADK equivalent—the closest analog is ADK's plugin system (`GlobalInstructionPlugin`, `ContextFilterPlugin`), but ADK plugins are limited to instructions and callbacks, while PydanticAI capabilities can also inject tools and model settings.

---

## Execution loops and multi-agent orchestration

The execution model is the architectural core that shapes everything else about a framework.

**PydanticAI** runs agents as an internal graph with three node types: `UserPromptNode` (assembles prompts), `ModelRequestNode` (calls LLM), and `CallToolsNode` (executes tools, checks for output). The loop is ReAct-style: model calls tools, tool results feed back to model, repeat until final output. The `agent.iter()` API exposes this graph for fine-grained control—you can inspect and intervene at each node. `UsageLimits` caps tokens and tool calls to prevent runaway loops.

**LangGraph** uses a **Pregel-inspired superstep model**. Within each superstep, all scheduled nodes execute (potentially in parallel). After completion, state synchronizes, edges evaluate to determine the next superstep's nodes, and the cycle repeats until reaching `END` or the recursion limit (default **25**). This model gives maximum control: you define exactly which nodes exist, how they connect, what conditions trigger which paths, and how state merges during parallel execution. The `Command` object elegantly combines state updates with routing instructions, enabling patterns like `return Command(goto="next_node", update={"key": "value"})`.

**Google ADK's** Runner uses a yield/pause/process/resume cycle. The agent runs as an async generator yielding `Event` objects. The Runner receives each event, processes actions (persists state via SessionService), and forwards the event. The cycle repeats until the generator exhausts. ADK's orchestration agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) are deterministic wrappers—no LLM decides the flow. For LLM-driven routing, ADK uses the parent `LlmAgent`'s `sub_agents` with `description`-based delegation (similar to LangGraph's supervisor pattern).

**Deep Agents** adds a planning layer on top of LangGraph. The key innovation is the **todo list tool**—a no-op that forces the LLM to decompose complex tasks into discrete steps. Combined with subagent spawning (context isolation for subtasks), filesystem access (shared workspace), and automatic context windowing, Deep Agents handle extended autonomous work sessions that would overflow standard agent context windows.

**Multi-agent patterns** converge across frameworks:

- **Supervisor/Coordinator**: One agent routes to specialists. LangGraph: `create_supervisor()`. ADK: Parent `LlmAgent` with `sub_agents`. PydanticAI: Programmatic hand-off or agent-in-tool.
- **Swarm/Peer-to-Peer**: Agents hand off to each other autonomously. LangGraph: `create_swarm()` with `create_handoff_tool()`. ADK: LLM-driven transfer via `transfer_to_agent` action.
- **Sequential Pipeline**: Deterministic chain. LangGraph: `add_sequence()`. ADK: `SequentialAgent`. PydanticAI: `pydantic_graph.Graph`.
- **Parallel Fan-out**: Independent work in parallel. LangGraph: Multiple edges from one node. ADK: `ParallelAgent`.

---

## Human-in-the-loop patterns across frameworks

HITL is where the frameworks' state models create the most visible differences. The core pattern is always: pause execution → present decision to human → resume with response. But the mechanics vary.

**PydanticAI** uses **deferred tools**. Mark a tool with `requires_approval=True`, and when the agent tries to call it, the run ends with a `DeferredToolRequests` output containing the pending tool calls. Your application presents these to the user, collects approvals, builds a `DeferredToolResults` object, and resumes with `agent.run(message_history=..., deferred_tool_results=results)`. The `ApprovalRequiredToolset` wrapper can dynamically require approval based on tool name or arguments. This is stateless—you must pass message history to resume.

**LangGraph** uses the `interrupt()` function, which pauses the graph and returns a value to the caller. Resumption uses `Command(resume=data)` with the same `thread_id`. Because LangGraph has built-in checkpointing, the graph state is automatically persisted at the interrupt point—no manual message history management needed. The older breakpoint approach (`interrupt_before=["node_name"]`) pauses before/after specific nodes.

**Google ADK** uses `require_confirmation=True` on `FunctionTool` or the more flexible `context.request_confirmation(prompt, expected_response_schema)` for structured approval data. The `ToolConfirmation` object carries the approval state. Resume by sending a `FunctionResponse` event with the same `invocation_id`.

The key insight: **LangGraph's approach is the most ergonomic** for complex HITL because checkpointing handles state automatically. PydanticAI's approach is the most explicit and testable. ADK's approach integrates most naturally with its event-driven architecture.

---

## Deep Agents: when standard agents aren't enough

Deep Agents (`pip install deepagents`) represent LangChain's answer to Claude Code-style autonomous agents. They sit above both LangChain and LangGraph in the stack:

```
deepagents  →  "Agent Harness" (batteries-included)
    ↓
langchain   →  "Agent Framework" (core loop + middleware)
    ↓
langgraph   →  "Agent Runtime" (state machine + persistence)
```

The four architectural pillars that distinguish Deep Agents from regular LangGraph agents are **detailed system prompts** (long, example-rich instructions), a **planning tool** (`write_todos` for task decomposition), **subagent spawning** (context isolation for subtasks), and **filesystem access** (shared workspace for notes, intermediate results, and context management).

Deep Agents v0.2 (October 2025) introduced **pluggable backends** that generalize the filesystem concept. The `Backend` abstraction supports LangGraph State (ephemeral), LangGraph Store (cross-thread), local filesystem, LangSmith Sandbox, Modal/Daytona/Deno sandboxes, and composite backends that route by path prefix. The **conversation history summarization** feature automatically compresses old messages when token usage grows large, and **large tool result eviction** dumps oversized results to the backend to keep the active context focused.

The relationship to "deep research" agents: LangChain's [open_deep_research](https://github.com/langchain-ai/open_deep_research) is a specific application built on the Deep Agents SDK, applying planning, subagents, and filesystem patterns specifically to research tasks.

**When to use Deep Agents vs standard LangGraph**: Use Deep Agents when the task is open-ended, long-running, requires decomposition into subtasks, generates intermediate artifacts, or would overflow a standard context window. Use standard LangGraph when you need precise control flow, deterministic routing, or the task is well-structured enough that explicit graph design is superior to autonomous planning.

---

## Cross-framework integration: A2A and MCP as the universal connectors

Two protocols enable agents from different frameworks to collaborate: **A2A (Agent-to-Agent)** for agent-to-agent communication and **MCP (Model Context Protocol)** for tool sharing. They are explicitly complementary.

**A2A** (now under the Linux Foundation, v0.3 with gRPC support) enables any agent to discover and communicate with any other agent regardless of framework. Each agent publishes an **Agent Card** at `/.well-known/agent-card.json` describing its capabilities and endpoint. Communication uses JSON-RPC 2.0 over HTTP with task lifecycle management (submitted → working → input_required → completed/failed). Over **150 organizations** support A2A including Google, Microsoft, LangChain, and Salesforce.

Framework A2A support status: **Google ADK** has native support via `to_a2a()` (server) and `RemoteA2aAgent` (client). **PydanticAI** has native support via `agent.to_a2a()` and the `fasta2a` library. **LangChain/LangGraph** does not yet have native A2A support—it's a community-requested feature (GitHub issue #5987), though workarounds exist using custom A2A client wrappers as tools.

**MCP** (now under the Agentic AI Foundation, **97 million monthly SDK downloads**) standardizes tool access. PydanticAI and ADK have native MCP support; LangChain uses the `langchain-mcp-adapters` package. Critically, LangGraph can **expose deployed agents as MCP servers** via its `/mcp` endpoint, making them consumable by any MCP client including PydanticAI and ADK.

```
┌──────────────────────────────────────────────────────────┐
│            CROSS-FRAMEWORK INTEGRATION MAP               │
│                                                          │
│   ┌──────────┐    A2A Protocol     ┌──────────┐         │
│   │PydanticAI│◄───────────────────►│Google ADK│         │
│   │  Agent   │                     │  Agent   │         │
│   └────┬─────┘                     └────┬─────┘         │
│        │                                │               │
│        │  MCP                     MCP   │               │
│        │  (native)              (native)│               │
│        ▼                                ▼               │
│   ┌─────────────────────────────────────────┐           │
│   │          MCP Tool Servers               │           │
│   │  (shared tools accessible by all)       │           │
│   └─────────────────────────────────────────┘           │
│        ▲                                ▲               │
│        │  MCP                     MCP   │               │
│        │  (adapter)            (/mcp    │               │
│        │                      endpoint) │               │
│   ┌────┴─────┐                ┌────┴─────┐             │
│   │LangChain │◄──subgraph────►│LangGraph │             │
│   │  Agent   │                │  Agent   │             │
│   └──────────┘                └──────────┘             │
│                                    ▲                    │
│                                    │ built on           │
│                               ┌────┴─────┐             │
│                               │  Deep    │             │
│                               │  Agents  │             │
│                               └──────────┘             │
└──────────────────────────────────────────────────────────┘
```

---

## Simple use case: customer support agent exercising all building blocks

This use case exercises all 17 building blocks in a single agent that handles customer support with order lookups, refund processing (with human approval), and structured ticket summaries. Below is the implementation pattern for each framework.

**Scenario**: A customer writes in about an order issue. The agent looks up the order (tool), determines if a refund is needed, requests human approval for refunds over $100 (HITL), processes the refund, and returns a structured `TicketSummary` with resolution details. Conversation history persists across messages. Input is validated for PII. The agent streams responses in real-time.

### PydanticAI implementation pattern

```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.capabilities import Hooks

class TicketSummary(BaseModel):
    order_id: str
    issue: str
    resolution: str
    refund_amount: float | None

@dataclass
class SupportDeps:
    db_client: DatabaseClient
    user_id: str

hooks = Hooks()

@hooks.on('before_model_request')
async def log_request(ctx, request):
    print(f"[TRACE] Model request for user {ctx.deps.user_id}")

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    deps_type=SupportDeps,
    output_type=TicketSummary,
    instructions="You are a customer support agent. Be empathetic and solution-oriented.",
    retries=3,
    capabilities=[hooks],
)

@agent.instructions
def add_user_context(ctx: RunContext[SupportDeps]) -> str:
    return f"Current user ID: {ctx.deps.user_id}"

@agent.tool
async def lookup_order(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """Look up order details by order ID."""
    return await ctx.deps.db_client.get_order(order_id)

@agent.tool(requires_approval=True)
async def process_refund(ctx: RunContext[SupportDeps], order_id: str, amount: float) -> str:
    """Process a refund for the given order. Requires human approval."""
    await ctx.deps.db_client.refund(order_id, amount)
    return f"Refund of ${amount} processed for order {order_id}"

@agent.output_validator
async def validate_summary(ctx: RunContext[SupportDeps], output: TicketSummary) -> TicketSummary:
    if output.refund_amount and output.refund_amount < 0:
        raise ModelRetry("Refund amount cannot be negative")
    return output

# Run with streaming, session continuity, and HITL
result = await agent.run_stream("My order #12345 arrived damaged", deps=SupportDeps(...))
# If DeferredToolRequests returned, present to human, resume with deferred_tool_results
```

### LangGraph implementation pattern

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool

class SupportState(MessagesState):
    ticket_summary: dict | None
    user_id: str

@tool
def lookup_order(order_id: str) -> str:
    """Look up order details by order ID."""
    return db.get_order(order_id)

@tool
def process_refund(order_id: str, amount: float, 
                   tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Process a refund. Requires approval for amounts over $100."""
    if amount > 100:
        decision = interrupt({"action": "refund", "amount": amount, "order": order_id})
        if not decision.get("approved"):
            return Command(update={"messages": [
                ToolMessage("Refund denied by supervisor", tool_call_id=tool_call_id)
            ]})
    db.refund(order_id, amount)
    return Command(update={"messages": [
        ToolMessage(f"Refund of ${amount} processed", tool_call_id=tool_call_id)
    ]})

# Build graph with guardrail node, agent node, tool node
builder = StateGraph(SupportState)
builder.add_node("guardrail", pii_filter_node)
builder.add_node("agent", agent_node)  # calls LLM with tools
builder.add_node("tools", ToolNode([lookup_order, process_refund]))
builder.add_edge(START, "guardrail")
builder.add_conditional_edges("guardrail", check_safe, {"safe": "agent", "unsafe": END})
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=PostgresSaver(...))
# Invoke with thread_id for session continuity
result = graph.invoke(input, config={"configurable": {"thread_id": "session-1"}})
```

### Google ADK implementation pattern

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from pydantic import BaseModel

class TicketSummary(BaseModel):
    order_id: str
    issue: str
    resolution: str
    refund_amount: float | None

def lookup_order(order_id: str, ctx: ToolContext) -> str:
    """Look up order details by order ID."""
    return db.get_order(order_id)

def process_refund(order_id: str, amount: float, ctx: ToolContext) -> str:
    """Process a refund for the given order."""
    return f"Refund of ${amount} processed for order {order_id}"

def input_guardrail(callback_context, llm_request):
    if contains_pii(llm_request.contents):
        return LlmResponse(content=Content(parts=[Part(text="PII detected, cannot process.")]))
    return None

support_agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="You are a customer support agent. Be empathetic.",
    tools=[lookup_order, FunctionTool(process_refund, require_confirmation=True)],
    output_schema=TicketSummary,
    output_key="ticket_summary",
    before_model_callback=input_guardrail,
)

runner = Runner(
    agent=support_agent,
    app_name="support",
    session_service=DatabaseSessionService(db_url="..."),
)
# Run with streaming events
async for event in runner.run_async(user_id="u1", session_id="s1", new_message=content):
    print(event)
```

---

## Complex use case: multi-framework research ecosystem

This scenario demonstrates agents from PydanticAI, LangGraph, and Google ADK collaborating. A user asks: "Research the competitive landscape of quantum computing startups and produce an investment memo."

### Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (Google ADK LlmAgent)                     │
│  - Decomposes task into subtasks                        │
│  - Routes to specialist agents via A2A                  │
│  - Aggregates results into final memo                   │
│                                                         │
│  sub_agents: [RemoteA2aAgent("research"),               │
│               RemoteA2aAgent("analysis"),               │
│               formatter_agent]                          │
└──────┬──────────────┬───────────────────┬───────────────┘
       │ A2A          │ A2A               │ local
       ▼              ▼                   ▼
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│RESEARCHER│   │  ANALYST     │   │  FORMATTER   │
│(Deep     │   │  (PydanticAI │   │  (ADK        │
│ Agents)  │   │   Agent)     │   │   LlmAgent)  │
│          │   │              │   │              │
│ Planning │   │ Structured   │   │ output_schema│
│ Subagents│   │ CompAnalysis │   │ = InvestMemo │
│ FileSystem│  │ output_type  │   │              │
│ MCP tools │  │ deps: market │   │              │
│          │   │ data client  │   │              │
└────┬─────┘   └──────┬───────┘   └──────────────┘
     │                │
     │ MCP            │ MCP
     ▼                ▼
┌─────────────────────────────────────┐
│  SHARED MCP SERVERS                 │
│  - Web search (Brave/Tavily)        │
│  - Database (company financials)    │
│  - Document store (SEC filings)     │
└─────────────────────────────────────┘
```

### Implementation sketch

**1. Shared MCP servers** (deployed once, used by all frameworks):

```python
# mcp_servers.py — FastMCP servers for shared tools
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("research-tools")

@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for information."""
    return brave_search(query)

@mcp.tool()  
def query_financials(company: str, metric: str) -> dict:
    """Query company financial data."""
    return financial_db.query(company, metric)
```

**2. Research Agent** (Deep Agents — autonomous research with planning):

```python
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({"research": {"url": "http://mcp:8000/mcp"}}) as client:
    research_agent = create_deep_agent(
        model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
        tools=await client.get_tools(),
        system_prompt="""You are a deep research agent specializing in quantum computing.
        Use the planning tool to decompose research tasks. Spawn subagents for 
        parallel research on different companies. Write findings to files.""",
    )
    # Expose via A2A
    app = FastAPI()
    # Wrap as A2A server using fasta2a or custom wrapper
```

**3. Analysis Agent** (PydanticAI — structured competitive analysis):

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP

class CompetitiveAnalysis(BaseModel):
    companies: list[CompanyProfile]
    market_size: str
    growth_rate: str
    key_trends: list[str]
    investment_thesis: str
    risks: list[RiskFactor]

@dataclass
class AnalysisDeps:
    research_data: str  # from research agent
    market_data_client: MarketDataClient

analysis_agent = Agent(
    'openai:gpt-4o',
    deps_type=AnalysisDeps,
    output_type=CompetitiveAnalysis,
    instructions="You are a financial analyst. Produce rigorous competitive analysis.",
    toolsets=[MCPServerStreamableHTTP(url="http://mcp:8000/mcp")],
)

# Expose via A2A
app = analysis_agent.to_a2a()  # PydanticAI native A2A
```

**4. Orchestrator** (Google ADK — coordinates everything):

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import RemoteA2aAgent

research_remote = RemoteA2aAgent(url="http://research-agent:8001")
analysis_remote = RemoteA2aAgent(url="http://analysis-agent:8002")

formatter = LlmAgent(
    name="formatter",
    model="gemini-2.5-pro",
    instruction="Format the analysis into a professional investment memo.",
    output_schema=InvestmentMemo,
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-pro",
    instruction="""You coordinate research tasks. First delegate deep research 
    to the researcher, then send findings to the analyst for structured analysis,
    then format the final memo.""",
    sub_agents=[research_remote, analysis_remote, formatter],
)

runner = Runner(agent=orchestrator, app_name="research-platform",
                session_service=DatabaseSessionService(db_url="..."))
```

This architecture demonstrates several key patterns: **A2A for agent discovery and communication** across framework boundaries, **MCP for shared tool access** (all agents use the same search and database tools), **Deep Agents for autonomous research** (planning, subagents, filesystem), **PydanticAI for type-safe structured analysis** (Pydantic output validation), and **ADK for orchestration** (leveraging its native A2A support and agent hierarchy).

---

## Choosing the right framework for each job

Rather than picking one framework for everything, the most effective approach combines frameworks based on their strengths:

**Use PydanticAI when** you need type-safe structured outputs, clean dependency injection, strong testing patterns (`TestModel`, `FunctionModel`, `ALLOW_MODEL_REQUESTS=False`), or you're building focused single-agent tools where Pydantic validation is critical. Its FastAPI-inspired design makes it the most Pythonic option. It excels at **precision agents** where output quality and type safety matter more than complex orchestration.

**Use LangGraph when** you need explicit control flow, complex multi-agent orchestration, built-in checkpointing/persistence, or sophisticated HITL patterns. The graph-based model makes state transitions visible and testable. It excels at **workflow agents** where the business process has clear steps, branches, and parallel paths. The supervisor and swarm prebuilts cover the most common multi-agent patterns.

**Use Deep Agents when** the task is open-ended, requires autonomous planning, generates intermediate artifacts, or would overflow a standard context window. Research tasks, coding tasks, and complex analysis are ideal fits. Deep Agents handle context management automatically—something you'd have to build manually in LangGraph.

**Use Google ADK when** you're in the Google Cloud ecosystem, need native A2A support for cross-framework orchestration, want built-in streaming (including audio/video via Live API), or prefer ADK's explicit session/state/memory/artifact separation. ADK's `SequentialAgent`/`ParallelAgent`/`LoopAgent` are the cleanest deterministic orchestration primitives across all frameworks.

**Integrate them when** your system needs strengths from multiple frameworks—use A2A for agent-to-agent communication and MCP for shared tool access. Deploy each agent as a microservice exposing A2A Agent Cards, and let any orchestrator (ADK, LangGraph, or custom) coordinate them.

---

## Conclusion: convergence is inevitable, protocols are the investment

The most striking finding across this research is how rapidly these frameworks are converging on the same abstractions. Tools, structured outputs, streaming, HITL, and multi-agent patterns look increasingly similar across PydanticAI, LangGraph, and ADK. The real differentiators are in **state management philosophy** (stateless vs. graph-state vs. event-sourced), **extension models** (capabilities vs. middleware vs. plugins), and **orchestration granularity** (implicit loop vs. explicit graph vs. deterministic wrappers).

The lasting investment isn't in any single framework—it's in the **protocols**. MCP (97M monthly downloads, governed by the Linux Foundation) and A2A (150+ supporting organizations, also Linux Foundation) are becoming the universal connectors. An agent built today with proper A2A and MCP integration will be callable from any future framework. Build your agents as protocol-native microservices, and the orchestration layer becomes interchangeable.

For the developer coming from ADK, the fastest path to productivity is: learn PydanticAI first (its dependency injection and type safety will feel natural, and the API surface is small), then learn LangGraph for complex orchestration (its graph model is the most flexible), then adopt Deep Agents for autonomous research/coding tasks. Keep ADK as your orchestrator—its native A2A support makes it the natural hub for a multi-framework ecosystem.