# Building AI agents with Pydantic AI and LangChain/LangGraph

**Pydantic AI offers a type-safe, Pythonic agent abstraction with first-class dependency injection and MCP support, while LangGraph provides graph-based stateful workflows with built-in persistence, human-in-the-loop, and multi-agent orchestration.** Both frameworks solve the same core problem — giving LLMs the ability to reason, use tools, and maintain state — but they approach it from fundamentally different angles. For an ADK engineer, Pydantic AI will feel like writing clean Python with strong typing, while LangGraph will feel like wiring an explicit state machine where you control every transition. This reference covers every building block you need to evaluate and adopt either framework.

---

## Universal agent building blocks

Before diving into framework-specific implementations, these are the core primitives every agent framework must address. Understanding them as abstractions makes it easier to map between ADK, Pydantic AI, and LangGraph.

**State** is the mutable data an agent or graph carries across execution steps. In ADK this is `session.state`; in LangGraph it's the `TypedDict` schema on `StateGraph`; in Pydantic AI, state lives implicitly in message history and explicitly in the graph module's `GraphRunContext.state`.

**Session and memory** span two dimensions. *Short-term memory* is the conversation history within a single session — message lists in all frameworks. *Long-term memory* persists across sessions: user preferences, learned facts, procedural rules. ADK uses `SessionService` and `ArtifactService`; LangGraph uses checkpointers (short-term) and `Store` (long-term); Pydantic AI relies on serialized `message_history` and external storage.

**Tools** are functions an LLM can invoke. Every framework wraps Python callables with a name, description, and JSON schema for arguments. MCP tools extend this by connecting to external tool servers over a standardized protocol. ADK has `FunctionTool`; Pydantic AI has `@agent.tool`; LangChain has `@tool` and `bind_tools()`.

**Instructions / system prompts** configure agent behavior. The key distinction is static vs. dynamic: ADK and Pydantic AI both support runtime-resolved instructions that can incorporate dependency data (e.g., the current user's name). LangGraph achieves this through prompt templates with `MessagesPlaceholder`.

**Context / dependency injection** separates agent definition from runtime state. Pydantic AI's `deps_type` + `RunContext` pattern is the most explicit — analogous to FastAPI's `Depends()`. ADK passes context through `Session` and `CallbackContext`. LangGraph passes `RunnableConfig` and a newer `Runtime` object to nodes.

**Runners and execution loops** drive the agent's reasoning cycle. The ReAct loop (reason → act → observe → repeat) is the dominant pattern. ADK's `Runner` yields event streams; Pydantic AI's `agent.run()` handles the loop internally; LangGraph makes the loop explicit as graph edges cycling between an LLM node and a tools node.

**Multi-agent communication** comes in three flavors: *orchestration* (a supervisor delegates to sub-agents), *delegation/handoff* (agents transfer control peer-to-peer), and *hierarchical* (nested supervisors). ADK uses `SequentialAgent`/`ParallelAgent`; Pydantic AI uses agent-as-tool; LangGraph uses supervisor and swarm patterns.

**Human-in-the-loop** means pausing execution for human approval or input. ADK handles this via callbacks; LangGraph has first-class `interrupt()` with checkpointed resume; Pydantic AI uses `DeferredToolRequests`.

**Structured outputs** force the LLM to return data conforming to a schema. All frameworks leverage Pydantic models, but the mechanism differs: Pydantic AI uses `output_type` natively, LangChain uses `with_structured_output()`, and ADK uses `output_schema`.

**Guardrails and validation** catch bad outputs before they reach the user. Pydantic AI's `@agent.output_validator` and `ModelRetry` are the most integrated — a failed validation sends feedback to the model for retry. LangGraph achieves similar behavior through conditional edges that loop back on validation failure.

---

## Pydantic AI: type-safe agents that feel like writing Python

Pydantic AI's philosophy is that building an agent should feel like writing a FastAPI endpoint — strongly typed, dependency-injected, and testable. The `Agent` class is the single core abstraction: it wraps a model, tools, instructions, and an output type into a callable unit.

### Agent class and model configuration

An agent is generic over two types: `Agent[DepsType, OutputType]`. Defaults are `Agent[None, str]`. You configure it with a model string, instructions, tools, and optional settings:

```python
from pydantic_ai import Agent

agent = Agent(
    'google-gla:gemini-2.5-flash',
    instructions='You are a concise research assistant.',
    output_type=str,
)
result = agent.run_sync('Summarize quantum computing in one paragraph.')
print(result.output)
```

> **ADK mapping**: `Agent(model='gemini-2.5-flash', instruction='...')` maps directly. The model string format differs — ADK uses just `'gemini-2.5-flash'` while Pydantic AI prefixes with the provider: `'google-gla:gemini-2.5-flash'`.

For explicit Gemini configuration with Vertex AI or custom credentials:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-key')  # or vertexai=True
model = GoogleModel('gemini-2.5-flash', provider=provider)
agent = Agent(model)
```

Run methods include **`run()`** (async), **`run_sync()`** (blocking), **`run_stream()`** (async streaming), and **`iter()`** (step-by-step graph node iteration).

### Dependency injection solves the testability problem

The DI pattern is Pydantic AI's most distinctive feature. You declare a dependency *type* on the agent, then pass an *instance* at runtime. Tools and dynamic instructions access dependencies through `RunContext[DepsType]`:

```python
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent('google-gla:gemini-2.5-flash', deps_type=MyDeps)

@agent.instructions
async def dynamic_prompt(ctx: RunContext[MyDeps]) -> str:
    # Fetch user-specific data to customize the system prompt
    resp = await ctx.deps.http_client.get('https://api.example.com/user')
    return f'User context: {resp.text}'

@agent.tool
async def search_api(ctx: RunContext[MyDeps], query: str) -> str:
    """Search the external API."""
    resp = await ctx.deps.http_client.get(
        f'https://api.example.com/search?q={query}',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    return resp.text

# At runtime, inject real or mock dependencies
async with httpx.AsyncClient() as client:
    deps = MyDeps(api_key='real-key', http_client=client)
    result = await agent.run('Find recent AI papers', deps=deps)
```

**Why this matters**: In ADK, you pass context through `Session` and `CallbackContext`, which makes unit testing require mocking the entire session infrastructure. Pydantic AI's `deps_type` lets you swap a `DatabaseConn` for a `MockDatabaseConn` in tests with zero agent code changes. The `agent.override(deps=mock_deps)` context manager makes this even cleaner.

### Structured output via Pydantic models

Setting `output_type` to a Pydantic model constrains the LLM to return validated, typed data. The framework handles schema generation, prompt augmentation, and response parsing automatically:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class InvoiceData(BaseModel):
    vendor: str
    total: float = Field(description='Total amount in USD')
    line_items: list[str]

agent = Agent('google-gla:gemini-2.5-flash', output_type=InvoiceData)
result = agent.run_sync('Extract: "Invoice from Acme Corp. Widget A $50, Widget B $30. Total $80."')
print(result.output)  # InvoiceData(vendor='Acme Corp', total=80.0, line_items=['Widget A $50', 'Widget B $30'])
```

Union types enable branching output:

```python
class Success(BaseModel):
    sql_query: str

class InvalidRequest(BaseModel):
    error_message: str

agent = Agent('google-gla:gemini-2.5-flash', output_type=Success | InvalidRequest)
```

Add validation with `@agent.output_validator` — if validation fails, `ModelRetry` sends the error message back to the LLM for correction:

```python
from pydantic_ai import ModelRetry

@agent.output_validator
async def validate_sql(ctx, output):
    if isinstance(output, Success):
        try:
            await ctx.deps.db.execute(f'EXPLAIN {output.sql_query}')
        except Exception as e:
            raise ModelRetry(f'Invalid SQL: {e}')
    return output
```

> **ADK mapping**: ADK uses `output_schema` on the agent and handles validation through callbacks. Pydantic AI's `ModelRetry` mechanism is more tightly integrated — the retry loop is automatic.

### Tools: context-aware and plain

Two decorator patterns exist. `@agent.tool` receives `RunContext` as the first parameter for dependency access. `@agent.tool_plain` skips context for stateless functions:

```python
@agent.tool
async def get_balance(ctx: RunContext[MyDeps], account_id: str) -> float:
    """Fetch account balance from the database."""
    return await ctx.deps.db.get_balance(account_id)

@agent.tool_plain
def get_current_date() -> str:
    """Return today's date."""
    from datetime import date
    return str(date.today())
```

**Conditional tool availability** uses `prepare` functions that can return `None` to hide a tool based on runtime context:

```python
from pydantic_ai.tools import ToolDefinition

def admin_only(ctx: RunContext[MyDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps.user_role == 'admin':
        return tool_def
    return None  # Tool hidden from non-admins

agent = Agent('google-gla:gemini-2.5-flash', tools=[Tool(delete_user, prepare=admin_only)])
```

### Multi-agent: agent-as-tool and programmatic handoff

Pydantic AI's primary multi-agent pattern is **agent delegation** — one agent invokes another inside a tool function:

```python
from pydantic_ai import Agent, RunContext

research_agent = Agent('google-gla:gemini-2.5-flash', output_type=list[str])
writer_agent = Agent('google-gla:gemini-2.5-flash')

@writer_agent.tool
async def gather_research(ctx: RunContext, topic: str) -> list[str]:
    """Delegate research to a specialist agent."""
    result = await research_agent.run(
        f'Find 5 key facts about {topic}',
        usage=ctx.usage,  # Share token budget
    )
    return result.output

result = writer_agent.run_sync('Write a brief report on quantum computing.')
```

> **ADK mapping**: This is analogous to ADK's `AgentTool` where one agent uses another as a tool. The key difference is that ADK manages the multi-agent hierarchy declaratively via `sub_agents`, while Pydantic AI makes it explicit in code.

For more complex workflows, **pydantic-graph** (separate package) provides a type-safe state machine:

```python
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from dataclasses import dataclass

@dataclass
class ResearchState:
    topic: str
    findings: list[str] = None

@dataclass
class Research(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> 'Write | End[str]':
        ctx.state.findings = ['fact1', 'fact2']  # Would call research_agent here
        return Write()

@dataclass
class Write(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        return End(f'Report on {ctx.state.topic}: {ctx.state.findings}')

graph = Graph(nodes=(Research, Write))
result = graph.run_sync(Research(), state=ResearchState(topic='AI'))
```

Edges are defined by return type hints — the type system enforces valid graph transitions at development time.

### MCP integration is first-class

Pydantic AI natively acts as an MCP client. MCP servers are registered as `toolsets`:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

# Local MCP server via subprocess
fs_server = MCPServerStdio('npx', args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])

# Remote MCP server via HTTP
api_server = MCPServerStreamableHTTP('http://localhost:8000/mcp')

agent = Agent('google-gla:gemini-2.5-flash', toolsets=[fs_server, api_server])

async with agent:  # Manages server connections
    result = await agent.run('List files in /tmp and check the weather')
```

Load multiple servers from a JSON config (same format as Claude Desktop):

```python
from pydantic_ai.mcp import load_mcp_servers
servers = load_mcp_servers('mcp_config.json')
agent = Agent('google-gla:gemini-2.5-flash', toolsets=servers)
```

> **ADK mapping**: ADK also has native MCP support, but Pydantic AI's `toolsets` parameter makes MCP servers feel identical to local tools — no special handling required.

### Memory is manual but flexible

Pydantic AI does not provide built-in persistence. Conversation continuity relies on passing `message_history`:

```python
result1 = agent.run_sync('My name is Alice.')
result2 = agent.run_sync('What is my name?', message_history=result1.all_messages())
# Result: "Your name is Alice."

# Serialize for storage
from pydantic_core import to_json
serialized = to_json(result1.all_messages())

# Restore
from pydantic_ai import ModelMessagesTypeAdapter
history = ModelMessagesTypeAdapter.validate_json(serialized)
```

> **ADK mapping**: ADK's `SessionService` handles this automatically. In Pydantic AI, you build the persistence layer yourself (typically with a database in FastAPI or file storage in notebooks). This is more work but gives complete control.

### Streaming

```python
async with agent.run_stream('Tell me a story') as result:
    async for chunk in result.stream_text(delta=True):
        print(chunk, end='', flush=True)
```

`stream_text(delta=True)` yields incremental chunks. Without `delta=True`, each item is the complete text so far. `stream_output()` streams validated structured output progressively.

---

## LangChain: the composition layer

LangChain is the glue between LLMs and everything else — prompts, tools, retrievers, output parsers. Its core innovation is **LCEL (LangChain Expression Language)**, a composable pipe syntax for building chains.

### LCEL replaces legacy chains

LCEL uses the `|` operator to compose `Runnable` objects. Any LCEL chain automatically supports `.invoke()`, `.stream()`, `.batch()`, and async variants:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt = ChatPromptTemplate.from_template('Explain {topic} in one paragraph.')
chain = prompt | model | StrOutputParser()

result = chain.invoke({'topic': 'quantum computing'})
for chunk in chain.stream({'topic': 'quantum computing'}):
    print(chunk, end='')
```

> **ADK mapping**: ADK doesn't have an equivalent to LCEL — its composition is through agent hierarchies (`SequentialAgent`, etc.). LCEL is more like functional composition of data transformations, while ADK is object-oriented orchestration.

**`RunnableParallel`** runs branches concurrently:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

**Why LCEL exists**: Legacy chains (`LLMChain`, `ConversationChain`, `SequentialChain`) had inconsistent interfaces, no streaming, and couldn't compose. LCEL provides a uniform interface where every component is a `Runnable` with identical methods. All legacy chains are deprecated as of LangChain 0.3+ and moved to `langchain-classic`.

### Tools and function calling

Three creation patterns exist, from simplest to most flexible:

```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for matching records."""
    return f'Found {limit} results for "{query}"'

# Bind tools to a model
model_with_tools = model.bind_tools([search_database])
response = model_with_tools.invoke('Search for langchain')
print(response.tool_calls)  # [{'name': 'search_database', 'args': {'query': 'langchain'}, ...}]
```

`bind_tools()` is provider-agnostic — works identically with Gemini, OpenAI, Anthropic. Tool calls appear in `AIMessage.tool_calls`.

### Structured output via `with_structured_output()`

The preferred approach for models with native structured output support (Gemini included):

```python
from pydantic import BaseModel, Field

class ResearchSummary(BaseModel):
    title: str = Field(description='Title of the research')
    key_findings: list[str]
    confidence: float = Field(ge=0, le=1)

structured_model = model.with_structured_output(ResearchSummary)
result = structured_model.invoke('Summarize recent advances in fusion energy')
# Returns a ResearchSummary instance
```

> **ADK mapping**: ADK's `output_schema` serves the same purpose. The key difference: Pydantic AI validates output and can retry on failure, `with_structured_output()` relies on the model's native JSON mode, and ADK uses output schema declarations.

### Google Gemini setup in LangChain

```python
import os
os.environ['GOOGLE_API_KEY'] = 'your-key'

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    max_retries=2,
)

# For Vertex AI
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    project='your-project-id',
    # Automatically uses Vertex AI when project is provided
)
```

The `langchain-google-genai` package (v4.0+) uses the consolidated `google-genai` SDK. Auto-detection: if `project` or `credentials` are provided, it uses Vertex AI; otherwise, it uses the Gemini Developer API.

### Memory: everything old is deprecated

**All legacy memory classes are deprecated**: `ConversationBufferMemory`, `ConversationSummaryMemory`, `ConversationEntityMemory`, `ConversationChain`. The current approach is **LangGraph checkpointers** — discussed in the next section.

---

## LangGraph: explicit state machines for agent workflows

LangGraph models agent behavior as a directed graph with typed state. Where Pydantic AI hides the execution loop, LangGraph makes every decision point an explicit edge. This is its strength for complex, stateful workflows.

### StateGraph and the typed state pattern

A graph is built by defining a state schema, adding nodes (functions), and connecting them with edges:

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f'Sunny, 24°C in {city}'

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model_with_tools = model.bind_tools([get_weather])

def call_model(state: MessagesState):
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}

builder = StateGraph(MessagesState)
builder.add_node('agent', call_model)
builder.add_node('tools', ToolNode([get_weather]))
builder.add_edge(START, 'agent')
builder.add_conditional_edges('agent', tools_condition)  # Routes to tools or END
builder.add_edge('tools', 'agent')

graph = builder.compile()
result = graph.invoke({'messages': [('user', 'Weather in Berlin?')]})
```

**`MessagesState`** is the standard prebuilt state with a `messages` key annotated with the `add_messages` reducer. It automatically handles message appending and ID-based updates.

> **ADK mapping**: ADK's event-driven architecture doesn't have an explicit graph. The `Runner` manages the loop internally. LangGraph's graph is closest to what you'd build with ADK's `SequentialAgent` and `LoopAgent` — but with arbitrary branching logic.

### Reducers control how state updates merge

By default, node return values **overwrite** state keys. Reducers change this behavior:

```python
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    count: int                            # Overwrite on update
    logs: Annotated[list[str], add]       # Append lists on update
    messages: Annotated[list, add_messages]  # Smart message merging
```

The `add_messages` reducer is critical: it appends new messages but updates existing ones by ID — essential for human-in-the-loop edits.

### Conditional edges and the Command pattern

**Conditional edges** route to different nodes based on state:

```python
from typing import Literal

def route_by_intent(state: State) -> Literal['search', 'calculate', 'respond']:
    last_msg = state['messages'][-1]
    if last_msg.tool_calls:
        return 'tools'
    return 'respond'

builder.add_conditional_edges('agent', route_by_intent)
```

The **`Command`** class combines state updates and routing in a single return:

```python
from langgraph.types import Command

def my_node(state: State) -> Command[Literal['next_node']]:
    return Command(update={'status': 'processed'}, goto='next_node')
```

**`Send`** enables dynamic fan-out for map-reduce patterns:

```python
from langgraph.types import Send

def fan_out(state):
    return [Send('process_item', {'item': x}) for x in state['items']]
```

### Checkpointers: persistence that enables everything

Checkpointers save state snapshots at every super-step, organized by `thread_id`. This single mechanism enables conversational memory, human-in-the-loop, time travel, and fault tolerance:

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
config = {'configurable': {'thread_id': 'user-123'}}

# Turn 1
result = graph.invoke({'messages': [('user', 'My name is Alice')]}, config)
# Turn 2 — same thread, agent remembers
result = graph.invoke({'messages': [('user', "What's my name?")]}, config)
```

For production, use **`PostgresSaver`** or **`SqliteSaver`**. `InMemorySaver` is for development only.

> **ADK mapping**: ADK's `SessionService` with `InMemorySessionService` or database-backed variants serves the same role. The key difference: LangGraph checkpointers save the *entire graph state* (not just messages), enabling time travel via `graph.get_state_history(config)`.

### Human-in-the-loop via interrupt()

The `interrupt()` function pauses graph execution and returns control to the caller. Resuming with `Command(resume=value)` restarts the interrupted node with the human's response as the return value of `interrupt()`:

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

def approval_node(state):
    human_response = interrupt({'question': 'Approve this action?', 'details': state['action']})
    if human_response == 'yes':
        return {'status': 'approved'}
    return {'status': 'rejected'}

# Build graph with approval_node, compile with checkpointer
graph = builder.compile(checkpointer=InMemorySaver())
config = {'configurable': {'thread_id': 'approval-1'}}

# Execution pauses at interrupt
result = graph.invoke({'action': 'Delete user data', 'messages': []}, config)

# Human approves — execution resumes
result = graph.invoke(Command(resume='yes'), config)
```

> **ADK mapping**: ADK handles human-in-the-loop through callbacks (`before_tool_callback`, `after_tool_callback`). LangGraph's approach is more structured — the interrupt is a first-class primitive with automatic state persistence and resume.

### Prebuilt agents: create_react_agent

For standard ReAct agents, `create_react_agent` builds a complete graph:

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

agent = create_react_agent(
    model,
    tools=[get_weather],
    prompt='You are a helpful weather assistant.',
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {'messages': [('user', 'Weather in Tokyo?')]},
    config={'configurable': {'thread_id': 'weather-1'}},
)
```

This creates a graph with two nodes (`agent` → `tools`) cycling until the LLM stops calling tools. It supports structured output via `response_format`, pre/post model hooks, and custom state schemas.

### Multi-agent: supervisor and swarm

**Supervisor pattern** — a central agent routes to specialists:

```python
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

research_agent = create_react_agent(model, tools=[web_search], name='researcher')
math_agent = create_react_agent(model, tools=[calculator], name='mathematician')

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt='Route to the appropriate expert.',
)
app = workflow.compile(checkpointer=InMemorySaver())
```

> **ADK mapping**: This is directly analogous to an ADK agent with `sub_agents` — the root agent is the supervisor, sub-agents are the specialists. ADK's `transfer_to_agent` instruction maps to LangGraph's handoff tools.

**Swarm pattern** — agents hand off directly to each other without a supervisor, **~40% less latency** due to fewer LLM round-trips:

```python
from langgraph_swarm import create_handoff_tool, create_swarm

sales = create_react_agent(model, tools=[
    lookup_pricing,
    create_handoff_tool(agent_name='support', description='Transfer to support'),
], name='sales')

support = create_react_agent(model, tools=[
    check_order_status,
    create_handoff_tool(agent_name='sales', description='Transfer to sales'),
], name='support')

app = create_swarm([sales, support], default_active_agent='sales').compile()
```

### Store: cross-thread long-term memory

While checkpointers provide within-thread persistence, **Store** enables cross-thread memory — information that persists across conversations:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={
    'dims': 768,
    'embed': 'google_genai:models/gemini-embedding-001',
})

graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# In a node, access the store
def personalize(state, *, store):
    memories = store.search(('user-123', 'preferences'), query='display settings')
    store.put(('user-123', 'preferences'), 'theme', {'content': 'dark mode'})
    return {'messages': [...]}
```

**LangMem SDK** adds intelligent memory on top of Store — automatic extraction, consolidation, and semantic search:

```python
from langmem import create_manage_memory_tool, create_search_memory_tool

agent = create_react_agent(
    model,
    tools=[
        create_manage_memory_tool(namespace=('memories',)),
        create_search_memory_tool(namespace=('memories',)),
    ],
    store=store,
)
```

### LangGraph streaming modes

LangGraph offers **five streaming modes** that can be combined:

- **`values`** — full state snapshot after each node
- **`updates`** — only changed keys per node
- **`messages`** — LLM token chunks with metadata (for real-time chat UI)
- **`custom`** — arbitrary data via `StreamWriter` in nodes
- **`debug`** — detailed execution traces

```python
for chunk in graph.stream(
    {'messages': [('user', 'Hello')]},
    config,
    stream_mode='messages',
):
    msg, metadata = chunk
    if msg.content:
        print(msg.content, end='', flush=True)
```

### MCP in LangGraph via langchain-mcp-adapters

LangGraph uses the `langchain-mcp-adapters` package to connect to MCP servers:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient({
    'filesystem': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
        'transport': 'stdio',
    },
    'api': {
        'url': 'http://localhost:8000/mcp',
        'transport': 'http',
    },
})

tools = await client.get_tools()  # Returns LangChain-compatible tools
agent = create_react_agent(model, tools)
```

> **ADK mapping**: ADK has native MCP support similar to Pydantic AI. LangGraph's MCP integration requires an additional package but provides the same functionality. Note that deployed LangGraph agents on LangGraph Platform automatically expose a `/mcp` endpoint.

---

## MCP and A2A: the interoperability protocols

### MCP connects agents to tools and data

**MCP (Model Context Protocol)** is an open standard by Anthropic (now under the Linux Foundation) that standardizes how agents access external tools, resources, and prompts. It uses JSON-RPC 2.0 over three transports: **stdio** (local subprocesses), **Streamable HTTP** (recommended for remote), and **SSE** (deprecated).

The three server capabilities are:
- **Tools** — executable functions (e.g., `query_database`, `send_email`)
- **Resources** — data the agent can read (file contents, API responses)
- **Prompts** — templated instructions for specific workflows

As of late 2025, MCP has **97 million monthly SDK downloads** and **10,000+ active servers**, with first-class support in Claude, ChatGPT, Cursor, Gemini, and VS Code.

### A2A connects agents to each other

**A2A (Agent-to-Agent)** is Google's open protocol (now under the Linux Foundation) for inter-agent communication. Where MCP is vertical (agent ↔ tools), **A2A is horizontal (agent ↔ agent)**. Core concepts:

- **Agent Card** — JSON at `/.well-known/agent.json` describing the agent's capabilities, like a business card for discovery
- **Task** — the unit of work with a lifecycle (submitted → working → completed/failed)
- **Message** — a communication turn containing Parts (text, images, data)
- **Artifact** — output produced by a completed task

**Current status**: Version 0.3 (July 2025), with v1.0 in active development. Backed by **150+ organizations**. The Python SDK is `a2a-sdk`.

> **ADK mapping**: ADK has deep native A2A support via `RemoteA2aAgent` (consuming remote agents) and `to_a2a()` (exposing agents). Pydantic AI supports A2A via `agent.to_a2a()` using the FastA2A library. LangGraph has sample implementations but no built-in A2A support.

---

## Side-by-side comparison across all three frameworks

| Building Block                  | Google ADK                                      | Pydantic AI                              | LangGraph                                                 |
| ------------------------------- | ----------------------------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Core abstraction**            | `Agent` / `LlmAgent`                            | `Agent[DepsType, OutputType]`            | `StateGraph` + compiled graph                             |
| **State**                       | `session.state` dict                            | `GraphRunContext.state` (graph module)   | `TypedDict` state schema with reducers                    |
| **Session / short-term memory** | `SessionService` (automatic)                    | `message_history` param (manual)         | Checkpointer with `thread_id` (automatic)                 |
| **Long-term memory**            | `ArtifactService`, state persistence            | External storage (manual)                | `Store` + `LangMem` SDK                                   |
| **Tools**                       | `FunctionTool`, `AgentTool`                     | `@agent.tool`, `@agent.tool_plain`       | `@tool`, `StructuredTool`, `BaseTool`                     |
| **MCP support**                 | Native (MCP Toolbox)                            | Native (`toolsets` param)                | Via `langchain-mcp-adapters`                              |
| **A2A support**                 | Native (`RemoteA2aAgent`, `to_a2a()`)           | Native (`agent.to_a2a()`)                | Samples only, no built-in                                 |
| **System prompt**               | `instruction` string                            | `instructions` + `@agent.instructions`   | `ChatPromptTemplate` / `prompt` param                     |
| **Dynamic prompts**             | Instruction with state references               | `@agent.instructions` with `RunContext`  | `MessagesPlaceholder` + template vars                     |
| **Dependency injection**        | Via `Session` / `CallbackContext`               | `deps_type` + `RunContext` (first-class) | `RunnableConfig` / `Runtime` object                       |
| **Structured output**           | `output_schema` on Agent                        | `output_type` with Pydantic model        | `with_structured_output()` on model                       |
| **Output validation + retry**   | Callbacks                                       | `@agent.output_validator` + `ModelRetry` | Conditional edges looping back                            |
| **Streaming**                   | Event stream from `Runner`                      | `run_stream()` / `stream_text()`         | 5 stream modes (values, updates, messages, custom, debug) |
| **Human-in-the-loop**           | Callbacks (`before_tool_callback`)              | `DeferredToolRequests`                   | `interrupt()` + `Command(resume=)`                        |
| **Multi-agent orchestration**   | `SequentialAgent`, `ParallelAgent`, `LoopAgent` | Agent-as-tool, `pydantic-graph`          | Supervisor, Swarm, Subgraphs                              |
| **Multi-agent delegation**      | `transfer_to_agent` instruction                 | Call agent inside `@agent.tool`          | `create_handoff_tool()`                                   |
| **Execution loop**              | `Runner.run()` yields events                    | `agent.run()` (internal loop)            | Explicit graph edges (ReAct visible)                      |
| **Observability**               | Built-in eval framework                         | OpenTelemetry + Logfire                  | LangSmith tracing                                         |
| **Graph visualization**         | ADK web UI                                      | `graph.mermaid_image()`                  | `graph.get_graph().draw_mermaid_png()`                    |
| **Deployment**                  | Cloud Run, Vertex AI                            | Any ASGI (FastAPI, Uvicorn)              | LangGraph Platform, Docker, cloud                         |
| **Testing**                     | Mock sessions                                   | `agent.override(deps=mock)`              | Mock checkpointers + state                                |

---

## Focused use cases for hands-on learning

### Pydantic AI: structured extraction with dependency injection

**Motivation**: A compliance team needs to extract structured entity data from legal documents, with different extraction configurations per client.

**Concepts exercised**: `output_type`, `deps_type`, `RunContext`, `@agent.instructions`, `ModelRetry`

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry

class ContractEntities(BaseModel):
    parties: list[str] = Field(description='Named parties in the contract')
    effective_date: str
    governing_law: str
    total_value: float | None = None

@dataclass
class ExtractionConfig:
    client_name: str
    jurisdiction_focus: str  # Which jurisdictions to prioritize

agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=ExtractionConfig,
    output_type=ContractEntities,
)

@agent.instructions
def customize_for_client(ctx: RunContext[ExtractionConfig]) -> str:
    return (
        f'Extract entities for client {ctx.deps.client_name}. '
        f'Pay special attention to {ctx.deps.jurisdiction_focus} law references.'
    )

@agent.output_validator
def validate_entities(ctx: RunContext[ExtractionConfig], output: ContractEntities) -> ContractEntities:
    if not output.parties:
        raise ModelRetry('No parties found — re-read the document header.')
    return output

# Jupyter usage
config = ExtractionConfig(client_name='Acme Corp', jurisdiction_focus='Delaware')
result = agent.run_sync('Extract from: "Agreement between Acme Corp and Beta LLC, effective Jan 1 2025, governed by Delaware law, total value $500,000."', deps=config)
print(result.output)
```

**Environment**: Jupyter notebook with `run_sync`. For FastAPI, use `async` endpoint with `await agent.run()`.

### Pydantic AI: MCP-powered research agent

**Motivation**: An engineer wants an agent that can browse the filesystem and query a database through standardized MCP servers, without writing custom tool integrations.

**Concepts exercised**: `MCPServerStdio`, `MCPServerStreamableHTTP`, `toolsets`, async context management

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

# Filesystem MCP server (reads local files)
fs_server = MCPServerStdio(
    'npx', args=['-y', '@modelcontextprotocol/server-filesystem', './docs']
)

# Database MCP server (running remotely)
db_server = MCPServerStreamableHTTP('http://localhost:3000/mcp')

agent = Agent(
    'google-gla:gemini-2.5-flash',
    instructions='You are a research assistant with access to local docs and a database.',
    toolsets=[fs_server, db_server],
)

# Must use async context manager to manage MCP connections
async def research(question: str) -> str:
    async with agent:
        result = await agent.run(question)
        return result.output

# Jupyter: import asyncio; asyncio.run(research('Summarize the README and list all users'))
```

**Environment**: Works in Jupyter with `asyncio.run()` or `nest_asyncio`. In FastAPI, use `lifespan` to manage the agent context.

### Pydantic AI: multi-agent report writer

**Motivation**: A content team needs a pipeline where a researcher gathers facts and a writer produces a polished report — two agents with different models and instructions.

**Concepts exercised**: Agent-as-tool delegation, shared `deps_type`, `usage` budget tracking

```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

class ResearchFindings(BaseModel):
    facts: list[str]
    sources: list[str]

@dataclass
class WriterDeps:
    topic: str
    max_facts: int = 5

researcher = Agent(
    'google-gla:gemini-2.5-flash',
    output_type=ResearchFindings,
    instructions='Find key facts about the given topic. Be specific and cite sources.',
)

writer = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=WriterDeps,
    instructions='Write a concise, well-structured report from research findings.',
)

@writer.tool
async def gather_research(ctx: RunContext[WriterDeps]) -> str:
    """Gather research findings on the topic."""
    result = await researcher.run(
        f'Research: {ctx.deps.topic}. Find {ctx.deps.max_facts} facts.',
        usage=ctx.usage,
    )
    findings = result.output
    return '\n'.join(f'- {f} (source: {s})' for f, s in zip(findings.facts, findings.sources))

# Usage
deps = WriterDeps(topic='Impact of AI on drug discovery', max_facts=5)
result = writer.run_sync('Write a report using the research tool.', deps=deps)
print(result.output)
```

### LangGraph: ReAct agent with Gemini and custom state

**Motivation**: Build a customer support agent that tracks not just messages but also the number of tool calls and escalation status — demonstrating custom state beyond `MessagesState`.

**Concepts exercised**: Custom `TypedDict` state, `Annotated` reducers, `add_messages`, `tools_condition`, Gemini integration

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int
    escalated: bool

@tool
def check_order(order_id: str) -> str:
    """Check the status of a customer order."""
    return f'Order {order_id}: Shipped, arriving Tuesday.'

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate this conversation to a human agent."""
    return f'Escalated: {reason}. A human agent will follow up.'

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model_with_tools = model.bind_tools([check_order, escalate_to_human])

def agent_node(state: SupportState):
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}

def track_tools(state: SupportState):
    # ToolNode runs tools, we also update our counter
    tool_node = ToolNode([check_order, escalate_to_human])
    result = tool_node.invoke(state)
    new_count = state['tool_call_count'] + len(state['messages'][-1].tool_calls)
    escalated = any(tc['name'] == 'escalate_to_human' for tc in state['messages'][-1].tool_calls)
    return {**result, 'tool_call_count': new_count, 'escalated': escalated or state['escalated']}

builder = StateGraph(SupportState)
builder.add_node('agent', agent_node)
builder.add_node('tools', track_tools)
builder.add_edge(START, 'agent')
builder.add_conditional_edges('agent', tools_condition)
builder.add_edge('tools', 'agent')

graph = builder.compile(checkpointer=InMemorySaver())

# Use in Jupyter
config = {'configurable': {'thread_id': 'support-001'}}
result = graph.invoke(
    {'messages': [('user', "Where's my order #12345?")], 'tool_call_count': 0, 'escalated': False},
    config,
)
```

### LangGraph: human-in-the-loop approval workflow

**Motivation**: A deployment pipeline agent must get human approval before executing destructive operations (database migrations, production deployments).

**Concepts exercised**: `interrupt()`, `Command(resume=)`, checkpointer persistence, conditional routing

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

@tool
def run_migration(migration_name: str) -> str:
    """Execute a database migration. Requires approval."""
    return f'Migration {migration_name} executed successfully.'

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

def plan_node(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': [response]}

def approval_gate(state: MessagesState):
    last_msg = state['messages'][-1]
    # Pause for human review
    decision = interrupt({
        'type': 'approval_required',
        'proposed_action': last_msg.content,
        'instructions': 'Reply "approve" or "reject"',
    })
    if decision == 'approve':
        return {'messages': [('system', 'Action approved by human reviewer.')]}
    return {'messages': [('system', f'Action rejected: {decision}')]}

def execute_node(state: MessagesState):
    return {'messages': [('assistant', 'Executing approved action...')]}

builder = StateGraph(MessagesState)
builder.add_node('plan', plan_node)
builder.add_node('approve', approval_gate)
builder.add_node('execute', execute_node)
builder.add_edge(START, 'plan')
builder.add_edge('plan', 'approve')
builder.add_edge('approve', 'execute')
builder.add_edge('execute', END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {'configurable': {'thread_id': 'deploy-1'}}

# Step 1: Execution pauses at interrupt
result = graph.invoke({'messages': [('user', 'Run the user_table_v2 migration')]}, config)
# The graph is now paused — inspect result['__interrupt__']

# Step 2: Human approves
result = graph.invoke(Command(resume='approve'), config)
# Execution continues through execute_node
```

**Environment**: Jupyter for testing. For production, build a FastAPI endpoint that stores the `thread_id`, returns the interrupt payload to a frontend, and resumes on POST.

### LangGraph: MCP-powered multi-tool agent

**Motivation**: Connect a single agent to multiple MCP servers (filesystem, database, web search) to build a versatile research assistant without writing tool integrations.

**Concepts exercised**: `MultiServerMCPClient`, MCP transports (stdio + HTTP), `create_react_agent`, tool discovery

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

async def build_agent():
    client = MultiServerMCPClient({
        'filesystem': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-filesystem', './data'],
            'transport': 'stdio',
        },
        'database': {
            'url': 'http://localhost:5000/mcp',
            'transport': 'http',
        },
    })
    tools = await client.get_tools()
    return create_react_agent(model, tools, checkpointer=InMemorySaver())

# Jupyter
import asyncio
agent = asyncio.run(build_agent())
result = agent.invoke(
    {'messages': [('user', 'List all CSV files in ./data and query the users table')]},
    config={'configurable': {'thread_id': 'research-1'}},
)
```

For FastAPI, initialize the `MultiServerMCPClient` in the app lifespan and reuse the compiled agent across requests.

---

## Conclusion: choosing the right framework

**Pydantic AI is the right choice when** you want type-safe, testable agent code with minimal abstraction overhead. Its dependency injection pattern makes it ideal for production services where agents are embedded in larger applications (FastAPI backends, microservices). First-class MCP and A2A support mean you can participate in the broader agent ecosystem without extra packages. The tradeoff: you build persistence, memory, and complex orchestration yourself.

**LangGraph is the right choice when** your workflow demands explicit state management, branching logic, human-in-the-loop approvals, or sophisticated multi-agent orchestration. Its checkpointer system provides persistence, time travel, and fault tolerance out of the box. The five streaming modes make it particularly strong for real-time chat UIs. The tradeoff: more boilerplate for simple agents, and MCP requires an adapter package.

**For an ADK engineer**, Pydantic AI will feel more familiar in its agent-centric model — one class, configure it, run it. LangGraph will feel more like building what ADK's workflow agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) do, but with arbitrary graph topology. Both frameworks interoperate with ADK through MCP and A2A, so the real question isn't which to choose exclusively — it's which to reach for based on the complexity of each specific workflow.