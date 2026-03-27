# ADK Tools, Sessions, State, Memory, A2A, Plugins Reference

## Table of Contents
1. [Function Tools](#function-tools)
2. [MCP Tools](#mcp-tools)
3. [Authentication](#authentication)
4. [Sessions](#sessions)
5. [State (session.state)](#state)
6. [Memory (cross-session)](#memory)
7. [Artifacts](#artifacts)
8. [A2A Protocol](#a2a-protocol)
9. [Plugins](#plugins)
10. [Callbacks Deep Dive](#callbacks-deep-dive)

---

## Function Tools

### Basic Function Tool
ADK auto-introspects Python functions. **Docstring = tool description sent to LLM**.

```python
def search_qdrant(query: str, collection: str = "docs", top_k: int = 5) -> dict:
    """Search the Qdrant vector database for relevant documents.

    Args:
        query: Natural language search query.
        collection: Name of the Qdrant collection to search.
        top_k: Number of results to return (max 20).

    Returns:
        dict with 'results' (list of dicts with 'content' and 'score') and 'total'.
    """
    # your implementation
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    results = client.search(collection_name=collection, query_text=query, limit=top_k)
    return {"results": [{"content": r.payload["text"], "score": r.score} for r in results],
            "total": len(results)}
```

**Rules:**
- All params must have type hints (used to build JSON schema for LLM)
- Return `dict` — LLM sees the entire dict as tool output
- Use `Optional[T]` or defaults for optional params
- Return error info in the dict, don't raise exceptions from tools (LLM can retry)

### Tool with ToolContext
```python
from google.adk.tools import ToolContext

def write_to_state(key: str, value: str, tool_context: ToolContext) -> dict:
    """Write a value to the session state.

    Args:
        key: State key to write.
        value: Value to store.

    Returns:
        Confirmation dict.
    """
    tool_context.state[key] = value
    return {"status": "ok", "key": key}

def read_from_state(key: str, tool_context: ToolContext) -> dict:
    """Read a value from session state."""
    value = tool_context.state.get(key)
    return {"key": key, "value": value, "found": value is not None}

# ToolContext also gives you:
# tool_context.session           → full session object
# tool_context.agent_name        → which agent called this tool
# tool_context.actions.escalate = True     → stop LoopAgent
# tool_context.actions.transfer_to_agent = "name"  → hand off to another agent
```

### Action Confirmation (human approval before tool executes)
```python
from google.adk.tools.tool_context import ToolContext

def delete_records(table: str, condition: str, tool_context: ToolContext) -> dict:
    """Delete records from the database. REQUIRES confirmation."""
    # This will pause execution and ask user to confirm
    tool_context.actions.request_human_confirmation = True
    tool_context.actions.confirmation_message = (
        f"About to DELETE from {table} WHERE {condition}. Confirm?"
    )
    # Code below runs only AFTER confirmation is received
    db.execute(f"DELETE FROM {table} WHERE {condition}")
    return {"status": "deleted", "table": table}
```

### Long-running Tool (background)
```python
from google.adk.tools import LongRunningTool

class DataExportTool(LongRunningTool):
    name = "export_data"
    description = "Export large datasets asynchronously."

    async def run_async(self, params: dict, tool_context: ToolContext):
        # Kicks off background job and returns immediately
        job_id = await start_export_job(params["dataset"])
        tool_context.state["export_job_id"] = job_id
        return {"status": "started", "job_id": job_id}

    async def check_status(self, job_id: str) -> dict:
        return await check_export_status(job_id)
```

---

## MCP Tools

### Subprocess (stdio) MCP Server
```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Filesystem MCP server
fs_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="uvx",
        args=["mcp-server-filesystem", "/data"],
    )
)

# Custom local MCP server
custom_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
        env={"DB_URL": os.environ["DB_URL"]},
    )
)

root_agent = LlmAgent(model="gemini-2.5-flash", tools=[fs_tools, custom_tools])
```

### SSE MCP Server (HTTP)
```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

remote_tools = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:8080/mcp",
        headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
    )
)
```

### Filter Specific Tools from MCP Server
```python
# Only expose certain tools from the MCP server to the agent
fs_tools = MCPToolset(
    connection_params=StdioServerParameters(command="uvx", args=["mcp-server-filesystem", "/"]),
    tool_filter=["read_file", "list_directory"],  # whitelist
)
```

### Build Your Own MCP Server (expose ADK agent as MCP)
```python
# my_mcp_server.py — expose Python functions as MCP tools
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Tool Server")

@mcp.tool()
def search_internal_kb(query: str, limit: int = 5) -> dict:
    """Search the internal knowledge base."""
    return {"results": kb.search(query, limit)}

@mcp.tool()
def get_system_status() -> dict:
    """Get current system health metrics."""
    return {"cpu": psutil.cpu_percent(), "memory": psutil.virtual_memory().percent}

if __name__ == "__main__":
    mcp.run()
```

---

## Authentication

### API Key Auth
```python
from google.adk.tools.rest_api_tool import RestApiTool
from google.adk.tools.openapi_tool.auth.auth_helpers import service_account_scheme_credential

# For tools requiring auth headers
class AuthenticatedApiTool(RestApiTool):
    def get_auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
```

### OAuth2 (for user-delegated auth)
```python
from google.adk.auth import OAuth2Config

oauth_config = OAuth2Config(
    client_id=os.environ["OAUTH_CLIENT_ID"],
    client_secret=os.environ["OAUTH_CLIENT_SECRET"],
    auth_uri="https://auth.example.com/oauth/authorize",
    token_uri="https://auth.example.com/oauth/token",
    scopes=["read:data", "write:data"],
)

def protected_api_call(resource_id: str, tool_context: ToolContext) -> dict:
    """Call protected API with OAuth2 token."""
    token = tool_context.auth.get_token(oauth_config)
    response = requests.get(
        f"https://api.example.com/resources/{resource_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    return response.json()
```

---

## Sessions

Sessions group a conversation (turns) for a user in an app.

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# In-memory (dev/testing)
session_service = InMemorySessionService()

# Persistent (production) — use your own DB
# session_service = DatabaseSessionService(db_url=os.environ["DATABASE_URL"])

from google.adk.runners import Runner

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="my_app",
)

# Create session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    state={"language": "en", "mode": "verbose"},  # initial state
)

# Get existing session
session = await session_service.get_session(
    app_name="my_app", user_id="user_123", session_id="sess_abc"
)

# List user's sessions
sessions = await session_service.list_sessions(app_name="my_app", user_id="user_123")
```

### Rewind Session
Roll back to a previous state in the session (useful for debugging/undo).
```python
# Get session history
events = await session_service.get_session_events(session_id=session.id)

# Rewind to event N
await session_service.rewind_session(session_id=session.id, event_index=N)
```

---

## State

State is a key-value store attached to a session. Persists across turns within a session.

### Prefix Scopes
```python
# session-scoped (default) — lives for this session only
ctx.state["current_task"] = "analyze"

# user-scoped — persists across sessions for this user
ctx.state["user:preferences"] = {"language": "fi", "verbose": True}

# app-scoped — shared across ALL users and sessions
ctx.state["app:config"] = {"max_results": 10}

# temp-scoped — cleared after each turn (good for intermediate results)
ctx.state["temp:raw_data"] = fetched_data
```

### State in Instructions (templating)
```python
# ADK auto-substitutes {key} from session.state into instruction strings
root_agent = LlmAgent(
    instruction="User {user:name} prefers {user:language}. Current task: {current_task}.",
    ...
)
```

### Update State in Tools (recommended way)
```python
def update_preferences(key: str, value: str, tool_context: ToolContext) -> dict:
    tool_context.state[f"user:{key}"] = value  # user-scoped
    return {"updated": key}
```

### Update State via output_key (agent output)
```python
agent = LlmAgent(
    output_schema=MyModel,
    output_key="analysis",  # writes parsed MyModel to state["analysis"]
)
# Downstream agents access via: {analysis} in instruction
```

---

## Memory

Memory persists **across multiple sessions** (long-term). Different from state which is per-session.

```python
from google.adk.memory import InMemoryMemoryService
# or: from google.adk.memory import VertexAiRagMemoryService  (Google-specific, skip if OSS)

memory_service = InMemoryMemoryService()

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    memory_service=memory_service,
    app_name="my_app",
)

# Agent uses load_memory / save_memory tools automatically when memory_service is set
# Or manually:
from google.adk.tools.memory_tool import load_memory, save_memory

root_agent = LlmAgent(
    tools=[load_memory, save_memory],
    instruction="Use load_memory to recall past interactions. Save important facts with save_memory.",
)
```

### Custom Memory Backend (e.g., Qdrant)
```python
from google.adk.memory import BaseMemoryService

class QdrantMemoryService(BaseMemoryService):
    def __init__(self, qdrant_url: str, collection: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection

    async def add_memory(self, app_name: str, user_id: str, content: str, metadata: dict = None):
        embedding = get_embedding(content)
        self.client.upsert(collection_name=self.collection, points=[
            PointStruct(id=str(uuid4()), vector=embedding,
                        payload={"content": content, "user_id": user_id, **(metadata or {})})
        ])

    async def search_memory(self, app_name: str, user_id: str, query: str, limit: int = 5):
        embedding = get_embedding(query)
        results = self.client.search(self.collection, query_vector=embedding, limit=limit,
                                     query_filter=Filter(must=[FieldCondition(
                                         key="user_id", match=MatchValue(value=user_id))]))
        return [r.payload["content"] for r in results]
```

---

## Artifacts

Store binary/file data produced during agent runs (images, PDFs, etc.).

```python
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools import ToolContext

artifact_service = InMemoryArtifactService()
runner = Runner(..., artifact_service=artifact_service)

def save_report(content: bytes, filename: str, tool_context: ToolContext) -> dict:
    """Save a generated report as an artifact."""
    artifact_id = tool_context.save_artifact(
        data=content,
        filename=filename,
        mime_type="application/pdf",
    )
    return {"artifact_id": artifact_id, "filename": filename}

def load_report(artifact_id: str, tool_context: ToolContext) -> dict:
    """Load a previously saved artifact."""
    data = tool_context.load_artifact(artifact_id=artifact_id)
    return {"data": data.decode(), "artifact_id": artifact_id}
```

---

## A2A Protocol

Agent-to-Agent: expose your ADK agent as an A2A server, or call external A2A agents.

### Expose ADK Agent as A2A Server
```python
# server.py
from google.adk.a2a import A2AServer
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="my_service")

server = A2AServer(runner=runner, host="0.0.0.0", port=9000)
server.start()  # Exposes /.well-known/agent.json + /tasks endpoint
```

### Consume External A2A Agent as a Tool
```python
from google.adk.tools.a2a_tool import A2ATool

# Load remote agent's capabilities from its well-known URL
remote_agent_tool = A2ATool.from_url("http://specialist-service:9000")

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="coordinator",
    instruction="Use the remote specialist agent for domain-specific tasks.",
    tools=[remote_agent_tool],
)
```

### A2A with Authentication
```python
remote_agent_tool = A2ATool.from_url(
    "http://specialist-service:9000",
    headers={"Authorization": f"Bearer {os.environ['SERVICE_TOKEN']}"},
)
```

---

## Plugins

Plugins hook into the agent event loop globally — good for logging, tracing, rate-limiting.

```python
from google.adk.plugins import BasePlugin
from google.adk.events import Event

class ObservabilityPlugin(BasePlugin):
    """Log all events to your observability stack."""

    async def on_event(self, event: Event, context) -> Event:
        # Called for every event in the agent run
        log_to_otel(
            event_type=event.type,
            agent=event.author,
            session_id=context.session.id,
        )
        return event  # must return the event (can modify it)

class RateLimitPlugin(BasePlugin):
    """Enforce rate limits per session."""

    def __init__(self, max_calls_per_minute: int = 60):
        self.limiter = RateLimiter(max_calls_per_minute)

    async def on_event(self, event: Event, context) -> Event:
        if event.type == "model_request":
            await self.limiter.acquire(key=context.session.user_id)
        return event

root_agent = LlmAgent(
    ...,
    plugins=[ObservabilityPlugin(), RateLimitPlugin(max_calls_per_minute=30)],
)
```

---

## Callbacks Deep Dive

Callbacks are agent-specific (vs plugins which are global).

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool

# BEFORE model call — can short-circuit by returning LlmResponse
def guard_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    user_input = str(llm_request.contents)
    if "forbidden_keyword" in user_input.lower():
        return LlmResponse(content=Content(parts=[Part(text="Sorry, that topic is restricted.")]))
    return None  # proceed normally

# AFTER model call — inspect/modify the response
def log_response_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    print(f"Model used {llm_response.usage.total_token_count} tokens")
    callback_context.state["last_token_count"] = llm_response.usage.total_token_count
    return None  # return modified response or None to keep original

# BEFORE tool call — can skip tool or modify args
def tool_audit_callback(
    callback_context: CallbackContext,
    tool: BaseTool,
    args: dict,
    tool_context
) -> dict | None:
    print(f"Tool called: {tool.name} with args: {args}")
    if tool.name == "delete_records" and not callback_context.state.get("admin_confirmed"):
        return {"error": "Admin confirmation required before destructive operations"}
    return None  # proceed with original args

root_agent = LlmAgent(
    ...,
    before_model_callback=guard_callback,
    after_model_callback=log_response_callback,
    before_tool_callback=tool_audit_callback,
)
```