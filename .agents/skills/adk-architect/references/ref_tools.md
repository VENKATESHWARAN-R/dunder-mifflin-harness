# ref_tools.md — Tools Reference

## Function Tools

### Basic function tool
```python
def search_documents(query: str, collection: str = "docs", top_k: int = 5) -> dict:
    """Search the document store for relevant content.

    Args:
        query: Natural language search query.
        collection: Collection name to search in. Defaults to 'docs'.
        top_k: Maximum results to return (1-20).

    Returns:
        Dict with 'results' list (each with 'content', 'score', 'source') and 'total'.
    """
    results = vector_db.search(collection, query, limit=top_k)
    return {
        "results": [{"content": r.text, "score": r.score, "source": r.metadata["source"]} for r in results],
        "total": len(results),
    }

# Register: tools=[search_documents]
```

**Rules:**
- All params need type hints → used to build JSON schema for the LLM
- Docstring body = description the LLM sees. Keep it accurate.
- `Args:` section = per-param descriptions. Critical for complex tools.
- Return `dict` always. Never raise exceptions — return error info in the dict.
- For Optional params: use `param: str = "default"` or `param: Optional[str] = None`
- Async tools: `async def my_tool(...) -> dict:` works natively

### Tool with ToolContext (state, actions)
```python
from google.adk.tools import ToolContext

def get_or_fetch_data(resource_id: str, tool_context: ToolContext) -> dict:
    """Fetch resource data, using session cache if available.

    Args:
        resource_id: Unique identifier for the resource.
    """
    # Check cache
    cache_key = f"temp:resource:{resource_id}"
    if cached := tool_context.state.get(cache_key):
        return {"data": cached, "from_cache": True}

    # Fetch fresh
    data = external_api.fetch(resource_id)
    tool_context.state[cache_key] = data   # temp: clears after turn
    tool_context.state[f"user:last_resource"] = resource_id   # persist for user
    return {"data": data, "from_cache": False}

def loop_done_if_good(score: int, tool_context: ToolContext) -> dict:
    """Signal loop completion if quality threshold met.

    Args:
        score: Quality score (0-10).
    """
    if score >= 8:
        tool_context.actions.escalate = True   # stops LoopAgent
    return {"score": score, "escalated": score >= 8}

def hand_off(agent_name: str, reason: str, tool_context: ToolContext) -> dict:
    """Transfer control to another agent.

    Args:
        agent_name: Name of the target agent.
        reason: Why transferring.
    """
    tool_context.actions.transfer_to_agent = agent_name
    return {"status": "transferring", "to": agent_name, "reason": reason}
```

### Action confirmation (pause for human approval)
```python
def execute_destructive_action(table: str, condition: str, tool_context: ToolContext) -> dict:
    """Delete records matching the condition. Requires user confirmation.

    Args:
        table: Table to delete from.
        condition: WHERE clause condition.
    """
    # Request confirmation — pauses agent and presents message to user
    if not tool_context.state.get("deletion_confirmed"):
        tool_context.actions.request_human_confirmation = True
        tool_context.actions.confirmation_message = (
            f"About to DELETE FROM {table} WHERE {condition}. This is irreversible. Confirm?"
        )
        tool_context.state["pending_deletion"] = {"table": table, "condition": condition}
        return {"status": "awaiting_confirmation"}

    # Runs after user confirms
    affected = db.execute(f"DELETE FROM {table} WHERE {condition}")
    tool_context.state.pop("deletion_confirmed", None)
    return {"status": "deleted", "rows_affected": affected}
```

---

## MCP Tools

### Stdio (subprocess) MCP server
```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Use uvx to run a published MCP server
filesystem_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="uvx",
        args=["mcp-server-filesystem", "/workspace"],
    )
)

# Run a local custom MCP server
custom_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
        env={"DB_URL": os.environ["DB_URL"], "API_KEY": os.environ["API_KEY"]},
    ),
    tool_filter=["search_kb", "get_status", "create_ticket"],  # whitelist specific tools
)

agent = LlmAgent(model="gemini-2.5-flash", tools=[filesystem_tools, custom_tools])
```

### SSE (HTTP) MCP server
```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

remote_tools = MCPToolset(
    connection_params=SseServerParams(
        url="http://mcp-server:8080/mcp",
        headers={
            "Authorization": f"Bearer {os.environ['MCP_TOKEN']}",
            "X-App-ID": "my_agent",
        },
    )
)
```

### Build your own MCP server (expose Python functions as MCP tools)
```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP
import psutil

mcp = FastMCP("Infrastructure Tools")

@mcp.tool()
def get_k8s_pod_status(namespace: str = "default", label_selector: str = "") -> dict:
    """Get Kubernetes pod status.

    Args:
        namespace: K8s namespace to query.
        label_selector: Optional label selector filter.
    """
    pods = k8s_client.list_namespaced_pod(namespace, label_selector=label_selector)
    return {"pods": [{"name": p.metadata.name, "phase": p.status.phase} for p in pods.items]}

@mcp.tool()
def get_system_metrics() -> dict:
    """Get current system CPU and memory metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }

if __name__ == "__main__":
    mcp.run()   # starts stdio MCP server
```

---

## OpenAPI Tools

```python
from google.adk.tools.openapi_tool import OpenApiToolset

# From URL
api_tools = OpenApiToolset.from_url(
    "https://api.example.com/openapi.json",
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {os.environ['API_TOKEN']}"},
    operation_filter=["getUser", "listOrders"],   # only expose these operations
)

# From file
api_tools = OpenApiToolset.from_file(
    "api_spec.yaml",
    base_url="http://localhost:8000",
)

agent = LlmAgent(model="gemini-2.5-flash", tools=[api_tools])
```

---

## ADK Skills (Experimental)

ADK Skills are reusable, packaged capability bundles — richer than MCP tools because
they can carry state, callbacks, and multi-step logic native to ADK.

```python
from google.adk.skills import BaseSkill

class SearchSkill(BaseSkill):
    """Reusable search capability with caching and rate limiting."""

    name = "search_skill"
    description = "Semantic search over a document collection with caching."

    def __init__(self, collection: str, cache_ttl: int = 300):
        self.collection = collection
        self.cache_ttl = cache_ttl

    def get_tools(self) -> list:
        """Return the tools this skill provides."""
        return [self._search_fn, self._index_fn]

    def _search_fn(self, query: str, top_k: int = 5) -> dict:
        """Search the collection."""
        return vector_db.search(self.collection, query, top_k)

    def _index_fn(self, content: str, metadata: dict) -> dict:
        """Index new content."""
        vector_db.upsert(self.collection, content, metadata)
        return {"indexed": True}

# Attach to agent
agent = LlmAgent(
    model="gemini-2.5-flash",
    skills=[SearchSkill(collection="incident_docs")],   # experimental API
)
```

> Note: Skills API is experimental. Check ADK release notes for stability changes.

---

## Authentication

### API key in tool
```python
import os

def call_external_api(endpoint: str, payload: dict) -> dict:
    """Call external REST API."""
    response = requests.post(
        f"https://api.example.com/{endpoint}",
        json=payload,
        headers={"X-API-Key": os.environ["EXTERNAL_API_KEY"]},
    )
    response.raise_for_status()
    return response.json()
```

### OAuth2 (user-delegated token, pauses for auth handshake)
```python
from google.adk.auth import OAuth2Config
from google.adk.tools import ToolContext

OAUTH_CONFIG = OAuth2Config(
    client_id=os.environ["OAUTH_CLIENT_ID"],
    client_secret=os.environ["OAUTH_CLIENT_SECRET"],
    auth_uri="https://auth.example.com/oauth/authorize",
    token_uri="https://auth.example.com/oauth/token",
    scopes=["read:data", "write:data"],
)

def call_protected_resource(resource_id: str, tool_context: ToolContext) -> dict:
    """Access protected resource via OAuth2.

    Args:
        resource_id: ID of the resource to access.
    """
    # get_token() handles token caching, refresh, and pausing for auth if needed
    token = tool_context.auth.get_token(OAUTH_CONFIG)
    response = requests.get(
        f"https://api.example.com/resources/{resource_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    return response.json()
```

### Service account (server-to-server)
```python
from google.oauth2 import service_account

def get_service_token(scopes: list[str]) -> str:
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=scopes,
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token
```