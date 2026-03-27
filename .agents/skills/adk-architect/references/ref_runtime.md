# ref_runtime.md — Runtime, Sessions, State, Memory, Artifacts, Events

## Runner — Running Agents Programmatically

```python
import asyncio
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.genai.types import Content, Part

# --- Full production-style setup ---
session_service = InMemorySessionService()       # swap with DB-backed for production
memory_service = InMemoryMemoryService()         # swap with Qdrant/ChromaDB for production
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=root_agent,
    app_name="my_app",
    session_service=session_service,
    memory_service=memory_service,               # optional
    artifact_service=artifact_service,           # optional
)

# --- Quick dev/test setup ---
runner = InMemoryRunner(agent=root_agent, app_name="my_app")

# --- Create a session with initial state ---
async def main():
    session = await session_service.create_session(
        app_name="my_app",
        user_id="user_123",
        state={
            "user:name": "Venkat",
            "user:lang": "en",
            "current_task": "analyze_logs",
        },
    )

    # --- Run a turn ---
    async for event in runner.run_async(
        user_id="user_123",
        session_id=session.id,
        new_message=Content(parts=[Part(text="Analyze the latest error logs.")]),
    ):
        # Stream partial events
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)

        # Detect final response
        if event.is_final_response():
            print("\n--- Done ---")
            final_text = event.content.parts[0].text

asyncio.run(main())
```

---

## Sessions

```python
from google.adk.sessions import InMemorySessionService

svc = InMemorySessionService()

# Create
session = await svc.create_session(app_name="app", user_id="u1", state={"key": "val"})

# Get by ID
session = await svc.get_session(app_name="app", user_id="u1", session_id="sess_abc")

# List user's sessions
sessions = await svc.list_sessions(app_name="app", user_id="u1")

# Delete
await svc.delete_session(app_name="app", user_id="u1", session_id="sess_abc")

# Rewind to a prior event (undo)
events = await svc.list_events(app_name="app", user_id="u1", session_id=session.id)
await svc.rewind_session(
    app_name="app", user_id="u1", session_id=session.id,
    event_id=events[3].id   # roll back to after event index 3
)
```

### Custom persistent session service
```python
from google.adk.sessions import BaseSessionService
from google.adk.sessions.session import Session
import json

class RedisSessionService(BaseSessionService):
    def __init__(self, redis_client):
        self.r = redis_client

    async def create_session(self, app_name, user_id, state=None, session_id=None):
        s = Session(app_name=app_name, user_id=user_id, state=state or {})
        await self.r.set(f"{app_name}:{user_id}:{s.id}", json.dumps(s.model_dump()))
        return s

    async def get_session(self, app_name, user_id, session_id):
        raw = await self.r.get(f"{app_name}:{user_id}:{session_id}")
        return Session.model_validate_json(raw) if raw else None
    # implement list_sessions, delete_session, append_event, ...
```

---

## State (session.state)

### Scope prefixes
```python
# From inside a tool (ToolContext) or callback (CallbackContext)

# Session-scoped (default) — gone when session ends
tool_context.state["current_task"] = "analyze"

# User-scoped — persists across sessions for this user
tool_context.state["user:preferences"] = {"verbose": True, "lang": "fi"}
tool_context.state["user:history_count"] = 42

# App-scoped — shared across ALL users
tool_context.state["app:feature_flags"] = {"beta_mode": False}
tool_context.state["app:rate_limit"] = 100

# Temp-scoped — cleared at end of each turn
tool_context.state["temp:raw_fetch"] = large_raw_response
```

### State in agent instructions (templating)
```python
agent = LlmAgent(
    # ADK auto-substitutes {key} from session.state
    instruction="""
    User: {user:name} | Language: {user:lang}
    Current task: {current_task}
    Prior analysis result: {analysis}
    """,
    ...
)
```

### Updating state from tools (preferred)
```python
from google.adk.tools import ToolContext

def save_result(key: str, value: str, tool_context: ToolContext) -> dict:
    """Save a result to session state.

    Args:
        key: State key (use user: prefix for cross-session persistence).
        value: Value to store.
    """
    tool_context.state[key] = value
    return {"saved": True, "key": key}
```

### Updating state via output_key (agents)
```python
from pydantic import BaseModel

class ExtractedData(BaseModel):
    entities: list[str]
    sentiment: str

agent = LlmAgent(
    output_schema=ExtractedData,
    output_key="extracted",   # auto-saves to session.state["extracted"]
)
# Downstream: reference as {extracted} in instructions
```

---

## Memory (long-term, cross-session)

```python
from google.adk.memory import InMemoryMemoryService

# Register with Runner
runner = Runner(agent=root_agent, memory_service=InMemoryMemoryService(), ...)

# Give agent memory tools
from google.adk.tools.memory_tool import load_memory, save_memory

root_agent = LlmAgent(
    tools=[load_memory, save_memory],
    instruction="""Before answering, call load_memory to check if you've seen
    this user's preferences before. After completing tasks, save important facts
    with save_memory.""",
)
```

### Custom Qdrant memory backend
```python
from google.adk.memory import BaseMemoryService
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from uuid import uuid4

class QdrantMemoryService(BaseMemoryService):
    def __init__(self, url: str, collection: str, embedder):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedder = embedder          # your embedding function

    async def add_memory(self, app_name: str, user_id: str, content: str, metadata: dict = None):
        vector = self.embedder(content)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={"content": content, "user_id": user_id, "app": app_name, **(metadata or {})},
            )]
        )

    async def search_memory(self, app_name: str, user_id: str, query: str, limit: int = 5) -> list[str]:
        vector = self.embedder(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            query_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="app", match=MatchValue(value=app_name)),
            ]),
        )
        return [r.payload["content"] for r in results]
```

---

## Artifacts

Named, versioned file storage. Not for structured data — for actual files (PDF, image, audio, CSV exports).

```python
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools import ToolContext

# Register with Runner
runner = Runner(agent=root_agent, artifact_service=InMemoryArtifactService(), ...)

# Save artifact from a tool
def generate_report(topic: str, tool_context: ToolContext) -> dict:
    """Generate and save a PDF report.

    Args:
        topic: Topic to generate report on.

    Returns:
        Artifact reference dict.
    """
    pdf_bytes = render_pdf(topic)          # your rendering logic
    artifact_id = tool_context.save_artifact(
        filename="report.pdf",
        data=pdf_bytes,
        mime_type="application/pdf",
    )
    return {"artifact_id": artifact_id, "filename": "report.pdf", "size": len(pdf_bytes)}

# Load artifact from a tool
def load_report(artifact_id: str, tool_context: ToolContext) -> dict:
    """Load a previously generated report.

    Args:
        artifact_id: ID returned when the artifact was saved.
    """
    artifact = tool_context.load_artifact(artifact_id=artifact_id)
    return {"data": artifact.data, "mime_type": artifact.mime_type}

# List artifacts for current session
def list_reports(tool_context: ToolContext) -> dict:
    """List all saved reports in this session."""
    artifacts = tool_context.list_artifacts()
    return {"artifacts": [{"id": a.id, "filename": a.filename} for a in artifacts]}
```

---

## Events

The event stream is the primary output of `runner.run_async()`. Each `Event` represents
one thing that happened (model spoke, tool called, tool returned, agent transferred, etc.).

```python
async for event in runner.run_async(user_id=..., session_id=..., new_message=...):

    # --- Who produced this event ---
    print(f"Author: {event.author}")       # agent name, "user", "tool", or "code_executor"

    # --- What type ---
    print(f"Type: {event.event_type}")     # "message", "tool_call", "tool_result", "agent_transfer"

    # --- Content ---
    if event.content:
        for part in event.content.parts:
            if part.text:
                print(f"Text: {part.text}")
            if hasattr(part, "function_call"):
                print(f"Tool call: {part.function_call.name}({part.function_call.args})")
            if hasattr(part, "function_response"):
                print(f"Tool result: {part.function_response.name} → {part.function_response.response}")

    # --- Actions carried by the event ---
    if event.actions:
        if event.actions.escalate:
            print("Loop escalating!")
        if event.actions.transfer_to_agent:
            print(f"Transferring to: {event.actions.transfer_to_agent}")

    # --- Terminal check ---
    if event.is_final_response():
        final = event.content.parts[0].text
        break
```

### Filtering events you care about
```python
async for event in runner.run_async(...):
    # Only process final text responses
    if event.is_final_response() and event.author != "user":
        yield event.content.parts[0].text

    # Only process tool calls (for logging/auditing)
    if event.content:
        for part in event.content.parts:
            if hasattr(part, "function_call"):
                log_tool_call(part.function_call.name, part.function_call.args)
```

---

## Context Compression

Prevents context window overflow in long-running / multi-turn agents.

```python
from google.adk.memory.compaction import LlmSummarizingCompactor, TokenCountCompactionTrigger

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="long_running",
    context_compaction_config={
        "trigger": TokenCountCompactionTrigger(
            token_threshold=8000,          # compress when context > 8k tokens
        ),
        "compactor": LlmSummarizingCompactor(
            model="gemini-2.5-flash",      # can use a cheaper/faster model for summarization
            summary_instruction="""Summarize the conversation so far. Preserve:
            - Key decisions made
            - Important facts discovered
            - Current task state
            - Pending action items""",
        ),
    },
)
```

**When to use:** Any agent expected to run for many turns or process large documents
where the accumulated conversation history will grow to exceed the model's context window.