# ref_cross_cutting.md — Callbacks, Plugins, A2A, Evaluation

## Callbacks — All 6 Types

Callbacks are agent-specific (set per agent). Return `None` to proceed normally.
Return a non-None value to **short-circuit** the normal step.

### before_agent_callback
Fires before the agent's `_run_async_impl` executes. Return `Content` to skip the agent entirely.

```python
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, Part
from typing import Optional

def auth_gate(callback_context: CallbackContext) -> Optional[Content]:
    """Block unauthorized users before agent runs."""
    user_id = callback_context.state.get("user:id")
    role = callback_context.state.get("user:role", "guest")

    if role not in ("admin", "analyst"):
        return Content(parts=[Part(text="Access denied. Insufficient permissions.")])

    # Log entry
    callback_context.state["temp:agent_start_ts"] = time.time()
    return None   # proceed

agent = LlmAgent(..., before_agent_callback=auth_gate)
```

### after_agent_callback
Fires after the agent produces its final output. Return `Content` to replace the output.

```python
def strip_pii(callback_context: CallbackContext, response: Content) -> Optional[Content]:
    """Scrub PII from agent response before returning to user."""
    if not response or not response.parts:
        return None

    cleaned_text = pii_scrubber.clean(response.parts[0].text)
    if cleaned_text != response.parts[0].text:
        callback_context.state["temp:pii_detected"] = True
        return Content(parts=[Part(text=cleaned_text)])
    return None   # no change needed
```

### before_model_callback
Fires before every LLM API call. Return `LlmResponse` to skip the model call entirely.

```python
from google.adk.models import LlmRequest, LlmResponse
import hashlib

_cache: dict = {}

def caching_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Return cached model response if the prompt hasn't changed."""
    prompt_hash = hashlib.md5(str(llm_request.contents).encode()).hexdigest()

    if prompt_hash in _cache:
        callback_context.state["temp:cache_hit"] = True
        return _cache[prompt_hash]

    callback_context.state["temp:prompt_hash"] = prompt_hash
    return None   # call the model

def input_guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Reject disallowed topics before they reach the model."""
    user_msg = str(llm_request.contents).lower()
    blocked = ["competitor name", "internal salary", "confidential project x"]

    for term in blocked:
        if term in user_msg:
            return LlmResponse(
                content=Content(parts=[Part(text=f"I can't discuss that topic.")])
            )
    return None

agent = LlmAgent(..., before_model_callback=input_guardrail)
```

### after_model_callback
Fires after the LLM responds. Return `LlmResponse` to replace the model's response.

```python
def token_logger(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Log token usage to session state for cost tracking."""
    usage = getattr(llm_response, "usage_metadata", None)
    if usage:
        callback_context.state["temp:tokens_in"] = usage.prompt_token_count
        callback_context.state["temp:tokens_out"] = usage.candidates_token_count
        callback_context.state["user:total_tokens"] = (
            callback_context.state.get("user:total_tokens", 0) + usage.total_token_count
        )

    # Also cache the response
    prompt_hash = callback_context.state.get("temp:prompt_hash")
    if prompt_hash:
        _cache[prompt_hash] = llm_response

    return None   # don't modify the response
```

### before_tool_callback
Fires before each tool function executes. Return `dict` to skip the tool and return that dict as the result.

```python
from google.adk.tools.base_tool import BaseTool

def tool_rate_limiter(
    callback_context: CallbackContext,
    tool: BaseTool,
    args: dict,
    tool_context,
) -> Optional[dict]:
    """Rate limit expensive external tool calls."""
    if tool.name in ("web_search", "external_api_call"):
        count_key = f"temp:tool_call_count:{tool.name}"
        count = callback_context.state.get(count_key, 0)
        if count >= 5:
            return {"error": f"Rate limit exceeded for {tool.name}. Max 5 calls per turn."}
        callback_context.state[count_key] = count + 1

    return None   # proceed with tool

def dry_run_mode(
    callback_context: CallbackContext,
    tool: BaseTool,
    args: dict,
    tool_context,
) -> Optional[dict]:
    """Intercept destructive tools in dry-run mode."""
    destructive = {"delete_records", "send_email", "deploy_service"}
    if callback_context.state.get("app:dry_run_mode") and tool.name in destructive:
        return {"status": "dry_run", "would_have_called": tool.name, "with_args": args}
    return None
```

### after_tool_callback
Fires after each tool function returns. Return `dict` to replace the tool result.

```python
def enrich_tool_result(
    callback_context: CallbackContext,
    tool: BaseTool,
    args: dict,
    tool_context,
    result: dict,
) -> Optional[dict]:
    """Enrich search results with additional metadata."""
    if tool.name == "search_documents" and "results" in result:
        # Add freshness info to each result
        for r in result["results"]:
            r["retrieved_at"] = time.time()
            r["agent"] = callback_context.agent_name
    return result   # return the modified result (or None to keep original)
```

### Combining callbacks on one agent
```python
agent = LlmAgent(
    model="gemini-2.5-flash",
    name="guarded_agent",
    before_agent_callback=auth_gate,
    after_agent_callback=strip_pii,
    before_model_callback=input_guardrail,
    after_model_callback=token_logger,
    before_tool_callback=tool_rate_limiter,
    after_tool_callback=enrich_tool_result,
)
```

---

## Plugins

Global event hooks — apply across all agents in a runner without configuring each agent.
Implement `BasePlugin`, override `on_event`. Must return the event (can modify it).

```python
from google.adk.plugins import BasePlugin
from google.adk.events import Event

class ObservabilityPlugin(BasePlugin):
    """Send all agent events to OpenTelemetry."""

    async def on_event(self, event: Event, context) -> Event:
        span_attributes = {
            "adk.agent": event.author,
            "adk.session_id": context.session.id,
            "adk.user_id": context.session.user_id,
            "adk.event_type": str(event.event_type),
        }
        otel_tracer.add_event("adk_event", attributes=span_attributes)
        return event

class SecurityPolicyPlugin(BasePlugin):
    """Block disallowed tool calls globally across all agents."""

    BLOCKED_TOOLS = {"drop_database", "rm_rf", "disable_firewall"}

    async def on_event(self, event: Event, context) -> Event:
        if event.content:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call.name in self.BLOCKED_TOOLS:
                    raise PermissionError(
                        f"Tool '{part.function_call.name}' is blocked by security policy."
                    )
        return event

class RateLimitPlugin(BasePlugin):
    """Global per-user rate limiting."""

    def __init__(self, max_turns_per_minute: int = 20):
        self.limiters: dict = {}
        self.max = max_turns_per_minute

    async def on_event(self, event: Event, context) -> Event:
        if event.author == "user":
            user_id = context.session.user_id
            limiter = self.limiters.setdefault(user_id, TokenBucket(self.max))
            if not limiter.consume():
                raise Exception(f"Rate limit exceeded for user {user_id}")
        return event

# Attach to agents at definition time
agent = LlmAgent(
    ...,
    plugins=[ObservabilityPlugin(), SecurityPolicyPlugin(), RateLimitPlugin(max_turns_per_minute=30)],
)
```

### Built-in: ReflectAndRetryPlugin
```python
from google.adk.plugins import ReflectAndRetryPlugin

# Makes agent reflect on tool failure and retry with better arguments
agent = LlmAgent(
    ...,
    plugins=[ReflectAndRetryPlugin(max_retries=3)],
)
```

---

## A2A Protocol

### Expose ADK agent as A2A server
```python
# server.py
from google.adk.a2a import A2AServer
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
runner = Runner(agent=specialist_agent, session_service=session_service, app_name="specialist_svc")

server = A2AServer(
    runner=runner,
    host="0.0.0.0",
    port=9000,
    # Agent card (capability discovery) auto-generated from agent.name + description
)

if __name__ == "__main__":
    server.start()
    # Exposes:
    # GET  /.well-known/agent.json  → agent capability card
    # POST /tasks                   → send a task, get async result
    # GET  /tasks/{task_id}         → poll task status
```

### Consume remote A2A agent as a tool
```python
from google.adk.tools.a2a_tool import A2ATool

# Load remote agent capabilities from well-known URL
remote_specialist = A2ATool.from_url(
    "http://specialist-service:9000",
    headers={"Authorization": f"Bearer {os.environ['SERVICE_TOKEN']}"},
)

orchestrator = LlmAgent(
    model="gemini-2.5-flash",
    name="orchestrator",
    instruction="""You coordinate work across specialist services.
    Use remote_specialist for domain-specific tasks.
    Handle general queries yourself.""",
    tools=[remote_specialist, local_tool],
)
```

### A2A in Kubernetes (service-to-service)
```python
# Assumes specialist-svc is a K8s Service in same namespace
remote_specialist = A2ATool.from_url(
    "http://specialist-svc.default.svc.cluster.local:9000",
    # In-cluster: no auth headers needed if using NetworkPolicy + ServiceAccount
)
```

---

## Evaluation

### Eval set format (JSON)
```json
[
  {
    "query": "What is the capital of Finland?",
    "expected_response": "Helsinki",
    "expected_tool_calls": []
  },
  {
    "query": "Search for recent incidents involving service X",
    "expected_response": null,
    "expected_tool_calls": [
      {"name": "search_incidents", "args": {"service": "service_x"}}
    ]
  }
]
```

### Running evals
```bash
# CLI
adk eval my_agent tests/eval_set.json --output results.json

# With specific criteria
adk eval my_agent tests/eval_set.json \
  --criteria response_accuracy tool_call_accuracy \
  --output results.json
```

### Programmatic eval
```python
from google.adk.evaluate import AgentEvaluator, EvalCriteria

evaluator = AgentEvaluator(
    runner=runner,
    criteria=[
        EvalCriteria.RESPONSE_ACCURACY,
        EvalCriteria.TOOL_CALL_ACCURACY,
        EvalCriteria.GROUNDEDNESS,
    ],
)

results = await evaluator.evaluate_from_file("tests/eval_set.json")
print(f"Accuracy: {results.response_accuracy:.2%}")
print(f"Tool accuracy: {results.tool_call_accuracy:.2%}")
```

### User simulation (automated multi-turn testing)
```python
from google.adk.evaluate import UserSimulator

# Let an LLM simulate the user for multi-turn eval
simulator = UserSimulator(
    model="gemini-2.5-flash",
    scenario="""You are a DevOps engineer testing an incident response agent.
    Start by reporting a service outage, then ask follow-up questions about RCA,
    and finally request a post-mortem report.""",
    max_turns=10,
)

result = await simulator.run(runner=runner, user_id="sim_user", session_id="sim_sess")
```

### Custom eval metric
```python
from google.adk.evaluate import BaseEvalMetric

class ResponseLengthMetric(BaseEvalMetric):
    name = "response_length_appropriate"

    async def score(self, query: str, response: str, expected: str, **kwargs) -> float:
        # Score 1.0 if response is within 50% of expected length, 0.0 if way off
        if not expected:
            return 1.0
        ratio = len(response) / max(len(expected), 1)
        return 1.0 if 0.5 <= ratio <= 2.0 else 0.0

evaluator = AgentEvaluator(runner=runner, criteria=[ResponseLengthMetric()])
```