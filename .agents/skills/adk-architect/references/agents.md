# ADK Agents Reference

## Table of Contents
1. [LlmAgent — Full Config](#llmagent--full-config)
2. [Workflow Agents](#workflow-agents)
3. [Multi-Agent Systems](#multi-agent-systems)
4. [Custom BaseAgent](#custom-baseagent)
5. [AgentConfig (YAML-driven)](#agentconfig-yaml-driven)
6. [Context Compression](#context-compression)
7. [Events](#events)

---

## LlmAgent — Full Config

```python
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    confidence: float

root_agent = LlmAgent(
    # Identity
    model="gemini-2.5-flash",
    name="my_agent",
    description="Used by orchestrators to route tasks here.",

    # Behavior
    instruction="""You are a data analysis assistant.
    Always return structured results using the output schema.
    Use the search tool when you need current data.""",

    # Tools
    tools=[my_fn_tool, mcp_toolset],

    # Structured output
    output_schema=OutputSchema,
    output_key="analysis_result",   # writes to session.state["analysis_result"]

    # Model tuning
    generate_content_config=GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
    ),

    # Context
    include_contents="default",     # "default" | "none"

    # Sub-agents (for orchestrator pattern)
    sub_agents=[specialist_a, specialist_b],

    # Lifecycle hooks
    before_model_callback=my_before_cb,
    after_model_callback=my_after_cb,
    before_tool_callback=my_tool_cb,

    # Plugins
    plugins=[my_plugin],
)
```

**Key params:**
- `output_schema` + `output_key`: Pydantic model → auto-saves parsed result to `session.state[output_key]`
- `include_contents="none"`: Stateless agents — don't pass conversation history to LLM
- `instruction` as callable: `instruction=lambda ctx: f"User prefs: {ctx.session.state['prefs']}"`

---

## Workflow Agents

### SequentialAgent
Runs sub-agents one after another. Pass data between steps via `output_key` → `session.state`.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from pydantic import BaseModel

class Step1Out(BaseModel):
    cleaned_data: str

class Step2Out(BaseModel):
    analysis: str

step1 = LlmAgent(
    model="gemini-2.5-flash",
    name="cleaner",
    instruction="Clean and normalize the input data.",
    output_schema=Step1Out,
    output_key="cleaned",
)

step2 = LlmAgent(
    model="gemini-2.5-flash",
    name="analyzer",
    # access step1's output via state templating:
    instruction="Analyze this data: {cleaned}",
    output_schema=Step2Out,
    output_key="analysis",
)

root_agent = SequentialAgent(
    name="pipeline",
    sub_agents=[step1, step2],
)
```

### LoopAgent
Loops until a sub-agent sets `escalate=True`. Use for generate → validate → retry.

```python
from google.adk.agents import LoopAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext

def check_quality(output: str) -> bool:
    return len(output) > 100  # your validation logic

# Validator agent uses a tool that escalates when done
from google.adk.tools import ToolContext

def validate_and_escalate(content: str, tool_context: ToolContext) -> dict:
    """Validate content quality and signal loop completion."""
    if check_quality(content):
        tool_context.actions.escalate = True
        return {"status": "approved", "content": content}
    return {"status": "retry", "feedback": "Content too short"}

generator = LlmAgent(
    model="gemini-2.5-flash",
    name="generator",
    instruction="Generate content based on: {topic}. Previous feedback: {feedback}",
    output_key="draft",
)

validator = LlmAgent(
    model="gemini-2.5-flash",
    name="validator",
    instruction="Validate this draft: {draft}",
    tools=[validate_and_escalate],
)

root_agent = LoopAgent(
    name="refine_loop",
    sub_agents=[generator, validator],
    max_iterations=5,
)
```

### ParallelAgent
Runs all sub-agents concurrently. Each should write to a unique `output_key`.

```python
from google.adk.agents import ParallelAgent

web_searcher = LlmAgent(..., output_key="web_results")
db_searcher = LlmAgent(..., output_key="db_results")
api_caller = LlmAgent(..., output_key="api_results")

fan_out = ParallelAgent(
    name="gather",
    sub_agents=[web_searcher, db_searcher, api_caller],
)

# Then a SequentialAgent to combine:
aggregator = LlmAgent(
    instruction="Synthesize: web={web_results}, db={db_results}, api={api_results}",
    output_key="final",
)

root_agent = SequentialAgent(
    name="research_pipeline",
    sub_agents=[fan_out, aggregator],
)
```

---

## Multi-Agent Systems

### Orchestrator Pattern (LLM-driven routing)
```python
# Each specialist has a clear description — orchestrator uses it for routing
billing_agent = LlmAgent(
    name="billing_agent",
    description="Handles billing inquiries, invoices, and payment issues.",
    model="gemini-2.5-flash",
    instruction="Answer billing questions. Query the billing DB as needed.",
    tools=[query_billing_db],
)

support_agent = LlmAgent(
    name="support_agent",
    description="Handles technical support, bugs, and feature requests.",
    model="gemini-2.5-flash",
    instruction="Diagnose and resolve technical issues.",
    tools=[search_kb, create_ticket],
)

root_agent = LlmAgent(
    name="router",
    model="gemini-2.5-flash",
    instruction="""You are the main customer interface.
    Route to billing_agent for financial questions.
    Route to support_agent for technical issues.
    Handle general queries yourself.""",
    sub_agents=[billing_agent, support_agent],
)
```

### Agent Transfer (explicit)
```python
# In a tool, explicitly transfer control:
from google.adk.tools import ToolContext

def escalate_to_human(reason: str, tool_context: ToolContext) -> dict:
    """Escalate to human agent."""
    tool_context.actions.transfer_to_agent = "human_agent"
    return {"status": "escalating", "reason": reason}
```

---

## Custom BaseAgent

For when you need arbitrary Python logic as an agent node.

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class DataValidatorAgent(BaseAgent):
    """Custom agent that validates data using pure Python logic."""

    required_fields: list[str]  # Pydantic fields work here

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        data = ctx.session.state.get("input_data", {})

        missing = [f for f in self.required_fields if f not in data]
        if missing:
            ctx.session.state["validation_error"] = f"Missing: {missing}"
            ctx.session.state["validation_passed"] = False
        else:
            ctx.session.state["validation_passed"] = True

        # Must yield at least one event
        from google.adk.events import Event
        yield Event(author=self.name, content=None)

validator = DataValidatorAgent(
    name="validator",
    required_fields=["user_id", "action", "timestamp"],
)
```

---

## AgentConfig (YAML-driven)

Define agents declaratively without Python. Good for config-driven deployments.

```yaml
# agent_config.yaml
agent:
  model: gemini-2.5-flash
  name: my_agent
  description: A config-driven agent
  instruction: You are a helpful assistant.
  tools:
    - name: search_docs
      description: Search documentation
```

```python
from google.adk.agents.config import AgentConfig

config = AgentConfig.from_yaml("agent_config.yaml")
root_agent = config.build_agent()
```

---

## Context Compression

Prevents context window overflow in long-running agents.

```python
from google.adk.memory.compaction import (
    LlmSummarizingCompactor,
    TokenCountCompactionTrigger,
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="long_chat",
    context_compaction_config={
        "trigger": TokenCountCompactionTrigger(token_threshold=8000),
        "compactor": LlmSummarizingCompactor(
            model="gemini-2.5-flash",
            summary_instruction="Summarize the conversation, preserving key decisions.",
        ),
    },
)
```

---

## Events

Events are the unit of communication in ADK's event loop.

```python
from google.adk.events import Event

# Reading events from runner:
async for event in runner.run_async(...):
    print(f"Author: {event.author}")          # which agent/tool produced this
    print(f"Type: {event.type}")               # message, tool_call, tool_result, etc.

    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.text:
                print(part.text)
            if hasattr(part, 'function_call'):
                print(f"Tool call: {part.function_call.name}")

    if event.is_final_response():              # last event in a turn
        final_text = event.content.parts[0].text
```

**Event authors:**
- `"user"` — human input
- `agent.name` — LLM response
- `"tool"` — tool result