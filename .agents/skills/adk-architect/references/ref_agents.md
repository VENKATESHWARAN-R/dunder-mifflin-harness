# ref_agents.md — Agent Implementation Reference

## LlmAgent — Full Configuration

```python
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

class AnalysisOutput(BaseModel):
    summary: str
    confidence: float
    tags: list[str]

root_agent = LlmAgent(
    # Core identity
    model="gemini-2.5-flash",          # or "ollama/llama3.2", "claude-sonnet-4-5", etc.
    name="analysis_agent",
    description="Analyzes input data and returns structured insights.",  # used by orchestrators

    # Instruction — static string OR callable for dynamic context
    instruction="Analyze the following. Return structured output.",
    # Dynamic version:
    # instruction=lambda ctx: f"User {ctx.session.state.get('user:name', 'unknown')} prefers {ctx.session.state.get('user:lang', 'en')}. Analyze:",

    # Tools
    tools=[my_fn_tool, mcp_toolset],

    # Structured output: forces Pydantic model and saves to state
    output_schema=AnalysisOutput,
    output_key="analysis",             # session.state["analysis"] = parsed AnalysisOutput

    # Context control
    include_contents="default",        # "none" = don't pass history to LLM (stateless mode)

    # Sub-agents (orchestrator pattern)
    sub_agents=[specialist_a, specialist_b],

    # LLM generation settings
    generate_content_config=GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
        top_p=0.95,
    ),

    # Callbacks (agent-specific hooks)
    before_agent_callback=my_before_agent_cb,
    after_agent_callback=my_after_agent_cb,
    before_model_callback=my_before_model_cb,
    after_model_callback=my_after_model_cb,
    before_tool_callback=my_before_tool_cb,
    after_tool_callback=my_after_tool_cb,

    # Plugins (global hooks, defined here apply only to this agent)
    plugins=[my_plugin],
)
```

### State templating in instructions
```python
# ADK substitutes {key} values from session.state automatically
agent = LlmAgent(
    instruction="User preference: {user:lang}. Current task: {current_task}. Prior result: {analysis}.",
    ...
)
# Dot notation for nested dicts not supported — flatten into state keys
```

---

## SequentialAgent

Runs sub-agents in order. Data flows via `output_key` → `session.state` → `{key}` in
next agent's instruction.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from pydantic import BaseModel

class CleanOutput(BaseModel):
    cleaned_text: str
    word_count: int

class AnalysisOutput(BaseModel):
    topics: list[str]
    sentiment: str

cleaner = LlmAgent(
    model="gemini-2.5-flash",
    name="cleaner",
    instruction="Clean and normalize this text. Remove noise, fix formatting.",
    output_schema=CleanOutput,
    output_key="cleaned",
)

analyzer = LlmAgent(
    model="gemini-2.5-flash",
    name="analyzer",
    instruction="Analyze this cleaned text: {cleaned}",   # receives cleaner's output
    output_schema=AnalysisOutput,
    output_key="analysis",
)

summarizer = LlmAgent(
    model="gemini-2.5-flash",
    name="summarizer",
    instruction="Summarize findings. Topics: {analysis}. Word count: {cleaned}",
    output_key="final_summary",
)

root_agent = SequentialAgent(
    name="text_pipeline",
    sub_agents=[cleaner, analyzer, summarizer],
)
```

---

## LoopAgent

Loops until sub-agent sets `actions.escalate = True` or `max_iterations` is hit.

```python
from google.adk.agents import LoopAgent, LlmAgent
from google.adk.tools import ToolContext

def evaluate_and_escalate(draft: str, tool_context: ToolContext) -> dict:
    """Evaluate draft quality. Escalate if good enough, otherwise give feedback.

    Args:
        draft: The current draft to evaluate.

    Returns:
        Evaluation result with status and optional feedback.
    """
    score = score_quality(draft)   # your quality function
    tool_context.state["quality_score"] = score

    if score >= 8:
        tool_context.actions.escalate = True
        return {"status": "approved", "score": score}
    return {
        "status": "retry",
        "score": score,
        "feedback": f"Score {score}/10. Needs improvement in: {identify_issues(draft)}"
    }

generator = LlmAgent(
    model="gemini-2.5-flash",
    name="generator",
    instruction="""Generate content about: {topic}.
    Previous feedback (if any): {feedback}
    Previous score (if any): {quality_score}
    Iteration: {iteration}""",
    output_key="draft",
)

evaluator = LlmAgent(
    model="gemini-2.5-flash",
    name="evaluator",
    instruction="Evaluate this draft critically: {draft}",
    tools=[evaluate_and_escalate],
)

root_agent = LoopAgent(
    name="refine_loop",
    sub_agents=[generator, evaluator],
    max_iterations=5,
)
```

---

## ParallelAgent

All sub-agents run concurrently. Each MUST use a unique `output_key`.

```python
from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent

web_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="web_searcher",
    instruction="Search for recent news about: {query}",
    tools=[web_search_tool],
    output_key="web_results",
)

db_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="db_searcher",
    instruction="Query internal database for: {query}",
    tools=[query_db_tool],
    output_key="db_results",
)

kb_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="kb_searcher",
    instruction="Search knowledge base for: {query}",
    tools=[search_kb_tool],
    output_key="kb_results",
)

# Fan-out: all three run at once
gather = ParallelAgent(
    name="gather",
    sub_agents=[web_agent, db_agent, kb_agent],
)

# Aggregate results
synthesizer = LlmAgent(
    model="gemini-2.5-flash",
    name="synthesizer",
    instruction="""Synthesize research into a coherent answer.
    Web: {web_results}
    Database: {db_results}
    Knowledge base: {kb_results}""",
    output_key="final_answer",
)

root_agent = SequentialAgent(
    name="research_pipeline",
    sub_agents=[gather, synthesizer],
)
```

---

## Multi-Agent (Orchestrator + Specialists)

```python
# Each specialist needs a precise description — the orchestrator LLM reads it
billing_agent = LlmAgent(
    name="billing_agent",
    description="Handles billing inquiries, invoice questions, payment failures, and refund requests.",
    model="gemini-2.5-flash",
    instruction="You are a billing specialist. Access billing DB for account details.",
    tools=[query_billing_db, create_refund],
)

tech_agent = LlmAgent(
    name="tech_support",
    description="Diagnoses and resolves technical issues, bugs, outages, and integration errors.",
    model="gemini-2.5-flash",
    instruction="You are a tech support engineer. Create tickets and search runbooks.",
    tools=[search_runbooks, create_ticket, check_system_status],
)

escalation_agent = LlmAgent(
    name="escalation_agent",
    description="Handles critical incidents, VIP customers, and cases requiring human manager review.",
    model="gemini-2.5-flash",
    instruction="Escalate to human team. Collect all context before escalating.",
    tools=[notify_on_call, create_incident],
)

root_agent = LlmAgent(
    name="customer_router",
    model="gemini-2.5-flash",
    instruction="""You are the main customer interface. Identify the nature of the request
    and route to the appropriate specialist. Handle simple greetings yourself.
    For ambiguous cases, ask a clarifying question before routing.""",
    sub_agents=[billing_agent, tech_agent, escalation_agent],
)
```

### Explicit agent transfer from a tool
```python
from google.adk.tools import ToolContext

def escalate_to_human(reason: str, context: str, tool_context: ToolContext) -> dict:
    """Escalate this conversation to a human agent.

    Args:
        reason: Why this needs human handling.
        context: Summary of the conversation so far.
    """
    tool_context.actions.transfer_to_agent = "human_handoff_agent"
    tool_context.state["escalation_reason"] = reason
    return {"status": "transferring", "reason": reason}
```

---

## Custom BaseAgent

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from typing import AsyncGenerator

class DataValidatorAgent(BaseAgent):
    """Validates session state data against required fields using pure Python."""

    required_fields: list[str]      # Pydantic field — set via constructor
    fail_fast: bool = True

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        data = ctx.session.state.get("input_data", {})
        missing = [f for f in self.required_fields if f not in data]

        if missing:
            ctx.session.state["validation_passed"] = False
            ctx.session.state["validation_errors"] = missing
            message = f"Validation failed. Missing fields: {missing}"
            if self.fail_fast:
                ctx.session.state["pipeline_aborted"] = True
        else:
            ctx.session.state["validation_passed"] = True
            message = "Validation passed."

        yield Event(
            author=self.name,
            content=Content(parts=[Part(text=message)]),
        )

# Usage
validator = DataValidatorAgent(
    name="validator",
    required_fields=["user_id", "action", "timestamp"],
    fail_fast=True,
)

root_agent = SequentialAgent(
    name="pipeline",
    sub_agents=[validator, processing_agent],  # validator runs first
)
```

---

## AgentConfig (YAML-driven)

```yaml
# agent_config.yaml
agent:
  model: gemini-2.5-flash
  name: support_agent
  description: Handles customer support inquiries.
  instruction: |
    You are a helpful support agent.
    Use the search tool to find answers in the knowledge base.
  tools:
    - type: function
      name: search_kb
      description: Search the knowledge base
  generate_content_config:
    temperature: 0.3
    max_output_tokens: 1024
```

```python
from google.adk.agents.config import AgentConfig

config = AgentConfig.from_yaml("agent_config.yaml")
root_agent = config.build_agent(
    # inject runtime tools not expressible in YAML:
    extra_tools=[mcp_toolset],
)
```
