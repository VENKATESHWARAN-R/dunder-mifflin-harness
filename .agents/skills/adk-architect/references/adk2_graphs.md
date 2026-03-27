# ADK 2.0 Graph Workflows Reference

> **Install**: `pip install google-adk --pre` (requires Python 3.11+)
> **Status**: Alpha — do NOT share storage with ADK 1.x projects

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Basic Graph Agent](#basic-graph-agent)
3. [Graph Routes (Conditional Branching)](#graph-routes)
4. [Human Input Nodes](#human-input-nodes)
5. [Data Handling](#data-handling)
6. [Dynamic Workflows](#dynamic-workflows)
7. [Collaborative Agents](#collaborative-agents)
8. [ADK 2.0 vs 1.x Comparison](#adk-20-vs-1x-comparison)

---

## Core Concepts

ADK 2.0 introduces graph-based workflows where you define **nodes** and **edges**:

- **Node**: A unit of work — LlmAgent, function, tool call, or human input
- **Edge**: A transition between nodes (static or conditional)
- **WorkflowGraph**: The container that wires nodes and edges together
- **Coordinator**: An LlmAgent that orchestrates subagents in the graph

Advantages over ADK 1.x workflow agents:
- Explicit routing logic (not just sequential/loop/parallel)
- Human-in-the-loop checkpoints as first-class nodes
- Mixed node types (LLM + pure Python functions + human input)
- Better observability of which branch executed

---

## Basic Graph Agent

```python
from google.adk.agents import LlmAgent
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode

# Define pure-function nodes
def classify_input(data: dict) -> dict:
    """Classify the incoming request type."""
    text = data.get("input", "")
    return {**data, "category": "billing" if "invoice" in text else "general"}

def format_output(data: dict) -> dict:
    """Final formatting step."""
    return {"response": data.get("llm_response", ""), "category": data["category"]}

# Define LLM agent nodes
responder = LlmAgent(
    model="gemini-2.5-flash",
    name="responder",
    instruction="Answer the user query based on the category: {category}",
    output_key="llm_response",
)

# Build the graph
graph = WorkflowGraph(name="my_workflow")

classifier_node = FunctionNode(name="classifier", fn=classify_input)
responder_node = AgentNode(name="responder", agent=responder)
formatter_node = FunctionNode(name="formatter", fn=format_output)

graph.add_node(classifier_node)
graph.add_node(responder_node)
graph.add_node(formatter_node)

# Static edges (always execute in this order)
graph.add_edge(classifier_node, responder_node)
graph.add_edge(responder_node, formatter_node)

graph.set_entry_point(classifier_node)
graph.set_finish_point(formatter_node)

root_agent = graph.compile()
```

---

## Graph Routes

Conditional branching based on state or node output.

```python
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode, ConditionalEdge

def triage(data: dict) -> dict:
    urgency = data.get("urgency_score", 0)
    return {**data, "route": "urgent" if urgency > 7 else "normal"}

def urgent_handler(data: dict) -> dict:
    return {**data, "response": "URGENT: Escalating immediately"}

urgent_agent = LlmAgent(name="urgent", model="gemini-2.5-flash",
                        instruction="Handle urgent case: {query}", output_key="response")
normal_agent = LlmAgent(name="normal", model="gemini-2.5-flash",
                        instruction="Handle normal case: {query}", output_key="response")

graph = WorkflowGraph(name="triage_workflow")

triage_node = FunctionNode(name="triage", fn=triage)
urgent_node = AgentNode(name="urgent", agent=urgent_agent)
normal_node = AgentNode(name="normal", agent=normal_agent)

graph.add_node(triage_node)
graph.add_node(urgent_node)
graph.add_node(normal_node)

# Conditional routing based on state["route"]
graph.add_conditional_edge(
    triage_node,
    condition=lambda state: state.get("route", "normal"),
    routes={
        "urgent": urgent_node,
        "normal": normal_node,
    }
)

graph.set_entry_point(triage_node)
# Both branches are finish points
graph.set_finish_point(urgent_node)
graph.set_finish_point(normal_node)

root_agent = graph.compile()
```

---

## Human Input Nodes

Pause graph execution to collect human input before proceeding.

```python
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode, HumanInputNode

# Step 1: agent generates a proposal
proposal_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="proposal_agent",
    instruction="Generate an action plan for: {task}",
    output_key="proposal",
)

# Step 2: human reviews the proposal
approval_node = HumanInputNode(
    name="approval",
    prompt="Review the proposal:\n{proposal}\n\nApprove? (yes/no/modify):",
    output_key="human_decision",
)

# Step 3: execute or revise based on human input
def route_on_decision(state: dict) -> str:
    decision = state.get("human_decision", "").lower()
    if "yes" in decision:
        return "execute"
    elif "no" in decision:
        return "abort"
    return "revise"

executor_agent = LlmAgent(name="executor", model="gemini-2.5-flash",
                          instruction="Execute: {proposal}", output_key="result")
reviser_agent = LlmAgent(name="reviser", model="gemini-2.5-flash",
                         instruction="Revise proposal based on: {human_decision}", output_key="proposal")

graph = WorkflowGraph(name="approval_workflow")

p_node = AgentNode(name="propose", agent=proposal_agent)
h_node = approval_node
e_node = AgentNode(name="execute", agent=executor_agent)
r_node = AgentNode(name="revise", agent=reviser_agent)
abort_node = FunctionNode(name="abort", fn=lambda d: {**d, "result": "Aborted by user"})

for node in [p_node, h_node, e_node, r_node, abort_node]:
    graph.add_node(node)

graph.add_edge(p_node, h_node)
graph.add_conditional_edge(h_node, route_on_decision, {
    "execute": e_node,
    "revise": r_node,
    "abort": abort_node,
})
graph.add_edge(r_node, h_node)  # Loop back to human after revision

graph.set_entry_point(p_node)
graph.set_finish_point(e_node)
graph.set_finish_point(abort_node)

root_agent = graph.compile()
```

### Running a graph with human input (async generator)
```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

async def run_with_human_input():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="app")
    session = await session_service.create_session(app_name="app", user_id="u1",
                                                    state={"task": "Deploy to production"})
    turn = 1
    async for event in runner.run_async(user_id="u1", session_id=session.id,
                                        new_message=Content(parts=[Part(text="start")])):
        if event.type == "human_input_request":
            # Graph is paused — get input from actual human
            human_response = input(f"[Human input requested]: {event.content.parts[0].text}\n> ")
            # Resume the graph by sending human input as next message
            async for resume_event in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=Content(parts=[Part(text=human_response)])
            ):
                if resume_event.is_final_response():
                    print(resume_event.content.parts[0].text)
        elif event.is_final_response():
            print(event.content.parts[0].text)
```

---

## Data Handling

Data flows through the graph via the shared **state dict**. Each node reads from and writes to state.

```python
from google.adk.workflows import WorkflowGraph, FunctionNode

# Nodes receive `state: dict` and return updated state dict
def extract_entities(state: dict) -> dict:
    text = state["input_text"]
    # ... extraction logic
    return {**state, "entities": ["entity1", "entity2"]}

def enrich_entities(state: dict) -> dict:
    entities = state["entities"]
    enriched = [fetch_entity_data(e) for e in entities]
    return {**state, "enriched_entities": enriched}

def aggregate(state: dict) -> dict:
    # Combine everything for final output
    return {
        **state,
        "final_output": {
            "entities": state["enriched_entities"],
            "source": state["input_text"][:100],
        }
    }

# State schema tip: use TypedDict or Pydantic for clarity
from typing import TypedDict

class PipelineState(TypedDict):
    input_text: str
    entities: list[str]
    enriched_entities: list[dict]
    final_output: dict
```

**Key rules:**
- Always return a new dict with `{**state, "new_key": value}` — don't mutate in-place
- LlmAgent nodes auto-write to `state[output_key]` when `output_key` is set
- Use `state.get("key", default)` defensively in early nodes

---

## Dynamic Workflows

Use code-based logic for loops and complex branching beyond what static edges support.

```python
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode

def should_continue(state: dict) -> str:
    """Dynamic routing: loop or exit based on iteration count and quality."""
    iteration = state.get("iteration", 0)
    quality_score = state.get("quality_score", 0)

    if quality_score >= 8 or iteration >= 5:
        return "done"
    return "continue"

def increment_iteration(state: dict) -> dict:
    return {**state, "iteration": state.get("iteration", 0) + 1}

generator = LlmAgent(name="gen", model="gemini-2.5-flash",
                     instruction="Iteration {iteration}. Generate content for: {topic}",
                     output_key="content")

scorer = LlmAgent(name="scorer", model="gemini-2.5-flash",
                  instruction="Score this content 1-10: {content}. Return JSON: {\"score\": N}",
                  output_key="quality_score")

graph = WorkflowGraph(name="dynamic_loop")
gen_node = AgentNode(name="gen", agent=generator)
score_node = AgentNode(name="score", agent=scorer)
inc_node = FunctionNode(name="increment", fn=increment_iteration)
done_node = FunctionNode(name="done", fn=lambda s: {**s, "final": s["content"]})

for n in [gen_node, score_node, inc_node, done_node]:
    graph.add_node(n)

graph.add_edge(gen_node, score_node)
graph.add_conditional_edge(score_node, should_continue, {
    "continue": inc_node,
    "done": done_node,
})
graph.add_edge(inc_node, gen_node)  # loop back

graph.set_entry_point(gen_node)
graph.set_finish_point(done_node)

root_agent = graph.compile()
```

---

## Collaborative Agents

Multiple specialized agents working under a coordinator.

```python
from google.adk.agents import LlmAgent
from google.adk.workflows import WorkflowGraph, AgentNode, CoordinatorNode

# Specialists
researcher = LlmAgent(name="researcher", model="gemini-2.5-flash",
                      description="Researches topics and finds facts.",
                      output_key="research_output")

writer = LlmAgent(name="writer", model="gemini-2.5-flash",
                  description="Writes clear prose from research notes.",
                  output_key="draft")

reviewer = LlmAgent(name="reviewer", model="gemini-2.5-flash",
                    description="Reviews drafts for accuracy and clarity.",
                    output_key="review_feedback")

# Coordinator orchestrates the specialists
coordinator = CoordinatorNode(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="""Coordinate the research and writing pipeline.
    Use researcher first, then writer, then reviewer.""",
    sub_agents=[researcher, writer, reviewer],
)

graph = WorkflowGraph(name="content_pipeline")
graph.add_node(coordinator)
graph.set_entry_point(coordinator)
graph.set_finish_point(coordinator)

root_agent = graph.compile()
```

---

## ADK 2.0 vs 1.x Comparison

| Feature | ADK 1.x | ADK 2.0 |
|---------|---------|---------|
| Sequential flow | `SequentialAgent` | `WorkflowGraph` + static edges |
| Loop | `LoopAgent` | `WorkflowGraph` + conditional back-edge |
| Parallel | `ParallelAgent` | `WorkflowGraph` + fan-out edges |
| Conditional routing | Not built-in | `ConditionalEdge` + route function |
| Human in the loop | Manual via tools | `HumanInputNode` (first-class) |
| Mixed node types | No | Yes (Function + Agent + Human) |
| Storage isolation | Shared OK | **Must isolate** from 1.x storage |
| Python version | 3.10+ | 3.11+ |
| Install | `pip install google-adk` | `pip install google-adk --pre` |

**When to stick with 1.x:**
- Production systems (2.0 is alpha)
- Simple sequential/loop/parallel patterns
- Need backwards compatibility

**When to use 2.0:**
- Complex branching that's hard to express with LoopAgent
- Human-in-the-loop checkpoints
- Mixed Python + LLM node graphs
- Need explicit, observable routing logic