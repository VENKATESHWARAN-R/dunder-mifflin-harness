# ref_adk2.md — ADK 2.0 Graph Workflows Reference

> **Alpha release.** Install: `pip install google-adk --pre` (Python 3.11+)
> **⚠️ Critical:** Never share storage (sessions, memory, artifacts) between ADK 1.x and 2.0 projects.

## Core Concepts

| Concept | What it is |
|---------|-----------|
| `WorkflowGraph` | Container for nodes + edges. Compiles to a runnable agent. |
| `FunctionNode` | Wraps a pure Python function. Input/output = state dict. |
| `AgentNode` | Wraps an `LlmAgent`. Reads/writes state via `output_key`. |
| `HumanInputNode` | Pauses graph execution, waits for human input. |
| `CoordinatorNode` | LLM that orchestrates sub-agents within the graph. |
| Static edge | Always go from node A to node B. |
| `ConditionalEdge` | Route to different nodes based on a function of state. |

**Data model:** All nodes share one state dict. Functions read from it and return an updated copy. Use `{**state, "new_key": value}` — never mutate in-place.

---

## Basic Graph (static edges)

```python
from google.adk.agents import LlmAgent
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode

# Pure Python node
def preprocess(state: dict) -> dict:
    """Normalize and validate input before sending to LLM."""
    text = state.get("input_text", "").strip().lower()
    if not text:
        return {**state, "error": "Empty input", "skip_analysis": True}
    return {**state, "normalized_text": text, "skip_analysis": False}

def postprocess(state: dict) -> dict:
    """Format LLM output for downstream consumption."""
    result = state.get("llm_output", "")
    return {**state, "final_result": result.strip(), "processed_at": time.time()}

# LLM node
analyzer = LlmAgent(
    model="gemini-2.5-flash",   # or "ollama/llama3.2" for local
    name="analyzer",
    instruction="Analyze this normalized text: {normalized_text}",
    output_key="llm_output",
)

# Assemble graph
graph = WorkflowGraph(name="analysis_pipeline")

pre_node = FunctionNode(name="preprocess", fn=preprocess)
analyze_node = AgentNode(name="analyze", agent=analyzer)
post_node = FunctionNode(name="postprocess", fn=postprocess)

for node in [pre_node, analyze_node, post_node]:
    graph.add_node(node)

graph.add_edge(pre_node, analyze_node)
graph.add_edge(analyze_node, post_node)

graph.set_entry_point(pre_node)
graph.set_finish_point(post_node)

root_agent = graph.compile()
```

---

## Conditional Routing (ConditionalEdge)

```python
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode, ConditionalEdge

def triage(state: dict) -> dict:
    """Classify and score the incoming request."""
    text = state.get("input_text", "")
    urgency = compute_urgency(text)       # your scoring logic
    category = classify_category(text)    # "billing" | "technical" | "general"
    return {**state, "urgency": urgency, "category": category}

def route_decision(state: dict) -> str:
    """Return the key of the next node to execute."""
    if state.get("urgency", 0) > 7:
        return "escalate"
    return state.get("category", "general")

escalation_agent = LlmAgent(name="escalate", model="gemini-2.5-flash",
                             instruction="Handle URGENT case: {input_text}", output_key="response")
billing_agent   = LlmAgent(name="billing",   model="gemini-2.5-flash",
                             instruction="Handle billing inquiry: {input_text}", output_key="response")
technical_agent = LlmAgent(name="technical", model="gemini-2.5-flash",
                             instruction="Diagnose technical issue: {input_text}", output_key="response")
general_agent   = LlmAgent(name="general",   model="gemini-2.5-flash",
                             instruction="Answer general question: {input_text}", output_key="response")

graph = WorkflowGraph(name="smart_router")

triage_node = FunctionNode(name="triage", fn=triage)
nodes = {
    "escalate": AgentNode(name="escalate", agent=escalation_agent),
    "billing":  AgentNode(name="billing",  agent=billing_agent),
    "technical":AgentNode(name="technical",agent=technical_agent),
    "general":  AgentNode(name="general",  agent=general_agent),
}

graph.add_node(triage_node)
for node in nodes.values():
    graph.add_node(node)

# Conditional edge: route_decision(state) returns a key → maps to a node
graph.add_conditional_edge(triage_node, route_decision, nodes)

graph.set_entry_point(triage_node)
for node in nodes.values():
    graph.set_finish_point(node)   # all branches are terminal

root_agent = graph.compile()
```

---

## Human Input Node (pause for human approval/input)

```python
from google.adk.workflows import WorkflowGraph, AgentNode, FunctionNode, HumanInputNode

# Step 1: Generate a plan
planner = LlmAgent(
    model="gemini-2.5-flash",
    name="planner",
    instruction="Generate a detailed action plan for: {task}",
    output_key="action_plan",
)

# Step 2: Pause for human review
approval = HumanInputNode(
    name="human_approval",
    prompt="Review this action plan:\n\n{action_plan}\n\nRespond: approve / reject / modify <reason>",
    output_key="human_decision",
)

# Step 3: Route based on decision
def parse_decision(state: dict) -> str:
    decision = state.get("human_decision", "").lower().strip()
    if decision.startswith("approve"):
        return "execute"
    elif decision.startswith("modify"):
        state["revision_request"] = decision.replace("modify", "").strip()
        return "revise"
    return "abort"

# Execution nodes
executor = LlmAgent(name="executor", model="gemini-2.5-flash",
                    instruction="Execute this approved plan: {action_plan}", output_key="result")
reviser  = LlmAgent(name="reviser",  model="gemini-2.5-flash",
                    instruction="Revise the plan per feedback: {revision_request}\nOriginal: {action_plan}",
                    output_key="action_plan")

def abort_fn(state: dict) -> dict:
    return {**state, "result": "Aborted by human reviewer."}

graph = WorkflowGraph(name="human_approval_flow")

plan_node    = AgentNode(name="plan",    agent=planner)
approve_node = approval
exec_node    = AgentNode(name="execute", agent=executor)
revise_node  = AgentNode(name="revise",  agent=reviser)
abort_node   = FunctionNode(name="abort", fn=abort_fn)

for node in [plan_node, approve_node, exec_node, revise_node, abort_node]:
    graph.add_node(node)

graph.add_edge(plan_node, approve_node)
graph.add_conditional_edge(approve_node, parse_decision, {
    "execute": exec_node,
    "revise":  revise_node,
    "abort":   abort_node,
})
graph.add_edge(revise_node, approve_node)   # loop back after revision

graph.set_entry_point(plan_node)
graph.set_finish_point(exec_node)
graph.set_finish_point(abort_node)

root_agent = graph.compile()
```

### Running graphs with human input
```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

async def run_with_human_in_loop():
    svc = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=svc, app_name="app")
    session = await svc.create_session(
        app_name="app", user_id="u1",
        state={"task": "Deploy service X to production cluster"},
    )

    async def run_turn(message: str):
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=Content(parts=[Part(text=message)]),
        ):
            if event.event_type == "human_input_request":
                # Graph is paused — get input from actual human
                prompt_text = event.content.parts[0].text
                return "PAUSED", prompt_text
            if event.is_final_response():
                return "DONE", event.content.parts[0].text
        return "DONE", ""

    # Start the workflow
    status, output = await run_turn("start")

    while status == "PAUSED":
        print(f"\n[Human input needed]:\n{output}\n")
        human_input = input("> ")             # in a real app: webhook, UI, Slack, etc.
        status, output = await run_turn(human_input)

    print(f"\n[Final result]: {output}")

asyncio.run(run_with_human_in_loop())
```

---

## Dynamic Workflows (code-controlled loops)

```python
from google.adk.workflows import WorkflowGraph, FunctionNode, AgentNode

def check_quality(state: dict) -> str:
    """Dynamic routing: continue loop or exit based on quality score."""
    score = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)
    if score >= 8.0 or iteration >= 5:
        return "done"
    return "continue"

def increment(state: dict) -> dict:
    return {**state, "iteration": state.get("iteration", 0) + 1}

def parse_score(state: dict) -> dict:
    """Extract numeric score from LLM's JSON output."""
    import json, re
    raw = state.get("score_output", '{"score": 5}')
    try:
        data = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        return {**state, "quality_score": float(data.get("score", 5))}
    except Exception:
        return {**state, "quality_score": 5.0}

generator = LlmAgent(
    model="gemini-2.5-flash",
    name="generator",
    instruction="""Generate content for topic: {topic}
    Iteration: {iteration}
    Previous feedback: {feedback}""",
    output_key="content",
)

scorer = LlmAgent(
    model="gemini-2.5-flash",
    name="scorer",
    instruction="""Score this content 1-10 and give feedback.
    Content: {content}
    Respond ONLY with valid JSON: {"score": <number>, "feedback": "<text>"}""",
    output_key="score_output",
)

graph = WorkflowGraph(name="iterative_refinement")

gen_node   = AgentNode(  name="generate",  agent=generator)
score_node = AgentNode(  name="score",     agent=scorer)
parse_node = FunctionNode(name="parse",    fn=parse_score)
inc_node   = FunctionNode(name="increment",fn=increment)
done_node  = FunctionNode(name="done",     fn=lambda s: {**s, "final_content": s.get("content")})

for n in [gen_node, score_node, parse_node, inc_node, done_node]:
    graph.add_node(n)

graph.add_edge(gen_node, score_node)
graph.add_edge(score_node, parse_node)
graph.add_conditional_edge(parse_node, check_quality, {
    "done":     done_node,
    "continue": inc_node,
})
graph.add_edge(inc_node, gen_node)   # loop back

graph.set_entry_point(gen_node)
graph.set_finish_point(done_node)

root_agent = graph.compile()
```

---

## Collaborative Agents (Coordinator Pattern)

```python
from google.adk.agents import LlmAgent
from google.adk.workflows import WorkflowGraph, CoordinatorNode

# Specialist agents — narrow, focused
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    description="Researches topics, finds facts, and retrieves source material.",
    tools=[web_search, fetch_url],
    output_key="research",
)

analyst = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    description="Analyzes data, identifies patterns, and draws conclusions.",
    tools=[run_sql, compute_stats],
    output_key="analysis",
)

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    description="Writes clear, structured prose from research and analysis notes.",
    output_key="draft",
)

reviewer = LlmAgent(
    name="reviewer",
    model="gemini-2.5-flash",
    description="Reviews drafts for accuracy, clarity, and completeness.",
    output_key="review",
)

# Coordinator orchestrates the specialists
coordinator = CoordinatorNode(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="""You coordinate a research and writing team.
    For each task:
    1. Use researcher to gather facts
    2. Use analyst if data needs processing
    3. Use writer to create the draft
    4. Use reviewer to check quality
    Return the final reviewed content.""",
    sub_agents=[researcher, analyst, writer, reviewer],
)

graph = WorkflowGraph(name="content_team")
graph.add_node(coordinator)
graph.set_entry_point(coordinator)
graph.set_finish_point(coordinator)

root_agent = graph.compile()
```

---

## ADK 2.0 vs 1.x — When to Choose Which

| Need | ADK 1.x | ADK 2.0 |
|------|---------|---------|
| Simple A→B→C pipeline | `SequentialAgent` ✅ | `WorkflowGraph` (overkill) |
| Retry until quality met | `LoopAgent` ✅ | Dynamic graph with back-edge |
| Independent parallel tasks | `ParallelAgent` ✅ | `WorkflowGraph` fan-out |
| Conditional branching | Workaround via tools | `ConditionalEdge` ✅ |
| Human approval checkpoint | Custom tool + pause | `HumanInputNode` ✅ |
| Mixed Python + LLM nodes | Custom `BaseAgent` | `FunctionNode` + `AgentNode` ✅ |
| Explicit routing visibility | Not built-in | Graph structure ✅ |
| Production stability | ✅ | ⚠️ Alpha |
| Python version | 3.10+ | 3.11+ |
| Install | `pip install google-adk` | `pip install google-adk --pre` |

**Bottom line:** Use ADK 1.x for production. Use ADK 2.0 for greenfield projects where
you need complex conditional routing, human-in-the-loop, or mixed node type graphs.
