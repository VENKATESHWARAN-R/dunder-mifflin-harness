# ref_adk2.md — ADK 2.0 Graph Workflows Reference

> **Alpha release.** Verified against `google-adk==2.0.0a2` (Python 3.11+)
> **⚠️ Critical:** Never share storage (sessions, memory, artifacts) between ADK 1.x and 2.0 projects.
> **⚠️ Module name:** `google.adk.workflow` (singular) — NOT `google.adk.workflows`.

## Core Concepts

| Concept | What it is |
|---------|-----------|
| `Workflow` | The root agent. Declared with `edges=[...]` tuple list — no `compile()`. |
| `FunctionNode` | Wraps a Python function. Constructor: `FunctionNode(func, *, name=None)`. |
| `AgentNode` | Wraps an `LlmAgent`. Constructor: `AgentNode(agent=..., name=...)`. |
| `START` | Sentinel constant marking the graph entry point in edge tuples. |
| Static edge | `(A, B, C)` tuple — always runs A → B → C in order. |
| Conditional edge | `(A, {"key1": B, "key2": C})` tuple — A emits `Event(route="key1")` to branch. |
| `DEFAULT_ROUTE` | Fallback route key when no specific route matches. |

**State model:** Session state (`session.state`) is the shared scratchpad.
- LlmAgent writes to state via `output_key` (auto).
- FunctionNode **reads** params by name from `session.state` (auto-injected by name).
- FunctionNode **writes** to state by yielding `Event(state={"key": value})`.
- Routing is set by yielding `Event(route="key")` — NOT by returning a string.
- Both can be combined: `yield Event(state={...}, route="write")`.

**NodeLike:** Edges accept bare callables, `LlmAgent`, `FunctionNode`, `AgentNode`, or `"START"` — the `Workflow` auto-wraps them.

---

## Basic Graph (static edges)

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.workflow import START, Workflow

# Pure Python node — params injected from session.state by name.
# Write to state via Event(state={...}). Return None or omit yield to write nothing.
def preprocess(node_input: str):  # node_input = the user's message from START
    """Normalize and seed session state before the LLM runs."""
    text = (node_input or "").strip().lower()
    yield Event(state={"normalized_text": text, "skip_analysis": not text})

def postprocess(llm_output: str):  # injected from session.state["llm_output"]
    """Format LLM output and finalize."""
    import time
    yield Event(state={"final_result": llm_output.strip(), "processed_at": time.time()})

# LlmAgent — output_key writes result to session.state automatically
analyzer = LlmAgent(
    model="gemini-2.5-flash",
    name="analyzer",
    instruction="Analyze this normalized text: {normalized_text}",
    output_key="llm_output",   # → session.state["llm_output"]
)

# Assemble as Workflow — edges list drives the graph, no compile() needed
root_agent = Workflow(
    name="root_agent",
    rerun_on_resume=True,
    edges=[
        # Linear chain: START → preprocess → analyzer → postprocess
        (START, preprocess, analyzer, postprocess),
    ],
)
```

**Key FunctionNode rules:**
- Params named `node_input` receive the upstream node's output.
- All other params are pulled **by name** from `session.state` — so name them to match state keys.
- Yield `Event(state={...})` to write to session state (dict return does NOT write state).
- Bare callables in edges work — Workflow auto-wraps them in `FunctionNode`.

---

## Conditional Routing

No `ConditionalEdge` class exists. Routing is driven by `Event(route="key")` yielded
from a function node. The edge tuple `(src, {"key": dest})` maps route keys to targets.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.workflow import START, Workflow

# Step 1: seed state from user input
def parse_input(node_input: str):
    yield Event(state={"input_text": node_input.strip()})

# Step 2: classify and emit a route — Event(route=...) drives the branch
def triage(input_text: str):   # injected from session.state
    urgency = 8 if "urgent" in input_text.lower() else 3
    category = "billing" if "invoice" in input_text.lower() else "general"
    route = "escalate" if urgency > 7 else category
    yield Event(state={"urgency": urgency}, route=route)

escalation_agent = LlmAgent(name="escalate", model="gemini-2.5-flash",
                             instruction="Handle URGENT case: {input_text}", output_key="response")
billing_agent    = LlmAgent(name="billing",  model="gemini-2.5-flash",
                             instruction="Handle billing inquiry: {input_text}", output_key="response")
general_agent    = LlmAgent(name="general",  model="gemini-2.5-flash",
                             instruction="Answer general question: {input_text}", output_key="response")

root_agent = Workflow(
    name="root_agent",
    rerun_on_resume=True,
    edges=[
        # Entry chain
        (START, parse_input, triage),
        # Conditional branch — triage's Event(route=...) selects the target
        (triage, {"escalate": escalation_agent,
                  "billing":  billing_agent,
                  "general":  general_agent}),
    ],
)
```

**Rules:**
- The function yielding `Event(route="key")` must immediately precede the `(src, {"key": dest})` edge.
- Route keys must be `str`, `int`, or `bool` (`RouteValue`).
- Use `DEFAULT_ROUTE` from `google.adk.workflow` as a catch-all fallback key.
- You can combine state write + route in one yield: `Event(state={...}, route="key")`.

---

## Human-in-the-Loop (RequestInput)

`HumanInputNode` does **not** exist in `2.0.0a2`. Use a `FunctionNode` that yields
`RequestInput` — this pauses the graph turn, surfaces a prompt to the user, and
resumes on the next user message.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, Workflow

planner = LlmAgent(
    model="gemini-2.5-flash",
    name="planner",
    instruction="Generate a detailed action plan for: {task}",
    output_key="action_plan",
)

# Yield RequestInput to pause and ask the human
def ask_human_approval(action_plan: str):  # injected from session.state
    yield RequestInput(
        message=(
            f"Review this plan:\n\n{action_plan}\n\n"
            "Reply: approve / reject / modify <reason>"
        ),
        response_schema={"decision": str},
    )

# Parse the human's reply (arrives as node_input on resume)
def parse_decision(node_input: str):
    decision = (node_input or "").strip().lower()
    if decision.startswith("approve"):
        yield Event(route="execute")
    elif decision.startswith("modify"):
        reason = decision.replace("modify", "").strip()
        yield Event(state={"revision_request": reason}, route="revise")
    else:
        yield Event(route="abort")

executor = LlmAgent(name="executor", model="gemini-2.5-flash",
                    instruction="Execute this approved plan: {action_plan}",
                    output_key="result")
reviser  = LlmAgent(name="reviser",  model="gemini-2.5-flash",
                    instruction="Revise plan for: {revision_request}\nOriginal: {action_plan}",
                    output_key="action_plan")

def abort_fn():
    yield Event(state={"result": "Aborted by human reviewer."})

def seed(node_input: str):
    yield Event(state={"task": node_input})

root_agent = Workflow(
    name="root_agent",
    rerun_on_resume=True,
    edges=[
        (START, seed, planner, ask_human_approval, parse_decision),
        (parse_decision, {"execute": executor,
                          "revise":  reviser,
                          "abort":   abort_fn}),
        (reviser, ask_human_approval),  # loop back for re-review
    ],
)
```

### Running a Workflow programmatically
```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

async def run():
    svc = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=svc, app_name="app")
    session = await svc.create_session(app_name="app", user_id="u1")

    async def run_turn(message: str) -> tuple[str, str]:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=Content(parts=[Part(text=message)]),
        ):
            if event.is_final_response() and event.content:
                return "DONE", event.content.parts[0].text
        return "DONE", ""

    await run_turn("Deploy service X to production")

asyncio.run(run())
```

---

## Dynamic Workflows (code-controlled loops)

Back-edges create loops. A function node emits `Event(route=...)` to exit or continue.

```python
import json, re
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.workflow import START, Workflow

# Seed state on entry
def seed(node_input: str):
    yield Event(state={"topic": node_input, "iteration": 0,
                       "feedback": "", "quality_score": 0.0})

# Parse LLM score output + decide route (combine into one node)
def parse_and_route(score_output: str, iteration: int):
    try:
        data = json.loads(re.search(r'\{.*\}', score_output, re.DOTALL).group())
        score = float(data.get("score", 5))
        feedback = data.get("feedback", "")
    except Exception:
        score, feedback = 5.0, ""
    route = "done" if (score >= 8.0 or iteration >= 5) else "continue"
    yield Event(state={"quality_score": score, "feedback": feedback}, route=route)

def increment(iteration: int):
    yield Event(state={"iteration": iteration + 1})

def done(content: str):
    yield Event(state={"final_content": content})

generator = LlmAgent(
    model="gemini-2.5-flash", name="generator",
    instruction="Generate content for: {topic}\nIteration: {iteration}\nFeedback: {feedback}",
    output_key="content",
)
scorer = LlmAgent(
    model="gemini-2.5-flash", name="scorer",
    instruction='Score 1-10. Content: {content}\nJSON only: {"score": N, "feedback": "..."}',
    output_key="score_output",
)

root_agent = Workflow(
    name="root_agent",
    rerun_on_resume=True,
    edges=[
        (START, seed, generator, scorer, parse_and_route),
        # Conditional exit or continue
        (parse_and_route, {"done": done, "continue": increment}),
        # Back-edge: increment → generator (loop)
        (increment, generator),
    ],
)
```

---

## Collaborative / Orchestrator Pattern

`CoordinatorNode` does **not** exist in `2.0.0a2`. The collaborative pattern is achieved
with an `LlmAgent` orchestrator that has `sub_agents` — same as ADK 1.x. Wrap it in a
`Workflow` if you need explicit graph edges around it.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import START, Workflow

# Specialists — the orchestrator routes to them by description
researcher = LlmAgent(
    name="researcher", model="gemini-2.5-flash",
    description="Researches topics, finds facts, and retrieves source material.",
    output_key="research",
)
analyst = LlmAgent(
    name="analyst", model="gemini-2.5-flash",
    description="Analyzes data, identifies patterns, draws conclusions.",
    output_key="analysis",
)
writer = LlmAgent(
    name="writer", model="gemini-2.5-flash",
    description="Writes clear, structured prose from research and analysis.",
    output_key="draft",
)

# Orchestrator — LLM decides which specialist to call
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction="""You coordinate a research and writing team. For each task:
    1. Use researcher to gather facts.
    2. Use analyst if data needs processing.
    3. Use writer to produce the final draft.
    Return the completed draft.""",
    sub_agents=[researcher, analyst, writer],
)

# Wrap in Workflow if you need explicit graph control around the orchestrator
root_agent = Workflow(
    name="root_agent",
    rerun_on_resume=True,
    edges=[(START, orchestrator)],
)
# — or just set root_agent = orchestrator directly for a pure LLM-orchestrated run
```

---

## ADK 2.0 vs 1.x — When to Choose Which

| Need | ADK 1.x | ADK 2.0 (`Workflow`) |
|------|---------|---------------------|
| Simple A→B→C pipeline | `SequentialAgent` ✅ | `Workflow` with linear edge (overkill) |
| Retry until condition | `LoopAgent` ✅ | `Workflow` with back-edge loop |
| Independent parallel tasks | `ParallelAgent` ✅ | `Workflow` fan-out (experimental) |
| Conditional branching | Workaround via tools | `Event(route=...)` + dict edge ✅ |
| Human pause/approval | Custom tool + pause | `RequestInput` in FunctionNode ✅ |
| Mixed Python + LLM nodes | Custom `BaseAgent` | `FunctionNode` + `LlmAgent` in edges ✅ |
| Explicit routing visibility | Not built-in | Edge tuple graph ✅ |
| Production stability | ✅ | ⚠️ Alpha |
| Python version | 3.10+ | 3.11+ |
| Install | `pip install google-adk` | `pip install google-adk --pre` |

## Quick Import Reference (2.0.0a2 verified)

```python
# Core workflow
from google.adk.workflow import START, Workflow          # main entry points
from google.adk.workflow import FunctionNode             # wrap a Python function explicitly
from google.adk.workflow._agent_node import AgentNode   # wrap an LlmAgent explicitly
from google.adk.workflow import DEFAULT_ROUTE            # catch-all fallback route key

# State + routing from FunctionNode
from google.adk.events.event import Event                # Event(state={}, route="key")
from google.adk.events.request_input import RequestInput # pause for human input

# Agents
from google.adk.agents.llm_agent import LlmAgent
```

**Bottom line:** Use ADK 1.x for production. Use ADK 2.0 for greenfield projects where
you need conditional routing, back-edge loops, or mixed Python + LLM node graphs.
