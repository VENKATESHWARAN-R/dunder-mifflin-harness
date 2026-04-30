# Event Contract

## Purpose

This document is the authoritative contract for all typed communication between the runtime
(agent layer) and any UI surface — terminal CLI, browser UI, or A2A server.

The runtime emits **events** (one-way notifications) and **requests** (two-way handshakes).
UI surfaces emit **commands** (inputs that mutate session or runtime state).

No UI-specific types belong in the runtime. No runtime logic belongs in UI surfaces.
The event bus is the only crossing point.

---

## Communication Directions

```
Runtime (agents, nodes, workflows, tools)
    │
    ├── Events ──────────────────────────────► UI surfaces (CLI / Browser / A2A)
    │    one-way, fire-and-forget
    │
    ├── Requests ────────────────────────────► UI surfaces
    │    runtime blocks, waiting for response ◄── UI returns structured response
    │
    └──────────────────────────────────────── Commands ◄── UI surfaces
         settings changes, user messages,
         slash command effects
```

---

## Events (Runtime → UI)

All events are frozen dataclasses inheriting `RuntimeEvent`. They are emitted via
`event_bus.emit(event)` and consumed by registered listeners in the UI layer.

### Context Fields

Events that belong to a specific run or task carry `run_id` and optionally `task_id`.
Events that are session-scoped (settings changes) carry neither.

---

### Run Lifecycle

```python
@dataclass(frozen=True, slots=True)
class RunStarted(RuntimeEvent):
    run_id: str
    prompt: str
    workflow_mode: str        # 'feature_by_feature'

@dataclass(frozen=True, slots=True)
class RunCompleted(RuntimeEvent):
    run_id: str
    status: str               # 'done' | 'failed' | 'cancelled'
    total_cost: float
    total_tokens: int
    duration_ms: int

@dataclass(frozen=True, slots=True)
class RunFailed(RuntimeEvent):
    run_id: str
    message: str
    exception: Exception | None = None
```

> **Current state:** `RunStarted` and `RunCompleted` exist but lack cost/token/workflow fields.
> These fields should be added when the state store is wired (W5).

---

### Task Lifecycle

```python
@dataclass(frozen=True, slots=True)
class TaskStarted(RuntimeEvent):
    run_id: str
    task_id: str
    title: str
    tier: str                 # scout | worker | architect
    model: str                # exact model id

@dataclass(frozen=True, slots=True)
class TaskCompleted(RuntimeEvent):
    run_id: str
    task_id: str
    eval_score: float         # 0.0–1.0

@dataclass(frozen=True, slots=True)
class TaskFailed(RuntimeEvent):
    run_id: str
    task_id: str
    attempt_count: int
    reason: str

@dataclass(frozen=True, slots=True)
class TaskEscalated(RuntimeEvent):
    run_id: str
    task_id: str
    from_tier: str
    to_tier: str
    reason: str               # 'max_attempts_reached' | 'eval_score_below_threshold'
```

> **Current state:** Not yet defined. Add to `runtime/events.py` when task routing is built (W9).

---

### Node Progress

```python
@dataclass(frozen=True, slots=True)
class NodeStarted(RuntimeEvent):     # already exists
    run_id: str                      # ADD: currently missing
    task_id: str                     # ADD: currently missing
    node_name: str

@dataclass(frozen=True, slots=True)
class NodeCompleted(RuntimeEvent):   # already exists
    run_id: str                      # ADD: currently missing
    task_id: str                     # ADD: currently missing
    node_name: str
    status: str                      # 'success' | 'failed'
    duration_ms: int                 # ADD: currently missing

@dataclass(frozen=True, slots=True)
class NodeFailed(RuntimeEvent):      # already exists
    run_id: str                      # ADD
    task_id: str                     # ADD
    node_name: str
    message: str
```

> **Current state:** `NodeStarted`, `NodeCompleted`, `NodeFailed` exist but lack `run_id`,
> `task_id`, and `duration_ms`. Add these fields when graph nodes are introduced (W8).

---

### Model / LLM Calls

Emitted around every LLM call — both full agent loops and direct model calls.

```python
@dataclass(frozen=True, slots=True)
class ModelCallStarted(RuntimeEvent):
    run_id: str
    task_id: str
    model: str                # e.g. 'anthropic:claude-sonnet-4-6'
    tier: str                 # scout | worker | architect
    call_type: str            # 'agent' | 'direct_llm'

@dataclass(frozen=True, slots=True)
class AgentTextDelta(RuntimeEvent):  # already exists
    run_id: str                      # ADD
    task_id: str                     # ADD
    text: str

@dataclass(frozen=True, slots=True)
class AgentMessageCompleted(RuntimeEvent):  # already exists
    run_id: str                             # ADD
    task_id: str                            # ADD
    message: str

@dataclass(frozen=True, slots=True)
class ModelCallCompleted(RuntimeEvent):
    run_id: str
    task_id: str
    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    cost: float
    duration_ms: int
```

> **Current state:** `AgentTextDelta` and `AgentMessageCompleted` exist without context fields.
> `ModelCallStarted` and `ModelCallCompleted` are new — add when model tier wiring is done (W6).

---

### Tool Calls

Covers both local tools and MCP tools. MCP tools include a `server_id` field.

```python
@dataclass(frozen=True, slots=True)
class ToolCallRequested(RuntimeEvent):  # already exists
    run_id: str                         # ADD
    task_id: str                        # ADD
    tool_name: str
    server_id: str | None = None        # ADD: set for MCP tools, None for local
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ToolCallCompleted(RuntimeEvent):  # already exists
    run_id: str                         # ADD
    task_id: str                        # ADD
    tool_name: str
    server_id: str | None = None        # ADD
    display_content: str = ""
    duration_ms: int = 0               # ADD
    is_error: bool = False
```

> MCP tools are distinguished from local tools by the presence of `server_id`.
> No separate `MCPToolCallRequested` event — the same events cover both.

---

### File Operations

```python
@dataclass(frozen=True, slots=True)
class FileEditPreviewed(RuntimeEvent):  # already exists
    run_id: str                         # ADD
    task_id: str                        # ADD
    path: Path
    diff: str

@dataclass(frozen=True, slots=True)
class FileEditApplied(RuntimeEvent):    # already exists
    run_id: str                         # ADD
    task_id: str                        # ADD
    path: Path
```

---

### Shell Commands

```python
@dataclass(frozen=True, slots=True)
class ShellCommandStarted(RuntimeEvent):  # already exists
    run_id: str                           # ADD
    task_id: str                          # ADD
    command: str
    cwd: Path
    timeout_seconds: float

@dataclass(frozen=True, slots=True)
class ShellCommandCompleted(RuntimeEvent):  # already exists
    run_id: str                             # ADD
    task_id: str                            # ADD
    command: str
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
```

---

### Evaluation

```python
@dataclass(frozen=True, slots=True)
class EvaluationStarted(RuntimeEvent):
    run_id: str
    task_id: str

@dataclass(frozen=True, slots=True)
class EvaluationCompleted(RuntimeEvent):
    run_id: str
    task_id: str
    score: float              # 0.0–1.0
    passed: bool
    feedback: str
```

> **Current state:** Not yet defined. Add when the evaluate node is built (W7).

---

### Cost

```python
@dataclass(frozen=True, slots=True)
class CostUpdated(RuntimeEvent):   # exists but only carries a summary string
    run_id: str
    task_id: str | None            # None = run-level rollup
    delta_cost: float
    total_cost: float
    model: str
    tokens_in: int
    tokens_out: int
```

> **Current state:** `CostUpdated` exists with only a `summary: str` field. Replace with
> structured fields when the state store is wired (W5).

---

### MCP Lifecycle

```python
@dataclass(frozen=True, slots=True)
class MCPServerConnected(RuntimeEvent):
    run_id: str
    server_id: str
    server_name: str

@dataclass(frozen=True, slots=True)
class MCPServerDisconnected(RuntimeEvent):
    run_id: str
    server_id: str
    reason: str | None = None

@dataclass(frozen=True, slots=True)
class MCPServerToggled(RuntimeEvent):
    run_id: str
    server_id: str
    enabled: bool              # True = re-enabled, False = disabled
```

> **Current state:** Not yet defined. Add when MCP wiring is introduced.

---

### Session Config

Emitted when slash commands change session-level settings. No `run_id` — session-scoped.

```python
@dataclass(frozen=True, slots=True)
class SessionConfigChanged(RuntimeEvent):  # replaces StateUpdated for config changes
    key: str                               # 'model' | 'tier' | 'approval_mode' | etc.
    old_value: Any
    new_value: Any
```

---

### Warnings

```python
@dataclass(frozen=True, slots=True)
class WarningRaised(RuntimeEvent):  # already exists
    run_id: str | None = None       # ADD: None for session-level warnings
    message: str
```

---

## Requests (Runtime → UI, with response)

Requests are two-way handshakes. The runtime blocks on `event_bus.request_approval()` or
`event_bus.request_question()` until the UI calls `resolve_approval()` or `answer_question()`.
Full type definitions are in `runtime/approvals.py` and `runtime/questions.py`.

### Approval Request

Used whenever an agent wants to take an action with side effects (file write, shell command,
network call, git commit). The UI renders the request and returns one of:

- `approve_once` — allow this specific action
- `deny` — block this action
- `allow_tool_for_session` — allow all future actions from this tool in this session
- `allow_exact_for_session` — allow this exact action string for the session

See `runtime/approvals.py` for full type definitions.

### Question Request

Used for HITL checkpoints and agent clarifications. Supports three forms:
- `free-text` — open-ended string answer
- `single-choice` — one option from a list
- `multi-choice` — multiple options from a list

Answers are returned as structured values, not raw strings, so browser UI and A2A can
implement the same contract without terminal-specific parsing.

See `runtime/questions.py` for full type definitions.

---

## Commands (UI → Runtime)

Commands flow from a UI surface into the runtime. For the CLI, most commands are direct
method calls on `SessionState` or `RunCoordinator`. For browser/A2A, these would be
serialized and dispatched.

| Command | Trigger | Runtime target |
|---|---|---|
| `run(prompt)` | User submits a message | `RunCoordinator.run(prompt)` |
| `set_model(model_id)` | `/model <id>` slash command | `SessionState.model_override` |
| `set_tier(tier)` | `/tier <scout\|worker\|architect>` | `SessionState.tier` |
| `set_approval_mode(mode)` | `/approval <mode>` | `SessionState.approval_policy.mode` |
| `set_params(key, value)` | `/params <key> <value>` | `SessionState.model_params` |
| `resolve_approval(response)` | User responds to approval prompt | `EventBus.resolve_approval()` |
| `answer_question(response)` | User answers a question | `EventBus.answer_question()` |
| `toggle_mcp(server_id, enabled)` | `/disable mcp:<name>` (v1+) | DB + `config_loader` rebuild |
| `toggle_skill(skill_id, enabled)` | `/disable skill:<name>` (v1+) | DB + `config_loader` rebuild |

---

## What Each UI Surface Must Implement

Any surface that connects to the runtime must:

1. Subscribe to the event types it wants to render via `event_bus.on(EventType, handler)`
2. Render or forward `ApprovalRequested` events and call `event_bus.resolve_approval()`
3. Render or forward `QuestionRequested` events and call `event_bus.answer_question()`
4. Translate user input into the appropriate runtime command

A surface that does NOT implement approval/question handling cannot be used in interactive
(HITL) mode. It is valid for Autopilot/benchmark-only surfaces to skip these.

---

## Implementation Status

| Area | Status | Notes |
|---|---|---|
| EventBus, emit, on, off | Done | `runtime/events.py` |
| Approval request/response | Done | `runtime/approvals.py` |
| Question request/response | Done | `runtime/questions.py` |
| Run lifecycle events | Partial | Missing cost/token fields on RunCompleted |
| Node events | Partial | Missing run_id, task_id, duration_ms |
| Text delta / message events | Partial | Missing run_id, task_id |
| Tool call events | Partial | Missing run_id, task_id, server_id, duration_ms |
| File / shell events | Partial | Missing run_id, task_id |
| CostUpdated | Partial | Summary string only; needs structured fields |
| Task lifecycle events | Not started | Add at W9 |
| Model call events | Not started | Add at W6 |
| Evaluation events | Not started | Add at W7 |
| MCP lifecycle events | Not started | Add when MCP wiring is introduced |
| SessionConfigChanged | Not started | Replaces generic StateUpdated |
