> **Status:** Reference · **Last revised:** 2026-05-07 · **Type:** developer documentation

# Runtime Layer

The runtime layer lives in `src/jac/runtime/`. It is the UI-agnostic core of JAC: it receives user messages, orchestrates agent construction, manages the event bus, and persists results. It has no dependency on `src/jac/cli/`.

See also: [`docs/contracts/EVENT_CONTRACT.md`](../contracts/EVENT_CONTRACT.md) — the typed event surface is a binding contract.

## RunCoordinator (`src/jac/runtime/coordinator.py`)

The orchestration unit. One coordinator per session.

**Constructor:**

```python
RunCoordinator(
    settings: Settings,
    state: StateStore,        # may be None for no-DB fallback
    session: SessionState,
    events: EventBus,
)
```

Run-scoped C6c helpers owned by the coordinator:

- `_tool_result_cache` — in-memory handles for summarized tool outputs.
- `_summariser` — Scout direct-call summarizer used by result-filter wrappers; when a state store is present, writes `direct_llm` attempt rows and merges `model_request` usage into the parent `ctx.usage` via `incr`.

**C7 usage capture:** Each `attempts` row stores a **delta** (not cumulative shared usage). Sub-agents snapshot `ctx.usage` before/after nested `agent.run(..., usage=ctx.usage)` and subtract direct child rows. Root rows subtract all direct children from the final `result.usage()`. See `src/jac/runtime/usage.py` for tuple helpers.

### `submit_message(UserMessage) -> str`

Main entry point. Called once per user turn. Steps:

1. Creates a run record in `state.runs` if this is the first message (or re-uses the existing `session.run_id`).
2. Calls `_ensure_agent()` to get a cached `pydantic_ai.Agent`.
3. Runs `async with agent: result = await agent.run(prompt, message_history=...)`.
4. Persists the user + assistant messages to `state.messages`.
5. Emits `AgentMessageCompleted` and `SessionUsageUpdated` (structured cumulative usage from root completions).
6. Returns the output string.

As of C6, manager attempt lifecycle tracking is factored into two internal helpers:
`_start_manager_attempt` and `_finish_manager_attempt`, which isolate attempt-row
bookkeeping from the rest of turn orchestration.

### `submit_slash_run(...) -> object`

Dedicated one-turn path for model-routed slash commands (`/plan`, `/init`).
It accepts a target `role`, runtime prompt, mode key, optional `output_type`,
and optional persisted user prompt text (for example `"/plan ..."` literal).

Flow:

1. Persists the slash prompt to `messages` as a user turn.
2. Builds a role-scoped agent with `config_loader(..., instructions_addendum=...)`.
3. Runs one model turn and persists the final assistant artifact.
4. Records a role-specific attempt row (`manager` or `planner`).
5. Replaces in-memory history with `filter_tool_noise(result.all_messages())` so
   intermediate tool call/return parts do not leak into subsequent turns.

### `_ensure_agent() -> Agent`

Async. Called internally on the first turn (or after `reset_agent()`). If `state`
is available: ensures manager + builder configs for the run, then calls
`config_loader(state, settings, run_id, role=session.config.role, approval_policy=...)`.
For manager role it injects `summon_jim` and native extras (`spawn_minion`,
`fetch_full_result`). Other native specialist roles also receive native extras.
If `state` is
`None`: calls `_build_fallback_agent()` which constructs a minimal agent directly
without DB-backed config. Caches the result.

### `reset_agent()`

Clears the cached agent. Called when the user changes the model string or tier via slash commands (`/model`, `/tier`) so the next `_ensure_agent()` call picks up the new settings.

### `resume_run(state, settings, run_id)` (module-level function)

Reconstructs a `RunCoordinator` from a prior run. Reads messages from `state.messages` for `run_id`, rebuilds `message_history` for the agent, creates a `SessionState` with the existing `run_id`. Returns a coordinator ready to continue the conversation.

---

## EventBus (`src/jac/runtime/events.py`)

Typed async pub/sub. The single channel through which the CLI observes runtime behaviour.

**API:**

```python
events = EventBus()

# Subscribe (sync or async handlers both work)
events.on(AgentTextDelta, lambda e: print(e.text, end="", flush=True))
events.on(ToolCallRequested, handle_tool_call)

# Publish
await events.emit(AgentTextDelta(text="hello"))
```

**All defined event types:**

| Event | Payload | Emitted by |
|---|---|---|
| `RunStarted` | `run_id`, `prompt` | Coordinator |
| `RunCompleted` | `run_id`, `output` | Coordinator |
| `RunFailed` | `run_id`, `error` | Coordinator |
| `AgentTextDelta` | `text` | Coordinator (streamed from model) |
| `AgentMessageCompleted` | `run_id`, `message` | Coordinator |
| `NodeStarted` | `node_name` | agents, coordinator |
| `NodeCompleted` | `node_name` | agents, coordinator |
| `ToolCallRequested` | `tool_name`, `args` | Coordinator |
| `ToolCallCompleted` | `tool_name`, `result` | Coordinator |
| `ApprovalRequested` | `request_id`, `description`, `preview` | Approvals module |
| `ApprovalResolved` | `request_id`, `approved` | Approvals module |
| `QuestionRequested` | `request_id`, `question` | Questions module |
| `QuestionAnswered` | `request_id`, `answer` | Questions module |
| `FileEditPreviewed` | `path`, `diff` | Filesystem tools |
| `FileEditApplied` | `path` | Filesystem tools |
| `ShellCommandStarted` | `command` | Shell tools |
| `ShellCommandCompleted` | `command`, `exit_code`, `output` | Shell tools |
| `SessionUsageUpdated` | cumulative tokens, last context vs max | Coordinator (root runs only) |
| `WarningRaised` | `message` | Any layer |
| `PlanGenerated` | `summary`, `dev_strategy`, `task_count` | CLI slash handler (`/plan`) |
| `WorkspaceSurveyCompleted` | `agents_md_path`, `line_count` | CLI slash handler (`/init`) |
| `MinionSpawned` | `parent_role`, `minion_role`, `task_summary`, `tools`, `tier`, `depth` | `spawn_minion` |
| `MinionReturned` | `parent_role`, `minion_role`, `duration_ms`, `success` | `spawn_minion` |

**Approval and question flow:**

Approvals and questions use dedicated methods on `EventBus`, not plain `emit`:

```python
# Request an approval (blocks until resolved)
response = await events.request_approval(ApprovalRequest(
    id="req-1",
    description="Run shell command",
    preview="ls -la /tmp",
))
# response.approved: bool

# Resolve from the UI side
await events.resolve_approval("req-1", approved=True)
```

Internally, `request_approval` emits `ApprovalRequested` then awaits an `asyncio.Future`. `resolve_approval` resolves the future, unblocking the coordinator. The question flow is symmetric.

---

## SessionState + SessionConfig (`src/jac/runtime/session.py`)

`SessionState` is the mutable per-session object shared between the CLI and the coordinator.

```python
@dataclass
class SessionState:
    run_id: str                        # UUID, set at creation
    config: SessionConfig
    attached_paths: list[Path]         # files attached via @path in current turn
    cumulative_tokens_in: int          # session-level counters (reset by /clear)
    cumulative_tokens_out: int
    cumulative_requests: int
    cumulative_tool_calls: int
    last_context_tokens: int           # last root LLM call input token count
    last_model: str | None
```

```python
@dataclass
class SessionConfig:
    cwd: Path
    mode: RunMode                      # hitl | autopilot
    model: str | None                  # explicit model override (None = use tier)
    tier: ModelTier | None             # scout | worker | architect (None = use default)
    role: str                          # default "manager"
    approval_mode: ApprovalMode        # interactive | auto-edit | yolo
    model_params: dict                 # {"temperature": 0.7, "max_tokens": 4096, ...}
    max_attachment_bytes: int
    shell_timeout_seconds: float
    shell_max_output_chars: int
    slash_mode: str | None             # transient slash-mode marker (init/plan)
```

`ModelTier` enum values: `scout`, `worker`, `architect`.

Slash commands mutate `session.config` directly. The coordinator reads `session.config` on each `submit_message` call.

---

## Approval flow (`src/jac/runtime/approvals.py`)

```python
@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    description: str
    preview: str | None = None

@dataclass(frozen=True)
class ApprovalResponse:
    approved: bool
```

`ApprovalPolicy(mode: ApprovalMode)` provides `auto_response_for(request) -> ApprovalResponse | None`:

- `interactive` → returns `None` (caller must prompt)
- `auto-edit` → returns `approved=True` for file edits, `None` for shell commands
- `yolo` → returns `approved=True` for everything

**Wiring (C5a, 2026-05-04):** `_resolve_local_tools` in `agents/base.py` wraps every entry from `TOOL_REGISTRY` with `agents/approval.py:make_approval_wrapper(fn, events, policy)`. The wrapper:

- Skips the gate for `RiskLevel.READ_ONLY` tools (read_file, list_directory, search_files, grep_files, list_processes, read_process_output).
- For `category == "file_write"` tools, computes the prospective diff before the gate (via `compute_edit` for `edit_file`, `preview_write` for `write_file`) and emits `FileEditPreviewed` so the renderer shows the diff *before* the prompt fires.
- Asks `policy.auto_response_for(request)` first; falls through to `await events.request_approval(...)` only when no auto-response applies.
- On denial, returns a typed `ToolResult(status=PERMISSION_DENIED)` matching the wrapped tool's return type — never raises — so pydantic_ai feeds a coherent error back to the model.
- On a successful file write, emits `FileEditApplied(path)`.

`config_loader` accepts an optional `approval_policy` kwarg and falls back to `ApprovalPolicy(mode=INTERACTIVE)` when none is supplied. `RunCoordinator` constructs its policy from `session.config.approval_mode` and threads the same instance through to `config_loader` — so `/approval yolo` flips behaviour for the next agent build without restarting the session. `ChatApp.from_resumed` re-uses the coordinator's policy rather than building a divergent one.

`edit_file` was split into `compute_edit` (returns a `PreparedEdit` or an error `FileEditResult`) and `apply_edit` so the wrapper can preview a diff without touching disk; the public `edit_file` async function chains the two for direct callers.

---

## Question flow (`src/jac/runtime/questions.py`)

Symmetric to approvals.

```python
@dataclass(frozen=True)
class QuestionRequest:
    id: str
    question: str
    choices: list[str] | None = None  # if set, user must pick from list

@dataclass(frozen=True)
class QuestionResponse:
    answer: str
```

Used when the agent or a tool needs a clarifying answer from the user (distinct from an approval — questions have a free-form or choice-based answer, not a yes/no).

---

## Model factory (`src/jac/runtime/models.py`)

`build_pydantic_model(selection: ModelSelection, settings: Settings) -> Model`

The **only** place provider-specific model objects are instantiated. Dispatches on `selection.provider`:

| Provider value | Model class |
|---|---|
| `"gateway"` | `GatewayModel` |
| `"anthropic"` | `AnthropicModel` |
| `"openai"` | `OpenAIModel` |
| `"google-gla"` | `GoogleModel` (GLA) |
| `"ollama"` | `OllamaModel` |
| `"openrouter"` | `OpenRouterModel` |
| `"litellm"` | `LiteLLMModel` |

Before construction, calls `settings.require_model_credentials(provider)` which raises `MissingCredentialsError` with a helpful message if the required env var is absent.

`ModelSelection` is produced by `settings.resolve_model_selection(model_override, tier)` — it checks `model_override` first, then falls back to the tier mapping in `model_tiers`.
