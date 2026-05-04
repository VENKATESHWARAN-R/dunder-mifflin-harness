# C5a — Tool Approval Middleware

> **Status:** Shipped (2026-05-04) · **Last revised:** 2026-05-04
>
> _Kept for historical reference. See `docs/dev/runtime-layer.md` "Approval flow" for the current behaviour and `docs/ROADMAP.md#c5a-—-tool-approval-middleware-2026-05-04` for the ship notes._
>
> **Deltas from this plan as written:**
> - The wrapper short-circuits with the compute_edit error when the file is missing or the match is ambiguous, instead of asking for approval first. There's no action to approve when the action is impossible.
> - `compute_edit` returns a `PreparedEdit` dataclass on success or a `FileEditResult` on error, rather than the `(old_content, new_content, diff)` tuple sketched here.
> - `FileEditApplied` keeps its existing `(path,)` shape (per `EVENT_CONTRACT.md`); the wrapper does not add a `diff` field.
> - `_safe_description` defends against `description_fn` raising — the gate prefers a generic summary over surfacing an internal error.

## Problem

C3 and C4 shipped tool implementations with `.approval` metadata and a fully wired
CLI/event-bus approval handshake — but the **link between the two was never built**.
When Pydantic AI calls a local tool, it invokes the raw function directly. No code reads
`.approval`, no `ApprovalRequest` is emitted, and no user prompt appears. Agents can
`write_file`, `edit_file`, and `run_shell` without any gate.

The acceptance criteria in the C3/C4 roadmap entries ("diff renders, approval prompt
fires") are not met. This component closes that gap.

## Goals

- Every local tool call from an agent goes through an approval gate before executing.
- `read_only` tools auto-approve in all modes; `low`/`medium`/`high` risk tools obey
  the session's `ApprovalMode` (`interactive | auto-edit | yolo`).
- File writes show the prospective diff **before** writing; the write only happens on
  approve. `FileEditPreviewed` is emitted pre-write, `FileEditApplied` post-write.
- Shell commands show the command string and cwd before executing.
- The rest of the codebase does not change. Only `agents/base.py:_resolve_local_tools`
  gains new logic; tools themselves stay stateless functions.

## Non-goals

- MCP tool approvals (those are negotiated at the MCP protocol level; out of scope here).
- Persisting approval decisions across runs (session-scoped allowances only, as
  `ApprovalPolicy` already implements).
- Changing the `ToolApprovalMeta` schema — it ships everything needed.

## Key files

| File | Role |
|---|---|
| `src/jac/agents/base.py` | `_resolve_local_tools` → returns approval-wrapped versions |
| `src/jac/tools/types.py` | `ToolApprovalMeta`, `RiskLevel` (no changes needed) |
| `src/jac/runtime/approvals.py` | `ApprovalRequest`, `ApprovalPolicy`, `ApprovalDecision` (no changes needed) |
| `src/jac/runtime/events.py` | `EventBus.request_approval`, `FileEditPreviewed`, `FileEditApplied` (no changes needed) |
| `src/jac/agents/approval.py` | **new** — `make_approval_wrapper(fn, events, policy)` factory |
| `tests/test_agent_approval.py` | **new** — integration tests |

## How the wrapper works

```
agent calls tool
       ↓
approval_wrapper(fn, events, policy)(**kwargs)
       ├─ read meta from fn.approval
       ├─ if risk_level == READ_ONLY → skip gate, call fn(**kwargs)
       ├─ else:
       │     build ApprovalRequest(summary=description_fn(**kwargs), ...)
       │     auto = policy.auto_response_for(request)
       │     if auto is None → response = await events.request_approval(request)
       │     else            → response = auto
       │     policy.record_response(request, response)
       │     if not response.approved → return denial result
       │     call fn(**kwargs) → result
       │     if tool is edit_file → emit FileEditApplied(diff=result.diff, path=...)
       └─ return result
```

For `write_file` and `edit_file` the wrapper also emits `FileEditPreviewed` **before**
requesting approval, giving the renderer a chance to show the diff. The diff for
`edit_file` must be computed before the write — so `edit_file` needs a small refactor:
split into `_compute_edit(...)` (returns `old_content, new_content, diff`) and
`_apply_edit(path, new_content)`. The wrapper calls compute, emits preview, gates, then
calls apply.

For `write_file` creating a new file, there is no "before" state; the preview just shows
the full content (truncated to a reasonable length if large).

## Implementation steps

1. **Refactor `edit_file` in `tools/filesystem.py`** — separate compute and apply phases
   so the wrapper can read the diff before writing. Keep the existing function signature
   intact (it delegates to the two helpers). Tests must still pass unchanged.

2. **Create `src/jac/agents/approval.py`** with `make_approval_wrapper`:

   ```python
   from __future__ import annotations
   from jac.tools.types import RiskLevel, ToolApprovalMeta
   from jac.runtime.approvals import ApprovalPolicy, ApprovalRequest, ApprovalActionKind
   from jac.runtime.events import EventBus, FileEditPreviewed, FileEditApplied

   def make_approval_wrapper(fn, events: EventBus, policy: ApprovalPolicy):
       meta: ToolApprovalMeta = fn.approval

       async def wrapper(**kwargs):
           if meta.risk_level == RiskLevel.READ_ONLY:
               return await fn(**kwargs)
           # compute preview diff for file edits before asking
           ...
           request = ApprovalRequest(
               summary=meta.description_fn(**kwargs),
               action_kind=_action_kind(meta.category),
               risk=_risk(meta.risk_level),
               tool_name=fn.__name__,
           )
           auto = policy.auto_response_for(request)
           response = auto or await events.request_approval(request)
           policy.record_response(request, response)
           if not response.approved:
               return _denial_result(fn, kwargs)
           result = await fn(**kwargs)
           if hasattr(result, "diff") and result.diff:
               await events.emit(FileEditApplied(path=result.path, diff=result.diff))
           return result

       # preserve the original function's name/schema for pydantic_ai tool registration
       wrapper.__name__ = fn.__name__
       wrapper.__doc__ = fn.__doc__
       wrapper.__annotations__ = fn.__annotations__
       return wrapper
   ```

   Key subtleties:
   - `pydantic_ai` uses `__name__` and `__annotations__` to infer the tool schema.
     Use `functools.wraps` or manually copy all relevant dunder attributes.
   - The denial return must be a valid `ToolResult` subtype so the model gets a
     coherent error, not a Python exception. Return
     `ToolResult(status=ToolStatus.ERROR, error="Action denied by user.")` cast to the
     right type, or define a `DenialResult` helper in `tools/types.py`.
   - `RiskLevel` in `tools/types.py` uses a `StrEnum` — compare against `RiskLevel.READ_ONLY`
     directly.

3. **Thread `events` and `policy` into `_resolve_local_tools`** in `agents/base.py`:

   ```python
   def _resolve_local_tools(
       allowed_tools: list[str],
       events: EventBus,
       policy: ApprovalPolicy,
   ) -> list[ToolFn]:
       ...
       local_tools.extend(make_approval_wrapper(fn, events, policy) for fn in tools)
   ```

   `config_loader` already receives `events: EventBus | None`. It does not currently
   receive an `ApprovalPolicy`. Two options:
   - Pass `policy` as a new optional param to `config_loader` (preferred — keeps the
     factory pure).
   - Build a default `ApprovalPolicy(mode=INTERACTIVE)` inside `config_loader` if none
     is provided (acceptable fallback for SDK callers who don't set up a CLI).

   `RunCoordinator.build_agent` already holds `self.session.config.approval_mode`; pass
   `ApprovalPolicy(mode=self.session.config.approval_mode)` through to `config_loader`.

4. **Wire `FileEditPreviewed`** for write/edit tools. Before calling `request_approval`,
   emit:

   ```python
   if meta.category == "file_write":
       await events.emit(FileEditPreviewed(path=..., diff=prospective_diff))
   ```

   The renderer already handles `FileEditPreviewed` (`cli/renderer.py:49`).

5. **Add tests** in `tests/test_agent_approval.py`:
   - `test_write_file_prompts_approval_in_interactive_mode` — mock `events.request_approval`
     to return approve; assert file written.
   - `test_write_file_denied_does_not_write` — mock to return deny; assert file absent.
   - `test_shell_high_risk_not_auto_approved_in_auto_edit_mode` — `auto-edit` approves
     file edits but not shell; assert `request_approval` called for shell.
   - `test_read_file_skips_gate_entirely` — assert `request_approval` never called.
   - `test_yolo_mode_no_prompts` — all tools auto-approve without `request_approval`.
   - `test_file_edit_previewed_emitted_before_write`.

6. **End-to-end smoke** (manual):
   - `jac "write a hello.py"` → diff panel renders → approval prompt → `y` → file lands.
   - `jac "run ls"` → shell approval prompt → `y` → output rendered.
   - `jac --approval yolo "write a hello.py"` → no prompt, file lands immediately.

## Acceptance checks

- [ ] `just test` green (all existing + new tests pass)
- [ ] `just typecheck` clean
- [ ] `jac "write a file called test.py"` → approval prompt fires before file exists on disk
- [ ] `jac "edit test.py"` → diff preview shown, then approval prompt
- [ ] `jac "run ls"` → shell approval prompt fires
- [ ] `jac --approval yolo "run ls"` → no prompt
- [ ] Denial correctly short-circuits execution (file is not created/modified)
- [ ] `FileEditPreviewed` event visible in renderer before the approval gate

## Contract refs

- `docs/contracts/TOOLS_CONTRACT.md` — `ToolApprovalMeta`, risk levels, denial result pattern
- `docs/contracts/CLI_DESIGN.md` — approval UX contract (interactive / auto-edit / yolo)
- `docs/contracts/EVENT_CONTRACT.md` — `FileEditPreviewed`, `FileEditApplied`, `ApprovalRequested`
