"""
Worker agent loop observer.

agent.run() is the right call for Scout and Architect because:
  - Scout: single-pass reads with no tool loops — a budget ceiling in UsageLimits is enough
  - Architect: produces one structured plan in one or two requests; there's no loop to spiral

Workers are different. They implement features across open-ended tool loops that can run
20+ steps. By the time agent.run() returns, a stuck or hallucinating Worker has already
burned tokens, made bad writes, and corrupted context. We need to observe *between steps*.

agent.iter() exposes the execution graph one node at a time and hands control back to us
between every state transition. This file wraps that loop with an observer that:
  - extracts per-step metrics at CallToolsNode (the only point with response + usage together)
  - injects corrective system context at ModelRequestNode (before the next model call)
  - aborts and signals escalation when hard limits are hit
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.messages import SystemPromptPart, ToolCallPart
from pydantic_graph import End


# ---------------------------------------------------------------------------
# Health signal
# ---------------------------------------------------------------------------


class HealthStatus(Enum):
    OK = auto()
    # Soft signal: something looks off but the agent can still self-correct.
    # We steer by injecting a system-level nudge before the next model call.
    WARNING = auto()
    # Hard signal: the run cannot recover on its own.
    # We abort cleanly and return control to the caller for escalation
    # (e.g. hand the accumulated context to the Architect for triage).
    CRITICAL = auto()


# ---------------------------------------------------------------------------
# Per-step and cumulative metrics
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    input_tokens: int
    output_tokens: int
    # Stable fingerprints of every (tool_name, args) pair the model requested.
    # Stored as a flat list so we can count repeats across steps via Counter.
    tool_call_fingerprints: list[str]


@dataclass
class ObserverState:
    """
    Accumulated view of a Worker run from the observer's perspective.

    Kept completely separate from the agent's own RunContext — the observer
    never touches what the agent sees except at the explicit injection point
    in ModelRequestNode.
    """

    steps: list[StepMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # How many steering injections we've made. Useful for deciding whether
    # to escalate (agent ignored two steerings → probably not self-correcting).
    steerings_injected: int = 0
    # Flat list of all tool call fingerprints across every step.
    # Counter over this tells us which calls are repeating and how often.
    all_tool_fingerprints: list[str] = field(default_factory=list)

    def record(self, metrics: StepMetrics) -> None:
        self.steps.append(metrics)
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        self.all_tool_fingerprints.extend(metrics.tool_call_fingerprints)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


# ---------------------------------------------------------------------------
# Thresholds
# These are intentionally hardcoded for now. In v1 they should come from
# agent_configs in the persistent state store (each Worker can have different
# complexity budgets depending on task tier assignment).
# ---------------------------------------------------------------------------

_MAX_STEPS = 20
_WARN_STEPS = 13

_MAX_TOKENS = 60_000
_WARN_TOKENS = 40_000

# How many times the same (tool, args) fingerprint can appear before we
# call it a tool loop. 2 could be a legitimate retry; 3+ is a stuck loop.
_LOOP_REPEAT_THRESHOLD = 3

# If the latest step's output tokens are this many times the running mean,
# the model is probably generating verbose filler — a precursor to hallucination.
_VERBOSITY_SPIKE_MULTIPLIER = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fingerprint(tool_name: str, args: str | dict[str, Any] | None) -> str:
    """
    Stable short hash of a (tool_name, args) pair for loop detection.

    Args can arrive as a JSON string or a dict depending on the provider.
    We normalise to a sorted JSON dump so call-order doesn't produce false negatives,
    then take the first 12 hex chars — enough to distinguish calls, short enough to store cheaply.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass  # keep as string if it won't parse
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _extract_step_metrics(node: CallToolsNode, step: int) -> StepMetrics:
    """
    Pull usage and tool call fingerprints out of a CallToolsNode.

    We do this at CallToolsNode rather than ModelRequestNode because it's the first
    point in the cycle where both the model's response content *and* its token usage
    are available in a single place. ModelRequestNode only has the outgoing request.
    """
    usage = node.model_response.usage
    fingerprints = [
        _fingerprint(part.tool_name, part.args)
        for part in node.model_response.parts
        if isinstance(part, ToolCallPart)
    ]
    return StepMetrics(
        step=step,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        tool_call_fingerprints=fingerprints,
    )


def _classify(state: ObserverState) -> tuple[HealthStatus, str]:
    """
    Classify the current health of the Worker run given accumulated observer state.

    Checks run from most severe to least — first match wins so we never downgrade
    a CRITICAL to a WARNING once the hard threshold is crossed.
    """
    # --- Hard limits (CRITICAL) ---

    if state.step_count >= _MAX_STEPS:
        return (
            HealthStatus.CRITICAL,
            f"step ceiling hit ({state.step_count}/{_MAX_STEPS})",
        )

    if state.total_tokens >= _MAX_TOKENS:
        return (
            HealthStatus.CRITICAL,
            f"token budget exhausted ({state.total_tokens:,}/{_MAX_TOKENS:,})",
        )

    if state.all_tool_fingerprints:
        top_fp, top_count = Counter(state.all_tool_fingerprints).most_common(1)[0]
        if top_count >= _LOOP_REPEAT_THRESHOLD:
            return (
                HealthStatus.CRITICAL,
                f"tool call loop: same call repeated {top_count}× ({top_fp})",
            )

    # Two ignored steerings means the agent is not self-correcting.
    # Hand off to Architect rather than keep burning budget.
    if state.steerings_injected >= 2:
        return HealthStatus.CRITICAL, "agent did not self-correct after two steerings"

    # --- Soft limits (WARNING) ---

    if state.step_count >= _WARN_STEPS:
        return (
            HealthStatus.WARNING,
            f"approaching step ceiling ({state.step_count}/{_MAX_STEPS})",
        )

    if state.total_tokens >= _WARN_TOKENS:
        return (
            HealthStatus.WARNING,
            f"approaching token budget ({state.total_tokens:,}/{_MAX_TOKENS:,})",
        )

    # Output verbosity spike — model generating padding, often precedes hallucination.
    # Only meaningful after we have a few steps to establish a baseline.
    if state.step_count >= 3:
        recent_out = state.steps[-1].output_tokens
        prior_mean = sum(s.output_tokens for s in state.steps[:-1]) / (
            state.step_count - 1
        )
        if prior_mean > 0 and recent_out > prior_mean * _VERBOSITY_SPIKE_MULTIPLIER:
            return (
                HealthStatus.WARNING,
                f"output spike: {recent_out} tokens vs {prior_mean:.0f} mean — possible verbosity spiral",
            )

    return HealthStatus.OK, "healthy"


def _build_steering(reason: str, step: int) -> SystemPromptPart:
    """
    Build a system prompt part to inject as corrective steering.

    We use SystemPromptPart rather than a user-turn message for two reasons:
    1. The model treats system context as background guidance, not a command —
       it degrades more gracefully when partially ignored.
    2. It doesn't appear in the visible conversation history, so the agent's
       next user-facing response won't try to "reply" to the correction.
    """
    return SystemPromptPart(
        content=(
            f"[Observer — step {step}] Warning: {reason}. "
            "Re-evaluate your current approach. If the same tool call is not making progress, "
            "try a different strategy or summarise what you have found so far and stop."
        )
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


@dataclass
class WorkerRunResult:
    output: Any | None
    observer: ObserverState
    # True if the run completed normally; False if the observer aborted it.
    completed: bool
    # Set when completed=False so the caller knows why and can escalate.
    abort_reason: str | None = None


async def run_worker(
    agent: Agent,
    prompt: str,
    *,
    deps: Any = None,
) -> WorkerRunResult:
    """
    Run a Worker (Tier 2) agent under observer supervision.

    The loop structure maps to the agent's execution graph:

        UserPromptNode
              ↓
        ModelRequestNode  ← inject pending steering here, before model call
              ↓
        CallToolsNode     ← evaluate health here, after model responds
              ↓
        ModelRequestNode  ← (loops back until End)
              ↓
            End

    pending_steering carries a SystemPromptPart from a WARNING detection at
    CallToolsNode forward to the next ModelRequestNode. This way the corrective
    context lands in the request *before* the model makes its next decision,
    not after the damage is done.

    On CRITICAL, we break out of the loop without calling agent_run.next().
    The async context manager closes cleanly and the caller receives a
    WorkerRunResult(completed=False) to trigger Architect escalation.
    """
    observer = ObserverState()
    pending_steering: SystemPromptPart | None = None
    abort_reason: str | None = None
    final_output: Any | None = None

    async with agent.iter(prompt, deps=deps) as agent_run:
        node = agent_run.next_node

        while not isinstance(node, End):
            # ── ModelRequestNode: before model call ───────────────────────────
            # This is our injection window. The request has been assembled but
            # not yet sent to the model. Appending to node.request.parts here
            # means the model receives the steering as part of its input context.
            if isinstance(node, ModelRequestNode) and pending_steering is not None:
                node.request.parts.append(pending_steering)
                pending_steering = None

            # ── CallToolsNode: after model responds, before tools execute ─────
            # The model has made its decision (what to say, what tools to call)
            # but tools have not run yet. This is the earliest point we can
            # inspect the response and intervene without losing a full round-trip.
            elif isinstance(node, CallToolsNode):
                metrics = _extract_step_metrics(node, observer.step_count)
                observer.record(metrics)

                status, reason = _classify(observer)

                if status == HealthStatus.CRITICAL:
                    # Do not advance — break cleanly so the context manager closes.
                    # The caller is expected to log the abort, write it to the
                    # attempts table, and route to Architect for triage.
                    abort_reason = reason
                    break

                if status == HealthStatus.WARNING:
                    # Queue a steering injection for the next ModelRequestNode.
                    # We do not inject here because tool execution still needs
                    # to proceed — the steering is for the model's *next* call.
                    pending_steering = _build_steering(reason, observer.step_count)
                    observer.steerings_injected += 1

            node = await agent_run.next(node)

        if isinstance(node, End):
            final_output = (
                node.data.output if hasattr(node.data, "output") else node.data
            )

    completed = abort_reason is None
    return WorkerRunResult(
        output=final_output,
        observer=observer,
        completed=completed,
        abort_reason=abort_reason,
    )
