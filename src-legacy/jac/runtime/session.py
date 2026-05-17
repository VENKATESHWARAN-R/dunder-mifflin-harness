"""Session state and runtime configuration overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from jac.runtime.approvals import ApprovalMode


class RunMode(StrEnum):
    """Top-level operation mode for a session."""

    AUTOPILOT = "autopilot"
    HITL = "hitl"


class ModelTier(StrEnum):
    """Preferred model tier labels used by the future router."""

    SCOUT = "scout"
    WORKER = "worker"
    ARCHITECT = "architect"


@dataclass(slots=True)
class SessionConfig:
    """Mutable per-session settings controlled by CLI flags and slash commands."""

    cwd: Path = field(default_factory=Path.cwd)
    mode: RunMode = RunMode.HITL
    model: str | None = None
    tier: ModelTier | None = None
    role: str = "manager"
    slash_mode: str | None = None
    approval_mode: ApprovalMode = ApprovalMode.INTERACTIVE
    model_params: dict[str, str] = field(default_factory=dict)
    max_attachment_bytes: int = 200_000
    shell_timeout_seconds: float = 10.0
    shell_max_output_chars: int = 20_000
    debug: bool = False


@dataclass(slots=True)
class SessionState:
    """Runtime state owned by a single interactive or headless session."""

    config: SessionConfig = field(default_factory=SessionConfig)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    active_attempt_id: str | None = None
    attached_paths: list[Path] = field(default_factory=list)
    cumulative_tokens_in: int = 0
    cumulative_tokens_out: int = 0
    cumulative_requests: int = 0
    cumulative_tool_calls: int = 0
    last_context_tokens: int = 0
    last_model: str | None = None

    def reset_usage_counters(self) -> None:
        """Reset cumulative usage counters (e.g. /clear). Do not call from /compact (C12)."""
        self.cumulative_tokens_in = 0
        self.cumulative_tokens_out = 0
        self.cumulative_requests = 0
        self.cumulative_tool_calls = 0
        self.last_context_tokens = 0

    def remember_attachments(self, paths: list[Path]) -> None:
        """Track file attachments seen in this session."""
        for path in paths:
            if path not in self.attached_paths:
                self.attached_paths.append(path)
