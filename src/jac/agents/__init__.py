"""JAC agents — persona-driven Pydantic AI agents.

`agents.base.build_agent` is the only site in the codebase that calls
`pydantic_ai.Agent.from_file()`. See `docs/reference/PHILOSOPHY.md` for the
dependency-direction rules.
"""

from jac.agents.approval_callbacks import ApprovalCallback, auto_deny_callback
from jac.agents.base import (
    ConfigurationError,
    ModelSpecs,
    TierResolution,
    UserSettings,
    build_agent,
    default_persona_path,
    load_per_run_override,
    resolve_tier_and_model,
)
from jac.agents.overrides import PerRunOverride
from jac.tools.types import ScottDeps

__all__ = [
    "ApprovalCallback",
    "ConfigurationError",
    "ModelSpecs",
    "PerRunOverride",
    "ScottDeps",
    "TierResolution",
    "UserSettings",
    "auto_deny_callback",
    "build_agent",
    "default_persona_path",
    "load_per_run_override",
    "resolve_tier_and_model",
]
