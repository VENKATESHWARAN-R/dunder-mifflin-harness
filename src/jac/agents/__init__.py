"""JAC agents — persona-driven Pydantic AI agents.

`agents.base.build_agent` is the only site in the codebase that calls
`pydantic_ai.Agent.from_file()`. See `docs/reference/PHILOSOPHY.md` for the
dependency-direction rules.
"""

from jac.agents.base import (
    ConfigurationError,
    ModelSpecs,
    PerRunOverride,
    TierResolution,
    UserSettings,
    build_agent,
    default_persona_path,
    load_per_run_override,
    resolve_tier_and_model,
)

__all__ = [
    "ConfigurationError",
    "ModelSpecs",
    "PerRunOverride",
    "TierResolution",
    "UserSettings",
    "build_agent",
    "default_persona_path",
    "load_per_run_override",
    "resolve_tier_and_model",
]
