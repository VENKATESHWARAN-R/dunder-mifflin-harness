"""Per-run override types shared between the agent factory and the state layer.

`PerRunOverride` lives in a neutral leaf module so `state/agent_configs.py`
can construct it without creating a `state -> agents.base` import edge that
would invert the dependency direction matrix in `docs/reference/PHILOSOPHY.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from jac.config import Tier


@dataclass(frozen=True, slots=True)
class PerRunOverride:
    """Per-run overrides sourced from `agent_configs`.

    Empty by default — the factory treats an empty override as "no per-run
    layer, fall through to user settings / shipped TOML." The state layer's
    `fetch_per_run_override` returns a populated instance when a matching
    `agent_configs` row exists.
    """

    tier: Tier | None = None
    model_override: str | None = None
