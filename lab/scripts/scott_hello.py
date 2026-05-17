"""Slice B smoke: build Scott via the factory and exchange one turn.

Usage:
    uv run python lab/scripts/scott_hello.py "say hello"

Resolution at run time:
    per-run override (stubbed -> empty)
        -> ~/.jac/settings.json (default_tier, model_overrides[tier])
            -> src/jac/data/model_specs.toml ([tiers], first non-null wins)

The factory currently builds Anthropic models only. If your
`~/.jac/settings.json` points a tier at a non-Anthropic provider, this
script will surface the resolution error rather than silently fall back.
"""

from __future__ import annotations

import sys

from jac.agents.base import (
    ConfigurationError,
    PerRunOverride,
    _load_model_specs,
    _load_user_settings,
    build_agent,
    resolve_tier_and_model,
)
from jac.config import Settings
from jac.workspace import discover_workspace


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or "say hello"
    settings = Settings()
    workspace = discover_workspace()

    resolution = resolve_tier_and_model(
        settings=settings,
        per_run=PerRunOverride(),
        user=_load_user_settings(workspace),
        shipped=_load_model_specs(),
    )
    print(
        f"[scott_hello] tier={resolution.tier} ({resolution.tier_source}) "
        f"model={resolution.model_ref} ({resolution.model_source})"
    )

    try:
        agent = build_agent(settings=settings, workspace=workspace)
    except ConfigurationError as exc:
        print(f"[scott_hello] configuration error: {exc}", file=sys.stderr)
        return 2

    result = agent.run_sync(prompt)
    print()
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
