"""Agent factory — the only site that calls `pydantic_ai.Agent.from_file()`.

Three-tier model resolution per the M1 spec:

1. **Per-run override** — `agent_configs.tier` / `agent_configs.model_override`
   from SQLite. Stubbed in Slice B (no state layer yet); the seam is real so
   that wiring it later is one call site, not a refactor.
2. **User settings** — `~/.jac/settings.json` (`default_tier`,
   `model_overrides[tier]`). File is optional; defaults apply when missing.
3. **Shipped TOML** — `src/jac/data/model_specs.toml` (`tiers[tier]`).
   First non-null wins, independently for the tier itself and for the
   model_ref bound to that tier.

The factory builds the Pydantic AI model explicitly (passing `api_key` via
the provider) so it does not depend on ambient env state at call time.
Tests inject a `TestModel` / `FunctionModel` via the `model=` kwarg to skip
the live provider.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from jac.agents.approval import make_approval_wrapper
from jac.agents.approval_callbacks import ApprovalCallback, auto_deny_callback
from jac.agents.overrides import PerRunOverride
from jac.config import Settings, Tier
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.tools import TOOL_REGISTRY
from jac.tools.types import ScottDeps
from jac.workspace import Workspace, discover_workspace

if TYPE_CHECKING:
    from pydantic_ai.models import Model


# Resource paths inside the installed `jac.data` package.
_PERSONAS_PACKAGE = "jac.data.personas"
_DATA_PACKAGE = "jac.data"
_MODEL_SPECS_NAME = "model_specs.toml"
DEFAULT_PERSONA = "scott.yaml"
DEFAULT_TIER_FALLBACK: Tier = "worker"

DEFAULT_TOOL_GROUPS: tuple[str, ...] = ("filesystem", "shell", "tasks")
"""Scott's default tool surface for M1.

Hard-coded here until Slice 4 (Runtime) writes the per-run `agent_configs`
row on run start and sources `allowed_tools` from SQLite.
"""


# ---- Per-run override seam (state layer fills this in later) ---------------


def load_per_run_override(run_id: str | None) -> PerRunOverride:
    """Return per-run overrides for a run.

    Always returns an empty override. The real read against `agent_configs`
    is async and lives in `jac.state.agent_configs.fetch_per_run_override` —
    runtime callers fetch it ahead of time and pass the result via the
    `per_run=` kwarg on `build_agent`. This sync stub stays so non-runtime
    callers (the smoke script, the existing tests) don't need an event loop.
    """
    _ = run_id
    return PerRunOverride()


# ---- Shipped TOML + user JSON loaders --------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpecs:
    tier_models: dict[str, str]
    max_context_per_model: dict[str, int]
    default_max_context: int


@dataclass(frozen=True, slots=True)
class UserSettings:
    default_tier: Tier | None = None
    model_overrides: dict[str, str] = field(default_factory=dict)


def _load_model_specs() -> ModelSpecs:
    raw = (resource_files(_DATA_PACKAGE) / _MODEL_SPECS_NAME).read_text(
        encoding="utf-8"
    )
    data: dict[str, Any] = tomllib.loads(raw)
    tiers = dict(data.get("tiers", {}))
    models_block = data.get("models", {})
    max_context: dict[str, int] = {
        name: int(spec.get("max_context", 0))
        for name, spec in models_block.items()
        if isinstance(spec, dict)
    }
    default_max = int(data.get("defaults", {}).get("max_context", 200_000))
    return ModelSpecs(
        tier_models=tiers,
        max_context_per_model=max_context,
        default_max_context=default_max,
    )


def _load_user_settings(workspace: Workspace) -> UserSettings:
    path = workspace.settings_path
    if not path.exists():
        return UserSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return UserSettings()
    if not isinstance(data, dict):
        return UserSettings()
    default_tier = data.get("default_tier")
    overrides_raw = data.get("model_overrides", {})
    overrides: dict[str, str] = {}
    if isinstance(overrides_raw, dict):
        overrides = {
            str(k): str(v) for k, v in overrides_raw.items() if isinstance(v, str)
        }
    tier: Tier | None = None
    if default_tier in ("scout", "worker", "architect"):
        tier = default_tier  # type: ignore[assignment]
    return UserSettings(default_tier=tier, model_overrides=overrides)


# ---- Resolution ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TierResolution:
    """The resolved (tier, model_ref) pair plus which layer supplied each."""

    tier: Tier
    model_ref: str
    tier_source: str  # "per_run" | "user_settings" | "shipped"
    model_source: str  # "per_run" | "user_settings" | "shipped"


class ConfigurationError(RuntimeError):
    """Raised when no tier or model can be resolved."""


def resolve_tier_and_model(
    *,
    settings: Settings,
    per_run: PerRunOverride,
    user: UserSettings,
    shipped: ModelSpecs,
) -> TierResolution:
    if per_run.tier is not None:
        tier, tier_source = per_run.tier, "per_run"
    elif user.default_tier is not None:
        tier, tier_source = user.default_tier, "user_settings"
    else:
        tier, tier_source = settings.default_tier, "shipped"

    if per_run.model_override is not None:
        model_ref, model_source = per_run.model_override, "per_run"
    elif tier in user.model_overrides:
        model_ref, model_source = user.model_overrides[tier], "user_settings"
    elif tier in shipped.tier_models:
        model_ref, model_source = shipped.tier_models[tier], "shipped"
    else:
        raise ConfigurationError(
            f"No model_ref bound to tier {tier!r}. Add an entry under "
            f"`[tiers]` in model_specs.toml, set `model_overrides[{tier}]` "
            f"in ~/.jac/settings.json, or supply a per-run override."
        )

    return TierResolution(
        tier=tier,
        model_ref=model_ref,
        tier_source=tier_source,
        model_source=model_source,
    )


# ---- Pydantic AI model construction ----------------------------------------


def build_pai_model(model_ref: str, settings: Settings) -> Model:
    """Construct the Pydantic AI model for a resolved `provider:model_id`."""
    provider_id, _, model_id = model_ref.partition(":")
    if not provider_id or not model_id:
        raise ConfigurationError(
            f"Invalid model_ref {model_ref!r}; expected 'provider:model_id'."
        )
    if provider_id != "anthropic":
        # M1 ships Anthropic only. Multi-provider lands when `jac init` does.
        raise ConfigurationError(
            f"Unsupported provider {provider_id!r}. M1 ships Anthropic only; "
            f"multi-provider arrives when `jac init` lands."
        )
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicModel(
        model_id,
        provider=AnthropicProvider(api_key=settings.require_anthropic_api_key()),
    )


# ---- Public entry points ---------------------------------------------------


def default_persona_path() -> Path:
    """Path to the shipped Scott persona YAML."""
    return Path(str(resource_files(_PERSONAS_PACKAGE) / DEFAULT_PERSONA))


def build_agent(
    *,
    persona_path: Path | str | None = None,
    settings: Settings | None = None,
    workspace: Workspace | None = None,
    run_id: str | None = None,
    per_run: PerRunOverride | None = None,
    model: Model | None = None,
    tools_groups: list[str] | None = None,
    approval_policy: ApprovalPolicy | None = None,
    approval_callback: ApprovalCallback | None = None,
    deps_type: type[Any] = ScottDeps,
) -> Agent[Any, Any]:
    """Build a Pydantic AI Agent from a persona YAML.

    This is the **only** site that calls `Agent.from_file()`. Tier and
    model are resolved by `resolve_tier_and_model`; tests can inject a
    `TestModel` / `FunctionModel` via the `model=` kwarg.

    Tools are resolved from `tools_groups` against `TOOL_REGISTRY`, wrapped
    with the approval middleware (`policy` decides what auto-approves;
    `callback` resolves anything that doesn't), and passed to
    `Agent.from_file(tools=...)`. `deps_type` is forwarded so stateful
    tools — task-CRUD — receive `RunContext[ScottDeps]` at call time; the
    runtime coordinator constructs the `ScottDeps` instance and supplies
    it via `agent.run(..., deps=...)`.

    `per_run` lets async callers (the runtime coordinator) pre-fetch the
    `agent_configs` row via `jac.state.agent_configs.fetch_per_run_override`
    and pass it in, keeping this factory sync.
    """
    settings = settings or Settings()
    workspace = workspace or discover_workspace()
    if per_run is None:
        per_run = load_per_run_override(run_id)
    shipped = _load_model_specs()
    user = _load_user_settings(workspace)
    resolution = resolve_tier_and_model(
        settings=settings,
        per_run=per_run,
        user=user,
        shipped=shipped,
    )

    pai_model: Model = (
        model if model is not None else build_pai_model(resolution.model_ref, settings)
    )
    policy = approval_policy or ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    callback = approval_callback or auto_deny_callback
    groups = (
        list(tools_groups) if tools_groups is not None else list(DEFAULT_TOOL_GROUPS)
    )
    wrapped_tools = _resolve_and_wrap_tools(groups, policy, callback)

    target = Path(persona_path) if persona_path else default_persona_path()
    return Agent.from_file(
        target,
        model=pai_model,
        tools=wrapped_tools,
        deps_type=deps_type,
    )


def _resolve_and_wrap_tools(
    groups: list[str],
    policy: ApprovalPolicy,
    callback: ApprovalCallback,
) -> list[Any]:
    """Resolve tool groups against `TOOL_REGISTRY` and wrap each raw tool.

    Unknown groups raise `ConfigurationError` so a typo in `allowed_tools`
    surfaces immediately rather than silently shipping a hobbled agent.
    Duplicate tools (a function appearing in two groups) are de-duplicated
    by identity so the agent only sees each tool once.
    """
    seen: set[int] = set()
    wrapped: list[Any] = []
    for group in groups:
        if group.startswith("mcp:"):
            # MCP servers resolve against the `mcp_servers` table elsewhere.
            continue
        if group not in TOOL_REGISTRY:
            raise ConfigurationError(
                f"Unknown tool group {group!r}; available: "
                f"{sorted(TOOL_REGISTRY.keys())}"
            )
        for fn in TOOL_REGISTRY[group]:
            if id(fn) in seen:
                continue
            seen.add(id(fn))
            wrapped.append(make_approval_wrapper(fn, policy, callback))
    return wrapped
