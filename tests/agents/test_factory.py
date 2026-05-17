"""Slice B factory tests — tier resolution and YAML loading.

Live providers are forbidden in unit tests (per CLAUDE.md). The factory
accepts a `model=` injection so tests can pass a `TestModel` and skip the
provider call entirely. The three-tier resolution is tested directly
against `resolve_tier_and_model` with hand-built inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from jac.agents.base import (
    DEFAULT_TIER_FALLBACK,
    ConfigurationError,
    ModelSpecs,
    PerRunOverride,
    UserSettings,
    _load_model_specs,
    _load_user_settings,
    build_agent,
    default_persona_path,
    resolve_tier_and_model,
)
from jac.config import Settings, Tier
from jac.workspace import Workspace


# ---- Shipped TOML ----------------------------------------------------------


def test_shipped_model_specs_loads_anthropic_tiers() -> None:
    shipped = _load_model_specs()
    assert shipped.tier_models["scout"].startswith("anthropic:")
    assert shipped.tier_models["worker"].startswith("anthropic:")
    assert shipped.tier_models["architect"].startswith("anthropic:")
    assert shipped.default_max_context > 0


# ---- Resolution: tier source priority --------------------------------------


def _shipped() -> ModelSpecs:
    return ModelSpecs(
        tier_models={
            "scout": "anthropic:claude-haiku-4-5",
            "worker": "anthropic:claude-sonnet-4-6",
            "architect": "anthropic:claude-opus-4-7",
        },
        max_context_per_model={},
        default_max_context=200_000,
    )


def _settings(default_tier: Tier = DEFAULT_TIER_FALLBACK) -> Settings:
    return Settings(default_tier=default_tier)


def test_tier_falls_back_to_shipped_when_no_user_or_per_run() -> None:
    resolution = resolve_tier_and_model(
        settings=_settings("worker"),
        per_run=PerRunOverride(),
        user=UserSettings(),
        shipped=_shipped(),
    )
    assert resolution.tier == "worker"
    assert resolution.tier_source == "shipped"
    assert resolution.model_ref == "anthropic:claude-sonnet-4-6"
    assert resolution.model_source == "shipped"


def test_tier_user_settings_beats_shipped() -> None:
    resolution = resolve_tier_and_model(
        settings=_settings("worker"),
        per_run=PerRunOverride(),
        user=UserSettings(default_tier="architect"),
        shipped=_shipped(),
    )
    assert resolution.tier == "architect"
    assert resolution.tier_source == "user_settings"
    assert resolution.model_ref == "anthropic:claude-opus-4-7"


def test_tier_per_run_beats_everything() -> None:
    resolution = resolve_tier_and_model(
        settings=_settings("worker"),
        per_run=PerRunOverride(tier="scout"),
        user=UserSettings(default_tier="architect"),
        shipped=_shipped(),
    )
    assert resolution.tier == "scout"
    assert resolution.tier_source == "per_run"
    assert resolution.model_source == "shipped"


# ---- Resolution: model source priority -------------------------------------


def test_model_user_override_beats_shipped() -> None:
    resolution = resolve_tier_and_model(
        settings=_settings("worker"),
        per_run=PerRunOverride(),
        user=UserSettings(model_overrides={"worker": "anthropic:claude-opus-4-7"}),
        shipped=_shipped(),
    )
    assert resolution.tier == "worker"
    assert resolution.model_ref == "anthropic:claude-opus-4-7"
    assert resolution.model_source == "user_settings"


def test_model_per_run_override_beats_user() -> None:
    resolution = resolve_tier_and_model(
        settings=_settings("worker"),
        per_run=PerRunOverride(model_override="anthropic:claude-haiku-4-5-20251001"),
        user=UserSettings(model_overrides={"worker": "anthropic:claude-opus-4-7"}),
        shipped=_shipped(),
    )
    assert resolution.model_ref == "anthropic:claude-haiku-4-5-20251001"
    assert resolution.model_source == "per_run"


def test_unbound_tier_raises_configuration_error() -> None:
    bare = ModelSpecs(tier_models={}, max_context_per_model={}, default_max_context=0)
    with pytest.raises(ConfigurationError):
        resolve_tier_and_model(
            settings=_settings("worker"),
            per_run=PerRunOverride(),
            user=UserSettings(),
            shipped=bare,
        )


# ---- User settings JSON ----------------------------------------------------


def test_user_settings_returns_empty_when_file_missing(tmp_path: Path) -> None:
    ws = Workspace(user_dir=tmp_path / "does-not-exist")
    user = _load_user_settings(ws)
    assert user == UserSettings()


def test_user_settings_parses_valid_json(tmp_path: Path) -> None:
    ws = Workspace(user_dir=tmp_path)
    tmp_path.mkdir(exist_ok=True)
    ws.settings_path.write_text(
        json.dumps(
            {
                "default_tier": "scout",
                "model_overrides": {
                    "scout": "anthropic:claude-haiku-4-5",
                    "worker": 123,  # invalid type, must be filtered out
                    "architect": "anthropic:claude-opus-4-7",
                },
            }
        )
    )
    user = _load_user_settings(ws)
    assert user.default_tier == "scout"
    assert user.model_overrides == {
        "scout": "anthropic:claude-haiku-4-5",
        "architect": "anthropic:claude-opus-4-7",
    }


def test_user_settings_tolerates_corrupt_json(tmp_path: Path) -> None:
    ws = Workspace(user_dir=tmp_path)
    tmp_path.mkdir(exist_ok=True)
    ws.settings_path.write_text("{ not valid json")
    user = _load_user_settings(ws)
    assert user == UserSettings()


def test_user_settings_ignores_unknown_tier_name(tmp_path: Path) -> None:
    ws = Workspace(user_dir=tmp_path)
    tmp_path.mkdir(exist_ok=True)
    ws.settings_path.write_text(json.dumps({"default_tier": "wizard"}))
    user = _load_user_settings(ws)
    assert user.default_tier is None


# ---- Agent.from_file integration (no live provider) ------------------------


def test_build_agent_with_test_model_loads_scott_yaml(tmp_path: Path) -> None:
    """Verifies Agent.from_file consumes scott.yaml as a valid AgentSpec.

    Using TestModel skips the provider; the YAML parse and Agent construction
    are the actual checks here.
    """
    empty_workspace = Workspace(user_dir=tmp_path)
    agent = build_agent(
        settings=_settings("worker"),
        workspace=empty_workspace,
        model=TestModel(),
    )
    assert agent.name == "scott"


def test_default_persona_path_resolves_to_scott_yaml() -> None:
    path = default_persona_path()
    assert path.name == "scott.yaml"
    assert path.exists()


def test_build_agent_runs_against_test_model(tmp_path: Path) -> None:
    """End-to-end deterministic exercise: build the agent, run it via TestModel."""
    empty_workspace = Workspace(user_dir=tmp_path)
    agent = build_agent(
        settings=_settings("worker"),
        workspace=empty_workspace,
        model=TestModel(custom_output_text="hello from scott"),
    )
    result = agent.run_sync("ping")
    assert result.output == "hello from scott"
