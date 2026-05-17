"""Agent-configs repo + `fetch_per_run_override` bridge."""

from __future__ import annotations

from jac.agents.overrides import PerRunOverride
from jac.state import StateStore, fetch_per_run_override


async def test_create_and_get(state: StateStore, run_id: str) -> None:
    row = await state.agent_configs.create(
        run_id=run_id, role="manager", model_tier="worker"
    )
    assert row.model_tier == "worker"
    assert row.model_override is None
    assert row.allowed_tools == "[]"
    assert row.max_context_tokens == 8000

    fetched = await state.agent_configs.get_by_run_and_role(run_id, "manager")
    assert fetched is not None
    assert fetched.role == "manager"


async def test_update_patches_only_provided_fields(
    state: StateStore, run_id: str
) -> None:
    await state.agent_configs.create(
        run_id=run_id, role="manager", model_tier="worker"
    )
    patched = await state.agent_configs.update(
        run_id, "manager", model_override="anthropic:claude-opus-4-7"
    )
    assert patched is not None
    assert patched.model_tier == "worker"  # unchanged
    assert patched.model_override == "anthropic:claude-opus-4-7"


async def test_unique_run_role_constraint(state: StateStore, run_id: str) -> None:
    import aiosqlite

    await state.agent_configs.create(
        run_id=run_id, role="manager", model_tier="worker"
    )
    try:
        await state.agent_configs.create(
            run_id=run_id, role="manager", model_tier="scout"
        )
    except aiosqlite.IntegrityError:
        return
    raise AssertionError("expected IntegrityError on duplicate (run_id, role)")


async def test_fetch_override_returns_empty_when_no_row(
    state: StateStore, run_id: str
) -> None:
    override = await fetch_per_run_override(state, run_id, role="manager")
    assert override == PerRunOverride()


async def test_fetch_override_round_trips_tier_and_model(
    state: StateStore, run_id: str
) -> None:
    await state.agent_configs.create(
        run_id=run_id,
        role="manager",
        model_tier="architect",
        model_override="anthropic:claude-opus-4-7",
    )
    override = await fetch_per_run_override(state, run_id, role="manager")
    assert override.tier == "architect"
    assert override.model_override == "anthropic:claude-opus-4-7"


async def test_fetch_override_ignores_unknown_tier_name(
    state: StateStore, run_id: str
) -> None:
    await state.agent_configs.create(
        run_id=run_id, role="manager", model_tier="wizard"
    )
    override = await fetch_per_run_override(state, run_id, role="manager")
    assert override.tier is None
