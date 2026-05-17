"""run_mcp_servers + run_skills toggle and active-listing semantics."""

from __future__ import annotations

from jac.state import StateStore


async def test_mcp_binding_active_listing(state: StateStore, run_id: str) -> None:
    server = await state.mcp_servers.upsert(
        name="playwright",
        description="browser",
        transport="stdio",
        config="{}",
        source_scope="user",
        source_path="/tmp/p.json",
    )
    await state.run_mcp_servers.create(
        run_id=run_id, mcp_server_id=server.mcp_server_id
    )
    active = await state.run_mcp_servers.list_active_for_run_with_details(run_id)
    assert [a.name for a in active] == ["playwright"]


async def test_mcp_binding_toggle_off_hides_from_active(
    state: StateStore, run_id: str
) -> None:
    server = await state.mcp_servers.upsert(
        name="playwright",
        description="browser",
        transport="stdio",
        config="{}",
        source_scope="user",
        source_path="/tmp/p.json",
    )
    binding = await state.run_mcp_servers.create(
        run_id=run_id, mcp_server_id=server.mcp_server_id
    )
    await state.run_mcp_servers.toggle(binding.id, enabled=0)
    active = await state.run_mcp_servers.list_active_for_run_with_details(run_id)
    assert active == []


async def test_skill_binding_active_listing(state: StateStore, run_id: str) -> None:
    skill = await state.skills.upsert(
        name="react-patterns",
        description="d",
        domain="frontend",
        content="body",
        source_scope="user",
        source_path="/tmp/s.md",
    )
    await state.run_skills.create(run_id=run_id, skill_id=skill.skill_id)
    active = await state.run_skills.list_active_for_run_with_details(run_id)
    assert [a.name for a in active] == ["react-patterns"]


async def test_role_scoped_binding_visible_for_matching_role(
    state: StateStore, run_id: str
) -> None:
    skill = await state.skills.upsert(
        name="react-patterns",
        description="d",
        domain="frontend",
        content="body",
        source_scope="user",
        source_path="/tmp/s.md",
    )
    await state.run_skills.create(
        run_id=run_id, skill_id=skill.skill_id, agent_role="builder"
    )
    active_for_builder = await state.run_skills.list_active_for_run_with_details(
        run_id, agent_role="builder"
    )
    assert [a.name for a in active_for_builder] == ["react-patterns"]
    active_for_manager = await state.run_skills.list_active_for_run_with_details(
        run_id, agent_role="manager"
    )
    assert active_for_manager == []
