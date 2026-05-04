"""Tests for the agent factory (config_loader)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from jac.agents import (
    AgentConfigNotFound,
    UnknownToolError,
    config_loader,
    ensure_default_run_config,
)
from jac.agents.personas import SCOTT_SYSTEM_PROMPT
from jac.config import Settings
from jac.runtime.coordinator import RunCoordinator, UserMessage
from jac.runtime.session import SessionConfig, SessionState
from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def _gateway_settings() -> Settings:
    return Settings(
        default_provider="gateway",
        model_tiers={
            "worker": ["gateway/google-vertex:gemini-3.1-flash-lite-preview"],
            "scout": ["gateway/google-vertex:gemini-3.1-flash-lite-preview"],
            "architect": ["gateway/anthropic:claude-opus-4-6"],
        },
    )


class TestConfigLoader:
    def test_builds_agent_from_default_config(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await ensure_default_run_config(state, "run-1")
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="manager"
                )
                return agent, settings
            finally:
                await state.close()

        agent, settings = _run(scenario())
        assert isinstance(agent, Agent)
        assert agent.model is not None
        assert agent._instructions == [SCOTT_SYSTEM_PROMPT]

    def test_resolves_local_tools(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="test",
                    allowed_tools=["filesystem:read"],
                )
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
                return agent
            finally:
                await state.close()

        agent = _run(scenario())
        tool_names = set(agent._function_toolset.tools.keys())
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "search_files" in tool_names
        assert "grep_files" in tool_names
        assert "write_file" not in tool_names

    def test_strips_mcp_prefix_and_skips_when_no_row(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="test",
                    allowed_tools=["mcp:nonexistent", "filesystem:read"],
                )
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
                return agent
            finally:
                await state.close()

        agent = _run(scenario())
        tool_names = set(agent._function_toolset.tools.keys())
        assert "read_file" in tool_names
        # No MCP toolsets should be present (function toolset is internal)
        from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

        assert not any(
            isinstance(t, (MCPServerStdio, MCPServerStreamableHTTP))
            for t in agent.toolsets
        )

    def test_builds_mcp_toolset_when_row_present(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                server = await state.mcp_servers.upsert(
                    name="stdio-server",
                    description="test",
                    transport="stdio",
                    config=json.dumps({"command": "python", "args": ["-m", "server"]}),
                    source_scope="seeded",
                    source_path=None,
                )
                await state.run_mcp_servers.create(
                    run_id="run-1", mcp_server_id=server.mcp_server_id
                )
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="test",
                    allowed_tools=["mcp:stdio-server"],
                )
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
                return agent
            finally:
                await state.close()

        agent = _run(scenario())
        from pydantic_ai.mcp import MCPServerStdio

        assert any(isinstance(t, MCPServerStdio) for t in agent.toolsets)

    def test_composes_skills(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                general = await state.skills.upsert(
                    name="general-style",
                    description="g",
                    domain="general",
                    content="Always use type hints.",
                    source_scope="seeded",
                    source_path=None,
                )
                python = await state.skills.upsert(
                    name="python-patterns",
                    description="p",
                    domain="python",
                    content="Prefer dataclasses.",
                    source_scope="seeded",
                    source_path=None,
                )
                await state.run_skills.create(run_id="run-1", skill_id=general.skill_id)
                await state.run_skills.create(run_id="run-1", skill_id=python.skill_id)
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="Base prompt.",
                    allowed_tools=[],
                )
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
                return agent
            finally:
                await state.close()

        agent = _run(scenario())
        instructions = agent._instructions[0]
        assert "Base prompt." in instructions
        assert "## Domain Knowledge" in instructions
        assert "Always use type hints." in instructions
        assert "Prefer dataclasses." in instructions
        assert instructions.index("Always use type hints.") < instructions.index(
            "Prefer dataclasses."
        )

    def test_uses_model_override_over_tier(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override="gateway/anthropic:claude-sonnet-4-6",
                    system_prompt="test",
                    allowed_tools=[],
                )
                agent = await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
                return agent
            finally:
                await state.close()

        agent = _run(scenario())
        assert isinstance(agent, Agent)
        # Gateway string gets resolved to the upstream provider model by Pydantic AI
        assert agent.model is not None

    def test_raises_on_unknown_tool(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="test",
                    allowed_tools=["bogus"],
                )
                await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
            finally:
                await state.close()

        with pytest.raises(UnknownToolError):
            _run(scenario())

    def test_raises_on_missing_config(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await config_loader(
                    state=state, settings=settings, run_id="run-1", role="chat"
                )
            finally:
                await state.close()

        with pytest.raises(AgentConfigNotFound):
            _run(scenario())

    def test_emits_node_events_when_event_bus_provided(self, tmp_path: Path) -> None:
        from jac.runtime.events import EventBus, NodeCompleted, NodeStarted

        async def scenario():
            settings = _gateway_settings()
            events = EventBus()
            recorded = []

            async def listener(event):
                recorded.append(type(event).__name__)

            events.on(NodeStarted, listener)
            events.on(NodeCompleted, listener)

            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                await ensure_default_run_config(state, "run-1")
                await config_loader(
                    state=state,
                    settings=settings,
                    run_id="run-1",
                    role="manager",
                    events=events,
                )
                return recorded
            finally:
                await state.close()

        recorded = _run(scenario())
        assert recorded == ["NodeStarted", "NodeCompleted"]


class TestEnsureDefaultRunConfig:
    def test_is_idempotent(self, tmp_path: Path) -> None:
        async def scenario():
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                first = await ensure_default_run_config(state, "run-1")
                second = await ensure_default_run_config(state, "run-1")
                return first, second
            finally:
                await state.close()

        first, second = _run(scenario())
        assert first.config_id == second.config_id

    def test_seeds_default_values(self, tmp_path: Path) -> None:
        async def scenario():
            state = await open_state_store(tmp_path / "state.db")
            try:
                await state.runs.create(run_id="run-1", prompt="p")
                cfg = await ensure_default_run_config(state, "run-1")
                return cfg
            finally:
                await state.close()

        cfg = _run(scenario())
        assert cfg.role == "manager"
        assert cfg.model_tier == "worker"
        assert cfg.model_override is None
        assert cfg.allowed_tools == ["filesystem", "shell"]
        assert cfg.persona == "Michael Scott"
        assert cfg.display_name == "Scott"
        assert "Michael Scott" in cfg.system_prompt


class TestCoordinatorFactory:
    def test_coordinator_uses_factory(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            state = await open_state_store(tmp_path / "state.db")
            try:
                # Seed a custom role with no tools — keeps TestModel from
                # auto-invoking write/shell tools (which would block on the
                # approval gate).
                await state.runs.create(run_id="run-1", prompt="p")
                await state.agent_configs.create(
                    run_id="run-1",
                    role="chat",
                    model_tier="worker",
                    model_override=None,
                    system_prompt="test",
                    allowed_tools=[],
                )
                with patch(
                    "jac.agents.base.build_pydantic_model",
                    return_value=TestModel(custom_output_text="factory ok"),
                ):
                    session = SessionState(config=SessionConfig(role="chat"))
                    session.run_id = "run-1"
                    coordinator = RunCoordinator(
                        settings=settings, state=state, session=session
                    )
                    coordinator._run_persisted = True
                    output = await coordinator.submit_message(UserMessage(text="hello"))
                    return output
            finally:
                await state.close()

        output = _run(scenario())
        assert output == "factory ok"

    def test_fallback_build_when_state_is_none(self, tmp_path: Path) -> None:
        async def scenario():
            settings = _gateway_settings()
            coordinator = RunCoordinator(settings=settings, state=None, session=None)
            coordinator._agent = Agent(
                TestModel(custom_output_text="fallback ok"), output_type=str
            )
            output = await coordinator.submit_message(UserMessage(text="hello"))
            return output

        output = _run(scenario())
        assert output == "fallback ok"
