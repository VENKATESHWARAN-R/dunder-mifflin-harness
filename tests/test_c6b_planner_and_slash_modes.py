"""C6b tests — planner role and slash mode addendums."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from jac.agents import config_loader, ensure_planner_config
from jac.agents.modes import build_instructions
from jac.agents.plans import Plan, PlannedTask
from jac.cli.app import ChatApp
from jac.config import Settings
from jac.runtime.events import EventBus, PlanGenerated
from jac.runtime.history import filter_tool_noise
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


def test_planner_seed_idempotent(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            first = await ensure_planner_config(store, "r1")
            second = await ensure_planner_config(store, "r1")
            row = await store.agent_configs.get_by_run_and_role("r1", "planner")
            return first, second, row
        finally:
            await store.close()

    first, second, row = _run(scenario())
    assert first.config_id == second.config_id
    assert row is not None
    assert row.persona == "Pam Beesly"
    assert row.model_tier == "architect"


def test_mode_addendum_compose() -> None:
    assert "SLASH MODE: /plan" in build_instructions("BASE", "plan")
    assert build_instructions("BASE", None) == "BASE"
    assert build_instructions("BASE", "missing") == "BASE"


def test_config_loader_threads_addendum(tmp_path: Path) -> None:
    async def scenario():
        settings = _gateway_settings()
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            await ensure_planner_config(store, "r1")
            with patch(
                "jac.agents.base.build_pydantic_model",
                return_value=TestModel(custom_output_text="ok"),
            ):
                agent = await config_loader(
                    state=store,
                    settings=settings,
                    run_id="r1",
                    role="planner",
                    instructions_addendum="ADDENDUM-X",
                )
            return agent
        finally:
            await store.close()

    agent = _run(scenario())
    instructions = agent._instructions[0]
    assert "You are Pam Beesly" in instructions
    assert instructions.endswith("ADDENDUM-X")


def test_filter_tool_noise_drops_tool_parts() -> None:
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content="/plan build auth"),
                ToolReturnPart(tool_name="list_directory", content="ignored"),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="list_directory", args={}),
                TextPart(content="final output"),
            ]
        ),
        ModelResponse(parts=[ToolCallPart(tool_name="read_file", args={})]),
    ]

    filtered = filter_tool_noise(messages)
    assert len(filtered) == 2
    assert isinstance(filtered[0], ModelRequest)
    assert isinstance(filtered[0].parts[0], UserPromptPart)
    assert isinstance(filtered[1], ModelResponse)
    assert isinstance(filtered[1].parts[0], TextPart)


def test_tasks_create_many_ordered(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            rows = await store.tasks.create_many(
                "r1",
                [
                    {"title": "one", "description": "d1", "acceptance_criteria": "a1"},
                    {"title": "two", "description": "d2", "acceptance_criteria": "a2"},
                    {"title": "three", "description": "d3", "acceptance_criteria": "a3"},
                ],
            )
            listed = await store.tasks.list_for_run("r1")
            return rows, listed
        finally:
            await store.close()

    rows, listed = _run(scenario())
    assert [row.order_index for row in rows] == [0, 1, 2]
    assert [row.run_id for row in listed] == ["r1", "r1", "r1"]


def test_plan_command_writes_tasks(tmp_path: Path) -> None:
    async def scenario():
        settings = _gateway_settings()
        state = await open_state_store(tmp_path / "state.db")
        plan = Plan(
            summary="Do it in two steps",
            dev_strategy="feature_by_feature",
            tasks=[
                PlannedTask(
                    title="step 1",
                    description="desc 1",
                    acceptance_criteria="acc 1",
                    complexity="simple",
                ),
                PlannedTask(
                    title="step 2",
                    description="desc 2",
                    acceptance_criteria="acc 2",
                    complexity="moderate",
                ),
            ],
        )
        events = EventBus()
        emitted: list[PlanGenerated] = []

        async def on_plan(event: PlanGenerated) -> None:
            emitted.append(event)

        events.on(PlanGenerated, on_plan)

        session = SessionState(config=SessionConfig())
        session.run_id = "r1"
        await state.runs.create(run_id="r1", prompt="start")
        app = ChatApp(settings=settings, state=state, session=session, events=events)
        try:
            original = app.coordinator.submit_slash_run

            async def fake_submit_slash_run(**_kwargs):
                return plan

            app.coordinator.submit_slash_run = fake_submit_slash_run  # type: ignore[method-assign]
            try:
                dispatched = await app.commands.dispatch("plan", "build auth")
                assert dispatched
            finally:
                app.coordinator.submit_slash_run = original  # type: ignore[method-assign]
            tasks = await state.tasks.list_for_run("r1")
            return tasks, emitted, app.session.config.slash_mode
        finally:
            await app.aclose()

    tasks, emitted, slash_mode = _run(scenario())
    assert len(tasks) == 2
    assert [task.order_index for task in tasks] == [0, 1]
    assert emitted and emitted[0].task_count == 2
    assert slash_mode is None


def test_slash_run_filters_history(tmp_path: Path, monkeypatch) -> None:
    async def scenario():
        settings = _gateway_settings()
        state = await open_state_store(tmp_path / "state.db")
        try:
            await state.runs.create(run_id="r1", prompt="start")
            await ensure_planner_config(state, "r1")
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            from jac.runtime.coordinator import RunCoordinator

            coordinator = RunCoordinator(settings=settings, state=state, session=session)
            coordinator._run_persisted = True

            class FakeResult:
                output = "ok"

                def usage(self):
                    return None

                def all_messages(self):
                    return [
                        ModelRequest(
                            parts=[
                                UserPromptPart(content="/plan x"),
                                ToolReturnPart(tool_name="list_directory", content="x"),
                            ]
                        ),
                        ModelResponse(
                            parts=[
                                ToolCallPart(tool_name="list_directory", args={}),
                                TextPart(content="final plan"),
                            ]
                        ),
                    ]

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _prompt, message_history=None):  # noqa: ARG002
                    return FakeResult()

            async def fake_loader(**_kwargs):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.config_loader", fake_loader)
            await coordinator.submit_slash_run(
                role="planner",
                prompt="build x",
                addendum_mode="plan",
                output_type=Plan,
                persist_user_prompt="/plan build x",
            )
            return coordinator._message_history
        finally:
            await state.close()

    history = _run(scenario())
    assert len(history) == 2
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert isinstance(history[1].parts[0], TextPart)
