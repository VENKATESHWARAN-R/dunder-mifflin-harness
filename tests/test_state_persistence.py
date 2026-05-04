"""Integration tests for state persistence wired into RunCoordinator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from jac.config import Settings
from jac.runtime.coordinator import RunCoordinator, UserMessage, resume_run
from jac.runtime.events import WarningRaised
from jac.runtime.session import SessionState
from jac.state import open_state_store


def _build_coordinator(settings: Settings, state, output_text: str) -> RunCoordinator:
    coordinator = RunCoordinator(settings=settings, state=state)
    coordinator._agent = Agent(
        TestModel(custom_output_text=output_text), output_type=str
    )
    return coordinator


def test_coordinator_writes_run_and_messages(tmp_path: Path) -> None:
    async def scenario():
        settings = Settings()
        state = await open_state_store(tmp_path / "state.db")
        coordinator = _build_coordinator(settings, state, "first reply")
        await coordinator.submit_message(UserMessage(text="first user"))
        coordinator._agent = Agent(
            TestModel(custom_output_text="second reply"), output_type=str
        )
        await coordinator.submit_message(UserMessage(text="second user"))

        run = await state.runs.get(coordinator.session.run_id)
        msgs = await state.messages.list_for_run(coordinator.session.run_id)
        await state.close()
        return run, msgs

    run, msgs = asyncio.run(scenario())
    assert run is not None
    assert run.prompt == "first user"
    assert run.status == "running"
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in msgs] == [
        "first user",
        "first reply",
        "second user",
        "second reply",
    ]


def test_run_id_stable_across_turns(tmp_path: Path) -> None:
    async def scenario():
        settings = Settings()
        state = await open_state_store(tmp_path / "state.db")
        coordinator = _build_coordinator(settings, state, "ack")
        await coordinator.submit_message(UserMessage(text="t1"))
        await coordinator.submit_message(UserMessage(text="t2"))
        runs = await state.runs.list_recent()
        run_id = coordinator.session.run_id
        await state.close()
        return runs, run_id

    runs, run_id = asyncio.run(scenario())
    assert len(runs) == 1
    assert runs[0].run_id == run_id


def test_resume_loads_message_history(tmp_path: Path) -> None:
    async def scenario():
        settings = Settings()
        db_path = tmp_path / "state.db"

        state = await open_state_store(db_path)
        coordinator = _build_coordinator(settings, state, "answer")
        await coordinator.submit_message(UserMessage(text="hello"))
        run_id = coordinator.session.run_id
        await state.close()

        state2 = await open_state_store(db_path)
        resumed = await resume_run(state=state2, settings=settings, run_id=run_id)
        await state2.close()
        return run_id, resumed

    run_id, resumed = asyncio.run(scenario())
    assert resumed.session.run_id == run_id
    assert len(resumed._message_history) == 2  # noqa: SLF001
    assert resumed._run_persisted is True  # noqa: SLF001


def test_resume_unknown_run_raises(tmp_path: Path) -> None:
    async def scenario():
        settings = Settings()
        state = await open_state_store(tmp_path / "state.db")
        try:
            await resume_run(state=state, settings=settings, run_id="nope")
        finally:
            await state.close()

    with pytest.raises(LookupError):
        asyncio.run(scenario())


def test_coordinator_without_state_still_works() -> None:
    async def scenario():
        settings = Settings()
        coordinator = RunCoordinator(settings=settings, session=SessionState())
        coordinator._agent = Agent(TestModel(custom_output_text="ok"), output_type=str)
        return await coordinator.submit_message(UserMessage(text="ping"))

    output = asyncio.run(scenario())
    assert output == "ok"


def test_coordinator_continues_when_logfire_config_fails() -> None:
    async def scenario():
        settings = Settings()
        coordinator = RunCoordinator(settings=settings, session=SessionState())
        coordinator._agent = Agent(TestModel(custom_output_text="ok"), output_type=str)
        warnings: list[str] = []

        async def capture_warning(event: WarningRaised) -> None:
            warnings.append(event.message)

        coordinator.events.on(WarningRaised, capture_warning)
        with patch(
            "jac.runtime.coordinator.logfire.configure",
            side_effect=RuntimeError("no auth"),
        ):
            output = await coordinator.submit_message(UserMessage(text="ping"))
        return output, warnings

    output, warnings = asyncio.run(scenario())
    assert output == "ok"
    assert len(warnings) == 1
    assert "Observability is disabled for this session" in warnings[0]
