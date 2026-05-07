"""C8 — model tier routing, SessionConfigChanged, manager-scoped /tier."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage

from jac.agents import (
    ensure_builder_config,
    ensure_manager_config,
    ensure_planner_config,
)
from jac.cli.app import ChatApp
from jac.config import Settings, tier_defaults_for
from jac.runtime.coordinator import RunCoordinator, UserMessage
from jac.runtime.events import EventBus, SessionConfigChanged, WarningRaised
from jac.runtime.session import ModelTier, SessionConfig, SessionState
from jac.state import open_state_store


def _run(coro):
    return asyncio.run(coro)


def _gateway_settings() -> Settings:
    return Settings(
        default_provider="gateway",
        model_tiers={
            "scout": ["gateway/google-vertex:scout-model"],
            "worker": ["gateway/google-vertex:worker-model"],
            "architect": ["gateway/anthropic:arch-model"],
        },
        default_tier="worker",
    )


def test_mid_session_tier_swap_updates_manager_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            coordinator = RunCoordinator(
                settings=settings, state=store, session=session
            )

            class FakeResult:
                output = "ok"

                def usage(self) -> RunUsage:
                    return RunUsage(
                        input_tokens=2,
                        output_tokens=1,
                        requests=1,
                        tool_calls=0,
                    )

                def all_messages(self):
                    return [
                        ModelRequest(parts=[UserPromptPart(content="hi")]),
                        ModelResponse(parts=[TextPart(content="ok")]),
                    ]

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _prompt, message_history=None):  # noqa: ARG002
                    return FakeResult()

            async def fake_loader(**_kw):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            monkeypatch.setattr("jac.agents.config_loader", fake_loader)

            await coordinator.submit_message(UserMessage(text="first"))
            rows1 = sorted(
                [
                    r
                    for r in await store.attempts.list_for_run("r1")
                    if r.role == "manager"
                ],
                key=lambda r: (r.created_at, r.attempt_id),
            )
            session.config.tier = ModelTier.SCOUT
            coordinator.reset_agent()
            await coordinator.submit_message(UserMessage(text="second"))
            rows2 = sorted(
                [
                    r
                    for r in await store.attempts.list_for_run("r1")
                    if r.role == "manager"
                ],
                key=lambda r: (r.created_at, r.attempt_id),
            )
            return rows1, rows2
        finally:
            await store.close()

    first_mgr, second_mgr = _run(scenario())
    assert len(first_mgr) == 1
    assert {(r.tier, r.model) for r in second_mgr} == {
        ("worker", "gateway/google-vertex:worker-model"),
        ("scout", "gateway/google-vertex:scout-model"),
    }


def test_tier_scout_does_not_change_planner_row(tmp_path: Path, monkeypatch) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            await ensure_planner_config(store, "r1")
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            events = EventBus()
            app = ChatApp(
                settings=settings, state=store, session=session, events=events
            )

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, *_a, **_k):
                    raise AssertionError("not called")

            async def fake_loader(**_kw):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            monkeypatch.setattr("jac.agents.config_loader", fake_loader)

            await app.commands.dispatch("tier", "scout")
            await app.coordinator._ensure_agent()
            row = await store.agent_configs.get_by_run_and_role("r1", "planner")
            return row
        finally:
            await store.close()

    row = _run(scenario())
    assert row is not None
    assert row.model_tier == "architect"


def test_model_override_propagates_to_all_roles(tmp_path: Path, monkeypatch) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            await ensure_manager_config(store, "r1")
            await ensure_builder_config(store, "r1")
            await ensure_planner_config(store, "r1")
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            events = EventBus()
            app = ChatApp(
                settings=settings, state=store, session=session, events=events
            )

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, *_a, **_k):
                    raise AssertionError("not called")

            async def fake_loader(**_kw):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            monkeypatch.setattr("jac.agents.config_loader", fake_loader)

            pin = "gateway/anthropic:claude-haiku-4-5"
            await app.commands.dispatch("model", pin)
            await app.coordinator._ensure_agent()
            mgr = await store.agent_configs.get_by_run_and_role("r1", "manager")
            bld = await store.agent_configs.get_by_run_and_role("r1", "builder")
            pln = await store.agent_configs.get_by_run_and_role("r1", "planner")
            return mgr, bld, pln
        finally:
            await store.close()

    mgr, bld, pln = _run(scenario())
    pin = "gateway/anthropic:claude-haiku-4-5"
    assert mgr is not None and mgr.model_override == pin
    assert bld is not None and bld.model_override == pin
    assert pln is not None and pln.model_override == pin


def test_session_config_changed_emitted_once_per_change(tmp_path: Path) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            events = EventBus()
            captured: list[SessionConfigChanged] = []

            async def on_cfg(event: SessionConfigChanged) -> None:
                captured.append(event)

            events.on(SessionConfigChanged, on_cfg)
            app = ChatApp(
                settings=settings, state=store, session=session, events=events
            )
            await app.commands.dispatch("tier", "worker")
            await app.commands.dispatch("tier", "worker")
            await app.commands.dispatch("tier", "scout")
            return captured
        finally:
            await store.close()

    captured = _run(scenario())
    keys = [e.key for e in captured]
    assert keys.count("tier") == 2
    assert captured[0].key == "tier" and captured[0].old_value is None
    assert captured[0].new_value == ModelTier.WORKER
    assert captured[1].key == "tier" and captured[1].old_value == ModelTier.WORKER
    assert captured[1].new_value == ModelTier.SCOUT


def test_tier_echo_includes_resolved_model(tmp_path: Path, monkeypatch) -> None:
    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            events = EventBus()
            app = ChatApp(
                settings=settings, state=store, session=session, events=events
            )
            lines: list[str] = []

            def capture_print(message: str) -> None:
                lines.append(message)

            monkeypatch.setattr(app.renderer, "print_info", capture_print)
            await app.commands.dispatch("tier", "worker")
            return lines
        finally:
            await store.close()

    lines = _run(scenario())
    assert any(
        "manager will use: gateway/google-vertex:worker-model" in ln for ln in lines
    )


def test_tier_defaults_for_two_providers() -> None:
    g = tier_defaults_for("gateway")
    a = tier_defaults_for("anthropic")
    for d in (g, a):
        assert set(d.keys()) == {"scout", "worker", "architect"}
        assert all(isinstance(d[k], list) and len(d[k]) > 0 for k in d)


def test_warn_missing_creds_emits_warning(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        Settings,
        "optional_env_var",
        lambda self, name: None,  # noqa: ARG002
    )

    async def scenario():
        store = await open_state_store(tmp_path / "state.db")
        try:
            await store.runs.create(run_id="r1", prompt="p")
            settings = _gateway_settings()
            session = SessionState(config=SessionConfig())
            session.run_id = "r1"
            events = EventBus()
            warnings: list[WarningRaised] = []

            async def on_warn(event: WarningRaised) -> None:
                warnings.append(event)

            events.on(WarningRaised, on_warn)
            app = ChatApp(
                settings=settings, state=store, session=session, events=events
            )
            await app._maybe_warn_missing_creds("gateway/google-vertex:gemini-3")
            return warnings
        finally:
            await store.close()

    warnings = _run(scenario())
    assert warnings and "set it before the next turn" in warnings[0].message
