"""C6c tests — spawn_minion and tool-layer hardening."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jac.agents.approval import make_approval_wrapper
from jac.agents.result_filter import make_result_filter_wrapper
from jac.agents.spawn import make_fetch_full_result_tool, make_spawn_minion_tool, native_agent_extras
from jac.config import Settings
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.runtime.events import EventBus, MinionReturned, MinionSpawned
from jac.runtime.session import SessionState
from jac.state import open_state_store
from jac.tools.cache import ToolResultCache
from jac.tools.filesystem import read_file_smart
from jac.tools.summarize import estimate_tokens
from jac.tools.types import RiskLevel, ShellToolResult, ToolApprovalMeta, ToolResult, ToolStatus


def _run(coro):
    return asyncio.run(coro)


def test_estimate_tokens_word_count_proxy() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2


def test_tool_result_cache_store_and_fetch_roundtrip() -> None:
    cache = ToolResultCache(max_entries=2)
    h1 = cache.store("a")
    h2 = cache.store("b")
    assert cache.fetch(h1) == "a"
    assert cache.fetch("missing") is None
    cache.store("c")
    assert cache.fetch(h2) == "b"


def test_read_file_smart_small_file_returns_full_content(tmp_path: Path) -> None:
    p = tmp_path / "small.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    result = _run(read_file_smart(str(p)))
    assert result.status == ToolStatus.OK
    assert result.metadata_only is False
    assert result.large is False
    assert "hello" in result.content


def test_result_filter_summarises_above_threshold() -> None:
    cache = ToolResultCache()

    async def fake_tool() -> ShellToolResult:
        return ShellToolResult(stdout="x " * 5000)

    setattr(fake_tool, "approval", getattr(read_file_smart, "approval"))

    async def summariser(_content: str, _hint: str) -> str:
        return "summarised"

    wrapped = make_result_filter_wrapper(fake_tool, cache, summariser, threshold_tokens=20)
    result = _run(wrapped())
    assert getattr(result, "summarized", False) is True
    handle = result.full_result_handle
    assert cache.fetch(handle) is not None


def test_fetch_full_result_missing_handle_returns_not_found() -> None:
    cache = ToolResultCache()
    fetch = make_fetch_full_result_tool(cache)
    result = _run(fetch("missing"))
    assert result.status == ToolStatus.NOT_FOUND


def test_approval_wrapper_per_tool_timeout() -> None:
    async def slow_tool() -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult()

    setattr(
        slow_tool,
        "approval",
        ToolApprovalMeta(
            category="file_read",
            risk_level=RiskLevel.READ_ONLY,
            reversible=True,
            description_fn=lambda **_: "slow",
            timeout_seconds=0.01,
        ),
    )
    wrapped = make_approval_wrapper(
        slow_tool, EventBus(), ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    )
    result = _run(wrapped())
    assert result.status == ToolStatus.TIMEOUT


def test_spawn_minion_refuses_when_parent_is_minion(tmp_path: Path) -> None:
    async def scenario() -> str:
        state = await open_state_store(tmp_path / "state.db")
        try:
            session = SessionState()
            await state.runs.create(run_id=session.run_id, prompt="p")
            tool = make_spawn_minion_tool(
                state=state,
                settings=Settings(),
                session=session,
                events=EventBus(),
                approval_policy=ApprovalPolicy(mode=ApprovalMode.INTERACTIVE),
                cache=ToolResultCache(),
                summariser=lambda _c, _h: asyncio.sleep(0, result="ok"),
                parent_role="minion:abc",
                parent_depth=1,
                parent_allowed_tools=["filesystem:read"],
            )

            class _Ctx:
                usage = None

            return await tool(_Ctx(), "do thing")
        finally:
            await state.close()

    result = _run(scenario())
    assert "cannot spawn further minions" in result


def test_spawn_minion_emits_events(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> tuple[int, int]:
        state = await open_state_store(tmp_path / "state.db")
        try:
            session = SessionState()
            await state.runs.create(run_id=session.run_id, prompt="p")
            events = EventBus()
            spawned: list[MinionSpawned] = []
            returned: list[MinionReturned] = []
            events.on(MinionSpawned, lambda e: spawned.append(e))
            events.on(MinionReturned, lambda e: returned.append(e))

            class FakeResult:
                output = "ok"

            class FakeAgent:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def run(self, _task, usage=None, usage_limits=None):  # noqa: ARG002
                    return FakeResult()

            async def fake_loader(**_kwargs):
                return FakeAgent()

            monkeypatch.setattr("jac.agents.base.config_loader", fake_loader)
            tool = make_spawn_minion_tool(
                state=state,
                settings=Settings(),
                session=session,
                events=events,
                approval_policy=ApprovalPolicy(mode=ApprovalMode.INTERACTIVE),
                cache=ToolResultCache(),
                summariser=lambda _c, _h: asyncio.sleep(0, result="ok"),
                parent_role="manager",
                parent_depth=0,
                parent_allowed_tools=["filesystem:read"],
            )

            class _Ctx:
                usage = None

            await tool(_Ctx(), "do thing")
            return len(spawned), len(returned)
        finally:
            await state.close()

    spawned, returned = _run(scenario())
    assert spawned == 1
    assert returned == 1


def test_native_agent_extras_attaches_two_tools() -> None:
    extras = native_agent_extras(
        state=object(),
        settings=Settings(),
        session=SessionState(),
        events=EventBus(),
        approval_policy=ApprovalPolicy(mode=ApprovalMode.INTERACTIVE),
        cache=ToolResultCache(),
        summariser=lambda _c, _h: asyncio.sleep(0, result="ok"),
        parent_role="manager",
        parent_depth=0,
        parent_allowed_tools=["filesystem:read"],
    )
    assert len(extras) == 2
    assert hasattr(extras[0], "approval")
    assert hasattr(extras[1], "approval")
