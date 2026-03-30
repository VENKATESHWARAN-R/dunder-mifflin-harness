"""Unit tests for pygemini.tools.registry.ToolRegistry.

Covers:
- register and get
- register duplicate raises ValueError
- unregister
- unregister missing raises KeyError
- get_function_declarations returns correct format
- get_filtered_declarations with excluded tools and core_tools filtering
- get_all
"""

from __future__ import annotations

import asyncio

import pytest

from pygemini.core.config import Config
from pygemini.tools.base import BaseTool, ToolResult
from pygemini.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test tool fixtures
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str = "A test tool.") -> BaseTool:
    """Factory that returns a minimal concrete BaseTool with the given name."""

    class _Tool(BaseTool):
        @property
        def name(self) -> str:
            return name

        @property
        def description(self) -> str:
            return description

        @property
        def parameter_schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
            }

        async def execute(
            self, params: dict, abort_signal: asyncio.Event | None = None
        ) -> ToolResult:
            return ToolResult(llm_content=f"{name} executed")

    return _Tool()


def _make_registry(
    *,
    core_tools: list[str] | None = None,
    excluded_tools: list[str] | None = None,
) -> ToolRegistry:
    """Return a ToolRegistry backed by a Config with the given settings."""
    cfg = Config(
        core_tools=core_tools if core_tools is not None else [],
        excluded_tools=excluded_tools if excluded_tools is not None else [],
    )
    return ToolRegistry(cfg)


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_and_get(self) -> None:
        registry = _make_registry()
        tool = _make_tool("alpha")
        registry.register(tool)
        assert registry.get("alpha") is tool

    def test_get_returns_none_for_unknown(self) -> None:
        registry = _make_registry()
        assert registry.get("nonexistent") is None

    def test_register_multiple_tools(self) -> None:
        registry = _make_registry()
        t1, t2, t3 = _make_tool("t1"), _make_tool("t2"), _make_tool("t3")
        registry.register(t1)
        registry.register(t2)
        registry.register(t3)
        assert registry.get("t1") is t1
        assert registry.get("t2") is t2
        assert registry.get("t3") is t3

    def test_register_duplicate_raises_value_error(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("dup"))
        with pytest.raises(ValueError, match="dup"):
            registry.register(_make_tool("dup"))

    def test_register_duplicate_does_not_overwrite(self) -> None:
        registry = _make_registry()
        original = _make_tool("orig")
        registry.register(original)
        try:
            registry.register(_make_tool("orig"))
        except ValueError:
            pass
        assert registry.get("orig") is original


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


class TestUnregister:
    def test_unregister_removes_tool(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("beta"))
        registry.unregister("beta")
        assert registry.get("beta") is None

    def test_unregister_missing_raises_key_error(self) -> None:
        registry = _make_registry()
        with pytest.raises(KeyError):
            registry.unregister("ghost")

    def test_can_re_register_after_unregister(self) -> None:
        registry = _make_registry()
        tool = _make_tool("reuse")
        registry.register(tool)
        registry.unregister("reuse")
        new_tool = _make_tool("reuse")
        registry.register(new_tool)
        assert registry.get("reuse") is new_tool

    def test_unregister_does_not_affect_others(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        registry.unregister("a")
        assert registry.get("b") is not None
        assert registry.get("a") is None


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    def test_empty_registry(self) -> None:
        registry = _make_registry()
        assert registry.get_all() == []

    def test_returns_all_registered(self) -> None:
        registry = _make_registry()
        tools = [_make_tool(f"tool_{i}") for i in range(3)]
        for t in tools:
            registry.register(t)
        result = registry.get_all()
        assert len(result) == 3
        assert set(t.name for t in result) == {"tool_0", "tool_1", "tool_2"}

    def test_returns_list(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("x"))
        assert isinstance(registry.get_all(), list)

    def test_reflects_unregister(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("keep"))
        registry.register(_make_tool("drop"))
        registry.unregister("drop")
        names = [t.name for t in registry.get_all()]
        assert "keep" in names
        assert "drop" not in names


# ---------------------------------------------------------------------------
# get_function_declarations
# ---------------------------------------------------------------------------


class TestGetFunctionDeclarations:
    def test_empty_registry(self) -> None:
        registry = _make_registry()
        assert registry.get_function_declarations() == []

    def test_returns_list_of_dicts(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("foo"))
        decls = registry.get_function_declarations()
        assert isinstance(decls, list)
        assert all(isinstance(d, dict) for d in decls)

    def test_declaration_keys(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("my_tool", description="Does things."))
        decl = registry.get_function_declarations()[0]
        assert set(decl.keys()) == {"name", "description", "parameters"}

    def test_declaration_name_and_description(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("named_tool", description="Special desc."))
        decl = registry.get_function_declarations()[0]
        assert decl["name"] == "named_tool"
        assert decl["description"] == "Special desc."

    def test_declaration_parameters(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("p_tool"))
        decl = registry.get_function_declarations()[0]
        assert decl["parameters"]["type"] == "object"

    def test_all_tools_included(self) -> None:
        registry = _make_registry()
        for i in range(4):
            registry.register(_make_tool(f"tool_{i}"))
        decls = registry.get_function_declarations()
        assert len(decls) == 4
        names = {d["name"] for d in decls}
        assert names == {"tool_0", "tool_1", "tool_2", "tool_3"}


# ---------------------------------------------------------------------------
# get_filtered_declarations
# ---------------------------------------------------------------------------


class TestGetFilteredDeclarations:
    def test_no_filters_returns_all(self) -> None:
        """When core_tools is empty and no excludes, all tools are returned."""
        registry = _make_registry(core_tools=[], excluded_tools=[])
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        decls = registry.get_filtered_declarations()
        assert {d["name"] for d in decls} == {"a", "b"}

    def test_core_tools_filter_includes_only_listed(self) -> None:
        registry = _make_registry(core_tools=["a", "c"])
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        registry.register(_make_tool("c"))
        decls = registry.get_filtered_declarations()
        names = {d["name"] for d in decls}
        assert names == {"a", "c"}
        assert "b" not in names

    def test_excluded_tools_config_removes_tool(self) -> None:
        registry = _make_registry(excluded_tools=["danger"])
        registry.register(_make_tool("safe"))
        registry.register(_make_tool("danger"))
        decls = registry.get_filtered_declarations()
        names = {d["name"] for d in decls}
        assert "safe" in names
        assert "danger" not in names

    def test_exclude_argument_removes_tool(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("keep"))
        registry.register(_make_tool("skip"))
        decls = registry.get_filtered_declarations(exclude={"skip"})
        names = {d["name"] for d in decls}
        assert "keep" in names
        assert "skip" not in names

    def test_exclude_argument_merges_with_config_excluded(self) -> None:
        registry = _make_registry(excluded_tools=["config_excluded"])
        registry.register(_make_tool("config_excluded"))
        registry.register(_make_tool("arg_excluded"))
        registry.register(_make_tool("allowed"))
        decls = registry.get_filtered_declarations(exclude={"arg_excluded"})
        names = {d["name"] for d in decls}
        assert names == {"allowed"}

    def test_core_tools_and_exclude_combined(self) -> None:
        registry = _make_registry(core_tools=["a", "b", "c"])
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        registry.register(_make_tool("c"))
        registry.register(_make_tool("d"))  # not in core_tools
        decls = registry.get_filtered_declarations(exclude={"b"})
        names = {d["name"] for d in decls}
        assert names == {"a", "c"}

    def test_empty_exclude_set(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("x"))
        decls = registry.get_filtered_declarations(exclude=set())
        assert len(decls) == 1

    def test_none_exclude_argument(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("y"))
        decls = registry.get_filtered_declarations(exclude=None)
        assert len(decls) == 1

    def test_core_tools_not_in_registry_excluded_gracefully(self) -> None:
        """core_tools lists a tool name that was never registered — no error."""
        registry = _make_registry(core_tools=["registered", "phantom"])
        registry.register(_make_tool("registered"))
        decls = registry.get_filtered_declarations()
        names = {d["name"] for d in decls}
        assert names == {"registered"}

    def test_returns_list(self) -> None:
        registry = _make_registry()
        registry.register(_make_tool("z"))
        assert isinstance(registry.get_filtered_declarations(), list)

    def test_empty_registry_returns_empty(self) -> None:
        registry = _make_registry(core_tools=["anything"])
        assert registry.get_filtered_declarations() == []
