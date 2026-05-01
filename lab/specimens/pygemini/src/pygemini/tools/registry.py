"""ToolRegistry — central registry for tool discovery and dispatch.

Manages registration, lookup, and filtered export of tool function
declarations for the Gemini API.  The AgentLoop uses this to resolve
tool calls by name, and to build the ``function_declarations`` list
sent with each request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pygemini.tools.base import BaseTool

if TYPE_CHECKING:
    from pygemini.core.config import Config

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry that holds tool instances and exposes them to the agent loop.

    Args:
        config: Application configuration used for filtering which tools
            are included in declarations sent to the model.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered"
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        del self._tools[name]  # raises KeyError if missing
        logger.debug("Unregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under *name*, or ``None``."""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Declarations for the Gemini API
    # ------------------------------------------------------------------

    def get_function_declarations(self) -> list[dict]:
        """Return function declarations for every registered tool."""
        return [tool.to_function_declaration() for tool in self._tools.values()]

    def get_filtered_declarations(
        self, exclude: set[str] | None = None
    ) -> list[dict]:
        """Return function declarations filtered by config and *exclude*.

        Filtering logic (applied in order):
        1. If ``config.core_tools`` is set, only include tools whose name
           appears in that list.
        2. Remove any tool whose name appears in ``config.excluded_tools``.
        3. Remove any tool whose name appears in *exclude*.
        """
        excluded: set[str] = set(self._config.excluded_tools)
        if exclude:
            excluded |= exclude

        core: set[str] | None = (
            set(self._config.core_tools) if self._config.core_tools else None
        )

        declarations: list[dict] = []
        for tool in self._tools.values():
            if core is not None and tool.name not in core:
                continue
            if tool.name in excluded:
                continue
            declarations.append(tool.to_function_declaration())

        return declarations

    # ------------------------------------------------------------------
    # Default tools
    # ------------------------------------------------------------------

    def register_defaults(
        self,
        event_emitter: object | None = None,
        memory_store: object | None = None,
    ) -> None:
        """Register all built-in tools.

        Args:
            event_emitter: EventEmitter instance for tools that need user
                interaction (e.g., ask_user).
            memory_store: MemoryStore instance for the save_memory tool.
        """
        from pygemini.tools.filesystem import get_filesystem_tools

        for tool in get_filesystem_tools():
            self.register(tool)

        # Shell tool
        from pygemini.tools.shell import ShellTool

        self.register(ShellTool())

        # Web tools
        from pygemini.tools.web_fetch import WebFetchTool

        self.register(WebFetchTool())

        from pygemini.tools.web_search import WebSearchTool

        self.register(WebSearchTool(api_key=self._config.api_key))

        # Ask user tool (requires event emitter)
        if event_emitter is not None:
            from pygemini.tools.ask_user import AskUserTool

            self.register(AskUserTool(event_emitter=event_emitter))  # type: ignore[arg-type]

        # Memory tool (requires memory store)
        if memory_store is not None:
            from pygemini.tools.memory import SaveMemoryTool

            self.register(SaveMemoryTool(memory_store=memory_store))  # type: ignore[arg-type]

        logger.debug("Registered %d default tools", len(self._tools))
