"""AgentLoop — the core ReAct loop orchestrating the agent.

Ties together the ContentGenerator, ToolRegistry, ConversationHistory,
and EventEmitter to implement a loop where the model generates text or
tool calls, tools are executed, and results are fed back until the model
produces a final text-only response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import types

from pygemini.core.config import Config
from pygemini.core.content_generator import ContentGenerator, FunctionCallData
from pygemini.core.events import (
    CoreEvent,
    ErrorEvent,
    EventEmitter,
    StreamTextEvent,
    ToolExecutingEvent,
    ToolOutputEvent,
    TurnCompleteEvent,
)
from pygemini.core.history import ConversationHistory
from pygemini.core.prompts import build_system_prompt
from pygemini.tools.base import ToolResult
from pygemini.tools.registry import ToolRegistry

__all__ = ["AgentLoop"]

logger = logging.getLogger(__name__)


class AgentLoop:
    """Core ReAct loop that orchestrates model generation and tool execution.

    Parameters
    ----------
    content_generator:
        Wrapper around the Gemini streaming API.
    tool_registry:
        Registry of available tools.
    history:
        Conversation history manager.
    event_emitter:
        Event bridge to the CLI layer.
    config:
        Application configuration.
    context_content:
        Pre-formatted content from GEMINI.md files and memory for the system prompt.
    compressor:
        Optional conversation compressor for auto-compression.
    hook_manager:
        Optional hook manager for lifecycle hooks.
    approval_manager:
        Optional approval manager for tool execution approval.
    policy_engine:
        Optional policy engine for tool execution rules.
    """

    def __init__(
        self,
        content_generator: ContentGenerator,
        tool_registry: ToolRegistry,
        history: ConversationHistory,
        event_emitter: EventEmitter,
        config: Config,
        context_content: str = "",
        compressor: object | None = None,
        hook_manager: object | None = None,
        approval_manager: object | None = None,
        policy_engine: object | None = None,
    ) -> None:
        self._content_generator = content_generator
        self._tool_registry = tool_registry
        self._history = history
        self._event_emitter = event_emitter
        self._config = config
        self._context_content = context_content
        self._compressor = compressor
        self._hook_manager = hook_manager
        self._approval_manager = approval_manager
        self._policy_engine = policy_engine
        self._abort_signal = asyncio.Event()

    @property
    def abort_signal(self) -> asyncio.Event:
        """Event that can be set externally to abort the current turn."""
        return self._abort_signal

    def set_context_content(self, content: str) -> None:
        """Update the context content injected into the system prompt."""
        self._context_content = content

    async def _maybe_compress(self) -> None:
        """Auto-compress history if a compressor is configured and threshold is hit."""
        if self._compressor is None:
            return
        try:
            should = self._compressor.should_compress(self._history)  # type: ignore[union-attr]
            if not should:
                return
            logger.info("Auto-compressing conversation history...")
            result = await self._compressor.compress(self._history)  # type: ignore[union-attr]
            keep_recent = getattr(self._compressor, "_keep_recent", 10)
            msg_count = self._history.message_count
            compress_end = msg_count - keep_recent
            if compress_end > 0:
                from google.genai import types as genai_types

                summary_content = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(
                        text=f"[Conversation summary]\n{result.summary}"  # type: ignore[union-attr]
                    )],
                )
                self._history.replace_messages(0, compress_end, [summary_content])
                logger.info(
                    "Compressed %d messages into summary (%d tokens est.)",
                    result.original_message_count,  # type: ignore[union-attr]
                    result.compressed_token_estimate,  # type: ignore[union-attr]
                )
        except Exception as exc:
            logger.warning("Auto-compression failed: %s", exc)

    async def _emit_hook(self, event_name: str, data: dict[str, Any] | None = None) -> bool:
        """Fire lifecycle hooks. Returns True if operation should be blocked."""
        if self._hook_manager is None:
            return False
        try:
            from pygemini.hooks.manager import HookEvent

            event = HookEvent(event_name)
            results = await self._hook_manager.emit(event, data)  # type: ignore[union-attr]
            return any(getattr(r, "should_block", False) for r in results)
        except Exception as exc:
            logger.warning("Hook emission failed for %s: %s", event_name, exc)
            return False

    def _check_policy(self, tool_name: str, params: dict[str, Any]) -> str:
        """Check policy engine for a tool call. Returns action: allow/deny/confirm."""
        if self._policy_engine is None:
            return "confirm"
        try:
            decision = self._policy_engine.evaluate(tool_name, params)  # type: ignore[union-attr]
            return decision.action  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Policy evaluation failed for %s: %s", tool_name, exc)
            return "confirm"

    def _check_approval(self, tool_name: str, confirmation: object | None) -> bool:
        """Check if a tool needs user confirmation via approval manager.

        Returns True if confirmation is needed.
        """
        if self._approval_manager is None:
            return confirmation is not None
        try:
            return self._approval_manager.needs_confirmation(tool_name, confirmation)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Approval check failed for %s: %s", tool_name, exc)
            return confirmation is not None

    async def run(self, user_message: str) -> None:
        """Execute one full agent turn.

        Adds the user message to history, then enters the ReAct loop:
        generate a model response, execute any tool calls, feed results
        back, and repeat until the model produces a text-only response.

        Parameters
        ----------
        user_message:
            The user's input text for this turn.
        """
        self._abort_signal.clear()

        # Step 1: Add user message to history
        self._history.add_user_message(user_message)

        # Step 1b: Auto-compress if needed
        await self._maybe_compress()

        # Step 2: Build system prompt (with GEMINI.md context)
        tool_names = [t.name for t in self._tool_registry.get_all()]
        system_prompt = build_system_prompt(
            context_content=self._context_content,
            tool_names=tool_names,
        )

        # Emit before_agent hook
        await self._emit_hook("before_agent", {"user_message": user_message})

        try:
            # Step 3: Enter the ReAct loop
            while not self._abort_signal.is_set():
                # 3a: Get tool declarations
                declarations = self._tool_registry.get_filtered_declarations()

                # 3b: Format tools for API
                tools: list[dict[str, Any]] = []
                if declarations:
                    tools = [{"function_declarations": declarations}]

                # Emit before_model hook
                await self._emit_hook("before_model")

                # 3c: Stream model response
                text_parts: list[str] = []
                function_calls: list[FunctionCallData] = []
                model_parts: list[types.Part] = []

                async for chunk in self._content_generator.generate_stream(
                    self._history.get_messages(), tools, system_prompt
                ):
                    if self._abort_signal.is_set():
                        break

                    # 3d: Collect text chunks and function calls
                    if chunk.text is not None:
                        await self._event_emitter.emit(
                            CoreEvent.STREAM_TEXT,
                            StreamTextEvent(text=chunk.text),
                        )
                        text_parts.append(chunk.text)

                    if chunk.function_calls:
                        function_calls.extend(chunk.function_calls)

                # Emit after_model hook
                await self._emit_hook("after_model")

                if self._abort_signal.is_set():
                    break

                # 3e: Build model response parts and add to history
                if text_parts:
                    full_text = "".join(text_parts)
                    model_parts.append(types.Part.from_text(text=full_text))

                for fc in function_calls:
                    model_parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=fc.name,
                                args=fc.args,
                            )
                        )
                    )

                if model_parts:
                    self._history.add_model_response(model_parts)

                # 3f: If no function calls, turn is done
                if not function_calls:
                    break

                # 3g: Execute each function call
                function_responses: list[dict[str, Any]] = []

                for fc in function_calls:
                    if self._abort_signal.is_set():
                        break

                    response = await self._execute_function_call(fc)
                    function_responses.append(response)

                if self._abort_signal.is_set():
                    break

                # 3h: Add tool results to history and continue loop
                self._history.add_tool_results(function_calls, function_responses)

        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            await self._event_emitter.emit(
                CoreEvent.ERROR,
                ErrorEvent(message=str(exc), exception=exc),
            )

        # Emit after_agent hook
        await self._emit_hook("after_agent")

        # Step 4: Emit turn complete
        await self._event_emitter.emit(
            CoreEvent.TURN_COMPLETE, TurnCompleteEvent()
        )

    async def _execute_function_call(
        self, fc: FunctionCallData
    ) -> dict[str, Any]:
        """Execute a single function call and return the response dict.

        Integrates policy engine, approval manager, and hooks.

        Returns a dict of the form::

            {"name": fc.name, "response": {"result": ...}}
        """
        # Look up tool
        tool = self._tool_registry.get(fc.name)
        if tool is None:
            error_msg = f"Tool '{fc.name}' not found"
            logger.warning(error_msg)
            return {"name": fc.name, "response": {"result": error_msg}}

        # Validate params
        validation_error = tool.validate_params(fc.args)
        if validation_error is not None:
            logger.warning(
                "Param validation failed for %s: %s", fc.name, validation_error
            )
            return {"name": fc.name, "response": {"result": validation_error}}

        # Check policy engine
        policy_action = self._check_policy(fc.name, fc.args)
        if policy_action == "deny":
            denial_msg = "Tool execution denied by policy"
            logger.info("Policy denied %s", fc.name)
            return {"name": fc.name, "response": {"result": denial_msg}}

        # Emit before_tool hook
        blocked = await self._emit_hook("before_tool", {
            "tool_name": fc.name, "params": fc.args,
        })
        if blocked:
            return {"name": fc.name, "response": {"result": "Blocked by hook"}}

        # Check confirmation (policy may override to allow/confirm)
        confirmation = tool.should_confirm(fc.args)
        needs_confirm = (
            policy_action == "confirm"
            and self._check_approval(fc.name, confirmation)
        )

        if needs_confirm and confirmation is not None:
            await self._event_emitter.emit(
                CoreEvent.TOOL_EXECUTING,
                ToolExecutingEvent(tool_name=fc.name, params=fc.args),
            )
            approved = await self._event_emitter.request_confirmation(
                description=confirmation.description,
                details=confirmation.details,
            )
            if not approved:
                denial_msg = "User denied execution"
                return {"name": fc.name, "response": {"result": denial_msg}}

        # Execute
        try:
            result: ToolResult = await tool.execute(fc.args, self._abort_signal)
        except Exception as exc:
            logger.error("Tool %s raised: %s", fc.name, exc, exc_info=True)
            result = ToolResult(
                llm_content=f"Error executing {fc.name}: {exc}",
                display_content=f"Error: {exc}",
                is_error=True,
            )

        # Emit after_tool hook
        await self._emit_hook("after_tool", {
            "tool_name": fc.name,
            "params": fc.args,
            "result": result.llm_content,
            "is_error": result.is_error,
        })

        # Emit tool output
        await self._event_emitter.emit(
            CoreEvent.TOOL_OUTPUT,
            ToolOutputEvent(
                display_content=result.display_content,
                is_error=result.is_error,
            ),
        )

        return {"name": fc.name, "response": {"result": result.llm_content}}
