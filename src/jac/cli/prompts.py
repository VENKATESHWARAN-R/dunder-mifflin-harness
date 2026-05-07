"""Interactive approval and question prompts."""

from __future__ import annotations

from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from jac.runtime.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.runtime.questions import (
    QuestionKind,
    QuestionRequest,
    QuestionResponse,
)

_PROMPT_STYLE = Style.from_dict({"prompt": "bold yellow", "dim": "italic dim"})
_REDIRECT_STYLE = Style.from_dict({"prompt": "bold cyan"})


class PromptViews:
    """Prompt the human for blocking runtime requests."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Render an interactive approval gate with arrow-key navigation."""
        # Rich panel
        body = [request.summary, "", f"risk: {request.risk.value}"]
        for key, value in request.details.items():
            body.append(f"{key}: {value}")
        self.console.print(
            Panel("\n".join(body), title="Approval Required", border_style="yellow")
        )

        # Build options list — redirect is always available
        allowed = set(request.allowed_decisions)
        options: list[tuple[str, str, ApprovalDecision | None]] = []
        if ApprovalDecision.APPROVE_ONCE in allowed:
            options.append(("a", "approve once", ApprovalDecision.APPROVE_ONCE))
        if ApprovalDecision.DENY in allowed:
            options.append(("d", "deny", ApprovalDecision.DENY))
        options.append(("r", "redirect — send feedback to model", None))
        if ApprovalDecision.ALLOW_TOOL_FOR_SESSION in allowed:
            options.append(
                ("t", "allow tool for session", ApprovalDecision.ALLOW_TOOL_FOR_SESSION)
            )
        if ApprovalDecision.ALLOW_EXACT_FOR_SESSION in allowed:
            options.append(
                (
                    "e",
                    "allow exact for session",
                    ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
                )
            )

        # Print static option list
        self.console.print()
        for key, label, _ in options:
            self.console.print(f"  [dim][{key}][/dim] {label}")
        self.console.print()

        # Interactive prompt_toolkit selector
        current: list[int] = [0]
        bindings: KeyBindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("c-p")
        def _up(event: Any) -> None:
            current[0] = (current[0] - 1) % len(options)
            event.app.invalidate()

        @bindings.add("down")
        @bindings.add("c-n")
        def _down(event: Any) -> None:
            current[0] = (current[0] + 1) % len(options)
            event.app.invalidate()

        @bindings.add("enter")
        def _confirm(event: Any) -> None:
            event.app.exit(result=options[current[0]][0])

        @bindings.add("c-c")
        @bindings.add("c-d")
        def _cancel(event: Any) -> None:
            event.app.exit(result="d")

        # Single-key shortcuts — select and confirm without Enter
        for opt_key, _, _ in options:

            @bindings.add(opt_key)
            def _shortcut(event: Any, k: str = opt_key) -> None:
                event.app.exit(result=k)

        def get_message() -> list[tuple[str, str]]:
            _, label, _ = options[current[0]]
            return [
                ("class:prompt", f"▶ {label}"),
                ("class:dim", "  (↑↓ navigate · enter · letter shortcut)"),
            ]

        session: PromptSession[str] = PromptSession(
            message=get_message,
            key_bindings=bindings,
            style=_PROMPT_STYLE,
        )
        try:
            selected_key = await session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            selected_key = "d"

        # Normalise: buffer may be empty if user exited via app.exit(result=key)
        valid_keys = {k for k, _, _ in options}
        if not selected_key or selected_key not in valid_keys:
            selected_key = options[current[0]][0]

        # Handle redirect: collect feedback then return
        if selected_key == "r":
            redirect_msg = await self._ask_redirect()
            if not redirect_msg.strip():
                # Empty feedback → treat as deny
                return ApprovalResponse(
                    request_id=request.id, decision=ApprovalDecision.DENY
                )
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.REDIRECT,
                redirect_message=redirect_msg,
            )

        # Map key to decision
        for opt_key, _, decision in options:
            if opt_key == selected_key and decision is not None:
                return ApprovalResponse(request_id=request.id, decision=decision)

        return ApprovalResponse(request_id=request.id, decision=ApprovalDecision.DENY)

    async def _ask_redirect(self) -> str:
        """Prompt for feedback to route back to the model."""
        self.console.print(
            "[dim]What should the model do instead? "
            "(e.g. 'use tail -20', 'use uv instead of python')[/dim]"
        )
        session: PromptSession[str] = PromptSession(
            message=[("class:prompt", "→ ")],
            style=_REDIRECT_STYLE,
        )
        try:
            return await session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            return ""

    async def ask_yn(self, prompt: str) -> bool:
        """Ask a yes/no question; returns False on interrupt or non-yes input."""
        bindings: KeyBindings = KeyBindings()

        @bindings.add("y")
        @bindings.add("Y")
        def _yes(event: Any) -> None:
            event.app.exit(result="y")

        @bindings.add("n")
        @bindings.add("N")
        @bindings.add("enter")
        @bindings.add("c-c")
        @bindings.add("c-d")
        def _no(event: Any) -> None:
            event.app.exit(result="n")

        session: PromptSession[str] = PromptSession(
            message=[("class:prompt", f"{prompt} [y/N] ")],
            key_bindings=bindings,
            style=Style.from_dict({"prompt": "dim"}),
        )
        try:
            answer = await session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    def ask_question(self, request: QuestionRequest) -> QuestionResponse:
        """Render a human question and return the structured answer."""
        self.console.print(Panel(request.prompt, title="Question", border_style="cyan"))

        if request.kind == QuestionKind.FREE_TEXT:
            try:
                answer = self.console.input("[bold cyan]Answer: [/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                answer = ""
            return QuestionResponse(request_id=request.id, answer=answer)

        for index, option in enumerate(request.options, start=1):
            suffix = f" - {option.description}" if option.description else ""
            self.console.print(
                f"  {index}. {option.label} [dim]({option.id})[/dim]{suffix}"
            )

        prompt = "Choices comma-separated: " if request.allow_multiple else "Choice: "
        try:
            raw = self.console.input(f"[bold cyan]{prompt}[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            raw = ""

        option_ids = [option.id for option in request.options]
        if request.kind == QuestionKind.SINGLE_CHOICE:
            answer = _coerce_choice(raw, option_ids) or ""
            return QuestionResponse(request_id=request.id, answer=answer)

        answers = tuple(
            choice
            for piece in raw.split(",")
            if (choice := _coerce_choice(piece.strip(), option_ids))
        )
        return QuestionResponse(request_id=request.id, answer=answers)


def _coerce_choice(raw: str, option_ids: list[str]) -> str | None:
    if raw in option_ids:
        return raw
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(option_ids):
            return option_ids[index]
    return None
