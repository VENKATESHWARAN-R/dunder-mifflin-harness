"""Interactive approval and question prompts."""

from __future__ import annotations

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


_APPROVAL_SHORTCUTS = {
    "a": ApprovalDecision.APPROVE_ONCE,
    "approve": ApprovalDecision.APPROVE_ONCE,
    "y": ApprovalDecision.APPROVE_ONCE,
    "yes": ApprovalDecision.APPROVE_ONCE,
    "d": ApprovalDecision.DENY,
    "deny": ApprovalDecision.DENY,
    "n": ApprovalDecision.DENY,
    "no": ApprovalDecision.DENY,
    "t": ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
    "tool": ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
    "e": ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
    "exact": ApprovalDecision.ALLOW_EXACT_FOR_SESSION,
}


class PromptViews:
    """Prompt the human for blocking runtime requests."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def ask_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Render an approval gate and return the selected decision."""
        body = [request.summary, "", f"risk: {request.risk.value}"]
        for key, value in request.details.items():
            body.append(f"{key}: {value}")

        allowed = {item for item in request.allowed_decisions}
        options = ["a=approve once", "d=deny"]
        if ApprovalDecision.ALLOW_TOOL_FOR_SESSION in allowed:
            options.append("t=allow tool for session")
        if ApprovalDecision.ALLOW_EXACT_FOR_SESSION in allowed:
            options.append("e=allow exact action for session")

        self.console.print(
            Panel("\n".join(body), title="Approval Required", border_style="yellow")
        )
        self.console.print("[dim]" + ", ".join(options) + "[/dim]")

        try:
            raw = self.console.input("[bold yellow]Decision: [/bold yellow]")
        except (EOFError, KeyboardInterrupt):
            return ApprovalResponse(
                request_id=request.id,
                decision=ApprovalDecision.DENY,
                reason="prompt interrupted",
            )

        decision = _APPROVAL_SHORTCUTS.get(raw.strip().lower(), ApprovalDecision.DENY)
        if decision not in allowed:
            decision = ApprovalDecision.DENY
        return ApprovalResponse(request_id=request.id, decision=decision)

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
