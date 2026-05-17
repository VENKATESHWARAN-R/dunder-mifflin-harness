import asyncio
from dataclasses import fields

from jac.runtime.approvals import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
)
from jac.runtime.events import (
    ApprovalRequested,
    EventBus,
    SessionConfigChanged,
)
from jac.runtime.questions import (
    ChoiceOption,
    QuestionKind,
    QuestionRequest,
    QuestionResponse,
    validate_response,
)


def test_session_config_changed_dataclass_shape() -> None:
    names = {f.name for f in fields(SessionConfigChanged)}
    assert names == {"key", "old_value", "new_value"}


def test_session_config_changed_has_contract_fields() -> None:
    event = SessionConfigChanged(key="tier", old_value=None, new_value="scout")
    assert event.key == "tier"
    assert event.old_value is None
    assert event.new_value == "scout"


def test_event_bus_approval_handshake() -> None:
    async def scenario() -> ApprovalResponse:
        bus = EventBus()
        request = ApprovalRequest(summary="Run pytest")

        async def approve(event: ApprovalRequested) -> None:
            await bus.resolve_approval(
                ApprovalResponse(
                    request_id=event.request.id,
                    decision=ApprovalDecision.APPROVE_ONCE,
                )
            )

        bus.on(ApprovalRequested, approve)
        return await bus.request_approval(request)

    response = asyncio.run(scenario())

    assert response.approved


def test_approval_policy_records_session_allowance() -> None:
    policy = ApprovalPolicy(mode=ApprovalMode.INTERACTIVE)
    request = ApprovalRequest(
        summary="Run shell",
        tool_name="shell",
        allowed_decisions=(
            ApprovalDecision.APPROVE_ONCE,
            ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
        ),
    )
    response = ApprovalResponse(
        request_id=request.id,
        decision=ApprovalDecision.ALLOW_TOOL_FOR_SESSION,
    )

    policy.record_response(request, response)

    auto_response = policy.auto_response_for(request)
    assert auto_response is not None
    assert auto_response.approved


def test_question_response_validation_for_multi_choice() -> None:
    request = QuestionRequest(
        prompt="Pick models",
        kind=QuestionKind.MULTI_CHOICE,
        options=(
            ChoiceOption(id="scout", label="Scout"),
            ChoiceOption(id="worker", label="Worker"),
        ),
    )
    response = QuestionResponse(request_id=request.id, answer=("scout", "worker"))

    validate_response(request, response)
