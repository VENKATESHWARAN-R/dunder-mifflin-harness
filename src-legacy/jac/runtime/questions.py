"""Human question types shared across terminal and future UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class QuestionKind(StrEnum):
    """Kinds of questions the runtime can ask the user."""

    FREE_TEXT = "free-text"
    SINGLE_CHOICE = "single-choice"
    MULTI_CHOICE = "multi-choice"


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """One selectable option in a choice question."""

    id: str
    label: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    """A structured question emitted by the runtime."""

    prompt: str
    kind: QuestionKind = QuestionKind.FREE_TEXT
    options: tuple[ChoiceOption, ...] = ()
    allow_multiple: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if self.kind != QuestionKind.FREE_TEXT and not self.options:
            msg = f"{self.kind.value} questions require at least one option"
            raise ValueError(msg)
        if self.kind == QuestionKind.MULTI_CHOICE and not self.allow_multiple:
            object.__setattr__(self, "allow_multiple", True)


@dataclass(frozen=True, slots=True)
class QuestionResponse:
    """User response to a runtime question."""

    request_id: str
    answer: str | tuple[str, ...]


def validate_response(
    request: QuestionRequest,
    response: QuestionResponse,
) -> None:
    """Validate that a question response matches the request shape."""
    if response.request_id != request.id:
        msg = "question response id does not match request id"
        raise ValueError(msg)

    if request.kind == QuestionKind.FREE_TEXT:
        if not isinstance(response.answer, str):
            msg = "free-text answers must be strings"
            raise ValueError(msg)
        return

    valid_ids = {option.id for option in request.options}

    if request.kind == QuestionKind.SINGLE_CHOICE:
        if not isinstance(response.answer, str) or response.answer not in valid_ids:
            msg = "single-choice answer must be one valid option id"
            raise ValueError(msg)
        return

    if not isinstance(response.answer, tuple):
        msg = "multi-choice answer must be a tuple of option ids"
        raise ValueError(msg)
    invalid = set(response.answer) - valid_ids
    if invalid:
        msg = f"unknown option ids: {', '.join(sorted(invalid))}"
        raise ValueError(msg)
