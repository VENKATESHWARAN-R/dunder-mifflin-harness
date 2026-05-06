"""Structured outputs emitted by the planner (Pam)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DevStrategy = Literal["feature_by_feature", "tdd", "spec_driven", "agile"]
Complexity = Literal["simple", "moderate", "complex"]


class PlannedTask(BaseModel):
    title: str = Field(..., description="Short imperative title")
    description: str = Field(
        ..., description="Detail sufficient for Jim to act without Pam"
    )
    acceptance_criteria: str = Field(
        ..., description="Checkable conditions for Dwight"
    )
    complexity: Complexity = "moderate"


class Plan(BaseModel):
    summary: str = Field(..., description="One-paragraph overview of the plan")
    dev_strategy: DevStrategy = "feature_by_feature"
    tasks: list[PlannedTask]
