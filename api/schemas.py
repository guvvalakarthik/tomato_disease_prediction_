from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelIdentity(BaseModel):
    version: str
    sha256: str


class ClassProbability(BaseModel):
    class_id: str
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class Explanation(BaseModel):
    method: str
    png_base64: str


class PredictionResponse(BaseModel):
    request_id: str
    status: Literal["predicted", "uncertain"]
    model: ModelIdentity
    prediction: ClassProbability | None
    top_predictions: list[ClassProbability]
    uncertainty_reason: str | None = None
    explanation: Explanation | None = None
    disclaimer: str


ParticipantRole = Literal["farmer", "agriculture_student", "domain_reviewer", "other"]
IssueTag = Literal[
    "upload_difficult",
    "result_unclear",
    "confidence_misleading",
    "uncertainty_unclear",
    "attention_map_unclear",
    "accessibility_problem",
]


class FeedbackRequest(BaseModel):
    consent: Literal[True]
    participant_role: ParticipantRole
    task_completed: bool
    interpretation_without_help: bool
    uncertainty_understood: bool
    expert_confirmation_intended: bool
    usefulness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    issue_tags: list[IssueTag] = Field(default_factory=list, max_length=6)


class FeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"
    feedback_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    model_version: str | None = None
    model_sha256: str | None = None
    class_count: int | None = None
    detail: str | None = None


class ErrorBody(BaseModel):
    request_id: str
    code: str
    message: str
