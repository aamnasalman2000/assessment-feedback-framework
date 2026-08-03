from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.extraction.semantic_models import SemanticExtractionResult


class FeedbackInputBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class InputSourceRange(FeedbackInputBaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_column: int | None = Field(default=None, ge=1)
    end_column: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "InputSourceRange":
        if self.end_line < self.start_line:
            raise ValueError(
                "end_line must be greater than or equal to start_line"
            )

        if (
            self.start_line == self.end_line
            and self.start_column is not None
            and self.end_column is not None
            and self.end_column < self.start_column
        ):
            raise ValueError(
                "end_column must be greater than or equal to "
                "start_column"
            )

        return self


class ProcessingToolRecord(FeedbackInputBaseModel):
    name: str = Field(min_length=1)
    version: str | None = None
    purpose: str | None = None


class ProcessingCheck(FeedbackInputBaseModel):
    check_type: str = Field(min_length=1)
    status: Literal[
        "passed",
        "failed",
        "partial",
        "not_run",
        "not_applicable",
    ]
    tool: str | None = None
    summary: str | None = None


class ProcessingDiagnostic(FeedbackInputBaseModel):
    diagnostic_id: str | None = None
    severity: Literal["info", "warning", "error"]
    diagnostic_type: str | None = None
    message: str = Field(min_length=1)
    source_range: InputSourceRange | None = None
    related_unit_ids: list[str] = Field(default_factory=list)


class ProcessingRecord(FeedbackInputBaseModel):
    status: Literal["success", "partial", "failed"]
    extraction_mode: Literal["parser", "textual", "hybrid", "manual"]
    detected_content_type: str | None = None
    tools: list[ProcessingToolRecord] = Field(default_factory=list)
    checks: list[ProcessingCheck] = Field(default_factory=list)
    diagnostics: list[ProcessingDiagnostic] = Field(default_factory=list)


class TaskMapping(FeedbackInputBaseModel):
    part_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    source: Literal["declared", "inferred", "mixed"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mapping(self) -> "TaskMapping":
        if not self.part_ids and not self.task_ids:
            raise ValueError(
                "Task mapping must contain at least one part ID or task ID"
            )

        return self


class AttemptRecord(FeedbackInputBaseModel):
    attempt_index: int = Field(ge=1)
    attempt_group_id: str | None = None
    declared_as_alternative: bool | None = None


class ContentBlock(FeedbackInputBaseModel):
    block_id: str | None = None
    block_type: str = Field(min_length=1)
    content: str
    language: str | None = None
    source_range: InputSourceRange | None = None


class ProcessedUnit(FeedbackInputBaseModel):
    unit_id: str = Field(min_length=1)
    unit_type: str = Field(min_length=1)
    label: str | None = None
    task_mapping: TaskMapping | None = None
    attempt: AttemptRecord | None = None
    source_range: InputSourceRange | None = None
    parent_unit_id: str | None = None
    context_unit_ids: list[str] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(min_length=1)
    structured_data: dict[str, Any] | None = None
    extraction_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ProcessedArtifact(FeedbackInputBaseModel):
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    processing: ProcessingRecord
    units: list[ProcessedUnit] = Field(default_factory=list)


class CrossArtifactLink(FeedbackInputBaseModel):
    link_id: str = Field(min_length=1)
    link_type: str = Field(min_length=1)
    source_artifact_id: str = Field(min_length=1)
    source_unit_id: str | None = None
    target_artifact_id: str = Field(min_length=1)
    target_unit_id: str | None = None
    description: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ProcessedSubmission(FeedbackInputBaseModel):
    schema_version: Literal["1.0"]
    processed_submission_id: str = Field(min_length=1)
    source_submission_id: str = Field(min_length=1)
    student_id: str = Field(min_length=1)
    assessment_id: str = Field(min_length=1)
    processing_version: str | None = None
    artifacts: list[ProcessedArtifact] = Field(min_length=1)
    cross_artifact_links: list[CrossArtifactLink] = Field(
        default_factory=list
    )


class FeedbackGenerationInput(FeedbackInputBaseModel):
    processed_submission: ProcessedSubmission
    semantic_extraction: SemanticExtractionResult
    assessment_specification: dict[str, Any]

    @model_validator(mode="after")
    def validate_identifiers(self) -> "FeedbackGenerationInput":
        processed = self.processed_submission
        semantic = self.semantic_extraction

        if semantic.processed_submission_id != processed.processed_submission_id:
            raise ValueError(
                "Semantic extraction processed_submission_id does not "
                "match the processed submission"
            )

        if semantic.source_submission_id != processed.source_submission_id:
            raise ValueError(
                "Semantic extraction source_submission_id does not match "
                "the processed submission"
            )

        specification_assessment_id = self.assessment_specification.get(
            "assessment_id"
        )

        if specification_assessment_id != processed.assessment_id:
            raise ValueError(
                "Assessment specification assessment_id does not match "
                "the processed submission"
            )

        return self