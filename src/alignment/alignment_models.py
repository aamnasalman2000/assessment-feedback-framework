from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyString = Annotated[str, Field(min_length=1)]
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]


class AlignmentBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class ComponentAlignment(AlignmentBaseModel):
    component_id: NonEmptyString

    primary_unit_ids: list[NonEmptyString] = Field(
        default_factory=list
    )
    supporting_unit_ids: list[NonEmptyString] = Field(
        default_factory=list
    )
    possibly_relevant_unit_ids: list[NonEmptyString] = Field(
        default_factory=list
    )

    reason: NonEmptyString
    confidence: ConfidenceScore

    unresolved: bool = False
    unresolved_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_alignment_state(self) -> ComponentAlignment:
        all_unit_ids = (
            self.primary_unit_ids
            + self.supporting_unit_ids
            + self.possibly_relevant_unit_ids
        )

        if len(all_unit_ids) != len(set(all_unit_ids)):
            raise ValueError(
                "A unit ID may appear in only one alignment category "
                "for a component."
            )

        has_aligned_units = bool(all_unit_ids)

        if has_aligned_units and self.unresolved:
            raise ValueError(
                "A component with aligned units cannot also be marked "
                "unresolved."
            )

        if not has_aligned_units and not self.unresolved:
            raise ValueError(
                "A component with no aligned units must be marked "
                "unresolved."
            )

        if self.unresolved and self.unresolved_reason is None:
            raise ValueError(
                "unresolved_reason is required when unresolved is true."
            )

        return self


class AlignmentLLMOutput(AlignmentBaseModel):
    component_alignments: list[ComponentAlignment] = Field(
        default_factory=list
    )

    summary: NonEmptyString
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_components(self) -> AlignmentLLMOutput:
        component_ids = [
            alignment.component_id
            for alignment in self.component_alignments
        ]

        if len(component_ids) != len(set(component_ids)):
            raise ValueError(
                "Duplicate component IDs are not allowed."
            )

        return self


class AlignmentResult(AlignmentLLMOutput):
    schema_version: NonEmptyString = "1.0"

    assessment_id: NonEmptyString
    source_submission_id: NonEmptyString
    processed_submission_id: NonEmptyString

    model_name: NonEmptyString
    aligner_version: NonEmptyString