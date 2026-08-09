from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


FeedbackMethod = Literal[
    "baseline",
    "criterion_decomposed",
    "self_reflective",
    "criterion_decomposed_self_reflective",
    "other",
]

CriterionStatus = Literal[
    "met",
    "partially_met",
    "not_met",
    "missing",
    "not_assessable",
]

FeedbackType = Literal[
    "strength",
    "error",
    "omission",
    "suggestion",
]

FeedbackScope = Literal[
    "overall",
    "part",
    "task",
    "task_group",
    "attempt",
]

SectionScope = Literal[
    "overall",
    "part",
    "task",
    "task_group",
]

VerificationStatus = Literal[
    "verified_by_tool",
    "supported_by_submission",
    "inferred",
    "not_verified",
]


NonEmptyString = Annotated[
    str,
    Field(min_length=1),
]

ConfidenceScore = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
    ),
]


class FeedbackBaseModel(BaseModel):
    """Base configuration shared by all generated-feedback models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class GenerationRecord(FeedbackBaseModel):
    method: FeedbackMethod
    provider: NonEmptyString
    model: NonEmptyString
    prompt_template_id: NonEmptyString
    pipeline_version: NonEmptyString


class SourceRange(FeedbackBaseModel):
    start_line: int = Field(
        ge=1
    )

    end_line: int = Field(
        ge=1
    )

    start_column: int | None = Field(
        default=None,
        ge=1,
    )

    end_column: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_range(
        self,
    ) -> "SourceRange":
        if (
            self.end_line
            < self.start_line
        ):
            raise ValueError(
                "end_line must be greater than "
                "or equal to start_line"
            )

        if (
            self.start_line
            == self.end_line
            and self.start_column
            is not None
            and self.end_column
            is not None
            and self.end_column
            < self.start_column
        ):
            raise ValueError(
                "end_column must be greater than "
                "or equal to start_column when "
                "the range is on one line"
            )

        return self


class SubmissionReferenceEvidence(
    FeedbackBaseModel
):
    evidence_type: Literal[
        "submission_reference"
    ] = "submission_reference"

    artifact_id: NonEmptyString

    unit_id: (
        NonEmptyString
        | None
    ) = None

    block_id: (
        NonEmptyString
        | None
    ) = None

    source_range: (
        SourceRange
        | None
    ) = None

    excerpt: (
        NonEmptyString
        | None
    ) = None

    description: (
        NonEmptyString
        | None
    ) = None

    @model_validator(
        mode="after"
    )
    def validate_reference_location(
        self,
    ) -> (
        "SubmissionReferenceEvidence"
    ):
        if (
            self.unit_id is None
            and self.block_id is None
            and self.source_range
            is None
        ):
            raise ValueError(
                "Submission-reference evidence "
                "must provide at least one of "
                "unit_id, block_id, or "
                "source_range"
            )

        return self


class AbsenceEvidence(
    FeedbackBaseModel
):
    evidence_type: Literal[
        "absence"
    ] = "absence"

    part_id: (
        NonEmptyString
        | None
    ) = None

    task_id: (
        NonEmptyString
        | None
    ) = None

    requirement_id: (
        NonEmptyString
        | None
    ) = None

    description: NonEmptyString

    @model_validator(
        mode="after"
    )
    def validate_absence_scope(
        self,
    ) -> "AbsenceEvidence":
        if (
            self.part_id is None
            and self.task_id is None
            and self.requirement_id
            is None
        ):
            raise ValueError(
                "Absence evidence must provide "
                "at least one of part_id, "
                "task_id, or requirement_id"
            )

        return self


Evidence = Annotated[
    (
        SubmissionReferenceEvidence
        | AbsenceEvidence
    ),
    Field(
        discriminator=(
            "evidence_type"
        )
    ),
]


class CriterionAssessment(
    FeedbackBaseModel
):
    criterion_assessment_id: (
        NonEmptyString
    )

    requirement_id: (
        NonEmptyString
    )

    part_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    task_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    status: CriterionStatus

    internal_finding: (
        NonEmptyString
    )

    verification_status: (
        VerificationStatus
    )

    confidence: (
        ConfidenceScore
    )

    evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )


class FeedbackObservation(
    FeedbackBaseModel
):
    observation_id: (
        NonEmptyString
    )

    feedback_type: (
        FeedbackType
    )

    scope: FeedbackScope

    part_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    task_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    attempt_unit_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    requirement_ids: list[
        NonEmptyString
    ] = Field(
        min_length=1
    )

    criterion_assessment_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    internal_finding: (
        NonEmptyString
    )

    student_feedback: (
        NonEmptyString
    )

    suggestion: (
        NonEmptyString
        | None
    ) = None

    verification_status: (
        VerificationStatus
    )

    confidence: (
        ConfidenceScore
    )

    evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )

    @model_validator(
        mode="after"
    )
    def validate_scope_references(
        self,
    ) -> "FeedbackObservation":
        if (
            self.scope == "part"
            and not self.part_ids
        ):
            raise ValueError(
                "part_ids must contain at "
                "least one identifier when "
                "scope is 'part'"
            )

        if (
            self.scope
            in {
                "task",
                "task_group",
            }
            and not self.task_ids
        ):
            raise ValueError(
                "task_ids must contain at "
                "least one identifier when "
                "scope is 'task' or "
                "'task_group'"
            )

        if (
            self.scope == "attempt"
            and not self.attempt_unit_ids
        ):
            raise ValueError(
                "attempt_unit_ids must "
                "contain at least one "
                "identifier when scope is "
                "'attempt'"
            )

        return self


# ============================================================
# Reflection models
# ============================================================


class ReflectionAnalysis(
    FeedbackBaseModel
):
    """
    Stage-1 audit of an existing requirement-level assessment.

    This stage decides whether the original assessment should be kept or
    revised. It does not generate revised feedback.
    """

    evidence_supported: bool

    unsupported_claims: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    overlooked_evidence: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    missing_rubric_points: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    preferred_solution_bias: bool

    confidence_assessment: Literal[
        "appropriate",
        "too_high",
        "too_low",
    ]

    should_revise: bool

    revision_reason: (
        NonEmptyString
    )

    # Required only when revision is requested.
    # Must be copied verbatim from the exact assessment requirement.
    requirement_basis: (
        NonEmptyString
        | None
    ) = None

    @model_validator(
        mode="after"
    )
    def validate_requirement_basis(
        self,
    ) -> "ReflectionAnalysis":
        if (
            self.should_revise
            and self.requirement_basis
            is None
        ):
            raise ValueError(
                "A revision decision must "
                "provide an exact "
                "requirement_basis."
            )

        if (
            not self.should_revise
            and self.requirement_basis
            is not None
        ):
            raise ValueError(
                "requirement_basis must be "
                "omitted when should_revise "
                "is false."
            )

        return self


class ReflectionAuditOutput(
    FeedbackBaseModel
):
    """
    Stage-1 self-reflection output.

    Only the audit decision is generated here. Revised feedback is generated
    separately only when the audit requests revision.
    """

    analysis: (
        ReflectionAnalysis
    )


class ReflectionRevisionPatch(
    FeedbackBaseModel
):
    """
    Stage-2 reflection output.

    Generated only when Stage 1 concludes that the original assessment
    contains a material defect.

    Python retains ownership of identifiers and structural metadata.
    """

    revised_status: (
        CriterionStatus
    )

    revised_internal_finding: (
        NonEmptyString
    )

    revised_student_feedback: (
        NonEmptyString
    )

    revised_suggestion: (
        NonEmptyString
        | None
    ) = None

    revised_verification_status: (
        VerificationStatus
    )

    revised_confidence: (
        ConfidenceScore
    )

    revised_evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )


class ReflectionDecision(
    FeedbackBaseModel
):
    """
    Combined persisted reflection result.

    Stage 1 supplies the analysis.

    Stage 2 supplies revised fields only when analysis.should_revise is true.

    Python combines the two stages into this stable representation so the
    rest of the feedback pipeline can continue using ReflectionDecision.
    """

    analysis: (
        ReflectionAnalysis
    )

    revised_status: (
        CriterionStatus
        | None
    ) = None

    revised_internal_finding: (
        NonEmptyString
        | None
    ) = None

    revised_student_feedback: (
        NonEmptyString
        | None
    ) = None

    revised_suggestion: (
        NonEmptyString
        | None
    ) = None

    revised_verification_status: (
        VerificationStatus
        | None
    ) = None

    revised_confidence: (
        ConfidenceScore
        | None
    ) = None

    revised_evidence: (
        list[Evidence]
        | None
    ) = None

    @model_validator(
        mode="after"
    )
    def validate_revision_fields(
        self,
    ) -> "ReflectionDecision":
        revision_values = (
            self.revised_status,
            self.revised_internal_finding,
            self.revised_student_feedback,
            self.revised_suggestion,
            self.revised_verification_status,
            self.revised_confidence,
            self.revised_evidence,
        )

        has_revision_value = any(
            value is not None
            for value
            in revision_values
        )

        if (
            not self.analysis
            .should_revise
            and has_revision_value
        ):
            raise ValueError(
                "Revised fields must be "
                "omitted when should_revise "
                "is false."
            )

        if (
            self.analysis
            .should_revise
        ):
            required_revision_values = (
                self.revised_status,
                self.revised_internal_finding,
                self.revised_student_feedback,
                self.revised_verification_status,
                self.revised_confidence,
                self.revised_evidence,
            )

            if any(
                value is None
                for value
                in required_revision_values
            ):
                raise ValueError(
                    "A revision decision must "
                    "provide revised_status, "
                    "revised_internal_finding, "
                    "revised_student_feedback, "
                    "revised_verification_status, "
                    "revised_confidence, and "
                    "revised_evidence."
                )

        return self


# ============================================================
# Final feedback output models
# ============================================================


class FeedbackSection(
    FeedbackBaseModel
):
    section_id: (
        NonEmptyString
    )

    label: (
        NonEmptyString
        | None
    ) = None

    scope: SectionScope

    part_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    task_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    summary: (
        NonEmptyString
    )

    observation_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )

    @model_validator(
        mode="after"
    )
    def validate_scope_references(
        self,
    ) -> "FeedbackSection":
        if (
            self.scope == "part"
            and not self.part_ids
        ):
            raise ValueError(
                "part_ids must contain at "
                "least one identifier when "
                "scope is 'part'"
            )

        if (
            self.scope
            in {
                "task",
                "task_group",
            }
            and not self.task_ids
        ):
            raise ValueError(
                "task_ids must contain at "
                "least one identifier when "
                "scope is 'task' or "
                "'task_group'"
            )

        return self


class OverallFeedback(
    FeedbackBaseModel
):
    summary: NonEmptyString

    strengths_summary: (
        NonEmptyString
    )

    improvement_summary: (
        NonEmptyString
    )

    observation_ids: list[
        NonEmptyString
    ] = Field(
        default_factory=list
    )


class FeedbackLLMOutput(
    FeedbackBaseModel
):
    criterion_assessments: list[
        CriterionAssessment
    ] = Field(
        default_factory=list
    )

    observations: list[
        FeedbackObservation
    ] = Field(
        default_factory=list
    )

    feedback_sections: list[
        FeedbackSection
    ] = Field(
        default_factory=list
    )

    overall_feedback: (
        OverallFeedback
    )

    @model_validator(
        mode="after"
    )
    def validate_identifiers_and_references(
        self,
    ) -> "FeedbackLLMOutput":
        self._validate_unique_ids()
        self._validate_criterion_references()
        self._validate_observation_references()

        return self

    def _validate_unique_ids(
        self,
    ) -> None:
        criterion_ids = [
            assessment
            .criterion_assessment_id
            for assessment
            in self.criterion_assessments
        ]

        observation_ids = [
            observation
            .observation_id
            for observation
            in self.observations
        ]

        section_ids = [
            section.section_id
            for section
            in self.feedback_sections
        ]

        self._raise_if_duplicates(
            criterion_ids,
            "criterion_assessment_id",
        )

        self._raise_if_duplicates(
            observation_ids,
            "observation_id",
        )

        self._raise_if_duplicates(
            section_ids,
            "section_id",
        )

    def _validate_criterion_references(
        self,
    ) -> None:
        valid_criterion_ids = {
            assessment
            .criterion_assessment_id
            for assessment
            in self.criterion_assessments
        }

        for observation in (
            self.observations
        ):
            unknown_ids = (
                set(
                    observation
                    .criterion_assessment_ids
                )
                - valid_criterion_ids
            )

            if unknown_ids:
                unknown = ", ".join(
                    sorted(
                        unknown_ids
                    )
                )

                raise ValueError(
                    "Observation "
                    f"'{observation.observation_id}' "
                    "references unknown criterion "
                    "assessment IDs: "
                    f"{unknown}"
                )

    def _validate_observation_references(
        self,
    ) -> None:
        valid_observation_ids = {
            observation
            .observation_id
            for observation
            in self.observations
        }

        for section in (
            self.feedback_sections
        ):
            unknown_ids = (
                set(
                    section
                    .observation_ids
                )
                - valid_observation_ids
            )

            if unknown_ids:
                unknown = ", ".join(
                    sorted(
                        unknown_ids
                    )
                )

                raise ValueError(
                    "Feedback section "
                    f"'{section.section_id}' "
                    "references unknown "
                    "observation IDs: "
                    f"{unknown}"
                )

        unknown_overall_ids = (
            set(
                self
                .overall_feedback
                .observation_ids
            )
            - valid_observation_ids
        )

        if unknown_overall_ids:
            unknown = ", ".join(
                sorted(
                    unknown_overall_ids
                )
            )

            raise ValueError(
                "overall_feedback references "
                "unknown observation IDs: "
                f"{unknown}"
            )

    @staticmethod
    def _raise_if_duplicates(
        values: list[str],
        field_name: str,
    ) -> None:
        seen: set[str] = set()

        duplicates: set[
            str
        ] = set()

        for value in values:
            if value in seen:
                duplicates.add(
                    value
                )

            seen.add(
                value
            )

        if duplicates:
            duplicate_text = ", ".join(
                sorted(
                    duplicates
                )
            )

            raise ValueError(
                f"Duplicate {field_name} "
                "values found: "
                f"{duplicate_text}"
            )


# ============================================================
# Component-level feedback outputs
# ============================================================


class ComponentCriterionAssessmentLLMOutput(
    FeedbackBaseModel
):
    requirement_id: (
        NonEmptyString
    )

    status: (
        CriterionStatus
    )

    internal_finding: (
        NonEmptyString
    )

    verification_status: (
        VerificationStatus
    )

    confidence: (
        ConfidenceScore
    )

    evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )


class ComponentObservationLLMOutput(
    FeedbackBaseModel
):
    feedback_type: (
        FeedbackType
    )

    requirement_ids: list[
        NonEmptyString
    ] = Field(
        min_length=1
    )

    internal_finding: (
        NonEmptyString
    )

    student_feedback: (
        NonEmptyString
    )

    suggestion: (
        NonEmptyString
        | None
    ) = None

    verification_status: (
        VerificationStatus
    )

    confidence: (
        ConfidenceScore
    )

    evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )


class ComponentFeedbackLLMOutput(
    FeedbackBaseModel
):
    criterion_assessments: list[
        ComponentCriterionAssessmentLLMOutput
    ] = Field(
        default_factory=list
    )

    observations: list[
        ComponentObservationLLMOutput
    ] = Field(
        default_factory=list
    )

    component_summary: (
        NonEmptyString
    )


# ============================================================
# Requirement-level feedback output
# ============================================================


class RequirementFeedbackLLMOutput(
    FeedbackBaseModel
):
    """
    Feedback generated for one known assessment requirement.

    Python controls the requirement ID and all record identifiers.
    """

    status: (
        CriterionStatus
    )

    internal_finding: (
        NonEmptyString
    )

    student_feedback: (
        NonEmptyString
    )

    suggestion: (
        NonEmptyString
        | None
    ) = None

    verification_status: (
        VerificationStatus
    )

    confidence: (
        ConfidenceScore
    )

    evidence: list[
        Evidence
    ] = Field(
        default_factory=list
    )