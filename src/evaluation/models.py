from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


FeedbackType = Literal[
    "strength",
    "error",
    "omission",
    "suggestion",
]


ScopeAlignment = Literal[
    "exact",
    "human_broader",
    "generated_broader",
    "compatible",
    "unknown",
]


FeedbackTypeCompatibility = Literal[
    "compatible",
    "potentially_compatible",
    "contradictory",
]


RequirementAlignmentStatus = Literal[
    "exact",
    "overlap",
    "no_overlap",
    "not_assessable",
]


HumanObservationStatus = Literal[
    "matched",
    "partially_matched",
    "not_matched",
    "excluded",
]


GeneratedMappingStatus = Literal[
    "matched",
    "partially_matched",
    "no_mapping",
]


GeneratedObservationClassification = Literal[
    "human_aligned",
    "additional_valid",
    "unsupported",
    "contradicted",
    "not_assessable",
    "unresolved",
]


GroundingStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "not_assessable",
    "unresolved",
]


EvidenceCoverage = Literal[
    "sufficient",
    "partial",
    "insufficient",
    "none",
    "not_applicable",
    "unresolved",
]


RubricAlignmentStatus = Literal[
    "aligned",
    "partially_aligned",
    "not_aligned",
    "not_applicable",
    "unresolved",
]


CoverageBasis = Literal[
    "explicit_generated_mapping",
    "matched_human_scope_propagation",
    "both",
    "not_covered",
]


SummaryComparisonType = Literal[
    "overall_summary",
    "strengths_summary",
    "improvement_summary",
]


SummaryComparisonStatus = Literal[
    "compared",
    "not_available",
]


DiagnosticSeverity = Literal[
    "info",
    "warning",
    "error",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class EmbeddingConfiguration(
    StrictBaseModel
):
    provider: Literal[
        "sentence-transformers"
    ] = "sentence-transformers"

    model: Literal[
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    ] = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    model_revision: str | None = None

    similarity_metric: Literal[
        "cosine"
    ] = "cosine"


class MatchingConfiguration(
    StrictBaseModel
):
    scope_filtering_enabled: bool = True

    requirement_overlap_mode: Literal[
        "supporting_only"
    ] = "supporting_only"

    feedback_type_mode: Literal[
        "compatibility_safeguard"
    ] = "compatibility_safeguard"

    full_match_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    partial_match_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    partial_match_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    exclude_generic_overall_observations: (
        bool
    ) = True

    excluded_human_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    @model_validator(
        mode="after"
    )
    def validate_threshold_order(
        self,
    ) -> MatchingConfiguration:
        if (
            self.partial_match_threshold
            >= self.full_match_threshold
        ):
            raise ValueError(
                "partial_match_threshold "
                "must be lower than "
                "full_match_threshold."
            )

        return self


class GroundingConfiguration(
    StrictBaseModel
):
    enabled: bool = True

    ground_unmatched_generated_observations: (
        bool
    ) = True

    distinguish_contradicted_from_unsupported: (
        bool
    ) = True

    allow_not_assessable: bool = True


class EvaluatorRecord(
    StrictBaseModel
):
    evaluation_method: Literal[
        "deterministic_semantic_"
        "structural_evidence_based"
    ] = (
        "deterministic_semantic_"
        "structural_evidence_based"
    )

    pipeline_version: str = Field(
        min_length=1,
    )

    embedding: EmbeddingConfiguration

    matching_configuration: (
        MatchingConfiguration
    )

    grounding_configuration: (
        GroundingConfiguration
    )


class RequirementAlignment(
    StrictBaseModel
):
    status: RequirementAlignmentStatus

    human_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    generated_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    shared_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    jaccard_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(
        mode="after"
    )
    def validate_requirement_alignment(
        self,
    ) -> RequirementAlignment:
        if (
            self.status
            == "not_assessable"
        ):
            if (
                self.jaccard_similarity
                is not None
            ):
                raise ValueError(
                    "jaccard_similarity must be "
                    "None when requirement "
                    "alignment is not_assessable."
                )

        return self


class ObservationMatch(
    StrictBaseModel
):
    match_id: str = Field(
        min_length=1,
    )

    human_observation_id: str = Field(
        min_length=1,
    )

    generated_observation_id: str = Field(
        min_length=1,
    )

    decision: Literal[
        "matched",
        "partial_match",
    ]

    semantic_similarity: float = Field(
        ge=0.0,
        le=1.0,
    )

    scope_alignment: ScopeAlignment

    feedback_type_compatibility: (
        FeedbackTypeCompatibility
    )

    requirement_alignment: (
        RequirementAlignment
    )


class HumanObservationResult(
    StrictBaseModel
):
    human_observation_id: str = Field(
        min_length=1,
    )

    feedback_type: FeedbackType

    included_in_primary_matching: bool

    status: HumanObservationStatus

    match_ids: list[str] = Field(
        default_factory=list,
    )

    coverage_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    best_candidate_generated_observation_id: (
        str | None
    ) = None

    best_candidate_similarity: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    exclusion_reason: str | None = None

    @model_validator(
        mode="after"
    )
    def validate_human_result(
        self,
    ) -> HumanObservationResult:
        if self.status == "excluded":
            if self.included_in_primary_matching:
                raise ValueError(
                    "Excluded human observations "
                    "cannot be included in "
                    "primary matching."
                )

            if self.coverage_score is not None:
                raise ValueError(
                    "Excluded human observations "
                    "must use coverage_score=None."
                )

            if not self.exclusion_reason:
                raise ValueError(
                    "Excluded human observations "
                    "must provide an "
                    "exclusion_reason."
                )

        else:
            if (
                not
                self.included_in_primary_matching
            ):
                raise ValueError(
                    "Non-excluded human "
                    "observations must be included "
                    "in primary matching."
                )

        if self.status == "matched":
            if not self.match_ids:
                raise ValueError(
                    "Matched human observations "
                    "must reference at least one "
                    "match."
                )

            if self.coverage_score != 1.0:
                raise ValueError(
                    "Matched human observations "
                    "must use coverage_score=1.0."
                )

        if (
            self.status
            == "partially_matched"
        ):
            if not self.match_ids:
                raise ValueError(
                    "Partially matched human "
                    "observations must reference "
                    "at least one match."
                )

            if self.coverage_score is None:
                raise ValueError(
                    "Partially matched human "
                    "observations require a "
                    "coverage_score."
                )

        if self.status == "not_matched":
            if self.match_ids:
                raise ValueError(
                    "Not-matched human "
                    "observations cannot reference "
                    "accepted matches."
                )

            if self.coverage_score != 0.0:
                raise ValueError(
                    "Not-matched human "
                    "observations must use "
                    "coverage_score=0.0."
                )

        return self


class GroundingEvaluation(
    StrictBaseModel
):
    status: GroundingStatus

    evidence_coverage: EvidenceCoverage

    rubric_alignment: (
        RubricAlignmentStatus
    )

    requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    artifact_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    unit_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    basis_codes: list[
        str
    ] = Field(
        default_factory=list,
    )

    rationale: str = Field(
        min_length=1,
    )


class GeneratedObservationResult(
    StrictBaseModel
):
    generated_observation_id: str = Field(
        min_length=1,
    )

    feedback_type: FeedbackType

    mapping_status: GeneratedMappingStatus

    match_ids: list[str] = Field(
        default_factory=list,
    )

    classification: (
        GeneratedObservationClassification
    )

    grounding_evaluation: (
        GroundingEvaluation | None
    ) = None

    @model_validator(
        mode="after"
    )
    def validate_generated_result(
        self,
    ) -> GeneratedObservationResult:
        if (
            self.classification
            == "human_aligned"
        ):
            if self.mapping_status not in {
                "matched",
                "partially_matched",
            }:
                raise ValueError(
                    "human_aligned generated "
                    "observations must have a "
                    "matched or partially_matched "
                    "mapping status."
                )

            if not self.match_ids:
                raise ValueError(
                    "human_aligned generated "
                    "observations must reference "
                    "at least one match."
                )

            if (
                self.grounding_evaluation
                is not None
            ):
                raise ValueError(
                    "human_aligned generated "
                    "observations must not contain "
                    "a grounding_evaluation."
                )

        else:
            if (
                self.mapping_status
                != "no_mapping"
            ):
                raise ValueError(
                    "Non-human-aligned generated "
                    "observations must use "
                    "mapping_status='no_mapping'."
                )

            if self.match_ids:
                raise ValueError(
                    "Generated observations with "
                    "no human mapping cannot "
                    "reference match IDs."
                )

            if (
                self.grounding_evaluation
                is None
            ):
                raise ValueError(
                    "Unmatched generated "
                    "observations require a "
                    "grounding_evaluation."
                )

        return self


class RequirementEvaluation(
    StrictBaseModel
):
    requirement_id: str = Field(
        min_length=1,
    )

    applicable: bool

    human_targeted: bool

    human_feedback_types: list[
        FeedbackType
    ] = Field(
        default_factory=list,
    )

    human_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    generated_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    covered: bool

    coverage_basis: CoverageBasis

    @model_validator(
        mode="after"
    )
    def validate_requirement_result(
        self,
    ) -> RequirementEvaluation:
        if (
            self.covered
            and self.coverage_basis
            == "not_covered"
        ):
            raise ValueError(
                "Covered requirements cannot "
                "use coverage_basis="
                "'not_covered'."
            )

        if (
            not self.covered
            and self.coverage_basis
            != "not_covered"
        ):
            raise ValueError(
                "Uncovered requirements must "
                "use coverage_basis="
                "'not_covered'."
            )

        return self


class ComponentEvaluation(
    StrictBaseModel
):
    component_id: str = Field(
        min_length=1,
    )

    human_targeted: bool

    human_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    generated_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    covered: bool


class RubricEvaluation(
    StrictBaseModel
):
    applicable_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    human_targeted_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    human_targeted_corrective_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    human_targeted_strength_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    generated_covered_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    human_targeted_component_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    generated_covered_component_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    requirement_results: list[
        RequirementEvaluation
    ] = Field(
        default_factory=list,
    )

    component_results: list[
        ComponentEvaluation
    ] = Field(
        default_factory=list,
    )


class SummaryComparison(
    StrictBaseModel
):
    comparison_type: (
        SummaryComparisonType
    )

    status: SummaryComparisonStatus

    semantic_similarity: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(
        mode="after"
    )
    def validate_summary_comparison(
        self,
    ) -> SummaryComparison:
        if self.status == "compared":
            if (
                self.semantic_similarity
                is None
            ):
                raise ValueError(
                    "Compared summaries require "
                    "semantic_similarity."
                )

        if (
            self.status
            == "not_available"
            and self.semantic_similarity
            is not None
        ):
            raise ValueError(
                "Unavailable summaries must use "
                "semantic_similarity=None."
            )

        return self


class RatioMetric(
    StrictBaseModel
):
    numerator: int = Field(
        ge=0,
    )

    denominator: int = Field(
        ge=0,
    )

    rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    applicable: bool

    @model_validator(
        mode="after"
    )
    def validate_ratio_metric(
        self,
    ) -> RatioMetric:
        if self.denominator == 0:
            if self.applicable:
                raise ValueError(
                    "A ratio metric with "
                    "denominator=0 must use "
                    "applicable=False."
                )

            if self.rate is not None:
                raise ValueError(
                    "A ratio metric with "
                    "denominator=0 must use "
                    "rate=None."
                )

            return self

        if not self.applicable:
            raise ValueError(
                "A ratio metric with a positive "
                "denominator must use "
                "applicable=True."
            )

        expected_rate = (
            self.numerator
            / self.denominator
        )

        if self.rate is None:
            raise ValueError(
                "An applicable ratio metric "
                "requires a rate."
            )

        tolerance = 1e-9

        if (
            abs(
                self.rate
                - expected_rate
            )
            > tolerance
        ):
            raise ValueError(
                "Ratio metric rate does not "
                "match numerator / denominator."
            )

        return self


class ScoredMetric(
    StrictBaseModel
):
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    applicable: bool

    @model_validator(
        mode="after"
    )
    def validate_scored_metric(
        self,
    ) -> ScoredMetric:
        if self.applicable:
            if self.score is None:
                raise ValueError(
                    "Applicable scored metrics "
                    "require a score."
                )

        else:
            if self.score is not None:
                raise ValueError(
                    "Non-applicable scored "
                    "metrics must use score=None."
                )

        return self


class HumanAlignmentMetrics(
    StrictBaseModel
):
    human_observation_recall: (
        RatioMetric
    )

    weighted_human_feedback_coverage: (
        ScoredMetric
    )

    human_agreement_precision: (
        RatioMetric
    )

    human_agreement_f1: ScoredMetric

    corrective_observation_recall: (
        RatioMetric
    )

    strength_observation_recall: (
        RatioMetric
    )


class GroundingMetrics(
    StrictBaseModel
):
    grounded_feedback_rate: RatioMetric

    additional_valid_observation_rate: (
        RatioMetric
    )

    novel_grounded_feedback_rate: (
        RatioMetric
    )

    unsupported_observation_rate: (
        RatioMetric
    )

    contradicted_observation_rate: (
        RatioMetric
    )

    not_assessable_observation_rate: (
        RatioMetric
    )


class RubricCoverageMetrics(
    StrictBaseModel
):
    human_targeted_requirement_recall: (
        RatioMetric
    )

    corrective_requirement_recall: (
        RatioMetric
    )

    strength_requirement_recall: (
        RatioMetric
    )

    human_targeted_component_recall: (
        RatioMetric
    )

    full_rubric_coverage: RatioMetric


class SummaryMetrics(
    StrictBaseModel
):
    average_summary_similarity: (
        ScoredMetric
    )


class EvaluationCounts(
    StrictBaseModel
):
    human_observations_total: int = Field(
        ge=0,
    )

    human_observations_eligible: int = Field(
        ge=0,
    )

    human_observations_excluded: int = Field(
        ge=0,
    )

    human_observations_matched: int = Field(
        ge=0,
    )

    human_observations_partially_matched: (
        int
    ) = Field(
        ge=0,
    )

    human_observations_not_matched: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_total: int = Field(
        ge=0,
    )

    generated_observations_fully_human_matched: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_partially_human_matched: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_additional_valid: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_unsupported: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_contradicted: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_not_assessable: (
        int
    ) = Field(
        ge=0,
    )

    generated_observations_unresolved: (
        int
    ) = Field(
        ge=0,
    )

    applicable_requirements: int = Field(
        ge=0,
    )

    human_targeted_requirements: int = Field(
        ge=0,
    )

    generated_covered_requirements: int = Field(
        ge=0,
    )

    human_targeted_components: int = Field(
        ge=0,
    )

    generated_covered_components: int = Field(
        ge=0,
    )


class AggregateMetrics(
    StrictBaseModel
):
    human_alignment: (
        HumanAlignmentMetrics
    )

    grounding: GroundingMetrics

    rubric_coverage: (
        RubricCoverageMetrics
    )

    summary_similarity: SummaryMetrics

    counts: EvaluationCounts


class Diagnostic(
    StrictBaseModel
):
    diagnostic_id: str = Field(
        min_length=1,
    )

    severity: DiagnosticSeverity

    code: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )

    related_human_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    related_generated_observation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    related_requirement_ids: list[
        str
    ] = Field(
        default_factory=list,
    )


class EvaluationResult(
    StrictBaseModel
):
    schema_version: str = Field(
        min_length=1,
    )

    evaluation_id: str = Field(
        min_length=1,
    )

    generated_feedback_id: str = Field(
        min_length=1,
    )

    human_feedback_id: str = Field(
        min_length=1,
    )

    processed_submission_id: str = Field(
        min_length=1,
    )

    assessment_id: str = Field(
        min_length=1,
    )

    student_id: str = Field(
        min_length=1,
    )

    evaluator: EvaluatorRecord

    observation_matches: list[
        ObservationMatch
    ] = Field(
        default_factory=list,
    )

    human_observation_results: list[
        HumanObservationResult
    ] = Field(
        default_factory=list,
    )

    generated_observation_results: list[
        GeneratedObservationResult
    ] = Field(
        default_factory=list,
    )

    rubric_evaluation: RubricEvaluation

    summary_comparisons: list[
        SummaryComparison
    ] = Field(
        default_factory=list,
    )

    aggregate_metrics: AggregateMetrics

    diagnostics: list[
        Diagnostic
    ] = Field(
        default_factory=list,
    )