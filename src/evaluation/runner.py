from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embedding_service import (
    EmbeddingService,
)
from .grounding import (
    GroundingResult,
    ObservationGrounder,
)
from .matching import (
    MatchingResult,
    ObservationMatcher,
)
from .models import (
    GroundingConfiguration,
    MatchingConfiguration,
    AggregateMetrics,
    EmbeddingConfiguration,
    EvaluationResult,
    EvaluatorRecord,
    RubricEvaluation,
    SummaryComparison,
)


@dataclass(frozen=True)
class EvaluationRuntime:
    """
    Runtime services and frozen configuration used
    by the deterministic evaluation pipeline.
    """

    embedding_service: EmbeddingService

    matching_configuration: (
        MatchingConfiguration
    )

    grounding_configuration: (
        GroundingConfiguration
    )

    matcher: ObservationMatcher

    grounder: ObservationGrounder


@dataclass(frozen=True)
class ObservationEvaluationResult:
    """
    Combined observation-level evaluation result.

    Matching establishes correspondence with human
    feedback. Grounding then assesses generated
    observations that have no human mapping.
    """

    matching: MatchingResult

    grounding: GroundingResult


def build_evaluation_runtime(
    *,
    embedding_service: (
        EmbeddingService | None
    ) = None,
) -> EvaluationRuntime:
    """
    Build the frozen deterministic evaluation
    runtime.

    Matching calibration
    --------------------

    Calibration used a manually adjudicated
    200-pair reference sample spanning both
    assessments and all four experimental
    configurations.

        full match:
            similarity >= 0.65

        partial match:
            similarity >= 0.28

        requirement rescue:
            similarity >= 0.22 and requirement
            alignment is exact or overlap
            -> partial match

    The 0.22 structural rescue is implemented in
    ObservationMatcher._calibrated_decision().

    Grounding policy
    ----------------

    Unmatched generated observations are grounded
    using independently inspectable evidence from
    the processed submission.

    Generator verification_status values are
    descriptive metadata and are not treated as
    ground truth.

    Resolvable submission references and structured
    absence evidence provide partial support only.

    Full support is reserved for explicit tool-check
    evidence that can be corroborated against a
    passed processed-submission check.
    """

    if embedding_service is None:
        embedding_service = (
            EmbeddingService()
        )

    matching_configuration = (
        MatchingConfiguration(
            full_match_threshold=0.65,
            partial_match_threshold=0.28,
        )
    )

    grounding_configuration = (
        GroundingConfiguration(
            enabled=True,
            ground_unmatched_generated_observations=True,
            distinguish_contradicted_from_unsupported=True,
            allow_not_assessable=True,
        )
    )

    matcher = ObservationMatcher(
        embedding_service=(
            embedding_service
        ),
        configuration=(
            matching_configuration
        ),
    )

    grounder = ObservationGrounder(
        configuration=(
            grounding_configuration
        ),
    )

    return EvaluationRuntime(
        embedding_service=(
            embedding_service
        ),
        matching_configuration=(
            matching_configuration
        ),
        grounding_configuration=(
            grounding_configuration
        ),
        matcher=matcher,
        grounder=grounder,
    )


def evaluate_observations(
    *,
    human_feedback: dict[str, Any],
    generated_feedback: dict[str, Any],
    processed_submission: dict[str, Any],
    runtime: (
        EvaluationRuntime | None
    ) = None,
) -> ObservationEvaluationResult:
    """
    Run calibrated observation matching followed by
    deterministic grounding.

    When evaluating multiple submissions, callers
    should build the runtime once and reuse it so
    that the embedding model is not repeatedly
    loaded.
    """

    if runtime is None:
        runtime = (
            build_evaluation_runtime()
        )

    matching_result = runtime.matcher.match(
        human_feedback=human_feedback,
        generated_feedback=(
            generated_feedback
        ),
    )

    grounding_result = runtime.grounder.ground(
        generated_feedback=(
            generated_feedback
        ),
        processed_submission=(
            processed_submission
        ),
        generated_match_states=(
            matching_result
            .generated_match_states
        ),
    )

    return ObservationEvaluationResult(
        matching=matching_result,
        grounding=grounding_result,
    )


def evaluate_observation_matching(
    *,
    human_feedback: dict[str, Any],
    generated_feedback: dict[str, Any],
    runtime: (
        EvaluationRuntime | None
    ) = None,
) -> MatchingResult:
    """
    Matching-only helper retained for calibration,
    diagnostics, and backward compatibility.
    """

    if runtime is None:
        runtime = (
            build_evaluation_runtime()
        )

    return runtime.matcher.match(
        human_feedback=human_feedback,
        generated_feedback=(
            generated_feedback
        ),
    )

def build_evaluation_result(
    *,
    evaluation_id: str,
    generated_feedback: dict[str, Any],
    human_feedback: dict[str, Any],
    processed_submission: dict[str, Any],
    observation_evaluation: (
        ObservationEvaluationResult
    ),
    rubric_evaluation: RubricEvaluation,
    summary_comparisons: list[
        SummaryComparison
    ],
    aggregate_metrics: AggregateMetrics,
    runtime: EvaluationRuntime,
    pipeline_version: str = "1.0.0",
) -> EvaluationResult:
    """
    Assemble the complete persisted evaluation
    result after matching, grounding, rubric
    evaluation, summary comparison, and metric
    aggregation.
    """

    generated_feedback_id = (
        generated_feedback.get(
            "generated_feedback_id"
        )
    )

    if not isinstance(
        generated_feedback_id,
        str,
    ) or not generated_feedback_id.strip():
        # Some existing experiment outputs do not
        # contain an explicit generated_feedback_id.
        # Construct a deterministic descriptive ID.
        generated_feedback_id = (
            f"{processed_submission['assessment_id']}_"
            f"{processed_submission['student_id']}_"
            "generated_feedback"
        )

    human_feedback_id = (
        human_feedback.get(
            "human_feedback_id"
        )
    )

    if not isinstance(
        human_feedback_id,
        str,
    ) or not human_feedback_id.strip():
        raise ValueError(
            "Human feedback is missing "
            "human_feedback_id."
        )

    processed_submission_id = (
        processed_submission.get(
            "processed_submission_id"
        )
    )

    if not isinstance(
        processed_submission_id,
        str,
    ) or not processed_submission_id.strip():
        raise ValueError(
            "Processed submission is missing "
            "processed_submission_id."
        )

    assessment_id = (
        processed_submission.get(
            "assessment_id"
        )
    )

    student_id = (
        processed_submission.get(
            "student_id"
        )
    )

    if not isinstance(
        assessment_id,
        str,
    ) or not assessment_id.strip():
        raise ValueError(
            "Processed submission is missing "
            "assessment_id."
        )

    if not isinstance(
        student_id,
        str,
    ) or not student_id.strip():
        raise ValueError(
            "Processed submission is missing "
            "student_id."
        )

    evaluator = EvaluatorRecord(
        pipeline_version=(
            pipeline_version
        ),
        embedding=(
            EmbeddingConfiguration()
        ),
        matching_configuration=(
            runtime
            .matching_configuration
        ),
        grounding_configuration=(
            runtime
            .grounding_configuration
        ),
    )

    return EvaluationResult(
        schema_version="1.0",
        evaluation_id=evaluation_id,
        generated_feedback_id=(
            generated_feedback_id
        ),
        human_feedback_id=(
            human_feedback_id
        ),
        processed_submission_id=(
            processed_submission_id
        ),
        assessment_id=assessment_id,
        student_id=student_id,
        evaluator=evaluator,
        observation_matches=list(
            observation_evaluation
            .matching
            .observation_matches
        ),
        human_observation_results=list(
            observation_evaluation
            .matching
            .human_observation_results
        ),
        generated_observation_results=list(
            observation_evaluation
            .grounding
            .generated_observation_results
        ),
        rubric_evaluation=(
            rubric_evaluation
        ),
        summary_comparisons=(
            summary_comparisons
        ),
        aggregate_metrics=(
            aggregate_metrics
        ),
        diagnostics=[],
    )
