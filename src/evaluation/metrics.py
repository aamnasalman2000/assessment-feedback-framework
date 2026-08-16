from __future__ import annotations

from typing import Iterable

from .models import (
    GeneratedObservationResult,
    HumanAlignmentMetrics,
    HumanObservationResult,
    RatioMetric,
    ScoredMetric,
    GroundingMetrics,
    ComponentEvaluation,
    RequirementEvaluation,
    RubricCoverageMetrics,
    RubricEvaluation,
    SummaryComparison,
    SummaryMetrics,
    AggregateMetrics,
    EvaluationCounts,
)


CORRECTIVE_TYPES = {
    "error",
    "omission",
    "suggestion",
}


def _ratio_metric(
    *,
    numerator: int,
    denominator: int,
) -> RatioMetric:
    if denominator == 0:
        return RatioMetric(
            numerator=0,
            denominator=0,
            rate=None,
            applicable=False,
        )

    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        rate=numerator / denominator,
        applicable=True,
    )


def _scored_metric(
    value: float | None,
) -> ScoredMetric:
    if value is None:
        return ScoredMetric(
            score=None,
            applicable=False,
        )

    return ScoredMetric(
        score=value,
        applicable=True,
    )


def _eligible_human_observations(
    human_results: Iterable[
        HumanObservationResult
    ],
) -> list[HumanObservationResult]:
    return [
        result
        for result in human_results
        if result.status != "excluded"
    ]


def _is_recalled(
    result: HumanObservationResult,
) -> bool:
    return result.status in {
        "matched",
        "partially_matched",
    }


def _coverage_value(
    result: HumanObservationResult,
) -> float:
    if result.status == "matched":
        return 1.0

    if result.status == "partially_matched":
        if result.coverage_score is None:
            raise ValueError(
                "Partially matched human observation "
                "is missing coverage_score."
            )

        return float(
            result.coverage_score
        )

    if result.status == "not_matched":
        return 0.0

    raise ValueError(
        "Excluded human observations must not "
        "be passed to _coverage_value()."
    )


def calculate_human_alignment_metrics(
    *,
    human_results: Iterable[
        HumanObservationResult
    ],
    generated_results: Iterable[
        GeneratedObservationResult
    ],
) -> HumanAlignmentMetrics:
    """
    Calculate observation-level agreement metrics
    against human feedback.

    Definitions
    -----------

    Human observation recall
        Fraction of eligible human observations that
        have at least one accepted full or partial
        generated-feedback match.

    Weighted human feedback coverage
        Mean human-observation coverage score over
        eligible human observations.

        matched            -> 1.0
        partially_matched  -> stored coverage_score
        not_matched        -> 0.0

    Human agreement precision
        Fraction of generated observations that are
        human-aligned among generated observations
        participating in the human-agreement
        comparison.

        At this stage, generated observations with
        classification='human_aligned' count as
        agreement. Unmatched generated observations
        do not.

    Human agreement F1
        Harmonic mean of human observation recall
        and human agreement precision.

    Corrective observation recall
        Human observation recall restricted to
        error, omission, and suggestion feedback.

    Strength observation recall
        Human observation recall restricted to
        strength feedback.
    """

    human_results = list(
        human_results
    )

    generated_results = list(
        generated_results
    )

    eligible = (
        _eligible_human_observations(
            human_results
        )
    )

    recalled = [
        result
        for result in eligible
        if _is_recalled(result)
    ]

    human_observation_recall = (
        _ratio_metric(
            numerator=len(recalled),
            denominator=len(eligible),
        )
    )

    # --------------------------------------------------------
    # Weighted human feedback coverage
    # --------------------------------------------------------

    if eligible:
        coverage_score = sum(
            _coverage_value(result)
            for result in eligible
        ) / len(eligible)

        weighted_coverage = (
            _scored_metric(
                coverage_score
            )
        )
    else:
        weighted_coverage = (
            _scored_metric(None)
        )

    # --------------------------------------------------------
    # Human agreement precision
    # --------------------------------------------------------

    human_aligned_generated = [
        result
        for result in generated_results
        if (
            result.classification
            == "human_aligned"
        )
    ]

    human_agreement_precision = (
        _ratio_metric(
            numerator=len(
                human_aligned_generated
            ),
            denominator=len(
                generated_results
            ),
        )
    )

    # --------------------------------------------------------
    # Human agreement F1
    # --------------------------------------------------------

    recall_rate = (
        human_observation_recall.rate
    )

    precision_rate = (
        human_agreement_precision.rate
    )

    if (
        recall_rate is None
        or precision_rate is None
    ):
        human_agreement_f1 = (
            _scored_metric(None)
        )

    elif (
        recall_rate
        + precision_rate
        == 0
    ):
        human_agreement_f1 = (
            _scored_metric(0.0)
        )

    else:
        f1 = (
            2
            * recall_rate
            * precision_rate
            / (
                recall_rate
                + precision_rate
            )
        )

        human_agreement_f1 = (
            _scored_metric(f1)
        )

    # --------------------------------------------------------
    # Corrective observation recall
    # --------------------------------------------------------

    corrective = [
        result
        for result in eligible
        if (
            result.feedback_type
            in CORRECTIVE_TYPES
        )
    ]

    corrective_recalled = [
        result
        for result in corrective
        if _is_recalled(result)
    ]

    corrective_observation_recall = (
        _ratio_metric(
            numerator=len(
                corrective_recalled
            ),
            denominator=len(
                corrective
            ),
        )
    )

    # --------------------------------------------------------
    # Strength observation recall
    # --------------------------------------------------------

    strengths = [
        result
        for result in eligible
        if (
            result.feedback_type
            == "strength"
        )
    ]

    strength_recalled = [
        result
        for result in strengths
        if _is_recalled(result)
    ]

    strength_observation_recall = (
        _ratio_metric(
            numerator=len(
                strength_recalled
            ),
            denominator=len(
                strengths
            ),
        )
    )

    return HumanAlignmentMetrics(
        human_observation_recall=(
            human_observation_recall
        ),
        weighted_human_feedback_coverage=(
            weighted_coverage
        ),
        human_agreement_precision=(
            human_agreement_precision
        ),
        human_agreement_f1=(
            human_agreement_f1
        ),
        corrective_observation_recall=(
            corrective_observation_recall
        ),
        strength_observation_recall=(
            strength_observation_recall
        ),
    )

def calculate_grounding_metrics(
    *,
    generated_results: Iterable[
        GeneratedObservationResult
    ],
) -> GroundingMetrics:
    """
    Calculate grounding metrics for generated
    observations with no human-feedback mapping.

    Human-aligned generated observations are excluded
    from unmatched-observation denominators.

    Grounded feedback includes both supported and
    partially-supported observations, because both
    have independently inspectable grounding evidence.

    additional_valid is stricter: it is reserved for
    observations whose grounding status is supported.
    """

    generated_results = list(
        generated_results
    )

    unmatched = [
        result
        for result in generated_results
        if result.mapping_status == "no_mapping"
    ]

    # --------------------------------------------------------
    # Grounded feedback
    # --------------------------------------------------------

    grounded = [
        result
        for result in unmatched
        if (
            result.grounding_evaluation
            is not None
            and result
            .grounding_evaluation
            .status
            in {
                "supported",
                "partially_supported",
            }
        )
    ]

    grounded_feedback_rate = (
        _ratio_metric(
            numerator=len(grounded),
            denominator=len(unmatched),
        )
    )

    # --------------------------------------------------------
    # Additional valid observations
    # --------------------------------------------------------

    additional_valid = [
        result
        for result in unmatched
        if (
            result.classification
            == "additional_valid"
        )
    ]

    additional_valid_observation_rate = (
        _ratio_metric(
            numerator=len(
                additional_valid
            ),
            denominator=len(
                unmatched
            ),
        )
    )

    # --------------------------------------------------------
    # Novel grounded feedback
    #
    # This deliberately uses all generated observations
    # as the denominator. It measures how much of the
    # total generated feedback consists of independently
    # supported novel observations.
    # --------------------------------------------------------

    novel_supported = [
        result
        for result in unmatched
        if (
            result.grounding_evaluation
            is not None
            and result
            .grounding_evaluation
            .status
            == "supported"
        )
    ]

    novel_grounded_feedback_rate = (
        _ratio_metric(
            numerator=len(
                novel_supported
            ),
            denominator=len(
                generated_results
            ),
        )
    )

    # --------------------------------------------------------
    # Unsupported
    # --------------------------------------------------------

    unsupported = [
        result
        for result in unmatched
        if (
            result.classification
            == "unsupported"
        )
    ]

    unsupported_observation_rate = (
        _ratio_metric(
            numerator=len(
                unsupported
            ),
            denominator=len(
                unmatched
            ),
        )
    )

    # --------------------------------------------------------
    # Contradicted
    # --------------------------------------------------------

    contradicted = [
        result
        for result in unmatched
        if (
            result.classification
            == "contradicted"
        )
    ]

    contradicted_observation_rate = (
        _ratio_metric(
            numerator=len(
                contradicted
            ),
            denominator=len(
                unmatched
            ),
        )
    )

    # --------------------------------------------------------
    # Not assessable
    #
    # Includes partially-supported observations because
    # the final classification deliberately reserves
    # additional_valid for independently supported claims.
    # --------------------------------------------------------

    not_assessable = [
        result
        for result in unmatched
        if (
            result.classification
            == "not_assessable"
        )
    ]

    not_assessable_observation_rate = (
        _ratio_metric(
            numerator=len(
                not_assessable
            ),
            denominator=len(
                unmatched
            ),
        )
    )

    return GroundingMetrics(
        grounded_feedback_rate=(
            grounded_feedback_rate
        ),
        additional_valid_observation_rate=(
            additional_valid_observation_rate
        ),
        novel_grounded_feedback_rate=(
            novel_grounded_feedback_rate
        ),
        unsupported_observation_rate=(
            unsupported_observation_rate
        ),
        contradicted_observation_rate=(
            contradicted_observation_rate
        ),
        not_assessable_observation_rate=(
            not_assessable_observation_rate
        ),
    )

def calculate_rubric_evaluation(
    *,
    human_feedback: dict,
    generated_feedback: dict,
    human_results: Iterable[
        HumanObservationResult
    ],
    generated_results: Iterable[
        GeneratedObservationResult
    ],
) -> RubricEvaluation:
    """
    Build requirement- and component-level coverage
    records from observation-level evaluation results.

    Human-targeted requirements/components are those
    referenced by eligible human observations.

    Generated coverage is credited only when a
    generated observation is human-aligned.

    Applicable requirements are derived from the
    generated criterion assessments, which represent
    the rubric criteria evaluated for this submission.
    """

    human_results = list(human_results)
    generated_results = list(
        generated_results
    )

    human_observation_lookup = {
        observation["observation_id"]:
            observation
        for observation in human_feedback.get(
            "observations",
            [],
        )
        if isinstance(observation, dict)
        and isinstance(
            observation.get("observation_id"),
            str,
        )
    }

    generated_observation_lookup = {
        observation["observation_id"]:
            observation
        for observation in generated_feedback.get(
            "observations",
            [],
        )
        if isinstance(observation, dict)
        and isinstance(
            observation.get("observation_id"),
            str,
        )
    }

    # ========================================================
    # Applicable rubric requirements
    # ========================================================

    applicable_requirement_ids = sorted({
        criterion["requirement_id"].strip()
        for criterion in generated_feedback.get(
            "criterion_assessments",
            [],
        )
        if isinstance(criterion, dict)
        and isinstance(
            criterion.get("requirement_id"),
            str,
        )
        and criterion["requirement_id"].strip()
        and criterion.get("status")
        != "not_assessable"
    })

    # ========================================================
    # Human-targeted requirements and components
    # ========================================================

    human_requirement_feedback_types = {}
    human_requirement_observation_ids = {}

    human_component_observation_ids = {}

    for result in human_results:

        if result.status == "excluded":
            continue

        observation = (
            human_observation_lookup.get(
                result.human_observation_id
            )
        )

        if observation is None:
            continue

        requirement_ids = observation.get(
            "requirement_ids",
            [],
        )

        if not isinstance(
            requirement_ids,
            list,
        ):
            requirement_ids = []

        for requirement_id in requirement_ids:

            if (
                not isinstance(
                    requirement_id,
                    str,
                )
                or not requirement_id.strip()
            ):
                continue

            requirement_id = (
                requirement_id.strip()
            )

            human_requirement_feedback_types.setdefault(
                requirement_id,
                set(),
            ).add(
                result.feedback_type
            )

            human_requirement_observation_ids.setdefault(
                requirement_id,
                set(),
            ).add(
                result.human_observation_id
            )

        part_ids = observation.get(
            "part_ids",
            [],
        )

        if not isinstance(
            part_ids,
            list,
        ):
            part_ids = []

        for component_id in part_ids:

            if (
                not isinstance(
                    component_id,
                    str,
                )
                or not component_id.strip()
            ):
                continue

            component_id = (
                component_id.strip()
            )

            human_component_observation_ids.setdefault(
                component_id,
                set(),
            ).add(
                result.human_observation_id
            )

    # ========================================================
    # Generated-covered requirements and components
    # ========================================================

    generated_requirement_observation_ids = {}

    generated_component_observation_ids = {}

    for result in generated_results:

        if (
            result.classification
            != "human_aligned"
        ):
            continue

        observation = (
            generated_observation_lookup.get(
                result
                .generated_observation_id
            )
        )

        if observation is None:
            continue

        requirement_ids = observation.get(
            "requirement_ids",
            [],
        )

        if not isinstance(
            requirement_ids,
            list,
        ):
            requirement_ids = []

        for requirement_id in requirement_ids:

            if (
                not isinstance(
                    requirement_id,
                    str,
                )
                or not requirement_id.strip()
            ):
                continue

            requirement_id = (
                requirement_id.strip()
            )

            generated_requirement_observation_ids.setdefault(
                requirement_id,
                set(),
            ).add(
                result.generated_observation_id
            )

        part_ids = observation.get(
            "part_ids",
            [],
        )

        if not isinstance(
            part_ids,
            list,
        ):
            part_ids = []

        for component_id in part_ids:

            if (
                not isinstance(
                    component_id,
                    str,
                )
                or not component_id.strip()
            ):
                continue

            component_id = (
                component_id.strip()
            )

            generated_component_observation_ids.setdefault(
                component_id,
                set(),
            ).add(
                result.generated_observation_id
            )

    human_targeted_requirement_ids = sorted(
        human_requirement_observation_ids
    )

    generated_covered_requirement_ids = sorted(
        generated_requirement_observation_ids
    )

    human_targeted_component_ids = sorted(
        human_component_observation_ids
    )

    generated_covered_component_ids = sorted(
        generated_component_observation_ids
    )

    # ========================================================
    # Requirement results
    # ========================================================

    all_requirement_ids = sorted(
        set(applicable_requirement_ids)
        | set(
            human_targeted_requirement_ids
        )
        | set(
            generated_covered_requirement_ids
        )
    )

    requirement_results = []

    for requirement_id in all_requirement_ids:

        human_ids = sorted(
            human_requirement_observation_ids.get(
                requirement_id,
                set(),
            )
        )

        generated_ids = sorted(
            generated_requirement_observation_ids.get(
                requirement_id,
                set(),
            )
        )

        human_targeted = bool(
            human_ids
        )

        explicit_generated_mapping = bool(
            generated_ids
        )

        covered = explicit_generated_mapping

        coverage_basis = (
            "explicit_generated_mapping"
            if covered
            else "not_covered"
        )

        requirement_results.append(
            RequirementEvaluation(
                requirement_id=(
                    requirement_id
                ),
                applicable=(
                    requirement_id
                    in applicable_requirement_ids
                ),
                human_targeted=(
                    human_targeted
                ),
                human_feedback_types=sorted(
                    human_requirement_feedback_types.get(
                        requirement_id,
                        set(),
                    )
                ),
                human_observation_ids=(
                    human_ids
                ),
                generated_observation_ids=(
                    generated_ids
                ),
                covered=covered,
                coverage_basis=(
                    coverage_basis
                ),
            )
        )

    # ========================================================
    # Component results
    # ========================================================

    all_component_ids = sorted(
        set(
            human_targeted_component_ids
        )
        | set(
            generated_covered_component_ids
        )
    )

    component_results = []

    for component_id in all_component_ids:

        human_ids = sorted(
            human_component_observation_ids.get(
                component_id,
                set(),
            )
        )

        generated_ids = sorted(
            generated_component_observation_ids.get(
                component_id,
                set(),
            )
        )

        component_results.append(
            ComponentEvaluation(
                component_id=(
                    component_id
                ),
                human_targeted=bool(
                    human_ids
                ),
                human_observation_ids=(
                    human_ids
                ),
                generated_observation_ids=(
                    generated_ids
                ),
                covered=bool(
                    generated_ids
                ),
            )
        )

    # ========================================================
    # Human-targeted requirement subgroups
    # ========================================================

    corrective_requirement_ids = sorted(
        requirement_id
        for requirement_id, types
        in human_requirement_feedback_types.items()
        if (
            set(types)
            & {
                "error",
                "omission",
                "suggestion",
            }
        )
    )

    strength_requirement_ids = sorted(
        requirement_id
        for requirement_id, types
        in human_requirement_feedback_types.items()
        if "strength" in types
    )

    return RubricEvaluation(
        applicable_requirement_ids=(
            applicable_requirement_ids
        ),
        human_targeted_requirement_ids=(
            human_targeted_requirement_ids
        ),
        human_targeted_corrective_requirement_ids=(
            corrective_requirement_ids
        ),
        human_targeted_strength_requirement_ids=(
            strength_requirement_ids
        ),
        generated_covered_requirement_ids=(
            generated_covered_requirement_ids
        ),
        human_targeted_component_ids=(
            human_targeted_component_ids
        ),
        generated_covered_component_ids=(
            generated_covered_component_ids
        ),
        requirement_results=(
            requirement_results
        ),
        component_results=(
            component_results
        ),
    )


def calculate_rubric_coverage_metrics(
    *,
    rubric_evaluation: RubricEvaluation,
) -> RubricCoverageMetrics:
    """
    Calculate requirement- and component-level
    coverage metrics from RubricEvaluation.
    """

    generated_requirements = set(
        rubric_evaluation
        .generated_covered_requirement_ids
    )

    # --------------------------------------------------------
    # Human-targeted requirement recall
    # --------------------------------------------------------

    targeted_requirements = set(
        rubric_evaluation
        .human_targeted_requirement_ids
    )

    human_targeted_requirement_recall = (
        _ratio_metric(
            numerator=len(
                targeted_requirements
                & generated_requirements
            ),
            denominator=len(
                targeted_requirements
            ),
        )
    )

    # --------------------------------------------------------
    # Corrective requirement recall
    # --------------------------------------------------------

    corrective_requirements = set(
        rubric_evaluation
        .human_targeted_corrective_requirement_ids
    )

    corrective_requirement_recall = (
        _ratio_metric(
            numerator=len(
                corrective_requirements
                & generated_requirements
            ),
            denominator=len(
                corrective_requirements
            ),
        )
    )

    # --------------------------------------------------------
    # Strength requirement recall
    # --------------------------------------------------------

    strength_requirements = set(
        rubric_evaluation
        .human_targeted_strength_requirement_ids
    )

    strength_requirement_recall = (
        _ratio_metric(
            numerator=len(
                strength_requirements
                & generated_requirements
            ),
            denominator=len(
                strength_requirements
            ),
        )
    )

    # --------------------------------------------------------
    # Human-targeted component recall
    # --------------------------------------------------------

    targeted_components = set(
        rubric_evaluation
        .human_targeted_component_ids
    )

    generated_components = set(
        rubric_evaluation
        .generated_covered_component_ids
    )

    human_targeted_component_recall = (
        _ratio_metric(
            numerator=len(
                targeted_components
                & generated_components
            ),
            denominator=len(
                targeted_components
            ),
        )
    )

    # --------------------------------------------------------
    # Full rubric coverage
    # --------------------------------------------------------

    applicable_requirements = set(
        rubric_evaluation
        .applicable_requirement_ids
    )

    full_rubric_coverage = (
        _ratio_metric(
            numerator=len(
                applicable_requirements
                & generated_requirements
            ),
            denominator=len(
                applicable_requirements
            ),
        )
    )

    return RubricCoverageMetrics(
        human_targeted_requirement_recall=(
            human_targeted_requirement_recall
        ),
        corrective_requirement_recall=(
            corrective_requirement_recall
        ),
        strength_requirement_recall=(
            strength_requirement_recall
        ),
        human_targeted_component_recall=(
            human_targeted_component_recall
        ),
        full_rubric_coverage=(
            full_rubric_coverage
        ),
    )

def calculate_summary_comparisons(
    *,
    human_feedback: dict,
    generated_feedback: dict,
    embedding_service,
) -> list[SummaryComparison]:
    """
    Compare corresponding human and generated
    overall-feedback summaries using cosine
    similarity from the configured embedding model.

    Three summary types are evaluated:

        overall_summary
        strengths_summary
        improvement_summary
    """

    human_overall = human_feedback.get(
        "overall_feedback",
        {},
    )

    generated_overall = generated_feedback.get(
        "overall_feedback",
        {},
    )

    if not isinstance(
        human_overall,
        dict,
    ):
        human_overall = {}

    if not isinstance(
        generated_overall,
        dict,
    ):
        generated_overall = {}

    field_map = {
        "overall_summary": "summary",
        "strengths_summary": (
            "strengths_summary"
        ),
        "improvement_summary": (
            "improvement_summary"
        ),
    }

    comparisons = []

    for comparison_type, field in (
        field_map.items()
    ):

        human_text = human_overall.get(
            field
        )

        generated_text = (
            generated_overall.get(
                field
            )
        )

        human_available = (
            isinstance(
                human_text,
                str,
            )
            and human_text.strip()
        )

        generated_available = (
            isinstance(
                generated_text,
                str,
            )
            and generated_text.strip()
        )

        if (
            not human_available
            or not generated_available
        ):
            comparisons.append(
                SummaryComparison(
                    comparison_type=(
                        comparison_type
                    ),
                    status=(
                        "not_available"
                    ),
                    semantic_similarity=(
                        None
                    ),
                )
            )

            continue

        human_text = (
            human_text.strip()
        )

        generated_text = (
            generated_text.strip()
        )

        embeddings = (
            embedding_service.encode(
                [
                    human_text,
                    generated_text,
                ]
            )
        )

        human_embedding = embeddings[0]
        generated_embedding = embeddings[1]

        similarity = float(
            human_embedding
            @ generated_embedding
        )

        similarity = max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

        comparisons.append(
            SummaryComparison(
                comparison_type=(
                    comparison_type
                ),
                status="compared",
                semantic_similarity=(
                    similarity
                ),
            )
        )

    return comparisons


def calculate_summary_metrics(
    *,
    comparisons: Iterable[
        SummaryComparison
    ],
) -> SummaryMetrics:
    """
    Aggregate available summary-comparison
    similarities into one mean score.
    """

    comparisons = list(
        comparisons
    )

    available = [
        comparison.semantic_similarity
        for comparison in comparisons
        if (
            comparison.status
            == "compared"
            and comparison
            .semantic_similarity
            is not None
        )
    ]

    if not available:
        average = (
            ScoredMetric(
                score=None,
                applicable=False,
            )
        )

    else:
        average = (
            ScoredMetric(
                score=(
                    sum(available)
                    / len(available)
                ),
                applicable=True,
            )
        )

    return SummaryMetrics(
        average_summary_similarity=(
            average
        )
    )

def calculate_evaluation_counts(
    *,
    human_results: Iterable[
        HumanObservationResult
    ],
    generated_results: Iterable[
        GeneratedObservationResult
    ],
    rubric_evaluation: RubricEvaluation,
) -> EvaluationCounts:
    """
    Calculate integer counts used alongside the
    aggregate evaluation metrics.
    """

    human_results = list(
        human_results
    )

    generated_results = list(
        generated_results
    )

    human_total = len(
        human_results
    )

    human_excluded = sum(
        result.status == "excluded"
        for result in human_results
    )

    human_eligible = (
        human_total
        - human_excluded
    )

    human_matched = sum(
        result.status == "matched"
        for result in human_results
    )

    human_partially_matched = sum(
        result.status
        == "partially_matched"
        for result in human_results
    )

    human_not_matched = sum(
        result.status
        == "not_matched"
        for result in human_results
    )

    generated_total = len(
        generated_results
    )

    generated_fully_human_matched = sum(
        result.mapping_status == "matched"
        for result in generated_results
    )

    generated_partially_human_matched = sum(
        result.mapping_status
        == "partially_matched"
        for result in generated_results
    )

    generated_additional_valid = sum(
        result.classification
        == "additional_valid"
        for result in generated_results
    )

    generated_unsupported = sum(
        result.classification
        == "unsupported"
        for result in generated_results
    )

    generated_contradicted = sum(
        result.classification
        == "contradicted"
        for result in generated_results
    )

    generated_not_assessable = sum(
        result.classification
        == "not_assessable"
        for result in generated_results
    )

    generated_unresolved = sum(
        result.classification
        == "unresolved"
        for result in generated_results
    )

    return EvaluationCounts(
        human_observations_total=(
            human_total
        ),
        human_observations_eligible=(
            human_eligible
        ),
        human_observations_excluded=(
            human_excluded
        ),
        human_observations_matched=(
            human_matched
        ),
        human_observations_partially_matched=(
            human_partially_matched
        ),
        human_observations_not_matched=(
            human_not_matched
        ),
        generated_observations_total=(
            generated_total
        ),
        generated_observations_fully_human_matched=(
            generated_fully_human_matched
        ),
        generated_observations_partially_human_matched=(
            generated_partially_human_matched
        ),
        generated_observations_additional_valid=(
            generated_additional_valid
        ),
        generated_observations_unsupported=(
            generated_unsupported
        ),
        generated_observations_contradicted=(
            generated_contradicted
        ),
        generated_observations_not_assessable=(
            generated_not_assessable
        ),
        generated_observations_unresolved=(
            generated_unresolved
        ),
        applicable_requirements=len(
            rubric_evaluation
            .applicable_requirement_ids
        ),
        human_targeted_requirements=len(
            rubric_evaluation
            .human_targeted_requirement_ids
        ),
        generated_covered_requirements=len(
            rubric_evaluation
            .generated_covered_requirement_ids
        ),
        human_targeted_components=len(
            rubric_evaluation
            .human_targeted_component_ids
        ),
        generated_covered_components=len(
            rubric_evaluation
            .generated_covered_component_ids
        ),
    )


def calculate_aggregate_metrics(
    *,
    human_alignment_metrics: (
        HumanAlignmentMetrics
    ),
    grounding_metrics: GroundingMetrics,
    rubric_coverage_metrics: (
        RubricCoverageMetrics
    ),
    summary_metrics: SummaryMetrics,
    counts: EvaluationCounts,
) -> AggregateMetrics:
    """
    Assemble the complete aggregate metric record.
    """

    return AggregateMetrics(
        human_alignment=(
            human_alignment_metrics
        ),
        grounding=(
            grounding_metrics
        ),
        rubric_coverage=(
            rubric_coverage_metrics
        ),
        summary_similarity=(
            summary_metrics
        ),
        counts=counts,
    )