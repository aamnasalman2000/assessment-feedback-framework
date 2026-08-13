from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .embedding_service import (
    EmbeddingService,
)
from .models import (
    FeedbackTypeCompatibility,
    HumanObservationResult,
    MatchingConfiguration,
    ObservationMatch,
    RequirementAlignment,
    ScopeAlignment,
)


GeneratedMappingStatus = Literal[
    "matched",
    "partially_matched",
    "no_mapping",
]


class ObservationMatchingError(
    RuntimeError
):
    """
    Raised when feedback observations cannot be
    matched because the structured input is
    malformed or internally inconsistent.
    """


@dataclass(frozen=True)
class GeneratedObservationMatchState:
    """
    Internal matching state for one generated
    observation.

    Grounding is performed later by the grounding
    stage for observations with no_mapping.
    """

    generated_observation_id: str
    mapping_status: GeneratedMappingStatus
    match_ids: tuple[str, ...]


@dataclass(frozen=True)
class MatchingResult:
    """
    Complete output of observation matching before
    grounding.
    """

    observation_matches: tuple[
        ObservationMatch,
        ...
    ]

    human_observation_results: tuple[
        HumanObservationResult,
        ...
    ]

    generated_match_states: tuple[
        GeneratedObservationMatchState,
        ...
    ]


class ObservationMatcher:
    """
    Deterministic structural + embedding-based
    matcher for normalised human feedback and
    generated feedback observations.

    Matching stages:

        1. Exclude explicitly configured
           non-diagnostic human observations.
        2. Determine structural scope
           compatibility.
        3. Compute cosine similarity.
        4. Apply full/partial thresholds.
        5. Apply feedback-type compatibility
           safeguard.
        6. Record requirement overlap as
           diagnostic metadata.
        7. Aggregate pairwise correspondences into
           human and generated mapping outcomes.

    No LLM judge is used.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        configuration: MatchingConfiguration,
    ) -> None:
        self.embedding_service = (
            embedding_service
        )
        self.configuration = configuration

        self._excluded_human_ids = set(
            configuration
            .excluded_human_observation_ids
        )

    def match(
        self,
        *,
        human_feedback: dict[str, Any],
        generated_feedback: dict[str, Any],
    ) -> MatchingResult:
        human_observations = (
            self._get_observations(
                human_feedback,
                source_name="human_feedback",
            )
        )

        generated_observations = (
            self._get_observations(
                generated_feedback,
                source_name="generated_feedback",
            )
        )

        self._validate_unique_ids(
            human_observations,
            id_field="observation_id",
            source_name="human_feedback",
        )

        self._validate_unique_ids(
            generated_observations,
            id_field="observation_id",
            source_name="generated_feedback",
        )

        eligible_human = [
            observation
            for observation
            in human_observations
            if self._is_primary_human_observation(
                observation
            )
        ]

        human_texts = [
            self._human_text(
                observation
            )
            for observation
            in eligible_human
        ]

        generated_texts = [
            self._generated_text(
                observation
            )
            for observation
            in generated_observations
        ]

        similarity_matrix = (
            self.embedding_service
            .similarity_matrix(
                human_texts,
                generated_texts,
            )
        )

        accepted_matches: list[
            ObservationMatch
        ] = []

        human_match_ids: dict[
            str,
            list[str],
        ] = {
            self._observation_id(
                observation
            ): []
            for observation
            in human_observations
        }

        human_match_decisions: dict[
            str,
            list[str],
        ] = {
            self._observation_id(
                observation
            ): []
            for observation
            in human_observations
        }

        generated_match_ids: dict[
            str,
            list[str],
        ] = {
            self._observation_id(
                observation
            ): []
            for observation
            in generated_observations
        }

        generated_match_decisions: dict[
            str,
            list[str],
        ] = {
            self._observation_id(
                observation
            ): []
            for observation
            in generated_observations
        }

        best_candidates: dict[
            str,
            tuple[
                str,
                float,
            ]
            | None,
        ] = {
            self._observation_id(
                observation
            ): None
            for observation
            in eligible_human
        }

        next_match_number = 1

        for human_index, human in enumerate(
            eligible_human
        ):
            human_id = (
                self._observation_id(
                    human
                )
            )

            for (
                generated_index,
                generated,
            ) in enumerate(
                generated_observations
            ):
                scope_alignment = (
                    determine_scope_alignment(
                        human,
                        generated,
                    )
                )

                if (
                    self.configuration
                    .scope_filtering_enabled
                    and scope_alignment
                    is None
                ):
                    continue

                if scope_alignment is None:
                    scope_alignment = "unknown"

                raw_similarity = float(
                    similarity_matrix[
                        human_index,
                        generated_index,
                    ]
                )

                similarity = (
                    _normalise_similarity_for_output(
                        raw_similarity
                    )
                )

                generated_id = (
                    self._observation_id(
                        generated
                    )
                )

                current_best = (
                    best_candidates[
                        human_id
                    ]
                )

                if (
                    current_best is None
                    or similarity
                    > current_best[1]
                ):
                    best_candidates[
                        human_id
                    ] = (
                        generated_id,
                        similarity,
                    )

                decision = (
                    self._semantic_decision(
                        similarity
                    )
                )

                if decision is None:
                    continue

                type_compatibility = (
                    determine_feedback_type_compatibility(
                        self._feedback_type(
                            human
                        ),
                        self._feedback_type(
                            generated
                        ),
                    )
                )

                decision = (
                    self._apply_type_safeguard(
                        decision=decision,
                        compatibility=(
                            type_compatibility
                        ),
                    )
                )

                if decision is None:
                    continue

                requirement_alignment = (
                    calculate_requirement_alignment(
                        human_requirement_ids=(
                            self._requirement_ids(
                                human
                            )
                        ),
                        generated_requirement_ids=(
                            self._requirement_ids(
                                generated
                            )
                        ),
                    )
                )

                match_id = (
                    f"match_"
                    f"{next_match_number:04d}"
                )

                next_match_number += 1

                match = ObservationMatch(
                    match_id=match_id,
                    human_observation_id=(
                        human_id
                    ),
                    generated_observation_id=(
                        generated_id
                    ),
                    decision=decision,
                    semantic_similarity=(
                        similarity
                    ),
                    scope_alignment=(
                        scope_alignment
                    ),
                    feedback_type_compatibility=(
                        type_compatibility
                    ),
                    requirement_alignment=(
                        requirement_alignment
                    ),
                )

                accepted_matches.append(
                    match
                )

                human_match_ids[
                    human_id
                ].append(
                    match_id
                )

                human_match_decisions[
                    human_id
                ].append(
                    decision
                )

                generated_match_ids[
                    generated_id
                ].append(
                    match_id
                )

                generated_match_decisions[
                    generated_id
                ].append(
                    decision
                )

        human_results = (
            self._build_human_results(
                human_observations=(
                    human_observations
                ),
                human_match_ids=(
                    human_match_ids
                ),
                human_match_decisions=(
                    human_match_decisions
                ),
                best_candidates=(
                    best_candidates
                ),
            )
        )

        generated_states = (
            self._build_generated_states(
                generated_observations=(
                    generated_observations
                ),
                generated_match_ids=(
                    generated_match_ids
                ),
                generated_match_decisions=(
                    generated_match_decisions
                ),
            )
        )

        return MatchingResult(
            observation_matches=tuple(
                accepted_matches
            ),
            human_observation_results=tuple(
                human_results
            ),
            generated_match_states=tuple(
                generated_states
            ),
        )

    def _semantic_decision(
        self,
        similarity: float,
    ) -> Literal[
        "matched",
        "partial_match",
    ] | None:
        if (
            similarity
            >= self.configuration
            .full_match_threshold
        ):
            return "matched"

        if (
            similarity
            >= self.configuration
            .partial_match_threshold
        ):
            return "partial_match"

        return None

    @staticmethod
    def _apply_type_safeguard(
        *,
        decision: Literal[
            "matched",
            "partial_match",
        ],
        compatibility: (
            FeedbackTypeCompatibility
        ),
    ) -> Literal[
        "matched",
        "partial_match",
    ] | None:
        """
        A contradictory feedback polarity cannot
        become a full semantic match purely because
        the texts have high embedding similarity.

        For example:

            "The proof is correct."

        versus:

            "The proof is incorrect."

        can have high cosine similarity.

        A contradictory pair is therefore capped at
        partial_match.
        """
        if (
            compatibility
            == "contradictory"
        ):
            return "partial_match"

        return decision

    def _build_human_results(
        self,
        *,
        human_observations: list[
            dict[str, Any]
        ],
        human_match_ids: dict[
            str,
            list[str],
        ],
        human_match_decisions: dict[
            str,
            list[str],
        ],
        best_candidates: dict[
            str,
            tuple[
                str,
                float,
            ]
            | None,
        ],
    ) -> list[
        HumanObservationResult
    ]:
        results: list[
            HumanObservationResult
        ] = []

        for human in human_observations:
            human_id = (
                self._observation_id(
                    human
                )
            )

            feedback_type = (
                self._feedback_type(
                    human
                )
            )

            if (
                not
                self._is_primary_human_observation(
                    human
                )
            ):
                results.append(
                    HumanObservationResult(
                        human_observation_id=(
                            human_id
                        ),
                        feedback_type=(
                            feedback_type
                        ),
                        included_in_primary_matching=(
                            False
                        ),
                        status="excluded",
                        match_ids=[],
                        coverage_score=None,
                        best_candidate_generated_observation_id=(
                            None
                        ),
                        best_candidate_similarity=(
                            None
                        ),
                        exclusion_reason=(
                            "Configured as a "
                            "non-diagnostic human "
                            "observation for "
                            "summary-level "
                            "evaluation only."
                        ),
                    )
                )

                continue

            match_ids = list(
                human_match_ids[
                    human_id
                ]
            )

            decisions = (
                human_match_decisions[
                    human_id
                ]
            )

            best_candidate = (
                best_candidates.get(
                    human_id
                )
            )

            if "matched" in decisions:
                status = "matched"
                coverage_score = 1.0

            elif (
                "partial_match"
                in decisions
            ):
                status = (
                    "partially_matched"
                )

                coverage_score = (
                    self.configuration
                    .partial_match_weight
                )

            else:
                status = "not_matched"
                coverage_score = 0.0

            if best_candidate is None:
                best_id = None
                best_similarity = None

            else:
                (
                    best_id,
                    best_similarity,
                ) = best_candidate

            results.append(
                HumanObservationResult(
                    human_observation_id=(
                        human_id
                    ),
                    feedback_type=(
                        feedback_type
                    ),
                    included_in_primary_matching=(
                        True
                    ),
                    status=status,
                    match_ids=match_ids,
                    coverage_score=(
                        coverage_score
                    ),
                    best_candidate_generated_observation_id=(
                        best_id
                    ),
                    best_candidate_similarity=(
                        best_similarity
                    ),
                    exclusion_reason=None,
                )
            )

        return results

    def _build_generated_states(
        self,
        *,
        generated_observations: list[
            dict[str, Any]
        ],
        generated_match_ids: dict[
            str,
            list[str],
        ],
        generated_match_decisions: dict[
            str,
            list[str],
        ],
    ) -> list[
        GeneratedObservationMatchState
    ]:
        results: list[
            GeneratedObservationMatchState
        ] = []

        for observation in (
            generated_observations
        ):
            generated_id = (
                self._observation_id(
                    observation
                )
            )

            decisions = (
                generated_match_decisions[
                    generated_id
                ]
            )

            if "matched" in decisions:
                mapping_status: (
                    GeneratedMappingStatus
                ) = "matched"

            elif (
                "partial_match"
                in decisions
            ):
                mapping_status = (
                    "partially_matched"
                )

            else:
                mapping_status = (
                    "no_mapping"
                )

            results.append(
                GeneratedObservationMatchState(
                    generated_observation_id=(
                        generated_id
                    ),
                    mapping_status=(
                        mapping_status
                    ),
                    match_ids=tuple(
                        generated_match_ids[
                            generated_id
                        ]
                    ),
                )
            )

        return results

    def _is_primary_human_observation(
        self,
        observation: dict[str, Any],
    ) -> bool:
        """
        Generic overall observations are excluded
        only when their IDs are explicitly supplied
        in the matching configuration.

        We intentionally do not automatically
        exclude every scope='overall' observation,
        because an overall observation may still
        contain a substantive diagnostic claim.
        """
        if (
            not self.configuration
            .exclude_generic_overall_observations
        ):
            return True

        observation_id = (
            self._observation_id(
                observation
            )
        )

        return (
            observation_id
            not in self._excluded_human_ids
        )

    @staticmethod
    def _get_observations(
        feedback: dict[str, Any],
        *,
        source_name: str,
    ) -> list[
        dict[str, Any]
    ]:
        observations = feedback.get(
            "observations"
        )

        if observations is None:
            raise ObservationMatchingError(
                f"{source_name} does not "
                "contain an observations field."
            )

        if not isinstance(
            observations,
            list,
        ):
            raise ObservationMatchingError(
                f"{source_name}.observations "
                "must be a list."
            )

        for index, observation in enumerate(
            observations
        ):
            if not isinstance(
                observation,
                dict,
            ):
                raise ObservationMatchingError(
                    f"{source_name}.observations"
                    f"[{index}] must be an "
                    "object."
                )

        return observations

    @staticmethod
    def _validate_unique_ids(
        observations: list[
            dict[str, Any]
        ],
        *,
        id_field: str,
        source_name: str,
    ) -> None:
        seen: set[str] = set()

        for observation in observations:
            value = observation.get(
                id_field
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ObservationMatchingError(
                    f"Every observation in "
                    f"{source_name} must have a "
                    f"non-empty {id_field}."
                )

            value = value.strip()

            if value in seen:
                raise ObservationMatchingError(
                    f"Duplicate observation ID "
                    f"{value!r} in "
                    f"{source_name}."
                )

            seen.add(
                value
            )

    @staticmethod
    def _observation_id(
        observation: dict[str, Any],
    ) -> str:
        value = observation.get(
            "observation_id"
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ObservationMatchingError(
                "Observation is missing a valid "
                "observation_id."
            )

        return value.strip()

    @staticmethod
    def _human_text(
        observation: dict[str, Any],
    ) -> str:
        text = observation.get(
            "text"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            raise ObservationMatchingError(
                "Human observation is missing "
                "non-empty text."
            )

        return text.strip()

    @staticmethod
    def _generated_text(
        observation: dict[str, Any],
    ) -> str:
        text = observation.get(
            "student_feedback"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            raise ObservationMatchingError(
                "Generated observation is missing "
                "non-empty student_feedback."
            )

        return text.strip()

    @staticmethod
    def _feedback_type(
        observation: dict[str, Any],
    ) -> str:
        feedback_type = observation.get(
            "feedback_type"
        )

        allowed = {
            "strength",
            "error",
            "omission",
            "suggestion",
        }

        if feedback_type not in allowed:
            raise ObservationMatchingError(
                "Observation has invalid "
                f"feedback_type: "
                f"{feedback_type!r}."
            )

        return feedback_type

    @staticmethod
    def _requirement_ids(
        observation: dict[str, Any],
    ) -> list[str]:
        value = observation.get(
            "requirement_ids",
            [],
        )

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            raise ObservationMatchingError(
                "requirement_ids must be a list."
            )

        result: list[str] = []

        for requirement_id in value:
            if (
                isinstance(
                    requirement_id,
                    str,
                )
                and requirement_id.strip()
            ):
                result.append(
                    requirement_id.strip()
                )

        return list(
            dict.fromkeys(
                result
            )
        )


def determine_scope_alignment(
    human: dict[str, Any],
    generated: dict[str, Any],
) -> ScopeAlignment | None:
    """
    Determine whether two observations are
    structurally compatible.

    Returns None when the structured metadata
    explicitly places the observations in
    incompatible assessment components.

    Otherwise returns diagnostic scope metadata:

        exact
        human_broader
        generated_broader
        compatible
        unknown

    Scope metadata is not converted into a
    numerical weight.
    """

    human_attempts = _id_set(
        human,
        "attempt_unit_ids",
    )

    generated_attempts = _id_set(
        generated,
        "attempt_unit_ids",
    )

    human_tasks = _id_set(
        human,
        "task_ids",
    )

    generated_tasks = _id_set(
        generated,
        "task_ids",
    )

    human_parts = _id_set(
        human,
        "part_ids",
    )

    generated_parts = _id_set(
        generated,
        "part_ids",
    )

    # --------------------------------------------------
    # Attempt-level evidence is the most specific scope.
    # --------------------------------------------------

    if (
        human_attempts
        and generated_attempts
    ):
        overlap = (
            human_attempts
            & generated_attempts
        )

        if not overlap:
            return None

        if (
            human_attempts
            == generated_attempts
        ):
            return "exact"

        if (
            generated_attempts
            <= human_attempts
        ):
            return "human_broader"

        if (
            human_attempts
            <= generated_attempts
        ):
            return "generated_broader"

        return "compatible"

    # --------------------------------------------------
    # Task-level scope.
    # --------------------------------------------------

    if (
        human_tasks
        and generated_tasks
    ):
        overlap = (
            human_tasks
            & generated_tasks
        )

        if not overlap:
            return None

        if human_tasks == generated_tasks:
            return "exact"

        if (
            generated_tasks
            <= human_tasks
        ):
            return "human_broader"

        if (
            human_tasks
            <= generated_tasks
        ):
            return "generated_broader"

        return "compatible"

    # --------------------------------------------------
    # Human has broader part scope while generated
    # observation is narrower.
    # --------------------------------------------------

    if (
        human_parts
        and generated_parts
    ):
        part_overlap = (
            human_parts
            & generated_parts
        )

        if not part_overlap:
            return None

        if (
            human_parts
            == generated_parts
        ):
            if (
                human_tasks
                and not generated_tasks
            ):
                return (
                    "generated_broader"
                )

            if (
                generated_tasks
                and not human_tasks
            ):
                return "human_broader"

            return "exact"

        if (
            generated_parts
            <= human_parts
        ):
            return "human_broader"

        if (
            human_parts
            <= generated_parts
        ):
            return "generated_broader"

        return "compatible"

    # --------------------------------------------------
    # Only the human side has task metadata.
    #
    # If the generated observation is mapped to the same
    # part we can still compare it, but it is broader.
    # --------------------------------------------------

    if (
        human_tasks
        and not generated_tasks
    ):
        if (
            human_parts
            and generated_parts
        ):
            if not (
                human_parts
                & generated_parts
            ):
                return None

            return "generated_broader"

        return "unknown"

    # --------------------------------------------------
    # Only the generated side has task metadata.
    # --------------------------------------------------

    if (
        generated_tasks
        and not human_tasks
    ):
        if (
            human_parts
            and generated_parts
        ):
            if not (
                human_parts
                & generated_parts
            ):
                return None

            return "human_broader"

        return "unknown"

    # --------------------------------------------------
    # Part metadata exists on only one side.
    # There is no explicit contradiction.
    # --------------------------------------------------

    if human_parts or generated_parts:
        return "unknown"

    # --------------------------------------------------
    # Neither observation has usable structural
    # metadata. Semantic comparison remains possible.
    # --------------------------------------------------

    return "unknown"


def determine_feedback_type_compatibility(
    human_type: str,
    generated_type: str,
) -> FeedbackTypeCompatibility:
    """
    Deterministic feedback-type compatibility
    safeguard.

    Exact communicative types are compatible.

    Error, omission and suggestion may express the
    same underlying deficiency in different ways.

    Strength versus error/omission is treated as
    contradictory.

    Strength versus suggestion is potentially
    compatible because a marker may praise an area
    while still recommending refinement.
    """

    allowed = {
        "strength",
        "error",
        "omission",
        "suggestion",
    }

    if human_type not in allowed:
        raise ObservationMatchingError(
            "Invalid human feedback type: "
            f"{human_type!r}."
        )

    if generated_type not in allowed:
        raise ObservationMatchingError(
            "Invalid generated feedback type: "
            f"{generated_type!r}."
        )

    if human_type == generated_type:
        return "compatible"

    corrective_types = {
        "error",
        "omission",
        "suggestion",
    }

    if (
        human_type in corrective_types
        and generated_type
        in corrective_types
    ):
        return "potentially_compatible"

    pair = {
        human_type,
        generated_type,
    }

    if pair in (
        {
            "strength",
            "error",
        },
        {
            "strength",
            "omission",
        },
    ):
        return "contradictory"

    if pair == {
        "strength",
        "suggestion",
    }:
        return "potentially_compatible"

    return "potentially_compatible"


def calculate_requirement_alignment(
    *,
    human_requirement_ids: list[str],
    generated_requirement_ids: list[str],
) -> RequirementAlignment:
    """
    Calculate requirement overlap for diagnostic
    purposes.

    Human requirement IDs are treated as unknown
    when absent. An empty human list therefore
    yields not_assessable rather than no_overlap.

    Requirement overlap does not determine whether
    two observations semantically match.
    """

    human_ids = {
        value.strip()
        for value
        in human_requirement_ids
        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        )
    }

    generated_ids = {
        value.strip()
        for value
        in generated_requirement_ids
        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        )
    }

    if not human_ids:
        return RequirementAlignment(
            status="not_assessable",
            human_requirement_ids=[],
            generated_requirement_ids=(
                sorted(
                    generated_ids
                )
            ),
            shared_requirement_ids=[],
            jaccard_similarity=None,
        )

    shared = (
        human_ids
        & generated_ids
    )

    union = (
        human_ids
        | generated_ids
    )

    if not generated_ids:
        return RequirementAlignment(
            status="no_overlap",
            human_requirement_ids=(
                sorted(
                    human_ids
                )
            ),
            generated_requirement_ids=[],
            shared_requirement_ids=[],
            jaccard_similarity=0.0,
        )

    if human_ids == generated_ids:
        status = "exact"

    elif shared:
        status = "overlap"

    else:
        status = "no_overlap"

    jaccard = (
        len(
            shared
        )
        / len(
            union
        )
        if union
        else 0.0
    )

    return RequirementAlignment(
        status=status,
        human_requirement_ids=(
            sorted(
                human_ids
            )
        ),
        generated_requirement_ids=(
            sorted(
                generated_ids
            )
        ),
        shared_requirement_ids=(
            sorted(
                shared
            )
        ),
        jaccard_similarity=(
            float(
                jaccard
            )
        ),
    )


def _id_set(
    observation: dict[str, Any],
    field: str,
) -> set[str]:
    value = observation.get(
        field,
        [],
    )

    if value is None:
        return set()

    if not isinstance(
        value,
        list,
    ):
        raise ObservationMatchingError(
            f"{field} must be a list."
        )

    return {
        item.strip()
        for item in value
        if (
            isinstance(
                item,
                str,
            )
            and item.strip()
        )
    }


def _normalise_similarity_for_output(
    similarity: float,
) -> float:
    """
    ObservationMatch stores similarities in the
    schema's [0, 1] interval.

    Cosine similarity may technically be negative,
    so negative values are clipped to zero.

    Values above one caused by floating-point
    precision are clipped to one.
    """
    if np.isnan(
        similarity
    ):
        raise ObservationMatchingError(
            "Embedding similarity produced NaN."
        )

    return float(
        max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )
    )