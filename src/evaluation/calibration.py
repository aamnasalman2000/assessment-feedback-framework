from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embedding_service import (
    EmbeddingService,
)
from .matching import (
    calculate_requirement_alignment,
    determine_feedback_type_compatibility,
    determine_scope_alignment,
)


@dataclass(frozen=True)
class CalibrationPair:
    human_observation_id: str
    generated_observation_id: str

    human_text: str
    generated_text: str

    human_feedback_type: str
    generated_feedback_type: str

    human_scope: str | None
    generated_scope: str | None

    human_part_ids: tuple[str, ...]
    generated_part_ids: tuple[str, ...]

    human_task_ids: tuple[str, ...]
    generated_task_ids: tuple[str, ...]

    human_requirement_ids: tuple[str, ...]
    generated_requirement_ids: tuple[str, ...]

    scope_alignment: str
    feedback_type_compatibility: str
    requirement_alignment: str

    semantic_similarity: float

    manual_label: str = ""
    annotation_notes: str = ""


class CalibrationExporter:
    """
    Build manually labelable observation pairs for
    calibration of semantic matching thresholds.

    The exporter applies the same deterministic
    candidate filtering used by the matcher:

        - structural scope compatibility;
        - rejection of contradictory feedback
          polarity;
        - rejection of broad-scope pairs with
          explicit requirement non-overlap.

    It does NOT apply semantic similarity thresholds.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
    ) -> None:
        self.embedding_service = (
            embedding_service
        )

    def build_pairs(
        self,
        *,
        human_feedback: dict[str, Any],
        generated_feedback: dict[str, Any],
        excluded_human_observation_ids: (
            set[str] | None
        ) = None,
    ) -> list[CalibrationPair]:
        excluded_ids = (
            excluded_human_observation_ids
            or set()
        )

        human_observations = (
            human_feedback.get(
                "observations",
                [],
            )
        )

        generated_observations = (
            generated_feedback.get(
                "observations",
                [],
            )
        )

        if not isinstance(
            human_observations,
            list,
        ):
            raise ValueError(
                "human_feedback.observations "
                "must be a list."
            )

        if not isinstance(
            generated_observations,
            list,
        ):
            raise ValueError(
                "generated_feedback.observations "
                "must be a list."
            )

        eligible_human = [
            observation
            for observation
            in human_observations
            if observation.get(
                "observation_id"
            )
            not in excluded_ids
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

        pairs: list[
            CalibrationPair
        ] = []

        for human_index, human in enumerate(
            eligible_human
        ):
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

                if scope_alignment is None:
                    continue

                human_type = (
                    self._feedback_type(
                        human
                    )
                )

                generated_type = (
                    self._feedback_type(
                        generated
                    )
                )

                type_compatibility = (
                    determine_feedback_type_compatibility(
                        human_type,
                        generated_type,
                    )
                )

                if (
                    type_compatibility
                    == "contradictory"
                ):
                    continue

                human_requirement_ids = (
                    self._ids(
                        human,
                        "requirement_ids",
                    )
                )

                generated_requirement_ids = (
                    self._ids(
                        generated,
                        "requirement_ids",
                    )
                )

                requirement_alignment = (
                    calculate_requirement_alignment(
                        human_requirement_ids=list(
                            human_requirement_ids
                        ),
                        generated_requirement_ids=list(
                            generated_requirement_ids
                        ),
                    )
                )

                if (
                    requirement_alignment.status
                    == "no_overlap"
                    and scope_alignment
                    != "exact"
                ):
                    continue

                similarity = float(
                    similarity_matrix[
                        human_index,
                        generated_index,
                    ]
                )

                similarity = max(
                    0.0,
                    min(
                        1.0,
                        similarity,
                    ),
                )

                pairs.append(
                    CalibrationPair(
                        human_observation_id=(
                            self._observation_id(
                                human
                            )
                        ),
                        generated_observation_id=(
                            self._observation_id(
                                generated
                            )
                        ),
                        human_text=(
                            self._human_text(
                                human
                            )
                        ),
                        generated_text=(
                            self._generated_text(
                                generated
                            )
                        ),
                        human_feedback_type=(
                            human_type
                        ),
                        generated_feedback_type=(
                            generated_type
                        ),
                        human_scope=(
                            human.get(
                                "scope"
                            )
                        ),
                        generated_scope=(
                            generated.get(
                                "scope"
                            )
                        ),
                        human_part_ids=tuple(
                            self._ids(
                                human,
                                "part_ids",
                            )
                        ),
                        generated_part_ids=tuple(
                            self._ids(
                                generated,
                                "part_ids",
                            )
                        ),
                        human_task_ids=tuple(
                            self._ids(
                                human,
                                "task_ids",
                            )
                        ),
                        generated_task_ids=tuple(
                            self._ids(
                                generated,
                                "task_ids",
                            )
                        ),
                        human_requirement_ids=tuple(
                            human_requirement_ids
                        ),
                        generated_requirement_ids=tuple(
                            generated_requirement_ids
                        ),
                        scope_alignment=(
                            scope_alignment
                        ),
                        feedback_type_compatibility=(
                            type_compatibility
                        ),
                        requirement_alignment=(
                            requirement_alignment
                            .status
                        ),
                        semantic_similarity=(
                            similarity
                        ),
                    )
                )

        return pairs

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
            raise ValueError(
                "Observation is missing a valid "
                "observation_id."
            )

        return value.strip()

    @staticmethod
    def _human_text(
        observation: dict[str, Any],
    ) -> str:
        value = observation.get(
            "text"
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                "Human observation is missing "
                "non-empty text."
            )

        return value.strip()

    @staticmethod
    def _generated_text(
        observation: dict[str, Any],
    ) -> str:
        value = observation.get(
            "student_feedback"
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                "Generated observation is missing "
                "non-empty student_feedback."
            )

        return value.strip()

    @staticmethod
    def _feedback_type(
        observation: dict[str, Any],
    ) -> str:
        value = observation.get(
            "feedback_type"
        )

        allowed = {
            "strength",
            "error",
            "omission",
            "suggestion",
        }

        if value not in allowed:
            raise ValueError(
                "Invalid feedback_type: "
                f"{value!r}"
            )

        return value

    @staticmethod
    def _ids(
        observation: dict[str, Any],
        field: str,
    ) -> list[str]:
        values = observation.get(
            field,
            [],
        )

        if values is None:
            return []

        if not isinstance(
            values,
            list,
        ):
            raise ValueError(
                f"{field} must be a list."
            )

        result: list[str] = []

        for value in values:
            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                result.append(
                    value.strip()
                )

        return list(
            dict.fromkeys(
                result
            )
        )