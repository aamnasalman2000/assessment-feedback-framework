from __future__ import annotations

from pathlib import Path
from typing import Any

from .feedback_generator import (
    _build_alignment_record,
    _collect_components,
    _collect_processing_evidence,
    _get_feedback_requirements,
    _get_relevant_specification_notes,
    _select_candidate_units,
    _select_semantic_annotations,
    _to_dict,
)
from .feedback_input_models import (
    FeedbackGenerationInput,
)
from .feedback_llm_client import (
    FeedbackStructuredClient,
)
from .feedback_models import (
    CriterionAssessment,
    FeedbackLLMOutput,
    FeedbackObservation,
    FeedbackSection,
)
from .feedback_reflection_service import (
    FeedbackReflectionService,
)
from .baseline_feedback_service import (
    BaselineFeedbackService,
)


class BaselineReflectionError(
    RuntimeError
):
    """
    Raised when Pipeline B cannot map or reflect
    Pipeline A feedback safely.
    """


class BaselineReflectionService:
    """
    Pipeline B:

        baseline component prompting
        -> existing requirement-level self-reflection

    The reflection mechanism is intentionally the same
    FeedbackReflectionService used by Pipeline D.

    Pipeline A produces all criterion assessments for a
    component in one generation call.

    Pipeline B then iterates over those criterion
    assessments and supplies each one to the existing
    reflection service.

    This keeps the experimental reflection mechanism
    constant between B and D while preserving the
    baseline generation strategy used by A.
    """

    def __init__(
        self,
        *,
        llm_client: FeedbackStructuredClient,
        checkpoint_dir: str | Path | None = None,
        resume_from_checkpoints: bool = True,
    ) -> None:
        self.llm_client = (
            llm_client
        )

        self.reflection_service = (
            FeedbackReflectionService(
                llm_client=(
                    llm_client
                ),
                checkpoint_dir=(
                    checkpoint_dir
                ),
                resume_from_checkpoints=(
                    resume_from_checkpoints
                ),
            )
        )

    # ========================================================
    # Public entry point
    # ========================================================

    def reflect_feedback(
        self,
        *,
        generation_input: FeedbackGenerationInput,
        baseline_result: FeedbackLLMOutput,
        log_prefix: str = "baseline_reflection",
    ) -> FeedbackLLMOutput:
        """
        Reflect a complete Pipeline A result.

        Each baseline CriterionAssessment is audited by the
        same requirement-level reflection service used by
        Pipeline D.

        Final criterion assessments and observations are
        reassembled into a Pipeline B FeedbackLLMOutput.
        """

        specification = _to_dict(
            generation_input
            .assessment_specification
        )

        if not isinstance(
            specification,
            dict,
        ):
            raise BaselineReflectionError(
                "assessment_specification must "
                "serialize to a dictionary."
            )

        component_lookup = (
            self._build_requirement_lookup(
                specification
            )
        )

        (
            processing_checks,
            processing_diagnostics,
        ) = (
            _collect_processing_evidence(
                generation_input
            )
        )

        global_requirements = (
            specification.get(
                "global_requirements",
                [],
            )
        )

        if not isinstance(
            global_requirements,
            list,
        ):
            global_requirements = []

        feedback_requirements = (
            _get_feedback_requirements(
                specification
            )
        )

        # Cache component evidence so all criteria belonging
        # to one component reuse the same baseline evidence
        # context.
        component_context_cache: dict[
            str,
            dict[str, Any],
        ] = {}

        final_criteria: list[
            CriterionAssessment
        ] = []

        final_observations: list[
            FeedbackObservation
        ] = []

        observation_counter = 1

        for initial_criterion in (
            baseline_result
            .criterion_assessments
        ):
            requirement_id = (
                initial_criterion
                .requirement_id
            )

            lookup_record = (
                component_lookup.get(
                    requirement_id
                )
            )

            if lookup_record is None:
                raise BaselineReflectionError(
                    "Could not locate requirement "
                    f"{requirement_id!r} in the "
                    "assessment specification."
                )

            part_id = lookup_record[
                "part_id"
            ]

            component = lookup_record[
                "component"
            ]

            requirement = lookup_record[
                "requirement"
            ]

            component_id = component[
                "component_id"
            ]

            # ----------------------------------------------
            # Reuse component-level baseline evidence
            # ----------------------------------------------

            if (
                component_id
                not in component_context_cache
            ):
                candidate_units = (
                    _select_candidate_units(
                        generation_input=(
                            generation_input
                        ),
                        component=component,
                        part_id=part_id,
                    )
                )

                semantic_annotations = (
                    _select_semantic_annotations(
                        generation_input=(
                            generation_input
                        ),
                        candidate_units=(
                            candidate_units
                        ),
                    )
                )

                alignment = (
                    _build_alignment_record(
                        component=component,
                        part_id=part_id,
                        candidate_units=(
                            candidate_units
                        ),
                    )
                )

                specification_notes = (
                    _get_relevant_specification_notes(
                        specification=(
                            specification
                        ),
                        component_id=(
                            component_id
                        ),
                    )
                )

                component_context_cache[
                    component_id
                ] = {
                    "candidate_units": (
                        candidate_units
                    ),
                    "semantic_annotations": (
                        semantic_annotations
                    ),
                    "alignment": (
                        alignment
                    ),
                    "specification_notes": (
                        specification_notes
                    ),
                }

            context = (
                component_context_cache[
                    component_id
                ]
            )

            # ----------------------------------------------
            # Build requirement-specific baseline
            # observation for the existing D reflector
            # ----------------------------------------------

            initial_observation = (
                self
                ._build_initial_observation(
                    initial_criterion=(
                        initial_criterion
                    ),
                    baseline_result=(
                        baseline_result
                    ),
                    component_id=(
                        component_id
                    ),
                    part_id=part_id,
                    observation_id=(
                        "b_initial_obs_"
                        f"{observation_counter:03d}"
                    ),
                )
            )

            # ----------------------------------------------
            # Existing D reflection service
            # ----------------------------------------------

            (
                _decision,
                final_criterion,
                final_observation,
            ) = (
                self.reflection_service
                .reflect_requirement_feedback(
                    component=component,
                    requirement=requirement,
                    part_id=part_id,
                    candidate_units=(
                        context[
                            "candidate_units"
                        ]
                    ),
                    semantic_annotations=(
                        context[
                            "semantic_annotations"
                        ]
                    ),
                    alignment=(
                        context[
                            "alignment"
                        ]
                    ),
                    processing_checks=(
                        processing_checks
                    ),
                    processing_diagnostics=(
                        processing_diagnostics
                    ),
                    initial_criterion_assessment=(
                        initial_criterion
                    ),
                    initial_observation=(
                        initial_observation
                    ),
                    global_requirements=(
                        global_requirements
                    ),
                    feedback_requirements=(
                        feedback_requirements
                    ),
                    specification_notes=(
                        context[
                            "specification_notes"
                        ]
                    ),
                    log_name=(
                        f"{log_prefix}_"
                        f"{component_id}_"
                        f"{requirement_id}"
                    ),
                )
            )

            # Give Pipeline B observations deterministic,
            # unique IDs while retaining reflected content.
            final_observation = (
                final_observation
                .model_copy(
                    update={
                        "observation_id": (
                            "b_obs_"
                            f"{observation_counter:03d}"
                        ),
                        "requirement_ids": [
                            requirement_id
                        ],
                        "criterion_assessment_ids": [
                            final_criterion
                            .criterion_assessment_id
                        ],
                    },
                    deep=True,
                )
            )

            final_criteria.append(
                final_criterion
            )

            final_observations.append(
                final_observation
            )

            observation_counter += 1

        # ----------------------------------------------
        # Rebuild sections around reflected observations
        # ----------------------------------------------

        final_sections = (
            self._rebuild_sections(
                baseline_sections=(
                    baseline_result
                    .feedback_sections
                ),
                observations=(
                    final_observations
                ),
            )
        )

        overall_feedback = (
            BaselineFeedbackService
            ._build_overall_feedback(
                component_summaries=[
                    section.summary
                    for section
                    in final_sections
                ],
                observations=(
                    final_observations
                ),
            )
        )

        return (
            baseline_result.model_copy(
                update={
                    "criterion_assessments": (
                        final_criteria
                    ),
                    "observations": (
                        final_observations
                    ),
                    "feedback_sections": (
                        final_sections
                    ),
                    "overall_feedback": (
                        overall_feedback
                    ),
                },
                deep=True,
            )
        )

    # ========================================================
    # Requirement lookup
    # ========================================================

    @staticmethod
    def _build_requirement_lookup(
        specification: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Build:

            requirement_id ->
                part_id
                component
                requirement

        Supports both Assessment 1 and Assessment 2 through
        the shared _collect_components() helper.
        """

        lookup: dict[
            str,
            dict[str, Any],
        ] = {}

        for (
            part_id,
            component,
        ) in _collect_components(
            specification
        ):
            component_id = (
                component.get(
                    "component_id"
                )
            )

            if not isinstance(
                component_id,
                str,
            ):
                continue

            requirements = (
                component.get(
                    "evaluation_requirements",
                    [],
                )
            )

            if not isinstance(
                requirements,
                list,
            ):
                continue

            for requirement in (
                requirements
            ):
                if not isinstance(
                    requirement,
                    dict,
                ):
                    continue

                requirement_id = (
                    requirement.get(
                        "id"
                    )
                )

                if not isinstance(
                    requirement_id,
                    str,
                ):
                    continue

                if (
                    requirement_id
                    in lookup
                ):
                    raise (
                        BaselineReflectionError(
                            "Duplicate requirement ID "
                            "found in specification: "
                            f"{requirement_id!r}."
                        )
                    )

                lookup[
                    requirement_id
                ] = {
                    "part_id": (
                        part_id
                    ),
                    "component": (
                        component
                    ),
                    "requirement": (
                        requirement
                    ),
                }

        return lookup

    # ========================================================
    # Initial observation adapter
    # ========================================================

    @staticmethod
    def _build_initial_observation(
        *,
        initial_criterion: CriterionAssessment,
        baseline_result: FeedbackLLMOutput,
        component_id: str,
        part_id: str | None,
        observation_id: str,
    ) -> FeedbackObservation:
        """
        Adapt Pipeline A criterion output into the
        requirement-level FeedbackObservation expected by
        FeedbackReflectionService.

        Prefer an A observation dedicated to this exact
        requirement.

        If A grouped multiple requirements into one
        observation, do not copy the grouped feedback because
        it may discuss unrelated criteria. Instead derive a
        requirement-specific observation from the criterion
        assessment itself.
        """

        requirement_id = (
            initial_criterion
            .requirement_id
        )

        exact_observation = None

        for observation in (
            baseline_result
            .observations
        ):
            if (
                observation.requirement_ids
                == [
                    requirement_id
                ]
            ):
                exact_observation = (
                    observation
                )
                break

        scope = (
            "part"
            if (
                part_id is not None
                and component_id
                == part_id
            )
            else "task"
        )

        if (
            exact_observation
            is not None
        ):
            student_feedback = (
                exact_observation
                .student_feedback
            )

            suggestion = (
                exact_observation
                .suggestion
            )

            attempt_unit_ids = list(
                exact_observation
                .attempt_unit_ids
            )

        else:
            # Pipeline A may intentionally group several
            # criteria into one student-facing observation.
            #
            # Reflection operates on one criterion at a time,
            # so use the criterion's own finding rather than
            # leaking other criteria into this audit.
            student_feedback = (
                initial_criterion
                .internal_finding
            )

            suggestion = None

            attempt_unit_ids = []

        return FeedbackObservation(
            observation_id=(
                observation_id
            ),
            feedback_type=(
                FeedbackReflectionService
                ._feedback_type_for_status(
                    initial_criterion
                    .status
                )
            ),
            scope=scope,
            part_ids=list(
                initial_criterion
                .part_ids
            ),
            task_ids=list(
                initial_criterion
                .task_ids
            ),
            attempt_unit_ids=(
                attempt_unit_ids
            ),
            requirement_ids=[
                requirement_id
            ],
            criterion_assessment_ids=[
                initial_criterion
                .criterion_assessment_id
            ],
            internal_finding=(
                initial_criterion
                .internal_finding
            ),
            student_feedback=(
                student_feedback
            ),
            suggestion=(
                suggestion
            ),
            verification_status=(
                initial_criterion
                .verification_status
            ),
            confidence=(
                initial_criterion
                .confidence
            ),
            evidence=list(
                initial_criterion
                .evidence
            ),
        )

    # ========================================================
    # Section rebuilding
    # ========================================================

    @staticmethod
    def _rebuild_sections(
        *,
        baseline_sections: list[
            FeedbackSection
        ],
        observations: list[
            FeedbackObservation
        ],
    ) -> list[
        FeedbackSection
    ]:
        """
        Preserve Pipeline A section structure while replacing
        observation references and summaries with reflected
        Pipeline B content.
        """

        rebuilt: list[
            FeedbackSection
        ] = []

        for section in (
            baseline_sections
        ):
            matching_observations = [
                observation
                for observation
                in observations
                if (
                    BaselineReflectionService
                    ._observation_matches_section(
                        observation=(
                            observation
                        ),
                        section=section,
                    )
                )
            ]

            observation_ids = [
                observation
                .observation_id
                for observation
                in matching_observations
            ]

            summary_parts = [
                observation
                .student_feedback
                .strip()
                for observation
                in matching_observations
                if (
                    isinstance(
                        observation
                        .student_feedback,
                        str,
                    )
                    and observation
                    .student_feedback
                    .strip()
                )
            ]

            if summary_parts:
                summary = " ".join(
                    summary_parts[
                        :3
                    ]
                )

            else:
                summary = (
                    section.summary
                )

            rebuilt.append(
                section.model_copy(
                    update={
                        "summary": (
                            summary
                        ),
                        "observation_ids": (
                            observation_ids
                        ),
                    },
                    deep=True,
                )
            )

        return rebuilt

    @staticmethod
    def _observation_matches_section(
        *,
        observation: FeedbackObservation,
        section: FeedbackSection,
    ) -> bool:
        """
        Match a reflected observation back to its original
        component section using Python-owned scope IDs.
        """

        section_task_ids = set(
            section.task_ids
        )

        observation_task_ids = set(
            observation.task_ids
        )

        if (
            section_task_ids
            and (
                section_task_ids
                & observation_task_ids
            )
        ):
            return True

        section_part_ids = set(
            section.part_ids
        )

        observation_part_ids = set(
            observation.part_ids
        )

        if (
            section_part_ids
            and (
                section_part_ids
                & observation_part_ids
            )
            and not section_task_ids
        ):
            return True

        return False