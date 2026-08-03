from __future__ import annotations

from typing import Any

from src.alignment.alignment_models import (
    AlignmentResult,
    ComponentAlignment,
)
from src.extraction.semantic_models import (
    SemanticExtractionResult,
)

from .feedback_input_models import (
    ProcessedSubmission,
)
from .feedback_llm_client import (
    FeedbackStructuredClient,
)
from .feedback_models import (
    CriterionAssessment,
    FeedbackLLMOutput,
    FeedbackObservation,
    FeedbackSection,
    OverallFeedback,
    RequirementFeedbackLLMOutput,
)
from .feedback_prompts import (
    build_requirement_feedback_prompt,
)
from .feedback_reflection_service import (
    FeedbackReflectionService,
)


class FeedbackServiceError(RuntimeError):
    """Raised when requirement-level feedback generation fails."""


class FeedbackService:
    """
    Generate feedback one assessment requirement at a time.

    Python controls all requirement IDs, feedback IDs, scopes, and
    cross-references.

    When self-reflection is enabled, the initial structured feedback records
    are audited by the reflection service. The reflection model returns only
    an audit and optional patch; Python retains ownership of all structural
    metadata.
    """

    def __init__(
        self,
        *,
        llm_client: FeedbackStructuredClient,
        reflection_service: (
            FeedbackReflectionService | None
        ) = None,
    ) -> None:
        self.llm_client = llm_client
        self.reflection_service = reflection_service

    def generate_feedback(
        self,
        *,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        alignment_result: AlignmentResult,
        assessment_specification: dict[str, Any],
        self_reflective: bool = False,
    ) -> FeedbackLLMOutput:
        self._validate_submission_identifiers(
            processed_submission=processed_submission,
            semantic_extraction=semantic_extraction,
            alignment_result=alignment_result,
        )

        if (
            self_reflective
            and self.reflection_service is None
        ):
            raise FeedbackServiceError(
                "Self-reflective feedback was requested, "
                "but no FeedbackReflectionService was "
                "provided."
            )

        component_lookup = (
            self._build_component_lookup(
                assessment_specification
            )
        )

        (
            unit_lookup,
            unit_artifact_lookup,
        ) = self._build_unit_lookups(
            processed_submission
        )

        semantic_lookup = {
            annotation.unit_id: annotation
            for annotation
            in semantic_extraction.unit_annotations
        }

        global_requirements = (
            assessment_specification.get(
                "global_requirements",
                [],
            )
        )

        evaluation_policy = (
            assessment_specification.get(
                "evaluation_policy",
                {},
            )
        )

        feedback_requirements = (
            evaluation_policy.get(
                "feedback_requirements",
                [],
            )
        )

        all_specification_notes = (
            assessment_specification.get(
                "specification_notes",
                [],
            )
        )

        criterion_assessments: list[
            CriterionAssessment
        ] = []

        observations: list[
            FeedbackObservation
        ] = []

        feedback_sections: list[
            FeedbackSection
        ] = []

        overall_strengths: list[str] = []
        overall_improvements: list[str] = []
        component_summaries: list[str] = []

        criterion_counter = 1
        observation_counter = 1
        section_counter = 1

        for alignment in (
            alignment_result.component_alignments
        ):
            component_id = alignment.component_id

            component_record = (
                component_lookup.get(component_id)
            )

            if component_record is None:
                raise FeedbackServiceError(
                    "Alignment references unknown "
                    f"component {component_id!r}."
                )

            part_id = component_record["part_id"]
            component = component_record["component"]

            candidate_unit_ids = (
                self._collect_candidate_unit_ids(
                    alignment
                )
            )

            candidate_units: list[
                dict[str, Any]
            ] = []

            semantic_annotations: list[Any] = []
            candidate_artifact_ids: set[str] = set()

            for unit_id in candidate_unit_ids:
                unit = unit_lookup.get(unit_id)

                if unit is None:
                    raise FeedbackServiceError(
                        f"Component {component_id!r} "
                        "references unknown unit "
                        f"{unit_id!r}."
                    )

                artifact_id = (
                    unit_artifact_lookup[unit_id]
                )

                candidate_artifact_ids.add(
                    artifact_id
                )

                unit_data = unit.model_dump(
                    mode="json",
                    exclude_none=True,
                )

                unit_data["artifact_id"] = (
                    artifact_id
                )

                candidate_units.append(unit_data)

                annotation = semantic_lookup.get(
                    unit_id
                )

                if annotation is not None:
                    semantic_annotations.append(
                        annotation
                    )

            (
                processing_checks,
                processing_diagnostics,
            ) = self._collect_processing_evidence(
                processed_submission=(
                    processed_submission
                ),
                artifact_ids=(
                    candidate_artifact_ids
                ),
            )

            specification_notes = [
                note
                for note in all_specification_notes
                if (
                    not isinstance(note, dict)
                    or note.get("component_id")
                    in {
                        None,
                        component_id,
                    }
                )
            ]

            alignment_data = alignment.model_dump(
                mode="json",
                exclude_none=True,
            )

            component_observation_ids: list[
                str
            ] = []

            component_feedback_texts: list[
                str
            ] = []

            requirements = component.get(
                "evaluation_requirements",
                [],
            )

            for requirement in requirements:
                requirement_id = requirement.get(
                    "id"
                )

                if not isinstance(
                    requirement_id,
                    str,
                ):
                    continue

                print(
                    "Generating feedback for "
                    f"{component_id}/"
                    f"{requirement_id}..."
                )

                system_prompt, user_prompt = (
                    build_requirement_feedback_prompt(
                        component=component,
                        requirement=requirement,
                        part_id=part_id,
                        candidate_units=(
                            candidate_units
                        ),
                        semantic_annotations=(
                            semantic_annotations
                        ),
                        alignment=alignment_data,
                        processing_checks=(
                            processing_checks
                        ),
                        processing_diagnostics=(
                            processing_diagnostics
                        ),
                        global_requirements=(
                            global_requirements
                        ),
                        feedback_requirements=(
                            feedback_requirements
                        ),
                        specification_notes=(
                            specification_notes
                        ),
                    )
                )

                initial_feedback = (
                    self.llm_client
                    .generate_requirement_feedback(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        candidate_artifact_ids=sorted(
                            candidate_artifact_ids
                        ),
                        candidate_unit_ids=(
                            candidate_unit_ids
                        ),
                        log_name=(
                            f"{component_id}_"
                            f"{requirement_id}_"
                            "initial"
                        ),
                    )
                )

                self._validate_requirement_feedback(
                    component_id=component_id,
                    requirement_id=requirement_id,
                    feedback=initial_feedback,
                    candidate_unit_ids=set(
                        candidate_unit_ids
                    ),
                    candidate_artifact_ids=(
                        candidate_artifact_ids
                    ),
                )

                criterion_id = (
                    f"ca_{criterion_counter:03d}"
                )

                criterion_counter += 1

                observation_id = (
                    f"obs_{observation_counter:03d}"
                )

                observation_counter += 1

                initial_criterion = (
                    CriterionAssessment(
                        criterion_assessment_id=(
                            criterion_id
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        part_ids=(
                            [part_id]
                            if isinstance(
                                part_id,
                                str,
                            )
                            else []
                        ),
                        task_ids=[component_id],
                        status=(
                            initial_feedback.status
                        ),
                        internal_finding=(
                            initial_feedback
                            .internal_finding
                        ),
                        verification_status=(
                            initial_feedback
                            .verification_status
                        ),
                        confidence=(
                            initial_feedback.confidence
                        ),
                        evidence=(
                            initial_feedback.evidence
                        ),
                    )
                )

                initial_observation = (
                    FeedbackObservation(
                        observation_id=(
                            observation_id
                        ),
                        feedback_type=(
                            self
                            ._feedback_type_for_status(
                                initial_feedback
                                .status
                            )
                        ),
                        scope="task",
                        part_ids=(
                            [part_id]
                            if isinstance(
                                part_id,
                                str,
                            )
                            else []
                        ),
                        task_ids=[component_id],
                        attempt_unit_ids=[],
                        requirement_ids=[
                            requirement_id
                        ],
                        criterion_assessment_ids=[
                            criterion_id
                        ],
                        internal_finding=(
                            initial_feedback
                            .internal_finding
                        ),
                        student_feedback=(
                            initial_feedback
                            .student_feedback
                        ),
                        suggestion=(
                            initial_feedback.suggestion
                        ),
                        verification_status=(
                            initial_feedback
                            .verification_status
                        ),
                        confidence=(
                            initial_feedback.confidence
                        ),
                        evidence=(
                            initial_feedback.evidence
                        ),
                    )
                )

                final_criterion = initial_criterion
                final_observation = (
                    initial_observation
                )

                if self_reflective:
                    reflection_service = (
                        self.reflection_service
                    )

                    if reflection_service is None:
                        raise FeedbackServiceError(
                            "Reflection service became "
                            "unavailable during feedback "
                            "generation."
                        )

                    print(
                        "Auditing feedback for "
                        f"{component_id}/"
                        f"{requirement_id}..."
                    )

                    (
                        decision,
                        final_criterion,
                        final_observation,
                    ) = (
                        reflection_service
                        .reflect_requirement_feedback(
                            component=component,
                            requirement=requirement,
                            part_id=part_id,
                            candidate_units=(
                                candidate_units
                            ),
                            semantic_annotations=(
                                semantic_annotations
                            ),
                            alignment=(
                                alignment_data
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
                                specification_notes
                            ),
                            log_name=(
                                f"{component_id}_"
                                f"{requirement_id}"
                            ),
                        )
                    )

                    self._validate_final_records(
                        component_id=component_id,
                        requirement_id=(
                            requirement_id
                        ),
                        criterion=final_criterion,
                        observation=(
                            final_observation
                        ),
                        expected_criterion_id=(
                            criterion_id
                        ),
                        expected_observation_id=(
                            observation_id
                        ),
                        candidate_unit_ids=set(
                            candidate_unit_ids
                        ),
                        candidate_artifact_ids=(
                            candidate_artifact_ids
                        ),
                    )

                    outcome = (
                        "revised"
                        if (
                            decision.analysis
                            .should_revise
                        )
                        else "unchanged"
                    )

                    print(
                        "✓ Completed feedback audit "
                        f"for {component_id}/"
                        f"{requirement_id}: "
                        f"{outcome}"
                    )

                criterion_assessments.append(
                    final_criterion
                )

                observations.append(
                    final_observation
                )

                component_observation_ids.append(
                    final_observation
                    .observation_id
                )

                component_feedback_texts.append(
                    final_observation
                    .student_feedback
                )

                if (
                    final_criterion.status
                    == "met"
                ):
                    overall_strengths.append(
                        final_observation
                        .student_feedback
                    )

                elif final_criterion.status in {
                    "partially_met",
                    "not_met",
                    "missing",
                }:
                    overall_improvements.append(
                        final_observation
                        .student_feedback
                    )

                print(
                    "✓ Completed feedback for "
                    f"{component_id}/"
                    f"{requirement_id}"
                )

            section_summary = (
                self._combine_summaries(
                    component_feedback_texts,
                    fallback=(
                        "No assessable feedback was "
                        "generated for "
                        f"{component_id}."
                    ),
                )
            )

            section_id = (
                f"section_{section_counter:03d}"
            )

            section_counter += 1

            feedback_sections.append(
                FeedbackSection(
                    section_id=section_id,
                    label=component.get("title"),
                    scope="task",
                    part_ids=(
                        [part_id]
                        if isinstance(part_id, str)
                        else []
                    ),
                    task_ids=[component_id],
                    summary=section_summary,
                    observation_ids=(
                        component_observation_ids
                    ),
                )
            )

            component_summaries.append(
                f"{component_id}: "
                f"{section_summary}"
            )

        all_observation_ids = [
            observation.observation_id
            for observation in observations
        ]

        return FeedbackLLMOutput(
            criterion_assessments=(
                criterion_assessments
            ),
            observations=observations,
            feedback_sections=feedback_sections,
            overall_feedback=OverallFeedback(
                summary=self._combine_summaries(
                    component_summaries,
                    fallback=(
                        "No component feedback was "
                        "generated."
                    ),
                ),
                strengths_summary=(
                    self._combine_summaries(
                        overall_strengths,
                        fallback=(
                            "No clearly supported "
                            "strengths were identified."
                        ),
                    )
                ),
                improvement_summary=(
                    self._combine_summaries(
                        overall_improvements,
                        fallback=(
                            "No clearly supported "
                            "improvements were "
                            "identified."
                        ),
                    )
                ),
                observation_ids=(
                    all_observation_ids
                ),
            ),
        )

    @staticmethod
    def _build_component_lookup(
        assessment_specification: dict[
            str,
            Any,
        ],
    ) -> dict[str, dict[str, Any]]:
        lookup: dict[
            str,
            dict[str, Any],
        ] = {}

        for part in (
            assessment_specification.get(
                "parts",
                [],
            )
        ):
            part_id = part.get("part_id")

            for component in part.get(
                "components",
                [],
            ):
                component_id = component.get(
                    "component_id"
                )

                if not isinstance(
                    component_id,
                    str,
                ):
                    continue

                lookup[component_id] = {
                    "part_id": part_id,
                    "component": component,
                }

        return lookup

    @staticmethod
    def _build_unit_lookups(
        processed_submission: (
            ProcessedSubmission
        ),
    ) -> tuple[
        dict[str, Any],
        dict[str, str],
    ]:
        unit_lookup: dict[str, Any] = {}
        artifact_lookup: dict[str, str] = {}

        for artifact in (
            processed_submission.artifacts
        ):
            for unit in artifact.units:
                unit_lookup[unit.unit_id] = unit

                artifact_lookup[
                    unit.unit_id
                ] = artifact.artifact_id

        return unit_lookup, artifact_lookup

    @staticmethod
    def _collect_candidate_unit_ids(
        alignment: ComponentAlignment,
    ) -> list[str]:
        ordered_ids = (
            alignment.primary_unit_ids
            + alignment.supporting_unit_ids
            + alignment
            .possibly_relevant_unit_ids
        )

        return list(
            dict.fromkeys(ordered_ids)
        )

    @staticmethod
    def _collect_processing_evidence(
        *,
        processed_submission: (
            ProcessedSubmission
        ),
        artifact_ids: set[str],
    ) -> tuple[list[Any], list[Any]]:
        checks: list[Any] = []
        diagnostics: list[Any] = []

        for artifact in (
            processed_submission.artifacts
        ):
            if (
                artifact_ids
                and artifact.artifact_id
                not in artifact_ids
            ):
                continue

            checks.extend(
                artifact.processing.checks
            )

            diagnostics.extend(
                artifact.processing.diagnostics
            )

        return checks, diagnostics

    @staticmethod
    def _validate_requirement_feedback(
        *,
        component_id: str,
        requirement_id: str,
        feedback: (
            RequirementFeedbackLLMOutput
        ),
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> None:
        errors = (
            FeedbackService
            ._validate_evidence_references(
                evidence_items=(
                    feedback.evidence
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
                candidate_artifact_ids=(
                    candidate_artifact_ids
                ),
            )
        )

        if errors:
            raise FeedbackServiceError(
                "Invalid feedback for "
                f"{component_id!r}/"
                f"{requirement_id!r}:\n- "
                + "\n- ".join(errors)
            )

    @staticmethod
    def _validate_final_records(
        *,
        component_id: str,
        requirement_id: str,
        criterion: CriterionAssessment,
        observation: FeedbackObservation,
        expected_criterion_id: str,
        expected_observation_id: str,
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> None:
        errors: list[str] = []

        if (
            criterion
            .criterion_assessment_id
            != expected_criterion_id
        ):
            errors.append(
                "Final criterion assessment ID "
                "was modified."
            )

        if (
            criterion.requirement_id
            != requirement_id
        ):
            errors.append(
                "Final criterion requirement ID "
                "was modified."
            )

        if (
            criterion.task_ids
            != [component_id]
        ):
            errors.append(
                "Final criterion component "
                "reference was modified."
            )

        if (
            observation.observation_id
            != expected_observation_id
        ):
            errors.append(
                "Final observation ID was "
                "modified."
            )

        if (
            observation.requirement_ids
            != [requirement_id]
        ):
            errors.append(
                "Final observation requirement "
                "reference was modified."
            )

        if (
            observation
            .criterion_assessment_ids
            != [expected_criterion_id]
        ):
            errors.append(
                "Final observation criterion "
                "reference was modified."
            )

        if (
            observation.task_ids
            != [component_id]
        ):
            errors.append(
                "Final observation component "
                "reference was modified."
            )

        errors.extend(
            FeedbackService
            ._validate_evidence_references(
                evidence_items=[
                    *criterion.evidence,
                    *observation.evidence,
                ],
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
                candidate_artifact_ids=(
                    candidate_artifact_ids
                ),
            )
        )

        if errors:
            raise FeedbackServiceError(
                "Invalid final reflective "
                "feedback for "
                f"{component_id!r}/"
                f"{requirement_id!r}:\n- "
                + "\n- ".join(errors)
            )

    @staticmethod
    def _validate_evidence_references(
        *,
        evidence_items: list[Any],
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> list[str]:
        errors: list[str] = []

        for evidence in evidence_items:
            if (
                evidence.evidence_type
                != "submission_reference"
            ):
                continue

            if (
                evidence.artifact_id
                not in candidate_artifact_ids
            ):
                errors.append(
                    "Evidence references "
                    "non-candidate artifact "
                    f"{evidence.artifact_id!r}."
                )

            if (
                evidence.unit_id is not None
                and evidence.unit_id
                not in candidate_unit_ids
            ):
                errors.append(
                    "Evidence references "
                    "non-candidate unit "
                    f"{evidence.unit_id!r}."
                )

        return errors

    @staticmethod
    def _feedback_type_for_status(
        status: str,
    ) -> str:
        if status == "met":
            return "strength"

        if status == "missing":
            return "omission"

        if status == "not_met":
            return "error"

        return "suggestion"

    @staticmethod
    def _validate_submission_identifiers(
        *,
        processed_submission: (
            ProcessedSubmission
        ),
        semantic_extraction: (
            SemanticExtractionResult
        ),
        alignment_result: AlignmentResult,
    ) -> None:
        errors: list[str] = []

        if (
            semantic_extraction
            .processed_submission_id
            != processed_submission
            .processed_submission_id
        ):
            errors.append(
                "Semantic extraction does not "
                "match the processed submission."
            )

        if (
            alignment_result
            .processed_submission_id
            != processed_submission
            .processed_submission_id
        ):
            errors.append(
                "Alignment result does not match "
                "the processed submission."
            )

        if (
            alignment_result.assessment_id
            != processed_submission
            .assessment_id
        ):
            errors.append(
                "Alignment result does not match "
                "the assessment."
            )

        if errors:
            raise FeedbackServiceError(
                "Feedback inputs are "
                "inconsistent:\n- "
                + "\n- ".join(errors)
            )

    @staticmethod
    def _combine_summaries(
        values: list[str],
        *,
        fallback: str,
    ) -> str:
        cleaned = [
            value.strip()
            for value in values
            if (
                isinstance(value, str)
                and value.strip()
            )
        ]

        if not cleaned:
            return fallback

        return " ".join(cleaned)