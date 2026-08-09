from __future__ import annotations

import json
from pathlib import Path
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

    Python controls requirement IDs, feedback IDs, scopes, structural
    references, and deterministic evidence routing.

    The LLM determines whether the supplied evidence satisfies the
    assessment requirement.

    When self-reflection is enabled, initial feedback is audited by the
    reflection service. Python retains ownership of all identifiers and
    structural metadata.
    """

    def __init__(
        self,
        *,
        llm_client: FeedbackStructuredClient,
        reflection_service: (
            FeedbackReflectionService | None
        ) = None,
        checkpoint_dir: str | Path | None = None,
        resume_from_checkpoints: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.reflection_service = reflection_service
        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        self.resume_from_checkpoints = (
            resume_from_checkpoints
        )

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

        feedback_requirements = (
            self._collect_feedback_requirements(
                assessment_specification
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
                component_lookup.get(
                    component_id
                )
            )

            if component_record is None:
                raise FeedbackServiceError(
                    "Alignment references unknown "
                    f"component {component_id!r}."
                )

            part_id = component_record[
                "part_id"
            ]

            component = component_record[
                "component"
            ]

            aligned_unit_ids = (
                self._collect_candidate_unit_ids(
                    alignment
                )
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

                print()
                print(
                    "Generating feedback for "
                    f"{component_id}/"
                    f"{requirement_id}..."
                )

                candidate_unit_ids = (
                    self
                    ._select_requirement_unit_ids(
                        component=component,
                        requirement=requirement,
                        aligned_unit_ids=(
                            aligned_unit_ids
                        ),
                        unit_lookup=(
                            unit_lookup
                        ),
                    )
                )

                candidate_unit_ids = (
                    self
                    ._augment_requirement_unit_ids(
                        component=component,
                        requirement=requirement,
                        candidate_unit_ids=(
                            candidate_unit_ids
                        ),
                        unit_lookup=(
                            unit_lookup
                        ),
                    )
                )

                print(
                    "Candidate units:",
                    len(candidate_unit_ids),
                )

                candidate_units: list[
                    dict[str, Any]
                ] = []

                semantic_annotations: list[
                    Any
                ] = []

                candidate_artifact_ids: set[
                    str
                ] = set()

                for unit_id in (
                    candidate_unit_ids
                ):
                    unit = unit_lookup.get(
                        unit_id
                    )

                    if unit is None:
                        raise FeedbackServiceError(
                            f"Component "
                            f"{component_id!r} "
                            "references unknown unit "
                            f"{unit_id!r}."
                        )

                    artifact_id = (
                        unit_artifact_lookup[
                            unit_id
                        ]
                    )

                    candidate_artifact_ids.add(
                        artifact_id
                    )

                    unit_data = (
                        unit.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    )

                    unit_data[
                        "artifact_id"
                    ] = artifact_id

                    candidate_units.append(
                        unit_data
                    )

                    annotation = (
                        semantic_lookup.get(
                            unit_id
                        )
                    )

                    if (
                        annotation
                        is not None
                    ):
                        semantic_annotations.append(
                            annotation
                        )

                (
                    processing_checks,
                    processing_diagnostics,
                ) = (
                    self
                    ._collect_processing_evidence(
                        processed_submission=(
                            processed_submission
                        ),
                        artifact_ids=(
                            candidate_artifact_ids
                        ),
                    )
                )

                specification_notes = [
                    note
                    for note
                    in all_specification_notes
                    if (
                        not isinstance(
                            note,
                            dict,
                        )
                        or note.get(
                            "component_id"
                        )
                        in {
                            None,
                            component_id,
                        }
                    )
                ]

                alignment_data = (
                    self
                    ._build_requirement_alignment_data(
                        alignment=alignment,
                        candidate_unit_ids=set(
                            candidate_unit_ids
                        ),
                    )
                )

                initial_feedback = (
                    self._load_requirement_checkpoint(
                        component_id=component_id,
                        requirement_id=requirement_id,
                    )
                )

                if initial_feedback is None:
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
                            alignment=(
                                alignment_data
                            ),
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
                            system_prompt=(
                                system_prompt
                            ),
                            user_prompt=(
                                user_prompt
                            ),
                            candidate_artifact_ids=(
                                sorted(
                                    candidate_artifact_ids
                                )
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
                        component_id=(
                            component_id
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        feedback=(
                            initial_feedback
                        ),
                        candidate_unit_ids=set(
                            candidate_unit_ids
                        ),
                        candidate_artifact_ids=(
                            candidate_artifact_ids
                        ),
                    )

                    self._save_requirement_checkpoint(
                        component_id=component_id,
                        requirement_id=requirement_id,
                        feedback=initial_feedback,
                    )

                else:
                    self._validate_requirement_feedback(
                        component_id=(
                            component_id
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        feedback=(
                            initial_feedback
                        ),
                        candidate_unit_ids=set(
                            candidate_unit_ids
                        ),
                        candidate_artifact_ids=(
                            candidate_artifact_ids
                        ),
                    )

                    print(
                        "✓ Loaded feedback checkpoint for "
                        f"{component_id}/{requirement_id}"
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
                        task_ids=[
                            component_id
                        ],
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
                        task_ids=[
                            component_id
                        ],
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

                final_criterion = (
                    initial_criterion
                )

                final_observation = (
                    initial_observation
                )

                if self_reflective:
                    reflection_service = (
                        self.reflection_service
                    )

                    if (
                        reflection_service
                        is None
                    ):
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
                        component_id=(
                            component_id
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        criterion=(
                            final_criterion
                        ),
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

                elif (
                    final_criterion.status
                    in {
                        "partially_met",
                        "not_met",
                        "missing",
                    }
                ):
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
                    section_id=(
                        section_id
                    ),
                    label=(
                        component.get(
                            "title"
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
                    task_ids=[
                        component_id
                    ],
                    summary=(
                        section_summary
                    ),
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
            observations=(
                observations
            ),
            feedback_sections=(
                feedback_sections
            ),
            overall_feedback=OverallFeedback(
                summary=(
                    self._combine_summaries(
                        component_summaries,
                        fallback=(
                            "No component feedback was "
                            "generated."
                        ),
                    )
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

    def _checkpoint_path(
        self,
        *,
        component_id: str,
        requirement_id: str,
    ) -> Path | None:
        if self.checkpoint_dir is None:
            return None

        filename = (
            f"{component_id}__"
            f"{requirement_id}.json"
        )

        return (
            self.checkpoint_dir
            / filename
        )

    def _load_requirement_checkpoint(
        self,
        *,
        component_id: str,
        requirement_id: str,
    ) -> RequirementFeedbackLLMOutput | None:
        if not self.resume_from_checkpoints:
            return None

        checkpoint_path = (
            self._checkpoint_path(
                component_id=component_id,
                requirement_id=requirement_id,
            )
        )

        if (
            checkpoint_path is None
            or not checkpoint_path.exists()
        ):
            return None

        try:
            with checkpoint_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise FeedbackServiceError(
                "Could not read feedback checkpoint "
                f"{checkpoint_path}."
            ) from exc

        if not isinstance(payload, dict):
            raise FeedbackServiceError(
                "Feedback checkpoint must contain "
                "a JSON object: "
                f"{checkpoint_path}."
            )

        if (
            payload.get("component_id")
            != component_id
            or payload.get("requirement_id")
            != requirement_id
        ):
            raise FeedbackServiceError(
                "Feedback checkpoint identifiers "
                "do not match the requested "
                "component/requirement: "
                f"{checkpoint_path}."
            )

        checkpoint_model = payload.get(
            "model_name"
        )

        if (
            isinstance(checkpoint_model, str)
            and checkpoint_model
            != self.llm_client.model_name
        ):
            return None

        feedback_data = payload.get(
            "feedback"
        )

        try:
            return (
                RequirementFeedbackLLMOutput
                .model_validate(
                    feedback_data
                )
            )
        except (
            ValueError,
            TypeError,
        ) as exc:
            raise FeedbackServiceError(
                "Feedback checkpoint contains "
                "invalid requirement feedback: "
                f"{checkpoint_path}."
            ) from exc

    def _save_requirement_checkpoint(
        self,
        *,
        component_id: str,
        requirement_id: str,
        feedback: RequirementFeedbackLLMOutput,
    ) -> None:
        checkpoint_path = (
            self._checkpoint_path(
                component_id=component_id,
                requirement_id=requirement_id,
            )
        )

        if checkpoint_path is None:
            return

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": "1.0",
            "component_id": component_id,
            "requirement_id": requirement_id,
            "model_name": (
                self.llm_client.model_name
            ),
            "feedback": feedback.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }

        temporary_path = (
            checkpoint_path.with_suffix(
                checkpoint_path.suffix
                + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(
            checkpoint_path
        )

        print(
            "✓ Saved feedback checkpoint for "
            f"{component_id}/{requirement_id}"
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

        # Assessment 1 layout:
        # parts -> components
        for part in (
            assessment_specification.get(
                "parts",
                [],
            )
        ):
            if not isinstance(
                part,
                dict,
            ):
                continue

            part_id = part.get(
                "part_id"
            )

            for component in part.get(
                "components",
                [],
            ):
                if not isinstance(
                    component,
                    dict,
                ):
                    continue

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

                lookup[
                    component_id
                ] = {
                    "part_id": (
                        part_id
                        if isinstance(
                            part_id,
                            str,
                        )
                        else None
                    ),
                    "component": component,
                }

        # Assessment 2 layout:
        # top-level components
        for component in (
            assessment_specification.get(
                "components",
                [],
            )
        ):
            if not isinstance(
                component,
                dict,
            ):
                continue

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

            part_id = (
                FeedbackService
                ._infer_component_part_id(
                    component_id=(
                        component_id
                    ),
                    assessment_specification=(
                        assessment_specification
                    ),
                )
            )

            lookup[
                component_id
            ] = {
                "part_id": part_id,
                "component": component,
            }

        return lookup

    @staticmethod
    def _infer_component_part_id(
        *,
        component_id: str,
        assessment_specification: dict[
            str,
            Any,
        ],
    ) -> str | None:
        assessment_structure = (
            assessment_specification.get(
                "assessment_structure",
                {},
            )
        )

        if not isinstance(
            assessment_structure,
            dict,
        ):
            return None

        part_ids = [
            part.get("id")
            for part
            in assessment_structure.values()
            if (
                isinstance(
                    part,
                    dict,
                )
                and isinstance(
                    part.get("id"),
                    str,
                )
            )
        ]

        for part_id in part_ids:
            if component_id == part_id:
                return part_id

        if component_id.startswith(
            "part_1"
        ):
            for part_id in part_ids:
                if part_id.startswith(
                    "part_1"
                ):
                    return part_id

        if component_id.startswith(
            "part_2"
        ):
            for part_id in part_ids:
                if part_id.startswith(
                    "part_2"
                ):
                    return part_id

        return None

    @staticmethod
    def _collect_feedback_requirements(
        assessment_specification: dict[
            str,
            Any,
        ],
    ) -> list[str]:
        collected: list[str] = []

        # Assessment 1
        evaluation_policy = (
            assessment_specification.get(
                "evaluation_policy",
                {},
            )
        )

        if isinstance(
            evaluation_policy,
            dict,
        ):
            requirements = (
                evaluation_policy.get(
                    "feedback_requirements",
                    [],
                )
            )

            if isinstance(
                requirements,
                list,
            ):
                for item in requirements:
                    if (
                        isinstance(
                            item,
                            str,
                        )
                        and item.strip()
                    ):
                        collected.append(
                            item.strip()
                        )

        # Assessment 2
        guidance = (
            assessment_specification.get(
                "feedback_generation_guidance",
                {},
            )
        )

        if isinstance(
            guidance,
            dict,
        ):
            general_principles = (
                guidance.get(
                    "general_principles",
                    [],
                )
            )

            if isinstance(
                general_principles,
                list,
            ):
                for item in (
                    general_principles
                ):
                    if (
                        isinstance(
                            item,
                            str,
                        )
                        and item.strip()
                    ):
                        collected.append(
                            item.strip()
                        )

            ontology_feedback = (
                guidance.get(
                    "ontology_feedback",
                    [],
                )
            )

            if isinstance(
                ontology_feedback,
                list,
            ):
                for item in (
                    ontology_feedback
                ):
                    if (
                        isinstance(
                            item,
                            str,
                        )
                        and item.strip()
                    ):
                        collected.append(
                            "Ontology feedback: "
                            + item.strip()
                        )

            prolog_feedback = (
                guidance.get(
                    "prolog_feedback",
                    [],
                )
            )

            if isinstance(
                prolog_feedback,
                list,
            ):
                for item in (
                    prolog_feedback
                ):
                    if (
                        isinstance(
                            item,
                            str,
                        )
                        and item.strip()
                    ):
                        collected.append(
                            "Prolog feedback: "
                            + item.strip()
                        )

            uncertainty_policy = (
                guidance.get(
                    "uncertainty_policy"
                )
            )

            if (
                isinstance(
                    uncertainty_policy,
                    str,
                )
                and uncertainty_policy.strip()
            ):
                collected.append(
                    "Uncertainty policy: "
                    + uncertainty_policy.strip()
                )

        return list(
            dict.fromkeys(
                collected
            )
        )

    @staticmethod
    def _select_requirement_unit_ids(
        *,
        component: dict[str, Any],
        requirement: dict[str, Any],
        aligned_unit_ids: list[str],
        unit_lookup: dict[str, Any],
    ) -> list[str]:
        """
        Narrow component-level alignment to evidence relevant to one
        requirement.

        This performs evidence routing only. It does not decide whether
        the requirement is satisfied.

        Assessment 1 keeps its existing aligned evidence.

        Assessment 2 Prolog components are already narrowly aligned.

        Assessment 2 Part 1 is narrowed by ontology/report unit type.
        """
        component_id = component.get(
            "component_id"
        )

        # Preserve Assessment 1 and already narrow
        # Assessment 2 Prolog alignment.
        if (
            component_id
            != "part_1_ontology"
        ):
            return list(
                aligned_unit_ids
            )

        criterion = requirement.get(
            "criterion"
        )

        if not isinstance(
            criterion,
            str,
        ):
            return list(
                aligned_unit_ids
            )

        criterion = criterion.strip()

        def unit_type(
            unit_id: str,
        ) -> str:
            unit = unit_lookup.get(
                unit_id
            )

            value = getattr(
                unit,
                "unit_type",
                "",
            )

            return (
                value
                if isinstance(
                    value,
                    str,
                )
                else ""
            )

        def unit_label(
            unit_id: str,
        ) -> str:
            unit = unit_lookup.get(
                unit_id
            )

            value = getattr(
                unit,
                "label",
                "",
            )

            return (
                value.lower()
                if isinstance(
                    value,
                    str,
                )
                else ""
            )

        selected: list[str] = []

        # --------------------------------------------------
        # P1-R1: classes
        # --------------------------------------------------

        if criterion == "classes":
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == "ontology_class"
                )
            ]

        # --------------------------------------------------
        # P1-R2: class hierarchy
        # --------------------------------------------------

        elif (
            criterion
            == "class_hierarchy"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == (
                        "ontology_subclass_axiom"
                    )
                )
            ]

        # --------------------------------------------------
        # P1-R3: properties
        # --------------------------------------------------

        elif criterion == "properties":
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if unit_type(
                    unit_id
                )
                in {
                    "ontology_object_property",
                    "ontology_data_property",
                }
            ]

        # --------------------------------------------------
        # P1-R4: property characteristics
        #
        # Keep declarations and hierarchy/characteristic
        # evidence, then cap repetitive domain/range units.
        # --------------------------------------------------

        elif (
            criterion
            == "property_characteristics"
        ):
            property_units = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if unit_type(
                    unit_id
                )
                in {
                    "ontology_object_property",
                    "ontology_data_property",
                    "ontology_subproperty_axiom",
                    "ontology_property_characteristic",
                }
            ]

            domain_range_units = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if unit_type(
                    unit_id
                )
                in {
                    "ontology_property_domain",
                    "ontology_property_range",
                }
            ][:8]

            selected = (
                property_units
                + domain_range_units
            )

        # --------------------------------------------------
        # P1-R5: Description Logic number restriction
        # --------------------------------------------------

        elif (
            criterion
            == "description_logic_axiom"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == "ontology_restriction"
                )
            ]

        # --------------------------------------------------
        # P1-R6: ontology consistency
        #
        # A reasoner result is the strongest evidence for
        # consistency. In this dataset consistency may be
        # marked not_run. We therefore keep only a bounded
        # structural sample and rely on deterministic
        # processing checks to establish whether consistency
        # was actually verified.
        # --------------------------------------------------

        elif (
            criterion
            == "ontology_consistency"
        ):
            structural_types = {
                "ontology_subclass_axiom",
                "ontology_restriction",
                "ontology_subproperty_axiom",
                "ontology_property_domain",
                "ontology_property_range",
                "ontology_property_characteristic",
            }

            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if unit_type(
                    unit_id
                )
                in structural_types
            ][:8]

        # --------------------------------------------------
        # P1-R7: report accuracy
        #
        # Keep all report sections plus a bounded collection
        # of ontology structures that the report is likely to
        # describe.
        # --------------------------------------------------

        elif (
            criterion
            == "report_accuracy"
        ):
            report_units = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == (
                        "ontology_report_section"
                    )
                )
            ]

            ontology_units = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if unit_type(
                    unit_id
                )
                in {
                    "ontology_subclass_axiom",
                    "ontology_restriction",
                    "ontology_subproperty_axiom",
                }
            ][:5]

            selected = (
                report_units
                + ontology_units
            )

        # --------------------------------------------------
        # P1-R8: report overview
        # --------------------------------------------------

        elif (
            criterion
            == "report_overview"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == (
                        "ontology_report_section"
                    )
                    and (
                        "overview"
                        in unit_label(
                            unit_id
                        )
                    )
                )
            ]

        # --------------------------------------------------
        # P1-R9: hierarchy justification
        # --------------------------------------------------

        elif (
            criterion
            == "report_hierarchy"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    (
                        unit_type(
                            unit_id
                        )
                        == (
                            "ontology_report_section"
                        )
                        and (
                            "justification"
                            in unit_label(
                                unit_id
                            )
                        )
                    )
                    or unit_type(
                        unit_id
                    )
                    in {
                        "ontology_subclass_axiom",
                        "ontology_subproperty_axiom",
                    }
                )
            ]

        # --------------------------------------------------
        # P1-R10: report axiom explanation
        # --------------------------------------------------

        elif (
            criterion
            == "report_axiom"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    (
                        unit_type(
                            unit_id
                        )
                        == (
                            "ontology_report_section"
                        )
                        and (
                            "axiom"
                            in unit_label(
                                unit_id
                            )
                        )
                    )
                    or (
                        unit_type(
                            unit_id
                        )
                        == (
                            "ontology_restriction"
                        )
                    )
                )
            ]

        # --------------------------------------------------
        # P1-R11: critical reflection
        # --------------------------------------------------

        elif (
            criterion
            == "critical_reflection"
        ):
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == (
                        "ontology_report_section"
                    )
                    and (
                        "advantage"
                        in unit_label(
                            unit_id
                        )
                        or "disadvantage"
                        in unit_label(
                            unit_id
                        )
                        or "limitation"
                        in unit_label(
                            unit_id
                        )
                        or "reflection"
                        in unit_label(
                            unit_id
                        )
                    )
                )
            ]

        # --------------------------------------------------
        # P1-R12: report word limit
        # --------------------------------------------------

        elif criterion == "word_limit":
            selected = [
                unit_id
                for unit_id
                in aligned_unit_ids
                if (
                    unit_type(
                        unit_id
                    )
                    == (
                        "ontology_report_section"
                    )
                )
            ]

        else:
            selected = list(
                aligned_unit_ids
            )

        # If a deterministic routing rule unexpectedly finds
        # nothing, do not silently remove evidence.
        if not selected:
            return list(
                aligned_unit_ids
            )

        return list(
            dict.fromkeys(
                selected
            )
        )

    @staticmethod
    def _augment_requirement_unit_ids(
        *,
        component: dict[str, Any],
        requirement: dict[str, Any],
        candidate_unit_ids: list[str],
        unit_lookup: dict[str, Any],
    ) -> list[str]:
        """
        Add narrowly scoped cross-task context when a requirement explicitly
        depends on work submitted for another task.

        P2-T5-R3 explicitly refers to the constraints established in Tasks 1-4.
        Give that requirement the submitted answer1/1 through answer4/1 formula
        units as comparison context in addition to its Task 5 evidence.
        """

        component_id = component.get(
            "component_id"
        )

        requirement_id = requirement.get(
            "id"
        )

        if (
            component_id != "part_2_task_5"
            or requirement_id != "P2-T5-R3"
        ):
            return list(
                candidate_unit_ids
            )

        context_unit_ids: list[str] = []

        expected_predicates = {
            "answer1",
            "answer2",
            "answer3",
            "answer4",
        }

        for unit_id, unit in unit_lookup.items():
            unit_type = getattr(
                unit,
                "unit_type",
                None,
            )

            structured_data = getattr(
                unit,
                "structured_data",
                None,
            )

            if hasattr(
                structured_data,
                "model_dump",
            ):
                structured_data = (
                    structured_data.model_dump(
                        mode="python",
                        exclude_none=True,
                    )
                )

            if not isinstance(
                structured_data,
                dict,
            ):
                structured_data = {}

            predicate_name = (
                structured_data.get(
                    "predicate_name"
                )
            )

            if (
                unit_type == "prolog_answer"
                and predicate_name
                in expected_predicates
            ):
                context_unit_ids.append(
                    unit_id
                )

        return list(
            dict.fromkeys(
                [
                    *candidate_unit_ids,
                    *context_unit_ids,
                ]
            )
        ) 

    @staticmethod
    def _build_requirement_alignment_data(
        *,
        alignment: ComponentAlignment,
        candidate_unit_ids: set[str],
    ) -> dict[str, Any]:
        data = alignment.model_dump(
            mode="json",
            exclude_none=True,
        )

        for field_name in (
            "primary_unit_ids",
            "supporting_unit_ids",
            "possibly_relevant_unit_ids",
        ):
            values = data.get(
                field_name,
                [],
            )

            if isinstance(
                values,
                list,
            ):
                data[field_name] = [
                    value
                    for value in values
                    if (
                        value
                        in candidate_unit_ids
                    )
                ]

        return data

    @staticmethod
    def _build_unit_lookups(
        processed_submission: (
            ProcessedSubmission
        ),
    ) -> tuple[
        dict[str, Any],
        dict[str, str],
    ]:
        unit_lookup: dict[
            str,
            Any,
        ] = {}

        artifact_lookup: dict[
            str,
            str,
        ] = {}

        for artifact in (
            processed_submission.artifacts
        ):
            for unit in artifact.units:
                unit_lookup[
                    unit.unit_id
                ] = unit

                artifact_lookup[
                    unit.unit_id
                ] = (
                    artifact.artifact_id
                )

        return (
            unit_lookup,
            artifact_lookup,
        )

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
            dict.fromkeys(
                ordered_ids
            )
        )

    @staticmethod
    def _collect_processing_evidence(
        *,
        processed_submission: (
            ProcessedSubmission
        ),
        artifact_ids: set[str],
    ) -> tuple[
        list[Any],
        list[Any],
    ]:
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
                artifact.processing
                .diagnostics
            )

        return (
            checks,
            diagnostics,
        )

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
                + "\n- ".join(
                    errors
                )
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
                + "\n- ".join(
                    errors
                )
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
    def _validate_submission_identifiers(
        *,
        processed_submission: (
            ProcessedSubmission
        ),
        semantic_extraction: (
            SemanticExtractionResult
        ),
        alignment_result: (
            AlignmentResult
        ),
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
            semantic_extraction
            .source_submission_id
            != processed_submission
            .source_submission_id
        ):
            errors.append(
                "Semantic extraction source "
                "submission does not match the "
                "processed submission."
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
            alignment_result
            .source_submission_id
            != processed_submission
            .source_submission_id
        ):
            errors.append(
                "Alignment result source "
                "submission does not match the "
                "processed submission."
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
                + "\n- ".join(
                    errors
                )
            )

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
    def _combine_summaries(
        values: list[str],
        *,
        fallback: str,
    ) -> str:
        cleaned = [
            value.strip()
            for value in values
            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            )
        ]

        if not cleaned:
            return fallback

        return " ".join(
            cleaned
        )