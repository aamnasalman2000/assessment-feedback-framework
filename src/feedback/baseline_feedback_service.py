from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baseline_feedback_prompts import (
    build_component_baseline_feedback_prompt,
)
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
    ComponentFeedbackLLMOutput,
    CriterionAssessment,
    FeedbackLLMOutput,
    FeedbackObservation,
    FeedbackSection,
    OverallFeedback,
)


class BaselineFeedbackError(RuntimeError):
    """Raised when baseline component feedback cannot be validated."""


class BaselineFeedbackService:
    """
    Pipeline A: non-decomposed, non-reflective feedback generation.

    The service is assessment-agnostic and supports both specification
    layouts currently used by the framework:

    Assessment 1:
        parts -> components

    Assessment 2:
        top-level components

    One LLM call is made per component. All evaluation requirements for that
    component are supplied together in a single prompt. Python retains
    ownership of record IDs, scopes, references, and final assembly.

    Optional component checkpoints allow long baseline runs to resume
    without regenerating completed components.
    """

    def __init__(
        self,
        *,
        llm_client: FeedbackStructuredClient,
        checkpoint_dir: str | Path | None = None,
        resume_from_checkpoints: bool = True,
    ) -> None:
        self.llm_client = llm_client

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
        generation_input: FeedbackGenerationInput,
    ) -> FeedbackLLMOutput:
        """
        Generate Pipeline A feedback for a complete assessment.

        Each component is evaluated once against all of its requirements.
        No criterion-level prompt decomposition and no reflection are used.
        """

        specification = _to_dict(
            generation_input.assessment_specification
        )

        if not isinstance(
            specification,
            dict,
        ):
            raise BaselineFeedbackError(
                "assessment_specification must serialize to a dictionary."
            )

        components = _collect_components(
            specification
        )

        if not components:
            raise BaselineFeedbackError(
                "No assessment components were found in the specification."
            )

        (
            processing_checks,
            processing_diagnostics,
        ) = _collect_processing_evidence(
            generation_input
        )

        global_requirements = specification.get(
            "global_requirements",
            [],
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

        criterion_assessments: list[
            CriterionAssessment
        ] = []

        observations: list[
            FeedbackObservation
        ] = []

        feedback_sections: list[
            FeedbackSection
        ] = []

        component_summaries: list[str] = []

        criterion_counter = 1
        observation_counter = 1
        section_counter = 1

        for part_id, component in components:
            component_id = component.get(
                "component_id"
            )

            if not isinstance(
                component_id,
                str,
            ):
                continue

            requirements = component.get(
                "evaluation_requirements",
                [],
            )

            if not isinstance(
                requirements,
                list,
            ):
                continue

            requirement_ids = [
                requirement.get(
                    "id"
                )
                for requirement in requirements
                if (
                    isinstance(
                        requirement,
                        dict,
                    )
                    and isinstance(
                        requirement.get(
                            "id"
                        ),
                        str,
                    )
                )
            ]

            if not requirement_ids:
                continue

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

            component_output = (
                self._load_component_checkpoint(
                    component_id=component_id,
                    requirement_ids=(
                        requirement_ids
                    ),
                )
            )

            if component_output is None:
                (
                    system_prompt,
                    user_prompt,
                ) = (
                    build_component_baseline_feedback_prompt(
                        component=component,
                        part_id=part_id,
                        candidate_units=(
                            candidate_units
                        ),
                        semantic_annotations=(
                            semantic_annotations
                        ),
                        alignment=alignment,
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

                candidate_artifact_ids = list(
                    dict.fromkeys(
                        unit.get(
                            "artifact_id"
                        )
                        for unit in candidate_units
                        if isinstance(
                            unit.get(
                                "artifact_id"
                            ),
                            str,
                        )
                    )
                )

                candidate_unit_ids = [
                    unit.get(
                        "unit_id"
                    )
                    for unit in candidate_units
                    if isinstance(
                        unit.get(
                            "unit_id"
                        ),
                        str,
                    )
                ]

                component_output = (
                    self.llm_client
                    .generate_component_feedback(
                        system_prompt=(
                            system_prompt
                        ),
                        user_prompt=(
                            user_prompt
                        ),
                        candidate_artifact_ids=(
                            candidate_artifact_ids
                        ),
                        candidate_unit_ids=(
                            candidate_unit_ids
                        ),
                        requirement_ids=(
                            requirement_ids
                        ),
                        component_id=(
                            component_id
                        ),
                        part_id=part_id,
                        log_name=(
                            f"baseline_{component_id}"
                        ),
                    )
                )

                self._validate_component_output(
                    component_output=(
                        component_output
                    ),
                    requirement_ids=set(
                        requirement_ids
                    ),
                    candidate_unit_ids=set(
                        candidate_unit_ids
                    ),
                    candidate_artifact_ids=set(
                        candidate_artifact_ids
                    ),
                )

                self._save_component_checkpoint(
                    component_id=(
                        component_id
                    ),
                    requirement_ids=(
                        requirement_ids
                    ),
                    component_output=(
                        component_output
                    ),
                )

            else:
                print(
                    "✓ Loaded baseline checkpoint for "
                    f"{component_id}"
                )

            (
                component_criteria,
                criterion_id_by_requirement,
                criterion_counter,
            ) = self._assemble_criterion_assessments(
                component_output=(
                    component_output
                ),
                component_id=(
                    component_id
                ),
                part_id=part_id,
                start_counter=(
                    criterion_counter
                ),
            )

            criterion_assessments.extend(
                component_criteria
            )

            (
                component_observations,
                component_observation_ids,
                observation_counter,
            ) = self._assemble_observations(
                component_output=(
                    component_output
                ),
                component_id=(
                    component_id
                ),
                part_id=part_id,
                criterion_id_by_requirement=(
                    criterion_id_by_requirement
                ),
                start_counter=(
                    observation_counter
                ),
            )

            observations.extend(
                component_observations
            )

            section_scope = (
                self._scope_for_component(
                    component_id=(
                        component_id
                    ),
                    part_id=part_id,
                )
            )

            feedback_sections.append(
                FeedbackSection(
                    section_id=(
                        f"section_{section_counter:03d}"
                    ),
                    label=(
                        component.get(
                            "title"
                        )
                        if isinstance(
                            component.get(
                                "title"
                            ),
                            str,
                        )
                        else component_id
                    ),
                    scope=(
                        section_scope
                    ),
                    part_ids=(
                        [part_id]
                        if (
                            section_scope
                            == "part"
                            and part_id
                            is not None
                        )
                        else (
                            [part_id]
                            if (
                                part_id
                                is not None
                                and section_scope
                                == "task"
                            )
                            else []
                        )
                    ),
                    task_ids=(
                        [component_id]
                        if section_scope
                        in {
                            "task",
                            "task_group",
                        }
                        else []
                    ),
                    summary=(
                        component_output
                        .component_summary
                    ),
                    observation_ids=(
                        component_observation_ids
                    ),
                )
            )

            component_summaries.append(
                component_output
                .component_summary
            )

            section_counter += 1

        if not criterion_assessments:
            raise BaselineFeedbackError(
                "Baseline generation produced no criterion assessments."
            )

        overall_feedback = (
            self._build_overall_feedback(
                component_summaries=(
                    component_summaries
                ),
                observations=(
                    observations
                ),
            )
        )

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
            overall_feedback=(
                overall_feedback
            ),
        )

    # ========================================================
    # Component output validation
    # ========================================================

    @staticmethod
    def _validate_component_output(
        *,
        component_output: ComponentFeedbackLLMOutput,
        requirement_ids: set[str],
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> None:
        errors: list[str] = []

        returned_requirement_ids = [
            item.requirement_id
            for item in (
                component_output
                .criterion_assessments
            )
        ]

        returned_set = set(
            returned_requirement_ids
        )

        missing_requirement_ids = (
            requirement_ids
            - returned_set
        )

        unknown_requirement_ids = (
            returned_set
            - requirement_ids
        )

        duplicate_requirement_ids = {
            requirement_id
            for requirement_id
            in returned_requirement_ids
            if (
                returned_requirement_ids.count(
                    requirement_id
                )
                > 1
            )
        }

        if missing_requirement_ids:
            errors.append(
                "Missing requirement assessments: "
                + ", ".join(
                    sorted(
                        missing_requirement_ids
                    )
                )
            )

        if unknown_requirement_ids:
            errors.append(
                "Unknown requirement assessments: "
                + ", ".join(
                    sorted(
                        unknown_requirement_ids
                    )
                )
            )

        if duplicate_requirement_ids:
            errors.append(
                "Duplicate requirement assessments: "
                + ", ".join(
                    sorted(
                        duplicate_requirement_ids
                    )
                )
            )

        for observation in (
            component_output.observations
        ):
            unknown_observation_requirements = (
                set(
                    observation
                    .requirement_ids
                )
                - requirement_ids
            )

            if unknown_observation_requirements:
                errors.append(
                    "Observation references unknown requirement IDs: "
                    + ", ".join(
                        sorted(
                            unknown_observation_requirements
                        )
                    )
                )

        evidence_values: list[Any] = []

        for assessment in (
            component_output
            .criterion_assessments
        ):
            evidence_values.extend(
                assessment.evidence
            )

        for observation in (
            component_output
            .observations
        ):
            evidence_values.extend(
                observation.evidence
            )

        for evidence in evidence_values:
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
                    "Component feedback references unknown artifact "
                    f"{evidence.artifact_id!r}."
                )

            if (
                evidence.unit_id
                is not None
                and evidence.unit_id
                not in candidate_unit_ids
            ):
                errors.append(
                    "Component feedback references unknown unit "
                    f"{evidence.unit_id!r}."
                )

        if errors:
            raise BaselineFeedbackError(
                "Invalid baseline component feedback:\n- "
                + "\n- ".join(
                    errors
                )
            )

    # ========================================================
    # Final-model assembly
    # ========================================================

    @staticmethod
    def _scope_for_component(
        *,
        component_id: str,
        part_id: str | None,
    ) -> str:
        if (
            part_id is not None
            and component_id == part_id
        ):
            return "part"

        return "task"

    @staticmethod
    def _part_ids_for_record(
        *,
        part_id: str | None,
    ) -> list[str]:
        if part_id is None:
            return []

        return [
            part_id
        ]

    @staticmethod
    def _task_ids_for_record(
        *,
        component_id: str,
        part_id: str | None,
    ) -> list[str]:
        if (
            part_id is not None
            and component_id == part_id
        ):
            return []

        return [
            component_id
        ]

    @staticmethod
    def _assemble_criterion_assessments(
        *,
        component_output: ComponentFeedbackLLMOutput,
        component_id: str,
        part_id: str | None,
        start_counter: int,
    ) -> tuple[
        list[CriterionAssessment],
        dict[str, str],
        int,
    ]:
        results: list[
            CriterionAssessment
        ] = []

        criterion_id_by_requirement: dict[
            str,
            str,
        ] = {}

        counter = start_counter

        for item in (
            component_output
            .criterion_assessments
        ):
            criterion_id = (
                f"ca_{counter:03d}"
            )

            criterion_id_by_requirement[
                item.requirement_id
            ] = criterion_id

            results.append(
                CriterionAssessment(
                    criterion_assessment_id=(
                        criterion_id
                    ),
                    requirement_id=(
                        item.requirement_id
                    ),
                    part_ids=(
                        BaselineFeedbackService
                        ._part_ids_for_record(
                            part_id=part_id
                        )
                    ),
                    task_ids=(
                        BaselineFeedbackService
                        ._task_ids_for_record(
                            component_id=(
                                component_id
                            ),
                            part_id=part_id,
                        )
                    ),
                    status=(
                        item.status
                    ),
                    internal_finding=(
                        item.internal_finding
                    ),
                    verification_status=(
                        item.verification_status
                    ),
                    confidence=(
                        item.confidence
                    ),
                    evidence=(
                        item.evidence
                    ),
                )
            )

            counter += 1

        return (
            results,
            criterion_id_by_requirement,
            counter,
        )

    @staticmethod
    def _assemble_observations(
        *,
        component_output: ComponentFeedbackLLMOutput,
        component_id: str,
        part_id: str | None,
        criterion_id_by_requirement: dict[str, str],
        start_counter: int,
    ) -> tuple[
        list[FeedbackObservation],
        list[str],
        int,
    ]:
        results: list[
            FeedbackObservation
        ] = []

        observation_ids: list[str] = []

        counter = start_counter

        scope = (
            BaselineFeedbackService
            ._scope_for_component(
                component_id=(
                    component_id
                ),
                part_id=part_id,
            )
        )

        for item in (
            component_output
            .observations
        ):
            observation_id = (
                f"obs_{counter:03d}"
            )

            criterion_assessment_ids = [
                criterion_id_by_requirement[
                    requirement_id
                ]
                for requirement_id
                in item.requirement_ids
                if requirement_id
                in criterion_id_by_requirement
            ]

            results.append(
                FeedbackObservation(
                    observation_id=(
                        observation_id
                    ),
                    feedback_type=(
                        item.feedback_type
                    ),
                    scope=scope,
                    part_ids=(
                        BaselineFeedbackService
                        ._part_ids_for_record(
                            part_id=part_id
                        )
                    ),
                    task_ids=(
                        BaselineFeedbackService
                        ._task_ids_for_record(
                            component_id=(
                                component_id
                            ),
                            part_id=part_id,
                        )
                    ),
                    attempt_unit_ids=[],
                    requirement_ids=list(
                        item.requirement_ids
                    ),
                    criterion_assessment_ids=(
                        criterion_assessment_ids
                    ),
                    internal_finding=(
                        item.internal_finding
                    ),
                    student_feedback=(
                        item.student_feedback
                    ),
                    suggestion=(
                        item.suggestion
                    ),
                    verification_status=(
                        item.verification_status
                    ),
                    confidence=(
                        item.confidence
                    ),
                    evidence=(
                        item.evidence
                    ),
                )
            )

            observation_ids.append(
                observation_id
            )

            counter += 1

        return (
            results,
            observation_ids,
            counter,
        )

    @staticmethod
    def _build_overall_feedback(
        *,
        component_summaries: list[str],
        observations: list[FeedbackObservation],
    ) -> OverallFeedback:
        summaries = [
            summary.strip()
            for summary in component_summaries
            if (
                isinstance(
                    summary,
                    str,
                )
                and summary.strip()
            )
        ]

        summary = (
            " ".join(
                summaries
            )
            if summaries
            else (
                "Feedback was generated across the submitted "
                "assessment components."
            )
        )

        strength_observations = [
            observation
            for observation in observations
            if observation.feedback_type
            == "strength"
        ]

        improvement_observations = [
            observation
            for observation in observations
            if observation.feedback_type
            in {
                "error",
                "omission",
                "suggestion",
            }
        ]

        strengths_summary = (
            " ".join(
                observation.student_feedback
                for observation
                in strength_observations[:3]
            )
            if strength_observations
            else (
                "No distinct strengths were summarised "
                "separately from the component feedback."
            )
        )

        improvement_summary = (
            " ".join(
                observation.student_feedback
                for observation
                in improvement_observations[:3]
            )
            if improvement_observations
            else (
                "No major improvement priorities were "
                "identified in the generated component feedback."
            )
        )

        return OverallFeedback(
            summary=summary,
            strengths_summary=(
                strengths_summary
            ),
            improvement_summary=(
                improvement_summary
            ),
            observation_ids=[
                observation.observation_id
                for observation in observations
            ],
        )

    # ========================================================
    # Checkpoint support
    # ========================================================

    def _checkpoint_path(
        self,
        *,
        component_id: str,
    ) -> Path | None:
        if self.checkpoint_dir is None:
            return None

        return (
            self.checkpoint_dir
            / f"{component_id}.json"
        )

    def _load_component_checkpoint(
        self,
        *,
        component_id: str,
        requirement_ids: list[str],
    ) -> ComponentFeedbackLLMOutput | None:
        if (
            not self.resume_from_checkpoints
        ):
            return None

        path = self._checkpoint_path(
            component_id=component_id
        )

        if (
            path is None
            or not path.exists()
        ):
            return None

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise BaselineFeedbackError(
                "Could not read baseline checkpoint "
                f"{path}."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise BaselineFeedbackError(
                "Baseline checkpoint must contain a JSON object."
            )

        if (
            payload.get(
                "component_id"
            )
            != component_id
        ):
            return None

        checkpoint_model = payload.get(
            "model_name"
        )

        if (
            isinstance(
                checkpoint_model,
                str,
            )
            and checkpoint_model
            != self.llm_client.model_name
        ):
            return None

        checkpoint_requirement_ids = (
            payload.get(
                "requirement_ids"
            )
        )

        if (
            checkpoint_requirement_ids
            != requirement_ids
        ):
            return None

        try:
            return (
                ComponentFeedbackLLMOutput
                .model_validate(
                    payload.get(
                        "component_output"
                    )
                )
            )

        except (
            ValueError,
            TypeError,
        ) as exc:
            raise BaselineFeedbackError(
                "Baseline checkpoint contains invalid "
                f"component feedback: {path}."
            ) from exc

    def _save_component_checkpoint(
        self,
        *,
        component_id: str,
        requirement_ids: list[str],
        component_output: ComponentFeedbackLLMOutput,
    ) -> None:
        path = self._checkpoint_path(
            component_id=component_id
        )

        if path is None:
            return

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": "1.0",
            "feedback_mode": (
                "baseline_component"
            ),
            "component_id": (
                component_id
            ),
            "requirement_ids": (
                requirement_ids
            ),
            "model_name": (
                self.llm_client.model_name
            ),
            "component_output": (
                component_output.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
        }

        temporary_path = (
            path.with_suffix(
                path.suffix
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
            path
        )

        print(
            "✓ Saved baseline checkpoint for "
            f"{component_id}"
        )