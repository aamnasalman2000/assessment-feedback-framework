from __future__ import annotations

from typing import Any

import json

from .feedback_llm_client import (
    FeedbackStructuredClient,
)
from .feedback_models import (
    CriterionAssessment,
    FeedbackObservation,
    ReflectionDecision,
)
from .feedback_reflection_prompts import (
    build_requirement_reflection_prompt,
)


class FeedbackReflectionError(RuntimeError):
    """Raised when a reflection decision or applied revision is invalid."""


class FeedbackReflectionService:
    """
    Audit an existing criterion assessment and feedback observation.

    The LLM returns a ReflectionDecision containing an analysis and an
    optional patch. Python applies that patch while preserving all
    identifiers and structural metadata.
    """

    def __init__(
        self,
        *,
        llm_client: FeedbackStructuredClient,
    ) -> None:
        self.llm_client = llm_client

    def reflect_requirement_feedback(
        self,
        *,
        component: dict[str, Any],
        requirement: dict[str, Any],
        part_id: str | None,
        candidate_units: list[Any],
        semantic_annotations: list[Any],
        alignment: dict[str, Any],
        processing_checks: list[Any],
        processing_diagnostics: list[Any],
        initial_criterion_assessment: CriterionAssessment,
        initial_observation: FeedbackObservation,
        global_requirements: list[dict[str, Any]],
        feedback_requirements: list[str],
        specification_notes: list[dict[str, Any]],
        log_name: str,
    ) -> tuple[
        ReflectionDecision,
        CriterionAssessment,
        FeedbackObservation,
    ]:
        """
        Audit one existing requirement-level evaluation.

        Returns:
            A tuple containing:

            1. the reflection audit and revision decision;
            2. the final criterion assessment;
            3. the final student-facing observation.

        When should_revise is false, the two final records are unchanged
        copies of the originals.
        """

        candidate_unit_ids = (
            self._collect_candidate_unit_ids(
                candidate_units
            )
        )

        candidate_artifact_ids = (
            self._collect_candidate_artifact_ids(
                candidate_units
            )
        )

        system_prompt, user_prompt = (
            build_requirement_reflection_prompt(
                component=component,
                requirement=requirement,
                part_id=part_id,
                candidate_units=candidate_units,
                semantic_annotations=semantic_annotations,
                alignment=alignment,
                processing_checks=processing_checks,
                processing_diagnostics=(
                    processing_diagnostics
                ),
                initial_criterion_assessment=(
                    initial_criterion_assessment
                ),
                initial_observation=initial_observation,
                global_requirements=global_requirements,
                feedback_requirements=(
                    feedback_requirements
                ),
                specification_notes=specification_notes,
            )
        )

        decision = (
            self.llm_client.generate_reflection_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                candidate_artifact_ids=(
                    candidate_artifact_ids
                ),
                candidate_unit_ids=candidate_unit_ids,
                requirement_id=requirement["id"],
                part_id=part_id,
                component_id=component["component_id"],
                log_name=f"{log_name}_reflection",
            )
        )
        
        self._validate_requirement_basis(
            decision=decision,
            requirement=requirement,
        )

        self._validate_reflection_decision(
            decision=decision,
            candidate_unit_ids=set(
                candidate_unit_ids
            ),
            candidate_artifact_ids=set(
                candidate_artifact_ids
            ),
        )

        final_criterion, final_observation = (
            self._apply_reflection_decision(
                decision=decision,
                initial_criterion_assessment=(
                    initial_criterion_assessment
                ),
                initial_observation=initial_observation,
            )
        )

        return (
            decision,
            final_criterion,
            final_observation,
        )

    @staticmethod
    def _collect_candidate_unit_ids(
        candidate_units: list[Any],
    ) -> list[str]:
        unit_ids: list[str] = []

        for unit in candidate_units:
            if isinstance(unit, dict):
                unit_id = unit.get("unit_id")
            else:
                unit_id = getattr(
                    unit,
                    "unit_id",
                    None,
                )

            if isinstance(unit_id, str):
                unit_ids.append(unit_id)

        return list(dict.fromkeys(unit_ids))

    @staticmethod
    def _collect_candidate_artifact_ids(
        candidate_units: list[Any],
    ) -> list[str]:
        artifact_ids: list[str] = []

        for unit in candidate_units:
            if isinstance(unit, dict):
                artifact_id = unit.get(
                    "artifact_id"
                )
            else:
                artifact_id = getattr(
                    unit,
                    "artifact_id",
                    None,
                )

            if isinstance(artifact_id, str):
                artifact_ids.append(artifact_id)

        return list(dict.fromkeys(artifact_ids))

    @staticmethod
    def _validate_reflection_decision(
        *,
        decision: ReflectionDecision,
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> None:
        errors: list[str] = []

        if not decision.analysis.should_revise:
            return

        revised_evidence = (
            decision.revised_evidence or []
        )

        for evidence in revised_evidence:
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
                    "Reflection references unknown "
                    "candidate artifact "
                    f"{evidence.artifact_id!r}."
                )

            if (
                evidence.unit_id is not None
                and evidence.unit_id
                not in candidate_unit_ids
            ):
                errors.append(
                    "Reflection references unknown "
                    "candidate unit "
                    f"{evidence.unit_id!r}."
                )

        if errors:
            raise FeedbackReflectionError(
                "Invalid reflection decision:\n- "
                + "\n- ".join(errors)
            )

    @staticmethod
    def _apply_reflection_decision(
        *,
        decision: ReflectionDecision,
        initial_criterion_assessment: CriterionAssessment,
        initial_observation: FeedbackObservation,
    ) -> tuple[
        CriterionAssessment,
        FeedbackObservation,
    ]:
        """
        Apply the optional LLM patch while preserving Python-controlled
        identifiers, references and scopes.
        """

        if not decision.analysis.should_revise:
            return (
                initial_criterion_assessment.model_copy(
                    deep=True
                ),
                initial_observation.model_copy(
                    deep=True
                ),
            )

        revised_status = decision.revised_status
        revised_internal_finding = (
            decision.revised_internal_finding
        )
        revised_student_feedback = (
            decision.revised_student_feedback
        )
        revised_verification_status = (
            decision.revised_verification_status
        )
        revised_confidence = (
            decision.revised_confidence
        )
        revised_evidence = decision.revised_evidence

        if (
            revised_status is None
            or revised_internal_finding is None
            or revised_student_feedback is None
            or revised_verification_status is None
            or revised_confidence is None
            or revised_evidence is None
        ):
            raise FeedbackReflectionError(
                "Reflection requested revision but did "
                "not provide a complete revision patch."
            )

        final_criterion = (
            initial_criterion_assessment.model_copy(
                update={
                    "status": revised_status,
                    "internal_finding": (
                        revised_internal_finding
                    ),
                    "verification_status": (
                        revised_verification_status
                    ),
                    "confidence": revised_confidence,
                    "evidence": revised_evidence,
                },
                deep=True,
            )
        )

        observation_update: dict[str, Any] = {
            "feedback_type": (
                FeedbackReflectionService
                ._feedback_type_for_status(
                    revised_status
                )
            ),
            "internal_finding": (
                revised_internal_finding
            ),
            "student_feedback": (
                revised_student_feedback
            ),
            "verification_status": (
                revised_verification_status
            ),
            "confidence": revised_confidence,
            "evidence": revised_evidence,
        }

        if decision.revised_suggestion is not None:
            observation_update["suggestion"] = (
                decision.revised_suggestion
            )
        else:
            observation_update["suggestion"] = None

        final_observation = (
            initial_observation.model_copy(
                update=observation_update,
                deep=True,
            )
        )

        return final_criterion, final_observation

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
    def _validate_requirement_basis(
        *,
        decision: ReflectionDecision,
        requirement: dict[str, Any],
    ) -> None:
        """
        Ensure that a requested revision is grounded in an exact phrase from
        the supplied assessment requirement.
        """
        if not decision.analysis.should_revise:
            return

        basis = decision.analysis.requirement_basis

        if not isinstance(basis, str) or not basis.strip():
            raise FeedbackReflectionError(
                "Reflection requested a revision without "
                "providing a requirement basis."
            )

        requirement_text = json.dumps(
            requirement,
            ensure_ascii=False,
        )

        normalised_basis = " ".join(
            basis.lower().split()
        )

        normalised_requirement = " ".join(
            requirement_text.lower().split()
        )

        if normalised_basis not in normalised_requirement:
            raise FeedbackReflectionError(
                "The reflection requirement_basis does not "
                "appear verbatim in the exact requirement."
            )