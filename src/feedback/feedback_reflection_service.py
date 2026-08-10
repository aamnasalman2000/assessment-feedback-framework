from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .feedback_llm_client import (
    FeedbackStructuredClient,
)
from .feedback_models import (
    CriterionAssessment,
    FeedbackObservation,
    ReflectionAnalysis,
    ReflectionDecision,
)
from .feedback_reflection_prompts import (
    build_requirement_reflection_audit_prompt,
    build_requirement_reflection_revision_prompt,
)


class FeedbackReflectionError(RuntimeError):
    """
    Raised when a reflection audit, revision, checkpoint,
    or applied patch is invalid.
    """


class FeedbackReflectionService:
    """
    Audit an existing criterion assessment and feedback observation.

    Reflection uses two stages:

    Stage 1:
        Audit the existing assessment and decide whether it should be
        retained or revised.

    Stage 2:
        Run only when Stage 1 identifies a material defect. Generate the
        smallest complete revision patch.

    Python combines both stages into a ReflectionDecision while retaining
    ownership of all identifiers and structural metadata.

    Deterministic guardrails are applied after Stage 2. If a generated
    revision is contradictory or insufficiently grounded, the revision is
    rejected and the original assessment is preserved.

    Completed reflection decisions may be checkpointed per requirement so
    long reflective runs can safely resume after interruption.
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

        A valid checkpoint is loaded first when resume support is enabled.

        Otherwise:

        1. run Stage-1 audit;
        2. stop immediately for KEEP;
        3. run Stage-2 revision only for REVISE;
        4. validate the combined ReflectionDecision;
        5. apply deterministic revision guardrails;
        6. apply the accepted patch or preserve the original;
        7. save a checkpoint.
        """

        component_id = component[
            "component_id"
        ]

        requirement_id = requirement[
            "id"
        ]

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

        # ====================================================
        # Resume from checkpoint when available
        # ====================================================

        checkpoint = (
            self._load_reflection_checkpoint(
                component_id=component_id,
                requirement_id=requirement_id,
                initial_criterion_assessment=(
                    initial_criterion_assessment
                ),
                initial_observation=(
                    initial_observation
                ),
            )
        )

        if checkpoint is not None:
            (
                decision,
                final_criterion,
                final_observation,
            ) = checkpoint

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

            checkpoint_guardrail_errors = (
                self._collect_revision_guardrail_errors(
                    decision=decision,
                    requirement=requirement,
                    processing_checks=(
                        processing_checks
                    ),
                    candidate_units=(
                        candidate_units
                    ),
                )
            )

            if not checkpoint_guardrail_errors:
                print(
                    "✓ Loaded reflection checkpoint for "
                    f"{component_id}/{requirement_id}"
                )

                return (
                    decision,
                    final_criterion,
                    final_observation,
                )

            print(
                "⚠ Existing reflection checkpoint failed "
                "current guardrails for "
                f"{component_id}/{requirement_id}; "
                "regenerating."
            )

            for error in checkpoint_guardrail_errors:
                print(
                    f"  - {error}"
                )

        # ====================================================
        # Stage 1: audit only
        # ====================================================

        (
            audit_system_prompt,
            audit_user_prompt,
        ) = (
            build_requirement_reflection_audit_prompt(
                component=component,
                requirement=requirement,
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
                initial_criterion_assessment=(
                    initial_criterion_assessment
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
            )
        )

        audit = (
            self.llm_client
            .generate_reflection_audit(
                system_prompt=(
                    audit_system_prompt
                ),
                user_prompt=(
                    audit_user_prompt
                ),
                log_name=log_name,
            )
        )

        self._validate_analysis_requirement_basis(
            analysis=(
                audit.analysis
            ),
            requirement=requirement,
        )

        # ====================================================
        # KEEP path
        # ====================================================

        if (
            not audit.analysis
            .should_revise
        ):
            decision = ReflectionDecision(
                analysis=(
                    audit.analysis
                ),
            )

            final_criterion = (
                initial_criterion_assessment
                .model_copy(
                    deep=True
                )
            )

            final_observation = (
                initial_observation
                .model_copy(
                    deep=True
                )
            )

            self._save_reflection_checkpoint(
                component_id=component_id,
                requirement_id=(
                    requirement_id
                ),
                decision=decision,
                final_criterion=(
                    final_criterion
                ),
                final_observation=(
                    final_observation
                ),
                initial_criterion_assessment=(
                    initial_criterion_assessment
                ),
                initial_observation=(
                    initial_observation
                ),
            )

            return (
                decision,
                final_criterion,
                final_observation,
            )

        # ====================================================
        # Stage 2: revision patch
        # ====================================================

        (
            revision_system_prompt,
            revision_user_prompt,
        ) = (
            build_requirement_reflection_revision_prompt(
                component=component,
                requirement=requirement,
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
                initial_criterion_assessment=(
                    initial_criterion_assessment
                ),
                initial_observation=(
                    initial_observation
                ),
                audit_analysis=(
                    audit.analysis
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

        revision = (
            self.llm_client
            .generate_reflection_revision(
                system_prompt=(
                    revision_system_prompt
                ),
                user_prompt=(
                    revision_user_prompt
                ),
                candidate_artifact_ids=(
                    candidate_artifact_ids
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
                requirement_id=(
                    requirement_id
                ),
                part_id=part_id,
                component_id=(
                    component_id
                ),
                log_name=log_name,
            )
        )

        # ====================================================
        # Combine Stage 1 + Stage 2
        # ====================================================

        decision = ReflectionDecision(
            analysis=(
                audit.analysis
            ),
            **revision.model_dump(
                mode="python",
                exclude_none=True,
            ),
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

        # ====================================================
        # Deterministic revision guardrails
        # ====================================================

        guardrail_errors = (
            self._collect_revision_guardrail_errors(
                decision=decision,
                requirement=requirement,
                processing_checks=(
                    processing_checks
                ),
                candidate_units=(
                    candidate_units
                ),
            )
        )

        if guardrail_errors:
            print(
                "⚠ Reflection revision rejected by "
                "deterministic guardrails for "
                f"{component_id}/{requirement_id}"
            )

            for error in guardrail_errors:
                print(
                    f"  - {error}"
                )

            decision = (
                self._build_guardrail_keep_decision(
                    audit_analysis=(
                        audit.analysis
                    ),
                    guardrail_errors=(
                        guardrail_errors
                    ),
                )
            )

            final_criterion = (
                initial_criterion_assessment
                .model_copy(
                    deep=True
                )
            )

            final_observation = (
                initial_observation
                .model_copy(
                    deep=True
                )
            )

        else:
            (
                final_criterion,
                final_observation,
            ) = (
                self._apply_reflection_decision(
                    decision=decision,
                    initial_criterion_assessment=(
                        initial_criterion_assessment
                    ),
                    initial_observation=(
                        initial_observation
                    ),
                )
            )

        self._save_reflection_checkpoint(
            component_id=component_id,
            requirement_id=requirement_id,
            decision=decision,
            final_criterion=(
                final_criterion
            ),
            final_observation=(
                final_observation
            ),
            initial_criterion_assessment=(
                initial_criterion_assessment
            ),
            initial_observation=(
                initial_observation
            ),
        )

        return (
            decision,
            final_criterion,
            final_observation,
        )

    # ========================================================
    # Checkpoint paths
    # ========================================================

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

    # ========================================================
    # Checkpoint loading
    # ========================================================

    def _load_reflection_checkpoint(
        self,
        *,
        component_id: str,
        requirement_id: str,
        initial_criterion_assessment: CriterionAssessment,
        initial_observation: FeedbackObservation,
    ) -> tuple[
        ReflectionDecision,
        CriterionAssessment,
        FeedbackObservation,
    ] | None:
        if (
            not self.resume_from_checkpoints
        ):
            return None

        checkpoint_path = (
            self._checkpoint_path(
                component_id=component_id,
                requirement_id=(
                    requirement_id
                ),
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
                payload = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise FeedbackReflectionError(
                "Could not read reflection "
                "checkpoint "
                f"{checkpoint_path}."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise FeedbackReflectionError(
                "Reflection checkpoint must "
                "contain a JSON object: "
                f"{checkpoint_path}."
            )

        if (
            payload.get(
                "component_id"
            )
            != component_id
            or payload.get(
                "requirement_id"
            )
            != requirement_id
        ):
            raise FeedbackReflectionError(
                "Reflection checkpoint "
                "identifiers do not match the "
                "requested component/requirement: "
                f"{checkpoint_path}."
            )

        checkpoint_model = (
            payload.get(
                "model_name"
            )
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

        initial_criterion_id = (
            payload.get(
                "initial_criterion_assessment_id"
            )
        )

        initial_observation_id = (
            payload.get(
                "initial_observation_id"
            )
        )

        if (
            initial_criterion_id
            != initial_criterion_assessment
            .criterion_assessment_id
            or initial_observation_id
            != initial_observation
            .observation_id
        ):
            return None

        try:
            decision = (
                ReflectionDecision
                .model_validate(
                    payload.get(
                        "decision"
                    )
                )
            )

            final_criterion = (
                CriterionAssessment
                .model_validate(
                    payload.get(
                        "final_criterion"
                    )
                )
            )

            final_observation = (
                FeedbackObservation
                .model_validate(
                    payload.get(
                        "final_observation"
                    )
                )
            )

        except (
            ValueError,
            TypeError,
        ) as exc:
            raise FeedbackReflectionError(
                "Reflection checkpoint "
                "contains invalid structured "
                "data: "
                f"{checkpoint_path}."
            ) from exc

        return (
            decision,
            final_criterion,
            final_observation,
        )

    # ========================================================
    # Checkpoint saving
    # ========================================================

    def _save_reflection_checkpoint(
        self,
        *,
        component_id: str,
        requirement_id: str,
        decision: ReflectionDecision,
        final_criterion: CriterionAssessment,
        final_observation: FeedbackObservation,
        initial_criterion_assessment: CriterionAssessment,
        initial_observation: FeedbackObservation,
    ) -> None:
        checkpoint_path = (
            self._checkpoint_path(
                component_id=component_id,
                requirement_id=(
                    requirement_id
                ),
            )
        )

        if checkpoint_path is None:
            return

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": "3.2",
            "reflection_mode": (
                "two_stage_guarded"
            ),
            "component_id": (
                component_id
            ),
            "requirement_id": (
                requirement_id
            ),
            "model_name": (
                self.llm_client
                .model_name
            ),
            "initial_criterion_assessment_id": (
                initial_criterion_assessment
                .criterion_assessment_id
            ),
            "initial_observation_id": (
                initial_observation
                .observation_id
            ),
            "decision": (
                decision.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "final_criterion": (
                final_criterion.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "final_observation": (
                final_observation.model_dump(
                    mode="json",
                    exclude_none=True,
                )
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
            "✓ Saved reflection checkpoint "
            "for "
            f"{component_id}/"
            f"{requirement_id}"
        )

    # ========================================================
    # Candidate IDs
    # ========================================================

    @staticmethod
    def _collect_candidate_unit_ids(
        candidate_units: list[Any],
    ) -> list[str]:
        unit_ids: list[str] = []

        for unit in candidate_units:
            if isinstance(
                unit,
                dict,
            ):
                unit_id = unit.get(
                    "unit_id"
                )

            else:
                unit_id = getattr(
                    unit,
                    "unit_id",
                    None,
                )

            if isinstance(
                unit_id,
                str,
            ):
                unit_ids.append(
                    unit_id
                )

        return list(
            dict.fromkeys(
                unit_ids
            )
        )

    @staticmethod
    def _collect_candidate_artifact_ids(
        candidate_units: list[Any],
    ) -> list[str]:
        artifact_ids: list[str] = []

        for unit in candidate_units:
            if isinstance(
                unit,
                dict,
            ):
                artifact_id = unit.get(
                    "artifact_id"
                )

            else:
                artifact_id = getattr(
                    unit,
                    "artifact_id",
                    None,
                )

            if isinstance(
                artifact_id,
                str,
            ):
                artifact_ids.append(
                    artifact_id
                )

        return list(
            dict.fromkeys(
                artifact_ids
            )
        )

    # ========================================================
    # Reflection structure validation
    # ========================================================

    @staticmethod
    def _validate_reflection_decision(
        *,
        decision: ReflectionDecision,
        candidate_unit_ids: set[str],
        candidate_artifact_ids: set[str],
    ) -> None:
        errors: list[str] = []

        if (
            not decision.analysis
            .should_revise
        ):
            return

        revised_evidence = (
            decision.revised_evidence
            or []
        )

        for evidence in (
            revised_evidence
        ):
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
                    "Reflection references "
                    "unknown candidate artifact "
                    f"{evidence.artifact_id!r}."
                )

            if (
                evidence.unit_id
                is not None
                and evidence.unit_id
                not in candidate_unit_ids
            ):
                errors.append(
                    "Reflection references "
                    "unknown candidate unit "
                    f"{evidence.unit_id!r}."
                )

        if errors:
            raise FeedbackReflectionError(
                "Invalid reflection "
                "decision:\n- "
                + "\n- ".join(
                    errors
                )
            )

    # ========================================================
    # Deterministic revision guardrails
    # ========================================================

    @staticmethod
    def _normalise_text(
        value: Any,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            return ""

        return " ".join(
            value.lower().split()
        )

    @staticmethod
    def _collect_evidence_descriptions(
        decision: ReflectionDecision,
    ) -> list[str]:
        descriptions: list[str] = []

        for evidence in (
            decision.revised_evidence
            or []
        ):
            description = getattr(
                evidence,
                "description",
                None,
            )

            if isinstance(
                description,
                str,
            ):
                descriptions.append(
                    description
                )

        return descriptions

    @staticmethod
    def _contains_negative_requirement_claim(
        text: str,
    ) -> bool:
        normalised = (
            FeedbackReflectionService
            ._normalise_text(
                text
            )
        )

        negative_phrases = (
            "the requirement is not met",
            "requirement is not satisfied",
            "does not meet the requirement",
            "does not satisfy the requirement",
            "fails to meet the requirement",
            "fails the requirement",
            "requirement has not been met",
            "requirement has not been satisfied",
        )

        return any(
            phrase in normalised
            for phrase in negative_phrases
        )

    @staticmethod
    def _contains_positive_requirement_claim(
        text: str,
    ) -> bool:
        normalised = (
            FeedbackReflectionService
            ._normalise_text(
                text
            )
        )

        positive_phrases = (
            "the requirement is met",
            "requirement is satisfied",
            "meets the requirement",
            "satisfies the requirement",
            "fully meets the requirement",
            "fulfills the requirement",
            "fulfils the requirement",
        )

        return any(
            phrase in normalised
            for phrase in positive_phrases
        )

    @staticmethod
    def _requirement_needs_deterministic_verification(
        requirement: dict[str, Any],
    ) -> bool:
        """
        Identify requirement types where a positive conclusion should not be
        inferred from structural evidence alone.
        """

        requirement_text = (
            requirement.get(
                "requirement",
                ""
            )
        )

        normalised = (
            FeedbackReflectionService
            ._normalise_text(
                requirement_text
            )
        )

        verification_phrases = (
            "logically coherent",
            "logical coherence",
            "internally consistent",
            "internal consistency",
            "logically consistent",
            "logical consistency",
            "ontology is consistent",
            "ontology remains consistent",
            "satisfiable",
            "unsatisfiable",
            "must compile",
            "must compile successfully",
            "compiles successfully",
            "compiles without",
            "compilation",
            "must execute",
            "executes successfully",
            "must run",
            "runs successfully",
            "must pass all tests",
            "must pass the tests",
            "passes all tests",
            "passes the tests",
        )

        return any(
            phrase in normalised
            for phrase in verification_phrases
        )

    @staticmethod
    def _required_check_types_for_requirement(
        requirement: dict[str, Any],
    ) -> set[str]:
        """
        Identify deterministic checks that can actually establish the
        property named by this requirement.

        A successful unrelated preprocessing check must never be treated as
        verification of the requirement.
        """

        requirement_text = (
            requirement.get(
                "requirement",
                ""
            )
        )

        normalised = (
            FeedbackReflectionService
            ._normalise_text(
                requirement_text
            )
        )

        if any(
            phrase in normalised
            for phrase in (
                "logically coherent",
                "logical coherence",
                "internally consistent",
                "internal consistency",
                "logically consistent",
                "logical consistency",
                "ontology is consistent",
                "ontology remains consistent",
                "satisfiable",
                "unsatisfiable",
            )
        ):
            return {
                "owl_consistency",
                "owl_reasoning",
                "ontology_consistency",
            }

        if any(
            phrase in normalised
            for phrase in (
                "must compile",
                "must compile successfully",
                "compile successfully",
                "compiles successfully",
                "compiles without",
                "compilation",
            )
        ):
            return {
                "compile",
                "compilation",
                "code_compile",
                "lean_compile",
            }

        if any(
            phrase in normalised
            for phrase in (
                "executes successfully",
                "runs successfully",
            )
        ):
            return {
                "execution",
                "run",
                "runtime",
            }

        if any(
            phrase in normalised
            for phrase in (
                "passes all tests",
                "passes the tests",
            )
        ):
            return {
                "tests",
                "test_run",
                "unit_tests",
            }

        return set()

    @staticmethod
    def _has_successful_required_deterministic_check(
        *,
        requirement: dict[str, Any],
        processing_checks: list[Any],
    ) -> bool:
        """
        Return true only when a successful deterministic check exists for the
        specific property named by the requirement.

        For example:

            owl_parse = passed

        does not verify ontology consistency.

        A consistency requirement requires a successful owl_consistency,
        owl_reasoning, or ontology_consistency check.
        """

        required_check_types = (
            FeedbackReflectionService
            ._required_check_types_for_requirement(
                requirement
            )
        )

        if not required_check_types:
            return False

        positive_statuses = {
            "passed",
            "pass",
            "success",
            "successful",
            "verified",
            "valid",
            "ok",
            "consistent",
            "satisfiable",
        }

        for check in processing_checks:
            if isinstance(
                check,
                dict,
            ):
                data = check

            elif hasattr(
                check,
                "model_dump",
            ):
                data = check.model_dump(
                    mode="json",
                    exclude_none=True,
                )

            else:
                continue

            check_type = data.get(
                "check_type"
            )

            status = data.get(
                "status"
            )

            if (
                not isinstance(
                    check_type,
                    str,
                )
                or not isinstance(
                    status,
                    str,
                )
            ):
                continue

            normalised_check_type = (
                check_type
                .strip()
                .lower()
            )

            normalised_status = (
                status
                .strip()
                .lower()
            )

            if (
                normalised_check_type
                in required_check_types
                and normalised_status
                in positive_statuses
            ):
                return True

        return False

    @staticmethod
    def _requirement_needs_task_5_description(
        requirement: dict[str, Any],
    ) -> bool:
        """Return true when the requirement depends on written explanation."""
        requirement_text = requirement.get("requirement", "")
        normalised = FeedbackReflectionService._normalise_text(requirement_text)
        phrases = (
            "description", "describe", "explanation", "explain",
            "justification", "justify", "discussion", "discuss",
        )
        return any(phrase in normalised for phrase in phrases)

    @staticmethod
    def _task_5_candidates_have_explanation(
        candidate_units: list[Any],
    ) -> bool:
        """Return true only if an aligned Task 5 unit records an explanation."""
        for unit in candidate_units:
            if isinstance(unit, dict):
                data = unit
            elif hasattr(unit, "model_dump"):
                data = unit.model_dump(mode="python", exclude_none=True)
            else:
                data = {
                    "unit_type": getattr(unit, "unit_type", None),
                    "structured_data": getattr(unit, "structured_data", None),
                }

            structured_data = data.get("structured_data")
            if not isinstance(structured_data, dict):
                structured_data = {}

            is_task_5 = (
                data.get("unit_type") == "prolog_task_5_answer"
                or structured_data.get("is_task_5_answer") is True
            )
            if (
                is_task_5
                and structured_data.get("has_explanation") is True
            ):
                return True

        return False

    @staticmethod
    def _collect_revision_guardrail_errors(
        *,
        decision: ReflectionDecision,
        requirement: dict[str, Any],
        processing_checks: list[Any],
        candidate_units: list[Any],
    ) -> list[str]:
        """
        Check a proposed Stage-2 patch for high-confidence deterministic
        failure modes.

        These rules intentionally reject dubious patches rather than trying
        to repair them automatically.
        """

        if (
            not decision.analysis
            .should_revise
        ):
            return []

        errors: list[str] = []

        revised_status = (
            decision.revised_status
        )

        if revised_status is None:
            errors.append(
                "Revision has no revised_status."
            )

            return errors

        revised_internal_finding = (
            decision.revised_internal_finding
            or ""
        )

        revised_student_feedback = (
            decision.revised_student_feedback
            or ""
        )

        evidence_descriptions = (
            FeedbackReflectionService
            ._collect_evidence_descriptions(
                decision
            )
        )

        # ----------------------------------------------------
        # 1. Status/text contradiction
        # ----------------------------------------------------

        if revised_status == "met":
            contradictory_texts = [
                revised_internal_finding,
                revised_student_feedback,
                *evidence_descriptions,
            ]

            if any(
                FeedbackReflectionService
                ._contains_negative_requirement_claim(
                    text
                )
                for text
                in contradictory_texts
            ):
                errors.append(
                    "Revision status is 'met' but its own "
                    "finding, student feedback, or evidence "
                    "states that the requirement is not met."
                )

        if revised_status in {
            "not_met",
            "missing",
        }:
            contradictory_texts = [
                revised_internal_finding,
                revised_student_feedback,
            ]

            if any(
                FeedbackReflectionService
                ._contains_positive_requirement_claim(
                    text
                )
                for text
                in contradictory_texts
            ):
                errors.append(
                    "Revision status is negative but its own "
                    "finding or student feedback states that "
                    "the requirement is met."
                )

        # ----------------------------------------------------
        # 2. Positive decisions cannot rely only on absence
        # ----------------------------------------------------

        if revised_status == "met":
            revised_evidence = (
                decision.revised_evidence
                or []
            )

            submission_references = [
                evidence
                for evidence
                in revised_evidence
                if (
                    evidence.evidence_type
                    == "submission_reference"
                )
            ]

            if (
                not submission_references
                and decision
                .revised_verification_status
                != "verified_by_tool"
            ):
                errors.append(
                    "A 'met' revision must be supported by "
                    "submission-reference evidence or an "
                    "explicit deterministic tool verification; "
                    "absence evidence alone cannot establish "
                    "that a requirement is met."
                )

        # ----------------------------------------------------
        # 3. Tool-dependent/global properties
        # ----------------------------------------------------

        if (
            revised_status == "met"
            and FeedbackReflectionService
            ._requirement_needs_deterministic_verification(
                requirement
            )
            and not FeedbackReflectionService
            ._has_successful_required_deterministic_check(
                requirement=requirement,
                processing_checks=(
                    processing_checks
                ),
            )
        ):
            errors.append(
                "The requirement concerns a property that "
                "requires deterministic verification, but no "
                "successful requirement-specific processing "
                "check establishes that property. The "
                "requirement cannot safely be marked as "
                "'met'."
            )

        # ----------------------------------------------------
        # 4. Task 5 description-dependent requirements
        # ----------------------------------------------------

        if (
            revised_status == "met"
            and FeedbackReflectionService
            ._requirement_needs_task_5_description(
                requirement
            )
            and not FeedbackReflectionService
            ._task_5_candidates_have_explanation(
                candidate_units
            )
        ):
            errors.append(
                "The requirement depends on a student-provided "
                "description or explanation, but the aligned Task 5 "
                "submission units contain no explanation. Formula "
                "evidence alone cannot establish that this requirement "
                "is met."
            )

        # ----------------------------------------------------
        # 5. Verification-status sanity
        # ----------------------------------------------------

        if (
            decision.revised_verification_status
            == "verified_by_tool"
            and not FeedbackReflectionService
            ._has_successful_required_deterministic_check(
                requirement=requirement,
                processing_checks=(
                    processing_checks
                ),
            )
        ):
            errors.append(
                "Revision claims verified_by_tool, but no "
                "successful requirement-specific deterministic "
                "processing check was supplied."
            )

        return errors

    @staticmethod
    def _build_guardrail_keep_decision(
        *,
        audit_analysis: ReflectionAnalysis,
        guardrail_errors: list[str],
    ) -> ReflectionDecision:
        """
        Fail safely when Stage 2 produces an invalid or contradictory patch.

        Debug logs still retain the original Stage-1 and Stage-2 model
        outputs. The persisted decision records that the generated revision
        was rejected and the original assessment was retained.
        """

        reason = (
            "A proposed reflection revision was rejected by "
            "deterministic guardrails, so the original "
            "assessment was preserved. "
            + " ".join(
                guardrail_errors
            )
        )

        safe_analysis = (
            ReflectionAnalysis(
                evidence_supported=(
                    audit_analysis
                    .evidence_supported
                ),
                unsupported_claims=list(
                    audit_analysis
                    .unsupported_claims
                ),
                overlooked_evidence=list(
                    audit_analysis
                    .overlooked_evidence
                ),
                missing_rubric_points=list(
                    audit_analysis
                    .missing_rubric_points
                ),
                preferred_solution_bias=(
                    audit_analysis
                    .preferred_solution_bias
                ),
                confidence_assessment=(
                    audit_analysis
                    .confidence_assessment
                ),
                should_revise=False,
                revision_reason=reason,
                requirement_basis=None,
            )
        )

        return ReflectionDecision(
            analysis=safe_analysis,
        )

    # ========================================================
    # Reflection application
    # ========================================================

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
        Apply the optional reflection patch while preserving
        Python-controlled IDs, scopes and references.
        """

        if (
            not decision.analysis
            .should_revise
        ):
            return (
                initial_criterion_assessment
                .model_copy(
                    deep=True
                ),
                initial_observation
                .model_copy(
                    deep=True
                ),
            )

        revised_status = (
            decision.revised_status
        )

        revised_internal_finding = (
            decision
            .revised_internal_finding
        )

        revised_student_feedback = (
            decision
            .revised_student_feedback
        )

        revised_verification_status = (
            decision
            .revised_verification_status
        )

        revised_confidence = (
            decision
            .revised_confidence
        )

        revised_evidence = (
            decision
            .revised_evidence
        )

        if (
            revised_status is None
            or revised_internal_finding
            is None
            or revised_student_feedback
            is None
            or revised_verification_status
            is None
            or revised_confidence
            is None
            or revised_evidence
            is None
        ):
            raise FeedbackReflectionError(
                "Reflection requested a "
                "revision but did not provide "
                "a complete revision patch."
            )

        final_criterion = (
            initial_criterion_assessment
            .model_copy(
                update={
                    "status": (
                        revised_status
                    ),
                    "internal_finding": (
                        revised_internal_finding
                    ),
                    "verification_status": (
                        revised_verification_status
                    ),
                    "confidence": (
                        revised_confidence
                    ),
                    "evidence": (
                        revised_evidence
                    ),
                },
                deep=True,
            )
        )

        observation_update: dict[
            str,
            Any,
        ] = {
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
            "confidence": (
                revised_confidence
            ),
            "evidence": (
                revised_evidence
            ),
        }

        if (
            decision.revised_suggestion
            is not None
        ):
            observation_update[
                "suggestion"
            ] = (
                decision
                .revised_suggestion
            )

        else:
            observation_update[
                "suggestion"
            ] = None

        final_observation = (
            initial_observation
            .model_copy(
                update=(
                    observation_update
                ),
                deep=True,
            )
        )

        return (
            final_criterion,
            final_observation,
        )

    # ========================================================
    # Feedback type
    # ========================================================

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

    # ========================================================
    # Requirement-basis validation
    # ========================================================

    @staticmethod
    def _validate_analysis_requirement_basis(
        *,
        analysis: Any,
        requirement: dict[str, Any],
    ) -> None:
        """
        Validate that a revision decision identifies its rubric basis.

        The basis may paraphrase the exact requirement wording. Requiring
        verbatim overlap is unnecessarily brittle for LLM-generated audit
        output because semantically faithful paraphrases are expected.
        """
        if not getattr(
            analysis,
            "should_revise",
            False,
        ):
            return

        requirement_basis = getattr(
            analysis,
            "requirement_basis",
            None,
        )

        if (
            not isinstance(
                requirement_basis,
                str,
            )
            or not requirement_basis.strip()
        ):
            raise FeedbackReflectionError(
                "A Stage-1 reflection that requests "
                "revision must identify the requirement "
                "basis for that revision."
            )

    @staticmethod
    def _validate_requirement_basis(
        *,
        decision: ReflectionDecision,
        requirement: dict[str, Any],
    ) -> None:
        """
        Validate the final combined reflection decision.
        """

        FeedbackReflectionService._validate_analysis_requirement_basis(
            analysis=(
                decision.analysis
            ),
            requirement=requirement,
        )