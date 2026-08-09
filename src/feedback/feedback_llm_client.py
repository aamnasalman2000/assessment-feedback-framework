from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.extraction.semantic_llm_client import (
    OpenAICompatibleStructuredClient,
)

from .feedback_models import (
    ReflectionAuditOutput,
    ReflectionRevisionPatch,
    RequirementFeedbackLLMOutput,
)


class FeedbackLLMError(RuntimeError):
    """Raised when feedback generation cannot produce valid output."""


# ============================================================
# General evidence normalisation
# ============================================================


def _normalise_evidence_discriminators(
    value: Any,
) -> Any:
    """
    Add a missing evidence_type discriminator when the evidence structure
    clearly identifies the intended variant.
    """
    if isinstance(
        value,
        list,
    ):
        return [
            _normalise_evidence_discriminators(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        normalised = {
            key: (
                _normalise_evidence_discriminators(
                    item
                )
            )
            for key, item
            in value.items()
        }

        if (
            "evidence_type"
            not in normalised
        ):
            is_submission_reference = any(
                key in normalised
                for key in (
                    "artifact_id",
                    "unit_id",
                    "block_id",
                    "source_range",
                    "excerpt",
                )
            )

            is_absence = (
                "description"
                in normalised
                and any(
                    key in normalised
                    for key in (
                        "part_id",
                        "task_id",
                        "requirement_id",
                    )
                )
                and "artifact_id"
                not in normalised
            )

            if is_submission_reference:
                normalised[
                    "evidence_type"
                ] = (
                    "submission_reference"
                )

            elif is_absence:
                normalised[
                    "evidence_type"
                ] = "absence"

        return normalised

    return value


def _normalise_submission_evidence_locations(
    value: Any,
    *,
    fallback_artifact_id: str | None,
    fallback_unit_id: str | None,
) -> Any:
    """
    Repair incomplete submission-reference evidence when exactly one
    candidate artifact or unit exists.

    This does not guess between multiple candidates.
    """
    if isinstance(
        value,
        list,
    ):
        return [
            _normalise_submission_evidence_locations(
                item,
                fallback_artifact_id=(
                    fallback_artifact_id
                ),
                fallback_unit_id=(
                    fallback_unit_id
                ),
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        normalised = {
            key: (
                _normalise_submission_evidence_locations(
                    item,
                    fallback_artifact_id=(
                        fallback_artifact_id
                    ),
                    fallback_unit_id=(
                        fallback_unit_id
                    ),
                )
            )
            for key, item
            in value.items()
        }

        if (
            normalised.get(
                "evidence_type"
            )
            == "submission_reference"
        ):
            if (
                fallback_artifact_id
                is not None
                and not normalised.get(
                    "artifact_id"
                )
            ):
                normalised[
                    "artifact_id"
                ] = fallback_artifact_id

            has_location = any(
                normalised.get(
                    field_name
                )
                is not None
                for field_name in (
                    "unit_id",
                    "block_id",
                    "source_range",
                )
            )

            if (
                not has_location
                and fallback_unit_id
                is not None
            ):
                normalised[
                    "unit_id"
                ] = fallback_unit_id

        return normalised

    return value


def _normalise_absence_evidence_scope(
    value: Any,
    *,
    fallback_requirement_id: str,
    fallback_part_id: str | None = None,
    fallback_task_id: str | None = None,
) -> Any:
    """
    Restore controlled scope identifiers for absence evidence.

    Python owns the assessment identifiers used to locate an absence.
    """
    if isinstance(
        value,
        list,
    ):
        return [
            _normalise_absence_evidence_scope(
                item,
                fallback_requirement_id=(
                    fallback_requirement_id
                ),
                fallback_part_id=(
                    fallback_part_id
                ),
                fallback_task_id=(
                    fallback_task_id
                ),
            )
            for item in value
        ]

    if not isinstance(
        value,
        dict,
    ):
        return value

    normalised = {
        key: (
            _normalise_absence_evidence_scope(
                item,
                fallback_requirement_id=(
                    fallback_requirement_id
                ),
                fallback_part_id=(
                    fallback_part_id
                ),
                fallback_task_id=(
                    fallback_task_id
                ),
            )
        )
        for key, item
        in value.items()
    }

    if (
        normalised.get(
            "evidence_type"
        )
        == "absence"
    ):
        has_scope = any(
            normalised.get(
                field_name
            )
            for field_name in (
                "part_id",
                "task_id",
                "requirement_id",
            )
        )

        if not has_scope:
            normalised[
                "requirement_id"
            ] = fallback_requirement_id

        elif (
            normalised.get(
                "requirement_id"
            )
            is not None
        ):
            normalised[
                "requirement_id"
            ] = fallback_requirement_id

        elif (
            normalised.get(
                "part_id"
            )
            is None
            and normalised.get(
                "task_id"
            )
            is None
        ):
            normalised[
                "requirement_id"
            ] = fallback_requirement_id

    return normalised


def _normalise_reflection_evidence_references(
    value: Any,
    *,
    candidate_artifact_ids: list[str],
    candidate_unit_ids: list[str],
) -> Any:
    """
    Repair or remove invented submission-reference identifiers returned by
    reflection revision generation.

    When exactly one valid candidate exists, an invalid or missing
    identifier can be restored deterministically.

    When multiple candidates exist, invalid evidence is removed rather than
    mapped arbitrarily.
    """
    valid_artifact_ids = set(
        candidate_artifact_ids
    )

    valid_unit_ids = set(
        candidate_unit_ids
    )

    fallback_artifact_id = (
        candidate_artifact_ids[0]
        if (
            len(
                candidate_artifact_ids
            )
            == 1
        )
        else None
    )

    fallback_unit_id = (
        candidate_unit_ids[0]
        if (
            len(
                candidate_unit_ids
            )
            == 1
        )
        else None
    )

    def normalise(
        item: Any,
    ) -> Any:
        if isinstance(
            item,
            list,
        ):
            cleaned_items: list[
                Any
            ] = []

            for child in item:
                normalised_child = (
                    normalise(
                        child
                    )
                )

                if (
                    normalised_child
                    is not None
                ):
                    cleaned_items.append(
                        normalised_child
                    )

            return cleaned_items

        if not isinstance(
            item,
            dict,
        ):
            return item

        normalised = {
            key: normalise(
                child
            )
            for key, child
            in item.items()
        }

        if (
            normalised.get(
                "evidence_type"
            )
            != "submission_reference"
        ):
            return normalised

        artifact_id = (
            normalised.get(
                "artifact_id"
            )
        )

        unit_id = (
            normalised.get(
                "unit_id"
            )
        )

        if (
            artifact_id
            not in valid_artifact_ids
        ):
            if (
                fallback_artifact_id
                is not None
            ):
                normalised[
                    "artifact_id"
                ] = fallback_artifact_id

            else:
                return None

        if (
            unit_id
            is not None
            and unit_id
            not in valid_unit_ids
        ):
            if (
                fallback_unit_id
                is not None
            ):
                normalised[
                    "unit_id"
                ] = fallback_unit_id

            else:
                normalised[
                    "unit_id"
                ] = None

        has_location = any(
            normalised.get(
                field_name
            )
            is not None
            for field_name in (
                "unit_id",
                "block_id",
                "source_range",
            )
        )

        if not has_location:
            if (
                fallback_unit_id
                is not None
            ):
                normalised[
                    "unit_id"
                ] = fallback_unit_id

            else:
                return None

        return normalised

    return normalise(
        value
    )


# ============================================================
# Stage-1 reflection audit normalisation
# ============================================================


def _normalise_reflection_audit_shape(
    value: Any,
) -> dict[str, Any]:
    """
    Normalise a Stage-1 reflection audit.

    The audit stage is intentionally allowed to contain only `analysis`.

    Small models may still emit revised_* fields even when the structured
    schema does not request them. Those fields are discarded here because
    revision generation belongs exclusively to Stage 2.
    """
    if not isinstance(
        value,
        dict,
    ):
        return {}

    analysis = value.get(
        "analysis"
    )

    if not isinstance(
        analysis,
        dict,
    ):
        return {
            "analysis": analysis
        }

    allowed_analysis_fields = {
        "evidence_supported",
        "unsupported_claims",
        "overlooked_evidence",
        "missing_rubric_points",
        "preferred_solution_bias",
        "confidence_assessment",
        "should_revise",
        "revision_reason",
        "requirement_basis",
    }

    cleaned_analysis = {
        key: item
        for key, item
        in analysis.items()
        if key
        in allowed_analysis_fields
    }

    should_revise = (
        cleaned_analysis.get(
            "should_revise"
        )
    )

    # --------------------------------------------------
    # KEEP decisions
    # --------------------------------------------------

    if should_revise is False:
        # requirement_basis belongs only to a revision decision.
        cleaned_analysis.pop(
            "requirement_basis",
            None,
        )

        revision_reason = (
            cleaned_analysis.get(
                "revision_reason"
            )
        )

        if (
            not isinstance(
                revision_reason,
                str,
            )
            or not revision_reason.strip()
        ):
            cleaned_analysis[
                "revision_reason"
            ] = (
                "The original assessment was "
                "retained because no material "
                "revision was required."
            )

    # --------------------------------------------------
    # REVISE decisions
    # --------------------------------------------------

    elif should_revise is True:
        revision_reason = (
            cleaned_analysis.get(
                "revision_reason"
            )
        )

        if (
            not isinstance(
                revision_reason,
                str,
            )
            or not revision_reason.strip()
        ):
            cleaned_analysis[
                "revision_reason"
            ] = (
                "The original assessment "
                "contains a material defect "
                "requiring revision."
            )

    return {
        "analysis": (
            cleaned_analysis
        )
    }


# ============================================================
# Feedback client
# ============================================================


class FeedbackStructuredClient:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
    ) -> None:
        # HuggingFaceStructuredClient already implements
        # generate_structured(), so preserve it directly.
        if hasattr(
            client,
            "generate_structured",
        ):
            self._client = client

        else:
            self._client = (
                OpenAICompatibleStructuredClient(
                    client=client,
                    model=model,
                )
            )

    @property
    def model_name(
        self,
    ) -> str:
        return (
            self._client
            .model_name
        )

    # ========================================================
    # Initial requirement feedback
    # ========================================================

    def generate_requirement_feedback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        candidate_artifact_ids: list[str],
        candidate_unit_ids: list[str],
        log_name: str,
    ) -> RequirementFeedbackLLMOutput:
        response = (
            self._client
            .generate_structured(
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                output_schema=(
                    RequirementFeedbackLLMOutput
                    .model_json_schema()
                ),
            )
        )

        normalised_response = (
            _normalise_evidence_discriminators(
                response
            )
        )

        fallback_artifact_id = (
            candidate_artifact_ids[0]
            if (
                len(
                    candidate_artifact_ids
                )
                == 1
            )
            else None
        )

        fallback_unit_id = (
            candidate_unit_ids[0]
            if (
                len(
                    candidate_unit_ids
                )
                == 1
            )
            else None
        )

        normalised_response = (
            _normalise_submission_evidence_locations(
                normalised_response,
                fallback_artifact_id=(
                    fallback_artifact_id
                ),
                fallback_unit_id=(
                    fallback_unit_id
                ),
            )
        )

        self._save_debug_responses(
            raw_response=response,
            normalised_response=(
                normalised_response
            ),
            log_name=log_name,
        )

        try:
            return (
                RequirementFeedbackLLMOutput
                .model_validate(
                    normalised_response
                )
            )

        except ValueError as exc:
            raise FeedbackLLMError(
                "The model returned invalid "
                "structured requirement feedback."
            ) from exc

    # ========================================================
    # Stage 1: reflection audit
    # ========================================================

    def generate_reflection_audit(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        log_name: str,
    ) -> ReflectionAuditOutput:
        """
        Generate the Stage-1 reflection audit.

        This call decides only whether the original assessment should be
        retained or revised.

        It does not generate revised feedback or evidence.
        """
        response = (
            self._client
            .generate_structured(
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                output_schema=(
                    ReflectionAuditOutput
                    .model_json_schema()
                ),
            )
        )

        normalised_response = (
            _normalise_reflection_audit_shape(
                response
            )
        )

        self._save_debug_responses(
            raw_response=response,
            normalised_response=(
                normalised_response
            ),
            log_name=(
                f"{log_name}_audit"
            ),
        )

        try:
            return (
                ReflectionAuditOutput
                .model_validate(
                    normalised_response
                )
            )

        except ValueError as exc:
            raise FeedbackLLMError(
                "The model returned an invalid "
                "structured reflection audit."
            ) from exc

    # ========================================================
    # Stage 2: reflection revision
    # ========================================================

    def generate_reflection_revision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        candidate_artifact_ids: list[str],
        candidate_unit_ids: list[str],
        requirement_id: str,
        part_id: str | None,
        component_id: str,
        log_name: str,
    ) -> ReflectionRevisionPatch:
        """
        Generate a Stage-2 revision patch.

        This method must be called only when Stage 1 returned
        analysis.should_revise=true.

        Python retains ownership of all structural identifiers.
        """
        response = (
            self._client
            .generate_structured(
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                output_schema=(
                    ReflectionRevisionPatch
                    .model_json_schema()
                ),
            )
        )

        normalised_response = (
            _normalise_evidence_discriminators(
                response
            )
        )

        fallback_artifact_id = (
            candidate_artifact_ids[0]
            if (
                len(
                    candidate_artifact_ids
                )
                == 1
            )
            else None
        )

        fallback_unit_id = (
            candidate_unit_ids[0]
            if (
                len(
                    candidate_unit_ids
                )
                == 1
            )
            else None
        )

        normalised_response = (
            _normalise_submission_evidence_locations(
                normalised_response,
                fallback_artifact_id=(
                    fallback_artifact_id
                ),
                fallback_unit_id=(
                    fallback_unit_id
                ),
            )
        )

        normalised_response = (
            _normalise_reflection_evidence_references(
                normalised_response,
                candidate_artifact_ids=(
                    candidate_artifact_ids
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
            )
        )

        normalised_response = (
            _normalise_absence_evidence_scope(
                normalised_response,
                fallback_requirement_id=(
                    requirement_id
                ),
                fallback_part_id=(
                    part_id
                ),
                fallback_task_id=(
                    component_id
                ),
            )
        )

        self._save_debug_responses(
            raw_response=response,
            normalised_response=(
                normalised_response
            ),
            log_name=(
                f"{log_name}_revision"
            ),
        )

        try:
            return (
                ReflectionRevisionPatch
                .model_validate(
                    normalised_response
                )
            )

        except ValueError as exc:
            raise FeedbackLLMError(
                "The model returned an invalid "
                "structured reflection revision."
            ) from exc

    # ========================================================
    # Debug output
    # ========================================================

    @staticmethod
    def _save_debug_responses(
        *,
        raw_response: dict[str, Any],
        normalised_response: dict[str, Any],
        log_name: str,
    ) -> None:
        debug_dir = Path(
            "results/logs"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path = (
            debug_dir
            / (
                f"{log_name}"
                "_raw_response.json"
            )
        )

        normalised_path = (
            debug_dir
            / (
                f"{log_name}"
                "_normalised_response.json"
            )
        )

        with raw_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                raw_response,
                file,
                ensure_ascii=False,
                indent=2,
            )

        with normalised_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                normalised_response,
                file,
                ensure_ascii=False,
                indent=2,
            )