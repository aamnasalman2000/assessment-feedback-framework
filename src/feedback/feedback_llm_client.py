from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.extraction.semantic_llm_client import (
    OpenAICompatibleStructuredClient,
)

from .feedback_models import (
    ReflectionDecision,
    RequirementFeedbackLLMOutput,
)


class FeedbackLLMError(RuntimeError):
    """Raised when feedback generation cannot produce valid output."""


def _normalise_evidence_discriminators(
    value: Any,
) -> Any:
    """
    Add a missing evidence_type discriminator when the evidence structure
    clearly identifies the intended variant.
    """
    if isinstance(value, list):
        return [
            _normalise_evidence_discriminators(item)
            for item in value
        ]

    if isinstance(value, dict):
        normalised = {
            key: _normalise_evidence_discriminators(item)
            for key, item in value.items()
        }

        if "evidence_type" not in normalised:
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
                "description" in normalised
                and any(
                    key in normalised
                    for key in (
                        "part_id",
                        "task_id",
                        "requirement_id",
                    )
                )
                and "artifact_id" not in normalised
            )

            if is_submission_reference:
                normalised["evidence_type"] = (
                    "submission_reference"
                )

            elif is_absence:
                normalised["evidence_type"] = "absence"

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
    if isinstance(value, list):
        return [
            _normalise_submission_evidence_locations(
                item,
                fallback_artifact_id=fallback_artifact_id,
                fallback_unit_id=fallback_unit_id,
            )
            for item in value
        ]

    if isinstance(value, dict):
        normalised = {
            key: _normalise_submission_evidence_locations(
                item,
                fallback_artifact_id=fallback_artifact_id,
                fallback_unit_id=fallback_unit_id,
            )
            for key, item in value.items()
        }

        if (
            normalised.get("evidence_type")
            == "submission_reference"
        ):
            if (
                fallback_artifact_id is not None
                and not normalised.get("artifact_id")
            ):
                normalised["artifact_id"] = (
                    fallback_artifact_id
                )

            has_location = any(
                normalised.get(field_name) is not None
                for field_name in (
                    "unit_id",
                    "block_id",
                    "source_range",
                )
            )

            if (
                not has_location
                and fallback_unit_id is not None
            ):
                normalised["unit_id"] = fallback_unit_id

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

    The LLM may describe missing work, but Python owns the assessment
    identifiers used to locate that absence.
    """
    if isinstance(value, list):
        return [
            _normalise_absence_evidence_scope(
                item,
                fallback_requirement_id=(
                    fallback_requirement_id
                ),
                fallback_part_id=fallback_part_id,
                fallback_task_id=fallback_task_id,
            )
            for item in value
        ]

    if not isinstance(value, dict):
        return value

    normalised = {
        key: _normalise_absence_evidence_scope(
            item,
            fallback_requirement_id=(
                fallback_requirement_id
            ),
            fallback_part_id=fallback_part_id,
            fallback_task_id=fallback_task_id,
        )
        for key, item in value.items()
    }

    if normalised.get("evidence_type") == "absence":
        has_scope = any(
            normalised.get(field_name)
            for field_name in (
                "part_id",
                "task_id",
                "requirement_id",
            )
        )

        if not has_scope:
            normalised["requirement_id"] = (
                fallback_requirement_id
            )

        elif normalised.get("requirement_id") is not None:
            normalised["requirement_id"] = (
                fallback_requirement_id
            )

        elif (
            normalised.get("part_id") is None
            and normalised.get("task_id") is None
        ):
            normalised["requirement_id"] = (
                fallback_requirement_id
            )

    return normalised


def _normalise_reflection_evidence_references(
    value: Any,
    *,
    candidate_artifact_ids: list[str],
    candidate_unit_ids: list[str],
) -> Any:
    """
    Repair or remove invented submission-reference identifiers returned by
    the reflection audit.

    When there is exactly one valid candidate, an invalid or missing
    identifier can be restored deterministically. When multiple candidates
    exist, invalid evidence is removed rather than mapped arbitrarily.
    """
    valid_artifact_ids = set(candidate_artifact_ids)
    valid_unit_ids = set(candidate_unit_ids)

    fallback_artifact_id = (
        candidate_artifact_ids[0]
        if len(candidate_artifact_ids) == 1
        else None
    )

    fallback_unit_id = (
        candidate_unit_ids[0]
        if len(candidate_unit_ids) == 1
        else None
    )

    def normalise(item: Any) -> Any:
        if isinstance(item, list):
            cleaned_items: list[Any] = []

            for child in item:
                normalised_child = normalise(child)

                if normalised_child is not None:
                    cleaned_items.append(
                        normalised_child
                    )

            return cleaned_items

        if not isinstance(item, dict):
            return item

        normalised = {
            key: normalise(child)
            for key, child in item.items()
        }

        if (
            normalised.get("evidence_type")
            != "submission_reference"
        ):
            return normalised

        artifact_id = normalised.get("artifact_id")
        unit_id = normalised.get("unit_id")

        if artifact_id not in valid_artifact_ids:
            if fallback_artifact_id is not None:
                normalised["artifact_id"] = (
                    fallback_artifact_id
                )
            else:
                return None

        if (
            unit_id is not None
            and unit_id not in valid_unit_ids
        ):
            if fallback_unit_id is not None:
                normalised["unit_id"] = fallback_unit_id
            else:
                normalised["unit_id"] = None

        has_location = any(
            normalised.get(field_name) is not None
            for field_name in (
                "unit_id",
                "block_id",
                "source_range",
            )
        )

        if not has_location:
            if fallback_unit_id is not None:
                normalised["unit_id"] = fallback_unit_id
            else:
                return None

        return normalised

    return normalise(value)


def _normalise_reflection_revision_shape(
    value: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove revised fields when no revision is requested.

    Small models may populate revised_* fields despite returning
    should_revise=false. The audit decision is treated as authoritative,
    and unused patch fields are removed before validation.
    """
    normalised = dict(value)

    analysis = normalised.get("analysis")

    if not isinstance(analysis, dict):
        return normalised

    should_revise = analysis.get("should_revise")

    revised_field_names = (
        "revised_status",
        "revised_internal_finding",
        "revised_student_feedback",
        "revised_suggestion",
        "revised_verification_status",
        "revised_confidence",
        "revised_evidence",
    )

    if should_revise is False:
        for field_name in revised_field_names:
            normalised.pop(field_name, None)

    return normalised

def _normalise_reflection_analysis_shape(
    value: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove revision-only analysis fields when the model decides to keep
    the original assessment.
    """
    normalised = dict(value)

    analysis = normalised.get("analysis")

    if not isinstance(analysis, dict):
        return normalised

    should_revise = analysis.get("should_revise")

    if should_revise is False:
        cleaned_analysis = dict(analysis)
        cleaned_analysis.pop(
            "requirement_basis",
            None,
        )
        normalised["analysis"] = cleaned_analysis

    return normalised

def _normalise_incomplete_reflection_revision(
    value: dict[str, Any],
) -> dict[str, Any]:
    """
    Fail safely when the model requests revision but does not provide a
    complete revision patch.

    An incomplete patch must never replace the original assessment.
    The decision is therefore downgraded to KEEP.
    """
    normalised = dict(value)

    analysis = normalised.get("analysis")

    if not isinstance(analysis, dict):
        return normalised

    if analysis.get("should_revise") is not True:
        return normalised

    required_revision_fields = (
        "revised_status",
        "revised_internal_finding",
        "revised_student_feedback",
        "revised_verification_status",
        "revised_confidence",
        "revised_evidence",
    )

    has_complete_patch = all(
        normalised.get(field_name) is not None
        for field_name in required_revision_fields
    )

    if has_complete_patch:
        return normalised

    revised_field_names = (
        *required_revision_fields,
        "revised_suggestion",
    )

    for field_name in revised_field_names:
        normalised.pop(field_name, None)

    revised_analysis = dict(analysis)
    revised_analysis["should_revise"] = False
    revised_analysis["evidence_supported"] = True
    revised_analysis["confidence_assessment"] = (
        "appropriate"
    )
    revised_analysis["preferred_solution_bias"] = False
    revised_analysis["unsupported_claims"] = []
    revised_analysis["overlooked_evidence"] = []
    revised_analysis["missing_rubric_points"] = []
    revised_analysis.pop("requirement_basis", None)
    revised_analysis["revision_reason"] = (
        "The model did not provide a complete valid revision "
        "patch, so the original assessment was preserved."
    )

    normalised["analysis"] = revised_analysis

    return normalised


class FeedbackStructuredClient:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
    ) -> None:
        if hasattr(client, "generate_structured"):
            self._client = client
        else:
            self._client = OpenAICompatibleStructuredClient(
                client=client,
                model=model,
            )

    @property
    def model_name(self) -> str:
        return self._client.model_name

    def generate_requirement_feedback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        candidate_artifact_ids: list[str],
        candidate_unit_ids: list[str],
        log_name: str,
    ) -> RequirementFeedbackLLMOutput:
        response = self._client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=(
                RequirementFeedbackLLMOutput
                .model_json_schema()
            ),
        )

        normalised_response = (
            _normalise_evidence_discriminators(
                response
            )
        )

        fallback_artifact_id = (
            candidate_artifact_ids[0]
            if len(candidate_artifact_ids) == 1
            else None
        )

        fallback_unit_id = (
            candidate_unit_ids[0]
            if len(candidate_unit_ids) == 1
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
            normalised_response=normalised_response,
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
                "The model returned invalid structured "
                "requirement feedback."
            ) from exc

    def generate_reflection_decision(
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
    ) -> ReflectionDecision:
        """
        Generate a rubric-guided audit and optional feedback patch.

        The model does not generate record IDs or structural metadata.
        """
        response = self._client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=(
                ReflectionDecision
                .model_json_schema()
            ),
        )

        normalised_response = (
            _normalise_evidence_discriminators(
                response
            )
        )

        fallback_artifact_id = (
            candidate_artifact_ids[0]
            if len(candidate_artifact_ids) == 1
            else None
        )

        fallback_unit_id = (
            candidate_unit_ids[0]
            if len(candidate_unit_ids) == 1
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
                fallback_part_id=part_id,
                fallback_task_id=component_id,
            )
        )

        normalised_response = (
            _normalise_incomplete_reflection_revision(
                normalised_response
            )
        )
        
        normalised_response = (
            _normalise_reflection_analysis_shape(
                normalised_response
            )
        )

        normalised_response = (
            _normalise_reflection_revision_shape(
                normalised_response
            )
        )

        self._save_debug_responses(
            raw_response=response,
            normalised_response=normalised_response,
            log_name=log_name,
        )

        try:
            return ReflectionDecision.model_validate(
                normalised_response
            )

        except ValueError as exc:
            raise FeedbackLLMError(
                "The model returned an invalid structured "
                "reflection decision."
            ) from exc

    @staticmethod
    def _save_debug_responses(
        *,
        raw_response: dict[str, Any],
        normalised_response: dict[str, Any],
        log_name: str,
    ) -> None:
        debug_dir = Path("results/logs")
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path = (
            debug_dir
            / f"{log_name}_raw_response.json"
        )

        normalised_path = (
            debug_dir
            / f"{log_name}_normalised_response.json"
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