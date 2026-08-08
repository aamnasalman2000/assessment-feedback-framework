from __future__ import annotations

from typing import Any

from .feedback_input_models import (
    FeedbackGenerationInput,
)
from .feedback_prompts import (
    build_requirement_feedback_prompt,
)


def _to_dict(value: Any) -> Any:
    """
    Convert Pydantic models to ordinary Python data
    while leaving existing dictionaries unchanged.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    return value


def _collect_components(
    specification: dict[str, Any],
) -> list[tuple[str | None, dict[str, Any]]]:
    """
    Normalise components from the supported assessment
    specification layouts.

    Assessment 1:
        parts -> components

    Assessment 2:
        components

    The returned representation is always:
        (part_id, component)
    """
    components: list[
        tuple[str | None, dict[str, Any]]
    ] = []

    # Assessment 1 layout:
    # specification["parts"][...]["components"]
    parts = specification.get(
        "parts",
        [],
    )

    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue

            part_id = part.get("part_id")

            if not isinstance(part_id, str):
                part_id = None

            part_components = part.get(
                "components",
                [],
            )

            if not isinstance(
                part_components,
                list,
            ):
                continue

            for component in part_components:
                if isinstance(component, dict):
                    components.append(
                        (
                            part_id,
                            component,
                        )
                    )

    # Assessment 2 layout:
    # specification["components"]
    top_level_components = specification.get(
        "components",
        [],
    )

    if isinstance(
        top_level_components,
        list,
    ):
        for component in top_level_components:
            if not isinstance(component, dict):
                continue

            part_id = _infer_part_id(
                specification=specification,
                component=component,
            )

            components.append(
                (
                    part_id,
                    component,
                )
            )

    return components


def _infer_part_id(
    *,
    specification: dict[str, Any],
    component: dict[str, Any],
) -> str | None:
    """
    Infer a part ID for specifications where components
    are stored at the top level.

    This is primarily needed by Assessment 2.

    Prefer explicit specification structure rather than
    relying only on component-name conventions.
    """
    component_id = component.get(
        "component_id"
    )

    if not isinstance(component_id, str):
        return None

    assessment_structure = specification.get(
        "assessment_structure",
        {},
    )

    if isinstance(
        assessment_structure,
        dict,
    ):
        for part in (
            assessment_structure.values()
        ):
            if not isinstance(part, dict):
                continue

            part_id = part.get("id")

            if not isinstance(part_id, str):
                continue

            # Assessment 2 uses IDs such as:
            #
            # part_1_ontology
            # part_2_prolog_modal_logic
            #
            # and component IDs such as:
            #
            # part_1_ontology
            # part_2_task_1
            #
            # First prefer an exact match.
            if component_id == part_id:
                return part_id

            if (
                component_id.startswith(
                    "part_1"
                )
                and part_id.startswith(
                    "part_1"
                )
            ):
                return part_id

            if (
                component_id.startswith(
                    "part_2"
                )
                and part_id.startswith(
                    "part_2"
                )
            ):
                return part_id

    return None


def _get_component_artifact_types(
    component: dict[str, Any],
) -> set[str]:
    """
    Read the artifact types declared by a component.

    Supports:

    Assessment 1:
        artifact_type

    Assessment 2:
        artifact
        artifacts
    """
    artifact_types: set[str] = set()

    direct_artifact_type = component.get(
        "artifact_type"
    )

    if isinstance(
        direct_artifact_type,
        str,
    ):
        artifact_types.add(
            direct_artifact_type
        )

    artifact = component.get(
        "artifact"
    )

    if isinstance(artifact, dict):
        artifact_type = artifact.get(
            "artifact_type"
        )

        if isinstance(
            artifact_type,
            str,
        ):
            artifact_types.add(
                artifact_type
            )

    artifacts = component.get(
        "artifacts",
        [],
    )

    if isinstance(artifacts, list):
        for artifact_spec in artifacts:
            if not isinstance(
                artifact_spec,
                dict,
            ):
                continue

            artifact_type = (
                artifact_spec.get(
                    "artifact_type"
                )
            )

            if isinstance(
                artifact_type,
                str,
            ):
                artifact_types.add(
                    artifact_type
                )

    return artifact_types


def _artifact_type_matches(
    *,
    processed_artifact_type: str,
    expected_artifact_type: str,
) -> bool:
    """
    Compare specification artifact types with processed
    artifact types.

    Most artifact types match directly.

    Assessment 1 uses semantic specification labels such
    as lean_proof_with_comments and
    lean_theorem_with_comments, while preprocessing uses
    lean_source for the actual Lean artifact.
    """
    if (
        processed_artifact_type
        == expected_artifact_type
    ):
        return True

    lean_specification_types = {
        "lean_proof_with_comments",
        "lean_theorem_with_comments",
    }

    if (
        expected_artifact_type
        in lean_specification_types
        and processed_artifact_type
        == "lean_source"
    ):
        return True

    # Assessment 2 may describe the report simply as
    # "report" while preprocessing identifies it as
    # "ontology_report".
    if (
        expected_artifact_type == "report"
        and processed_artifact_type
        == "ontology_report"
    ):
        return True

    return False


def _unit_matches_component_mapping(
    *,
    unit: Any,
    component_id: str,
    part_id: str | None,
) -> bool:
    """
    Determine whether deterministic task mapping explicitly
    links a processed unit to this component or part.
    """
    task_mapping = getattr(
        unit,
        "task_mapping",
        None,
    )

    if task_mapping is None:
        return False

    task_ids = set(
        getattr(
            task_mapping,
            "task_ids",
            [],
        )
        or []
    )

    part_ids = set(
        getattr(
            task_mapping,
            "part_ids",
            [],
        )
        or []
    )

    if component_id in task_ids:
        return True

    if (
        part_id is not None
        and part_id in part_ids
    ):
        return True

    return False


def _select_candidate_units(
    *,
    generation_input: FeedbackGenerationInput,
    component: dict[str, Any],
    part_id: str | None,
) -> list[dict[str, Any]]:
    """
    Select candidate processed units for a component.

    Deterministic task mappings are preferred.

    Artifact type is used as a fallback so that the
    generator remains compatible with both Assessment 1
    and Assessment 2.
    """
    component_id = component.get(
        "component_id"
    )

    if not isinstance(component_id, str):
        return []

    expected_artifact_types = (
        _get_component_artifact_types(
            component
        )
    )

    mapped_candidates: list[
        dict[str, Any]
    ] = []

    artifact_candidates: list[
        dict[str, Any]
    ] = []

    for artifact in (
        generation_input
        .processed_submission
        .artifacts
    ):
        processed_artifact_type = (
            artifact.artifact_type
        )

        artifact_type_matches = any(
            _artifact_type_matches(
                processed_artifact_type=(
                    processed_artifact_type
                ),
                expected_artifact_type=(
                    expected_type
                ),
            )
            for expected_type
            in expected_artifact_types
        )

        for unit in artifact.units:
            unit_data = unit.model_dump(
                mode="json",
                exclude_none=True,
            )

            # Add artifact identity so the LLM can produce
            # valid SubmissionReferenceEvidence.
            unit_data["artifact_id"] = (
                artifact.artifact_id
            )

            unit_data["artifact_type"] = (
                processed_artifact_type
            )

            if _unit_matches_component_mapping(
                unit=unit,
                component_id=component_id,
                part_id=part_id,
            ):
                mapped_candidates.append(
                    unit_data
                )

            if artifact_type_matches:
                artifact_candidates.append(
                    unit_data
                )

    # Prefer explicit deterministic mappings.
    if mapped_candidates:
        return _deduplicate_units(
            mapped_candidates
        )

    return _deduplicate_units(
        artifact_candidates
    )


def _deduplicate_units(
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    for unit in units:
        unit_id = unit.get("unit_id")

        if not isinstance(unit_id, str):
            continue

        if unit_id in seen:
            continue

        seen.add(unit_id)
        result.append(unit)

    return result


def _select_semantic_annotations(
    *,
    generation_input: FeedbackGenerationInput,
    candidate_units: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    candidate_unit_ids = {
        unit["unit_id"]
        for unit in candidate_units
        if isinstance(
            unit.get("unit_id"),
            str,
        )
    }

    annotations: list[
        dict[str, Any]
    ] = []

    for annotation in (
        generation_input
        .semantic_extraction
        .unit_annotations
    ):
        if (
            annotation.unit_id
            not in candidate_unit_ids
        ):
            continue

        annotations.append(
            annotation.model_dump(
                mode="json",
                exclude_none=True,
            )
        )

    return annotations


def _collect_processing_evidence(
    generation_input: FeedbackGenerationInput,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    processing_checks: list[
        dict[str, Any]
    ] = []

    processing_diagnostics: list[
        dict[str, Any]
    ] = []

    for artifact in (
        generation_input
        .processed_submission
        .artifacts
    ):
        for check in (
            artifact.processing.checks
        ):
            check_data = check.model_dump(
                mode="json",
                exclude_none=True,
            )

            check_data["artifact_id"] = (
                artifact.artifact_id
            )

            check_data["artifact_type"] = (
                artifact.artifact_type
            )

            processing_checks.append(
                check_data
            )

        for diagnostic in (
            artifact.processing.diagnostics
        ):
            diagnostic_data = (
                diagnostic.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )

            diagnostic_data["artifact_id"] = (
                artifact.artifact_id
            )

            diagnostic_data[
                "artifact_type"
            ] = artifact.artifact_type

            processing_diagnostics.append(
                diagnostic_data
            )

    return (
        processing_checks,
        processing_diagnostics,
    )


def _get_feedback_requirements(
    specification: dict[str, Any],
) -> list[str]:
    """
    Read feedback-generation requirements from either
    assessment specification layout.
    """
    evaluation_policy = specification.get(
        "evaluation_policy",
        {},
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

        if isinstance(requirements, list):
            return [
                value
                for value in requirements
                if isinstance(value, str)
            ]

    feedback_guidance = specification.get(
        "feedback_generation_guidance",
        {},
    )

    if isinstance(
        feedback_guidance,
        dict,
    ):
        principles = feedback_guidance.get(
            "general_principles",
            [],
        )

        if isinstance(principles, list):
            return [
                value
                for value in principles
                if isinstance(value, str)
            ]

    return []


def _get_relevant_specification_notes(
    *,
    specification: dict[str, Any],
    component_id: str,
) -> list[dict[str, Any]]:
    notes = specification.get(
        "specification_notes",
        [],
    )

    if not isinstance(notes, list):
        return []

    relevant_notes: list[
        dict[str, Any]
    ] = []

    for note in notes:
        if not isinstance(note, dict):
            continue

        note_component_id = note.get(
            "component_id"
        )

        # Include component-specific notes and any
        # genuinely global notes without a component ID.
        if (
            note_component_id is None
            or note_component_id
            == component_id
        ):
            relevant_notes.append(note)

    return relevant_notes


def _build_alignment_record(
    *,
    component: dict[str, Any],
    part_id: str | None,
    candidate_units: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Record the deterministic candidate selection supplied
    to the LLM.

    This does not claim that the selected units prove the
    requirement. The prompt explicitly requires the model
    to verify their relevance.
    """
    return {
        "component_id": component.get(
            "component_id"
        ),
        "part_id": part_id,
        "candidate_unit_ids": [
            unit["unit_id"]
            for unit in candidate_units
            if isinstance(
                unit.get("unit_id"),
                str,
            )
        ],
        "selection_method": (
            "deterministic_task_mapping_"
            "with_artifact_type_fallback"
        ),
        "authoritative": False,
    }


def build_requirement_prompts_for_assessment(
    generation_input: FeedbackGenerationInput,
) -> list[dict[str, Any]]:
    """
    Build one feedback prompt for every evaluation
    requirement in the assessment specification.

    Supports both current specification layouts:

    Assessment 1:
        parts -> components -> evaluation_requirements

    Assessment 2:
        components -> evaluation_requirements

    The returned records retain Python-owned identifiers
    so generated RequirementFeedbackLLMOutput objects can
    later be assembled into final feedback records.
    """
    specification = (
        generation_input
        .assessment_specification
    )

    specification_data = _to_dict(
        specification
    )

    if not isinstance(
        specification_data,
        dict,
    ):
        raise ValueError(
            "assessment_specification must "
            "serialize to a dictionary."
        )

    components = _collect_components(
        specification_data
    )

    if not components:
        raise ValueError(
            "No assessment components were found "
            "in the specification."
        )

    (
        processing_checks,
        processing_diagnostics,
    ) = _collect_processing_evidence(
        generation_input
    )

    global_requirements = (
        specification_data.get(
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
            specification_data
        )
    )

    prompt_records: list[
        dict[str, Any]
    ] = []

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
                    specification_data
                ),
                component_id=component_id,
            )
        )

        for requirement in requirements:
            if not isinstance(
                requirement,
                dict,
            ):
                continue

            requirement_id = (
                requirement.get("id")
            )

            if not isinstance(
                requirement_id,
                str,
            ):
                continue

            (
                system_prompt,
                user_prompt,
            ) = (
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

            prompt_records.append(
                {
                    "assessment_id": (
                        specification_data.get(
                            "assessment_id"
                        )
                    ),
                    "part_id": part_id,
                    "component_id": (
                        component_id
                    ),
                    "requirement_id": (
                        requirement_id
                    ),
                    "candidate_unit_ids": (
                        alignment[
                            "candidate_unit_ids"
                        ]
                    ),
                    "system_prompt": (
                        system_prompt
                    ),
                    "user_prompt": (
                        user_prompt
                    ),
                }
            )

    return prompt_records


def build_prompt_for_assessment(
    generation_input: FeedbackGenerationInput,
) -> tuple[str, str]:
    """
    Backwards-compatible wrapper.

    Retained so existing imports do not fail.

    A whole assessment now produces multiple
    requirement-level prompts, so callers that need the
    complete assessment should migrate to
    build_requirement_prompts_for_assessment().
    """
    prompt_records = (
        build_requirement_prompts_for_assessment(
            generation_input
        )
    )

    if not prompt_records:
        raise ValueError(
            "No requirement-level feedback "
            "prompts were generated."
        )

    if len(prompt_records) != 1:
        raise ValueError(
            "This assessment contains multiple "
            "evaluation requirements. Use "
            "build_requirement_prompts_for_assessment() "
            "instead of build_prompt_for_assessment()."
        )

    first_prompt = prompt_records[0]

    return (
        first_prompt["system_prompt"],
        first_prompt["user_prompt"],
    )