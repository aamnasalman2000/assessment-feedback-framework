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

def _normalise_scope_identifier(
    value: str,
) -> str:
    """
    Normalise specification/component identifiers and extracted scope
    labels for deterministic comparison.

    Examples:

        scenario_1 -> scenario1
        theorem_1 -> theorem1
        Part1Scenario1 -> part1scenario1
    """
    return "".join(
        character.lower()
        for character in value
        if character.isalnum()
    )


def _unit_matches_component_scope(
    *,
    unit: Any,
    component_id: str,
    part_id: str | None,
) -> bool:
    """
    Match a processed unit to a component using deterministic scope_path
    metadata when explicit task_mapping is unavailable.

    This supports extracted source scopes such as:

        Part1Scenario1
        Part2Theorem1

    without hard-coding assessment-specific component IDs.
    """

    structured_data = getattr(
        unit,
        "structured_data",
        None,
    )

    if structured_data is None:
        return False

    if hasattr(
        structured_data,
        "model_dump",
    ):
        structured_data = (
            structured_data.model_dump(
                mode="json",
                exclude_none=True,
            )
        )

    if not isinstance(
        structured_data,
        dict,
    ):
        return False

    scope_path = structured_data.get(
        "scope_path"
    )

    if not isinstance(
        scope_path,
        list,
    ):
        return False

    scope_values = [
        value
        for value in scope_path
        if isinstance(
            value,
            str,
        )
    ]

    if not scope_values:
        return False

    normalised_component = (
        _normalise_scope_identifier(
            component_id
        )
    )

    normalised_part = (
        _normalise_scope_identifier(
            part_id
        )
        if isinstance(
            part_id,
            str,
        )
        else ""
    )

    expected_values = {
        normalised_component,
    }

    if normalised_part:
        expected_values.add(
            normalised_part
            + normalised_component
        )

    for scope_value in scope_values:
        normalised_scope = (
            _normalise_scope_identifier(
                scope_value
            )
        )

        if normalised_scope in expected_values:
            return True

        # Extracted scopes commonly include the part prefix, e.g.
        # Part1Scenario1. Matching the component suffix remains
        # deterministic because scope_path itself came from source
        # structure rather than LLM inference.
        if normalised_scope.endswith(
            normalised_component
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
    Select component-level candidate units using the strongest available
    deterministic routing signal.

    Priority:

    1. exact task_mapping;
    2. extracted scope_path;
    3. part mapping for a true part-level component;
    4. artifact-type fallback only when compatible units contain no finer
       routing metadata.

    This prevents unrelated units from the same source artifact being
    assigned to a component merely because they share an artifact type.
    """

    component_id = component.get(
        "component_id"
    )

    if not isinstance(
        component_id,
        str,
    ):
        return []

    expected_artifact_types = (
        _get_component_artifact_types(
            component
        )
    )

    exact_task_candidates: list[
        dict[str, Any]
    ] = []

    scope_candidates: list[
        dict[str, Any]
    ] = []

    part_candidates: list[
        dict[str, Any]
    ] = []

    artifact_candidates: list[
        dict[str, Any]
    ] = []

    compatible_units_have_routing_metadata = False

    is_part_level_component = (
        part_id is not None
        and component_id == part_id
    )

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

        if not artifact_type_matches:
            continue

        for unit in artifact.units:
            unit_data = unit.model_dump(
                mode="json",
                exclude_none=True,
            )

            unit_data[
                "artifact_id"
            ] = artifact.artifact_id

            unit_data[
                "artifact_type"
            ] = processed_artifact_type

            task_mapping = getattr(
                unit,
                "task_mapping",
                None,
            )

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

            if task_mapping is not None:
                compatible_units_have_routing_metadata = True

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
                        mode="json",
                        exclude_none=True,
                    )
                )

            if (
                isinstance(
                    structured_data,
                    dict,
                )
                and structured_data.get(
                    "scope_path"
                )
            ):
                compatible_units_have_routing_metadata = True

            # ----------------------------------------------
            # 1. Exact task mapping
            # ----------------------------------------------

            if component_id in task_ids:
                exact_task_candidates.append(
                    unit_data
                )

            # ----------------------------------------------
            # 2. Deterministic extracted scope
            # ----------------------------------------------

            if _unit_matches_component_scope(
                unit=unit,
                component_id=component_id,
                part_id=part_id,
            ):
                scope_candidates.append(
                    unit_data
                )

            # ----------------------------------------------
            # 3. True part-level component
            # ----------------------------------------------

            if (
                is_part_level_component
                and part_id in part_ids
            ):
                part_candidates.append(
                    unit_data
                )

            artifact_candidates.append(
                unit_data
            )

    if exact_task_candidates:
        return _deduplicate_units(
            exact_task_candidates
        )

    if scope_candidates:
        return _deduplicate_units(
            scope_candidates
        )

    if part_candidates:
        return _deduplicate_units(
            part_candidates
        )

    # If the compatible source units already contain deterministic
    # routing metadata but none matches this component, returning the
    # entire artifact would actively introduce unrelated evidence.
    if compatible_units_have_routing_metadata:
        return []

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