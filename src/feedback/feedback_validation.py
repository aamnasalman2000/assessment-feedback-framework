from __future__ import annotations

from typing import Any

from .feedback_input_models import (
    FeedbackGenerationInput,
)
from .feedback_models import (
    FeedbackLLMOutput,
)


def _collect_specification_identifiers(
    specification_data: dict[str, Any],
) -> tuple[
    set[str],
    set[str],
    set[str],
]:
    """
    Collect valid part, component/task, and requirement identifiers
    from either supported assessment-specification layout.

    Assessment 1:
        parts -> components -> evaluation_requirements

    Assessment 2:
        assessment_structure
        top-level components -> evaluation_requirements
    """
    valid_part_ids: set[str] = set()
    valid_task_ids: set[str] = set()
    valid_requirement_ids: set[str] = set()

    for part in specification_data.get(
        "parts",
        [],
    ):
        if not isinstance(
            part,
            dict,
        ):
            continue

        part_id = part.get(
            "part_id"
        )

        if isinstance(
            part_id,
            str,
        ):
            valid_part_ids.add(
                part_id
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

            component_id = component.get(
                "component_id"
            )

            if isinstance(
                component_id,
                str,
            ):
                valid_task_ids.add(
                    component_id
                )

            for requirement in component.get(
                "evaluation_requirements",
                [],
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

                if isinstance(
                    requirement_id,
                    str,
                ):
                    valid_requirement_ids.add(
                        requirement_id
                    )

    assessment_structure = (
        specification_data.get(
            "assessment_structure",
            {},
        )
    )

    if isinstance(
        assessment_structure,
        dict,
    ):
        for part in (
            assessment_structure.values()
        ):
            if not isinstance(
                part,
                dict,
            ):
                continue

            part_id = part.get(
                "id"
            )

            if isinstance(
                part_id,
                str,
            ):
                valid_part_ids.add(
                    part_id
                )

    for component in specification_data.get(
        "components",
        [],
    ):
        if not isinstance(
            component,
            dict,
        ):
            continue

        component_id = component.get(
            "component_id"
        )

        if isinstance(
            component_id,
            str,
        ):
            valid_task_ids.add(
                component_id
            )

        for requirement in component.get(
            "evaluation_requirements",
            [],
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

            if isinstance(
                requirement_id,
                str,
            ):
                valid_requirement_ids.add(
                    requirement_id
                )

    return (
        valid_part_ids,
        valid_task_ids,
        valid_requirement_ids,
    )


def validate_feedback_against_input(
    *,
    feedback: FeedbackLLMOutput,
    generation_input: FeedbackGenerationInput,
) -> None:
    """
    Validate that identifiers produced in feedback exist in the supplied
    assessment specification and processed submission.

    Supports both Assessment 1 and Assessment 2 specification layouts.
    """
    specification = (
        generation_input
        .assessment_specification
    )

    if hasattr(
        specification,
        "model_dump",
    ):
        specification_data = (
            specification.model_dump(
                mode="json",
                exclude_none=True,
            )
        )
    else:
        specification_data = specification

    if not isinstance(
        specification_data,
        dict,
    ):
        raise ValueError(
            "Assessment specification must "
            "be a dictionary-like object."
        )

    (
        valid_part_ids,
        valid_task_ids,
        valid_requirement_ids,
    ) = _collect_specification_identifiers(
        specification_data
    )

    valid_unit_ids = {
        unit.unit_id
        for artifact
        in generation_input
        .processed_submission
        .artifacts
        for unit in artifact.units
    }

    errors: list[str] = []

    for assessment in (
        feedback.criterion_assessments
    ):
        if (
            assessment.requirement_id
            not in valid_requirement_ids
        ):
            errors.append(
                "Criterion assessment "
                f"{assessment.criterion_assessment_id!r} "
                "references unknown requirement "
                f"{assessment.requirement_id!r}."
            )

        unknown_part_ids = (
            set(
                assessment.part_ids
            )
            - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                "Criterion assessment "
                f"{assessment.criterion_assessment_id!r} "
                "references unknown part IDs: "
                + ", ".join(
                    sorted(
                        unknown_part_ids
                    )
                )
                + "."
            )

        unknown_task_ids = (
            set(
                assessment.task_ids
            )
            - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                "Criterion assessment "
                f"{assessment.criterion_assessment_id!r} "
                "references unknown task IDs: "
                + ", ".join(
                    sorted(
                        unknown_task_ids
                    )
                )
                + "."
            )

    for observation in (
        feedback.observations
    ):
        unknown_requirement_ids = (
            set(
                observation.requirement_ids
            )
            - valid_requirement_ids
        )

        if unknown_requirement_ids:
            errors.append(
                "Observation "
                f"{observation.observation_id!r} "
                "references unknown requirement IDs: "
                + ", ".join(
                    sorted(
                        unknown_requirement_ids
                    )
                )
                + "."
            )

        unknown_part_ids = (
            set(
                observation.part_ids
            )
            - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                "Observation "
                f"{observation.observation_id!r} "
                "references unknown part IDs: "
                + ", ".join(
                    sorted(
                        unknown_part_ids
                    )
                )
                + "."
            )

        unknown_task_ids = (
            set(
                observation.task_ids
            )
            - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                "Observation "
                f"{observation.observation_id!r} "
                "references unknown task IDs: "
                + ", ".join(
                    sorted(
                        unknown_task_ids
                    )
                )
                + "."
            )

        unknown_unit_ids = (
            set(
                observation.attempt_unit_ids
            )
            - valid_unit_ids
        )

        if unknown_unit_ids:
            errors.append(
                "Observation "
                f"{observation.observation_id!r} "
                "references unknown attempt unit IDs: "
                + ", ".join(
                    sorted(
                        unknown_unit_ids
                    )
                )
                + "."
            )

    for section in (
        feedback.feedback_sections
    ):
        unknown_part_ids = (
            set(
                section.part_ids
            )
            - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                "Feedback section "
                f"{section.section_id!r} "
                "references unknown part IDs: "
                + ", ".join(
                    sorted(
                        unknown_part_ids
                    )
                )
                + "."
            )

        unknown_task_ids = (
            set(
                section.task_ids
            )
            - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                "Feedback section "
                f"{section.section_id!r} "
                "references unknown task IDs: "
                + ", ".join(
                    sorted(
                        unknown_task_ids
                    )
                )
                + "."
            )

    if errors:
        raise ValueError(
            "Feedback output contains "
            "invalid references:\n- "
            + "\n- ".join(
                errors
            )
        )