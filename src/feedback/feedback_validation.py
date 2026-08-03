from __future__ import annotations

from .feedback_input_models import FeedbackGenerationInput
from .feedback_models import FeedbackLLMOutput


def validate_feedback_against_input(
    *,
    feedback: FeedbackLLMOutput,
    generation_input: FeedbackGenerationInput,
) -> None:
    """
    Validate that identifiers produced by the LLM actually exist in the
    supplied assessment specification and processed submission.
    """

    specification = generation_input.assessment_specification

    if hasattr(specification, "model_dump"):
        specification_data = specification.model_dump(
            mode="json",
            exclude_none=True,
        )
    else:
        specification_data = specification

    valid_part_ids: set[str] = set()
    valid_task_ids: set[str] = set()
    valid_requirement_ids: set[str] = set()

    for part in specification_data.get("parts", []):
        part_id = part.get("part_id")
        if part_id:
            valid_part_ids.add(part_id)

        for component in part.get("components", []):
            component_id = component.get("component_id")
            if component_id:
                valid_task_ids.add(component_id)

            for requirement in component.get(
                "evaluation_requirements",
                [],
            ):
                requirement_id = requirement.get("id")
                if requirement_id:
                    valid_requirement_ids.add(requirement_id)

    valid_unit_ids = {
        unit.unit_id
        for artifact in generation_input.processed_submission.artifacts
        for unit in artifact.units
    }

    errors: list[str] = []

    #
    # Criterion assessments
    #

    for assessment in feedback.criterion_assessments:
        if (
            assessment.requirement_id
            not in valid_requirement_ids
        ):
            errors.append(
                f"Criterion assessment "
                f"'{assessment.criterion_assessment_id}' "
                f"references unknown requirement "
                f"'{assessment.requirement_id}'."
            )

        unknown_part_ids = (
            set(assessment.part_ids) - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                f"Criterion assessment "
                f"'{assessment.criterion_assessment_id}' "
                f"references unknown part IDs: "
                f"{', '.join(sorted(unknown_part_ids))}."
            )

        unknown_task_ids = (
            set(assessment.task_ids) - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                f"Criterion assessment "
                f"'{assessment.criterion_assessment_id}' "
                f"references unknown task IDs: "
                f"{', '.join(sorted(unknown_task_ids))}."
            )

    #
    # Observations
    #

    for observation in feedback.observations:
        unknown_requirement_ids = (
            set(observation.requirement_ids)
            - valid_requirement_ids
        )

        if unknown_requirement_ids:
            errors.append(
                f"Observation "
                f"'{observation.observation_id}' "
                f"references unknown requirement IDs: "
                f"{', '.join(sorted(unknown_requirement_ids))}."
            )

        unknown_part_ids = (
            set(observation.part_ids) - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                f"Observation "
                f"'{observation.observation_id}' "
                f"references unknown part IDs: "
                f"{', '.join(sorted(unknown_part_ids))}."
            )

        unknown_task_ids = (
            set(observation.task_ids) - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                f"Observation "
                f"'{observation.observation_id}' "
                f"references unknown task IDs: "
                f"{', '.join(sorted(unknown_task_ids))}."
            )

        unknown_unit_ids = (
            set(observation.attempt_unit_ids)
            - valid_unit_ids
        )

        if unknown_unit_ids:
            errors.append(
                f"Observation "
                f"'{observation.observation_id}' "
                f"references unknown attempt unit IDs: "
                f"{', '.join(sorted(unknown_unit_ids))}."
            )

    #
    # Feedback sections
    #

    for section in feedback.feedback_sections:
        unknown_part_ids = (
            set(section.part_ids) - valid_part_ids
        )

        if unknown_part_ids:
            errors.append(
                f"Feedback section "
                f"'{section.section_id}' "
                f"references unknown part IDs: "
                f"{', '.join(sorted(unknown_part_ids))}."
            )

        unknown_task_ids = (
            set(section.task_ids) - valid_task_ids
        )

        if unknown_task_ids:
            errors.append(
                f"Feedback section "
                f"'{section.section_id}' "
                f"references unknown task IDs: "
                f"{', '.join(sorted(unknown_task_ids))}."
            )

    if errors:
        raise ValueError(
            "Feedback output contains invalid references:\n- "
            + "\n- ".join(errors)
        )