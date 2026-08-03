from __future__ import annotations

from typing import Any

from .feedback_input_models import FeedbackGenerationInput
from .feedback_prompts import build_assessment_feedback_prompt


def build_prompt_for_assessment(
    generation_input: FeedbackGenerationInput,
) -> tuple[str, str]:
    specification = generation_input.assessment_specification

    if hasattr(specification, "model_dump"):
        specification_data = specification.model_dump(
            mode="json",
            exclude_none=True,
        )
    else:
        specification_data = specification

    processed_units = [
        unit
        for artifact in generation_input.processed_submission.artifacts
        for unit in artifact.units
    ]

    semantic_annotations = list(
        generation_input.semantic_extraction.unit_annotations
    )

    processing_checks: list[Any] = []
    processing_diagnostics: list[Any] = []

    for artifact in generation_input.processed_submission.artifacts:
        processing_checks.extend(artifact.processing.checks)
        processing_diagnostics.extend(artifact.processing.diagnostics)

    evaluation_policy = specification_data.get(
        "evaluation_policy",
        {},
    )

    return build_assessment_feedback_prompt(
        assessment_specification=specification_data,
        processed_units=processed_units,
        semantic_annotations=semantic_annotations,
        processing_checks=processing_checks,
        processing_diagnostics=processing_diagnostics,
        global_requirements=specification_data.get(
            "global_requirements",
            [],
        ),
        feedback_requirements=evaluation_policy.get(
            "feedback_requirements",
            [],
        ),
        specification_notes=specification_data.get(
            "specification_notes",
            [],
        ),
    )