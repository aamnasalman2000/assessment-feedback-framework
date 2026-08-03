from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.alignment.alignment_models import (
    AlignmentResult,
    ComponentAlignment,
)
from src.extraction.semantic_models import (
    SemanticExtractionResult,
)
from src.feedback.feedback_input_models import (
    ProcessedSubmission,
)
from src.feedback.feedback_llm_client import (
    FeedbackStructuredClient,
)
from src.feedback.feedback_models import (
    CriterionAssessment,
    FeedbackLLMOutput,
    FeedbackObservation,
    FeedbackSection,
    OverallFeedback,
    ReflectionDecision,
)
from src.feedback.feedback_reflection_service import (
    FeedbackReflectionService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_ID = "assessment_1"
STUDENT_ID = "student_3"
MODEL = "qwen3:8b"

SEMANTIC_VERSION = "v1"
ALIGNMENT_VERSION = "v1"

TARGET_REQUIREMENTS = {
    ("theorem_1", "T1-R2"),
    ("theorem_4", "T4-R6"),
}

# Existing criterion-decomposed feedback version.
CRITERION_FEEDBACK_VERSION = "v2"

# Version for the new audit-based reflective output.
REFLECTIVE_FEEDBACK_VERSION = "v2"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"but found {type(data).__name__}."
        )

    return data


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(value, "model_dump"):
        serializable_value = value.model_dump(
            mode="json",
            exclude_none=True,
        )
    else:
        serializable_value = value

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def create_sdk_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
        timeout=1200.0,
        max_retries=0,
    )


def build_component_lookup(
    assessment_specification: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for part in assessment_specification.get(
        "parts",
        [],
    ):
        part_id = part.get("part_id")

        for component in part.get(
            "components",
            [],
        ):
            component_id = component.get(
                "component_id"
            )

            if not isinstance(component_id, str):
                continue

            lookup[component_id] = {
                "part_id": part_id,
                "component": component,
            }

    return lookup


def build_requirement_lookup(
    component_lookup: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for component_id, record in (
        component_lookup.items()
    ):
        component = record["component"]

        for requirement in component.get(
            "evaluation_requirements",
            [],
        ):
            requirement_id = requirement.get("id")

            if isinstance(requirement_id, str):
                lookup[
                    (component_id, requirement_id)
                ] = requirement

    return lookup


def build_alignment_lookup(
    alignment_result: AlignmentResult,
) -> dict[str, ComponentAlignment]:
    return {
        alignment.component_id: alignment
        for alignment
        in alignment_result.component_alignments
    }


def collect_candidate_unit_ids(
    alignment: ComponentAlignment,
) -> list[str]:
    unit_ids = (
        alignment.primary_unit_ids
        + alignment.supporting_unit_ids
        + alignment.possibly_relevant_unit_ids
    )

    return list(dict.fromkeys(unit_ids))


def build_unit_lookups(
    processed_submission: ProcessedSubmission,
) -> tuple[
    dict[str, Any],
    dict[str, str],
]:
    unit_lookup: dict[str, Any] = {}
    artifact_lookup: dict[str, str] = {}

    for artifact in processed_submission.artifacts:
        for unit in artifact.units:
            unit_lookup[unit.unit_id] = unit
            artifact_lookup[unit.unit_id] = (
                artifact.artifact_id
            )

    return unit_lookup, artifact_lookup


def collect_candidate_inputs(
    *,
    candidate_unit_ids: list[str],
    unit_lookup: dict[str, Any],
    artifact_lookup: dict[str, str],
    semantic_lookup: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[Any],
    set[str],
]:
    candidate_units: list[dict[str, Any]] = []
    semantic_annotations: list[Any] = []
    candidate_artifact_ids: set[str] = set()

    for unit_id in candidate_unit_ids:
        unit = unit_lookup.get(unit_id)

        if unit is None:
            raise ValueError(
                "Alignment references unknown unit: "
                f"{unit_id}"
            )

        artifact_id = artifact_lookup[unit_id]
        candidate_artifact_ids.add(artifact_id)

        unit_data = unit.model_dump(
            mode="json",
            exclude_none=True,
        )

        unit_data["artifact_id"] = artifact_id
        candidate_units.append(unit_data)

        annotation = semantic_lookup.get(unit_id)

        if annotation is not None:
            semantic_annotations.append(annotation)

    return (
        candidate_units,
        semantic_annotations,
        candidate_artifact_ids,
    )


def collect_processing_evidence(
    *,
    processed_submission: ProcessedSubmission,
    artifact_ids: set[str],
) -> tuple[list[Any], list[Any]]:
    checks: list[Any] = []
    diagnostics: list[Any] = []

    for artifact in processed_submission.artifacts:
        if (
            artifact_ids
            and artifact.artifact_id
            not in artifact_ids
        ):
            continue

        checks.extend(
            artifact.processing.checks
        )

        diagnostics.extend(
            artifact.processing.diagnostics
        )

    return checks, diagnostics


def build_observation_lookup(
    feedback: FeedbackLLMOutput,
) -> dict[str, FeedbackObservation]:
    """
    Map each criterion-assessment ID to the observation that references it.
    """
    lookup: dict[str, FeedbackObservation] = {}

    for observation in feedback.observations:
        for criterion_id in (
            observation.criterion_assessment_ids
        ):
            if criterion_id in lookup:
                raise ValueError(
                    "Multiple observations reference "
                    "criterion assessment "
                    f"{criterion_id!r}."
                )

            lookup[criterion_id] = observation

    return lookup


def combine_text(
    values: list[str],
    *,
    fallback: str,
) -> str:
    cleaned = [
        value.strip()
        for value in values
        if isinstance(value, str)
        and value.strip()
    ]

    if not cleaned:
        return fallback

    return " ".join(cleaned)


def build_reflection_summary(
    decisions: list[ReflectionDecision],
) -> dict[str, Any]:
    revised_decisions = [
        decision
        for decision in decisions
        if decision.analysis.should_revise
    ]

    unchanged_decisions = [
        decision
        for decision in decisions
        if not decision.analysis.should_revise
    ]

    unsupported_claim_count = sum(
        len(
            decision.analysis.unsupported_claims
        )
        for decision in decisions
    )

    overlooked_evidence_count = sum(
        len(
            decision.analysis.overlooked_evidence
        )
        for decision in decisions
    )

    missing_rubric_point_count = sum(
        len(
            decision.analysis.missing_rubric_points
        )
        for decision in decisions
    )

    preferred_solution_bias_count = sum(
        1
        for decision in decisions
        if decision.analysis.preferred_solution_bias
    )

    confidence_too_high_count = sum(
        1
        for decision in decisions
        if (
            decision.analysis
            .confidence_assessment
            == "too_high"
        )
    )

    confidence_too_low_count = sum(
        1
        for decision in decisions
        if (
            decision.analysis
            .confidence_assessment
            == "too_low"
        )
    )

    return {
        "assessment_id": ASSESSMENT_ID,
        "student_id": STUDENT_ID,
        "model": MODEL,
        "total_requirements_reviewed": len(
            decisions
        ),
        "requirements_revised": len(
            revised_decisions
        ),
        "requirements_unchanged": len(
            unchanged_decisions
        ),
        "unsupported_claims_identified": (
            unsupported_claim_count
        ),
        "overlooked_evidence_items": (
            overlooked_evidence_count
        ),
        "missing_rubric_points_identified": (
            missing_rubric_point_count
        ),
        "preferred_solution_bias_cases": (
            preferred_solution_bias_count
        ),
        "confidence_judged_too_high": (
            confidence_too_high_count
        ),
        "confidence_judged_too_low": (
            confidence_too_low_count
        ),
    }


def main() -> None:
    model_directory = MODEL.split(":")[0]

    processed_path = (
        PROJECT_ROOT
        / "data"
        / ASSESSMENT_ID
        / "processed"
        / f"{STUDENT_ID}.json"
    )

    semantic_path = (
        PROJECT_ROOT
        / "data"
        / ASSESSMENT_ID
        / "semantic"
        / SEMANTIC_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    alignment_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "alignment"
        / ALIGNMENT_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    specification_path = (
        PROJECT_ROOT
        / "specs"
        / f"{ASSESSMENT_ID}_spec.json"
    )

    initial_feedback_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "llm_feedback"
        / CRITERION_FEEDBACK_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "llm_feedback_reflective"
        / REFLECTIVE_FEEDBACK_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    audit_output_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "reflection_audits"
        / REFLECTIVE_FEEDBACK_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    summary_output_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "reflection_audits"
        / REFLECTIVE_FEEDBACK_VERSION
        / model_directory
        / f"{STUDENT_ID}_summary.json"
    )

    print("Loading processed submission...")

    processed_submission = (
        ProcessedSubmission.model_validate(
            load_json(processed_path)
        )
    )

    print("Loading semantic extraction...")

    semantic_extraction = (
        SemanticExtractionResult.model_validate(
            load_json(semantic_path)
        )
    )

    print("Loading alignment result...")

    alignment_result = (
        AlignmentResult.model_validate(
            load_json(alignment_path)
        )
    )

    print("Loading assessment specification...")

    assessment_specification = load_json(
        specification_path
    )

    print("Loading existing criterion feedback...")

    initial_feedback = (
        FeedbackLLMOutput.model_validate(
            load_json(initial_feedback_path)
        )
    )

    component_lookup = build_component_lookup(
        assessment_specification
    )

    requirement_lookup = build_requirement_lookup(
        component_lookup
    )

    alignment_lookup = build_alignment_lookup(
        alignment_result
    )

    unit_lookup, artifact_lookup = (
        build_unit_lookups(
            processed_submission
        )
    )

    semantic_lookup = {
        annotation.unit_id: annotation
        for annotation
        in semantic_extraction.unit_annotations
    }

    observation_lookup = build_observation_lookup(
        initial_feedback
    )

    global_requirements = (
        assessment_specification.get(
            "global_requirements",
            [],
        )
    )

    evaluation_policy = (
        assessment_specification.get(
            "evaluation_policy",
            {},
        )
    )

    feedback_requirements = (
        evaluation_policy.get(
            "feedback_requirements",
            [],
        )
    )

    all_specification_notes = (
        assessment_specification.get(
            "specification_notes",
            [],
        )
    )

    sdk_client = create_sdk_client()

    feedback_client = FeedbackStructuredClient(
        client=sdk_client,
        model=MODEL,
    )

    reflection_service = FeedbackReflectionService(
        llm_client=feedback_client,
    )

    reflection_decisions: list[
        ReflectionDecision
    ] = []

    reflected_criteria: list[
        CriterionAssessment
    ] = []

    reflected_observations: list[
        FeedbackObservation
    ] = []

    print(
        "Auditing existing criterion feedback "
        f"for {ASSESSMENT_ID}/{STUDENT_ID}..."
    )

    total = len(
        initial_feedback.criterion_assessments
    )

    for index, criterion in enumerate(
        initial_feedback.criterion_assessments,
        start=1,
    ):
        if not criterion.task_ids:
            raise ValueError(
                "Criterion assessment "
                f"{criterion.criterion_assessment_id!r} "
                "does not contain a task/component ID."
            )

        component_id = criterion.task_ids[0]
        requirement_id = criterion.requirement_id
        
        # Temporary: only run one requirement.
        completed_target_count = 0
        if (
            component_id,
            requirement_id,
        ) not in TARGET_REQUIREMENTS:
            continue

        component_record = component_lookup.get(
            component_id
        )

        if component_record is None:
            raise ValueError(
                f"Unknown component ID: "
                f"{component_id}"
            )

        requirement = requirement_lookup.get(
            (component_id, requirement_id)
        )

        if requirement is None:
            raise ValueError(
                "Could not find requirement "
                f"{requirement_id!r} in component "
                f"{component_id!r}."
            )

        alignment = alignment_lookup.get(
            component_id
        )

        if alignment is None:
            raise ValueError(
                "Could not find alignment for "
                f"component {component_id!r}."
            )

        original_observation = (
            observation_lookup.get(
                criterion.criterion_assessment_id
            )
        )

        if original_observation is None:
            raise ValueError(
                "Could not find an observation for "
                f"{criterion.criterion_assessment_id!r}."
            )

        candidate_unit_ids = (
            collect_candidate_unit_ids(
                alignment
            )
        )

        (
            candidate_units,
            semantic_annotations,
            candidate_artifact_ids,
        ) = collect_candidate_inputs(
            candidate_unit_ids=(
                candidate_unit_ids
            ),
            unit_lookup=unit_lookup,
            artifact_lookup=artifact_lookup,
            semantic_lookup=semantic_lookup,
        )

        (
            processing_checks,
            processing_diagnostics,
        ) = collect_processing_evidence(
            processed_submission=(
                processed_submission
            ),
            artifact_ids=(
                candidate_artifact_ids
            ),
        )

        part_id = component_record["part_id"]
        component = component_record["component"]

        specification_notes = [
            note
            for note in all_specification_notes
            if (
                not isinstance(note, dict)
                or note.get("component_id") in {
                    None,
                    component_id,
                }
            )
        ]

        print(
            f"[{index}/{total}] Auditing "
            f"{component_id}/{requirement_id}..."
        )

        (
            decision,
            final_criterion,
            final_observation,
        ) = (
            reflection_service
            .reflect_requirement_feedback(
                component=component,
                requirement=requirement,
                part_id=part_id,
                candidate_units=candidate_units,
                semantic_annotations=(
                    semantic_annotations
                ),
                alignment=alignment.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                processing_checks=(
                    processing_checks
                ),
                processing_diagnostics=(
                    processing_diagnostics
                ),
                initial_criterion_assessment=(
                    criterion
                ),
                initial_observation=(
                    original_observation
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
                log_name=(
                    f"{component_id}_"
                    f"{requirement_id}"
                ),
            )
        )

        reflection_decisions.append(decision)
        reflected_criteria.append(
            final_criterion
        )
        reflected_observations.append(
            final_observation
        )

        outcome = (
            "revised"
            if decision.analysis.should_revise
            else "unchanged"
        )

        print(
            f"✓ Completed audit for "
            f"{component_id}/{requirement_id}: "
            f"{outcome}"
        )
        completed_target_count += 1

        if (
            TARGET_REQUIREMENTS
            and completed_target_count
            == len(TARGET_REQUIREMENTS)
        ):
            print()
            print("Targeted audit sample completed successfully.")
            print(
                "Skipping full feedback reconstruction "
                "for this test."
            )
            return
        
        if TARGET_REQUIREMENTS:
            completed_target = (
            component_id,
            requirement_id,
        )

        if completed_target == next(
            reversed(TARGET_REQUIREMENTS)
        ):
            print()
            print("Targeted audit sample completed.")
            print(
                "Skipping full feedback reconstruction "
                "for this test."
            )
            return
        
    reflected_observation_lookup = {
        observation.observation_id: observation
        for observation
        in reflected_observations
    }

    reflected_sections: list[
        FeedbackSection
    ] = []

    for section in initial_feedback.feedback_sections:
        section_feedback = [
            reflected_observation_lookup[
                observation_id
            ].student_feedback
            for observation_id
            in section.observation_ids
            if observation_id
            in reflected_observation_lookup
        ]

        reflected_sections.append(
            section.model_copy(
                update={
                    "summary": combine_text(
                        section_feedback,
                        fallback=section.summary,
                    )
                },
                deep=True,
            )
        )

    strengths = [
        observation.student_feedback
        for observation in reflected_observations
        if observation.feedback_type == "strength"
    ]

    improvements = [
        observation.student_feedback
        for observation in reflected_observations
        if observation.feedback_type
        in {
            "error",
            "omission",
            "suggestion",
        }
    ]

    overall_summary = combine_text(
        [
            section.summary
            for section in reflected_sections
        ],
        fallback=(
            initial_feedback
            .overall_feedback.summary
        ),
    )

    reflected_output = FeedbackLLMOutput(
        criterion_assessments=(
            reflected_criteria
        ),
        observations=reflected_observations,
        feedback_sections=reflected_sections,
        overall_feedback=OverallFeedback(
            summary=overall_summary,
            strengths_summary=combine_text(
                strengths,
                fallback=(
                    "No clearly supported strengths "
                    "were identified."
                ),
            ),
            improvement_summary=combine_text(
                improvements,
                fallback=(
                    "No clearly supported areas for "
                    "improvement were identified."
                ),
            ),
            observation_ids=[
                observation.observation_id
                for observation
                in reflected_observations
            ],
        ),
    )

    audit_records = []

    for (
        criterion,
        decision,
    ) in zip(
        initial_feedback.criterion_assessments,
        reflection_decisions,
        strict=True,
    ):
        audit_records.append(
            {
                "criterion_assessment_id": (
                    criterion
                    .criterion_assessment_id
                ),
                "requirement_id": (
                    criterion.requirement_id
                ),
                "task_ids": criterion.task_ids,
                "decision": decision.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
        )

    audit_output = {
        "assessment_id": ASSESSMENT_ID,
        "student_id": STUDENT_ID,
        "model": MODEL,
        "reflection_version": (
            REFLECTIVE_FEEDBACK_VERSION
        ),
        "audits": audit_records,
    }

    reflection_summary = (
        build_reflection_summary(
            reflection_decisions
        )
    )

    save_json(
        output_path,
        reflected_output,
    )

    save_json(
        audit_output_path,
        audit_output,
    )

    save_json(
        summary_output_path,
        reflection_summary,
    )

    print()
    print(
        "Rubric-guided feedback audit "
        "completed successfully."
    )
    print(
        "Initial criterion feedback was not regenerated."
    )
    print(
        f"Saved reflected feedback to: "
        f"{output_path}"
    )
    print(
        f"Saved reflection audits to: "
        f"{audit_output_path}"
    )
    print(
        f"Saved reflection summary to: "
        f"{summary_output_path}"
    )


if __name__ == "__main__":
    main()