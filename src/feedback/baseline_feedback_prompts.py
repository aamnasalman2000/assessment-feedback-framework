from __future__ import annotations

from typing import Any

from .feedback_prompts import (
    _compact_candidate_units,
    _compact_component,
    _compact_processing_checks,
    _compact_requirement,
    _filter_feedback_requirements,
    _filter_global_requirements,
    _json_block,
)


def _compact_baseline_component(
    component: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep only component metadata needed for baseline evaluation.

    Pipeline A intentionally uses less prompt scaffolding than the
    criterion-decomposed pipelines.
    """

    compact = _compact_component(
        component
    )

    if not isinstance(
        compact,
        dict,
    ):
        return {}

    # The exact evaluation requirements are supplied separately,
    # so verbose evaluation guidance is unnecessary here.
    compact.pop(
        "evaluation_guidance",
        None,
    )

    task = compact.get(
        "task"
    )

    if isinstance(
        task,
        dict,
    ):
        useful_task_fields = (
            "summary",
            "description",
            "target_statement",
            "semantic_goal",
            "required_examples",
            "required_cases",
            "stated_conditions",
        )

        compact_task = {
            key: value
            for key, value
            in task.items()
            if (
                key in useful_task_fields
                and value not in (
                    None,
                    "",
                    [],
                    {},
                )
            )
        }

        if compact_task:
            compact[
                "task"
            ] = compact_task

        else:
            compact.pop(
                "task",
                None,
            )

    return compact


def build_component_baseline_feedback_prompt(
    *,
    component: dict[str, Any],
    part_id: str | None,
    candidate_units: list[Any],
    semantic_annotations: list[Any],
    alignment: dict[str, Any],
    processing_checks: list[Any],
    processing_diagnostics: list[Any],
    global_requirements: list[dict[str, Any]],
    feedback_requirements: list[str],
    specification_notes: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Build Pipeline A's non-decomposed component-level prompt.

    One prompt evaluates all requirements for one component together.

    This implementation is generic across Assessment 1 and Assessment 2.
    It contains no assessment-specific component IDs or requirement IDs.

    Pipeline A intentionally does not inject:

    - criterion-specific evidence routing;
    - semantic annotations;
    - processing diagnostics;
    - reflection context.

    Direct processed student evidence and deterministic processing checks
    remain available.
    """

    # Retained in the function signature so Pipeline A can share the same
    # service-level interface as the other feedback pipelines.
    #
    # They are deliberately not inserted into the baseline prompt.
    _ = semantic_annotations
    _ = alignment
    _ = processing_diagnostics

    component_id = component.get(
        "component_id"
    )

    if not isinstance(
        component_id,
        str,
    ):
        raise ValueError(
            "Component must contain a string component_id."
        )

    requirements = component.get(
        "evaluation_requirements",
        [],
    )

    if not isinstance(
        requirements,
        list,
    ):
        requirements = []

    compact_requirements = [
        _compact_requirement(
            requirement
        )
        for requirement in requirements
        if isinstance(
            requirement,
            dict,
        )
    ]

    compact_requirements = [
        requirement
        for requirement
        in compact_requirements
        if isinstance(
            requirement.get(
                "id"
            ),
            str,
        )
    ]

    if not compact_requirements:
        raise ValueError(
            "Component contains no valid evaluation requirements."
        )

    compact_component = (
        _compact_baseline_component(
            component
        )
    )

    # No criterion-specific compaction in Pipeline A.
    compact_units = (
        _compact_candidate_units(
            candidate_units,
            criterion=None,
        )
    )

    compact_checks = (
        _compact_processing_checks(
            processing_checks
        )
    )

    relevant_global_requirements = (
        _filter_global_requirements(
            global_requirements=(
                global_requirements
            ),
            component_id=(
                component_id
            ),
            part_id=(
                part_id
            ),
        )
    )

    relevant_feedback_requirements = (
        _filter_feedback_requirements(
            component_id=(
                component_id
            ),
            feedback_requirements=(
                feedback_requirements
            ),
        )
    )

    valid_requirement_ids = [
        requirement[
            "id"
        ]
        for requirement
        in compact_requirements
    ]

    valid_unit_ids = [
        unit.get(
            "unit_id"
        )
        for unit in compact_units
        if isinstance(
            unit.get(
                "unit_id"
            ),
            str,
        )
    ]

    valid_artifact_ids = list(
        dict.fromkeys(
            unit.get(
                "artifact_id"
            )
            for unit in compact_units
            if isinstance(
                unit.get(
                    "artifact_id"
                ),
                str,
            )
        )
    )

    compact_specification_notes = [
        note
        for note in specification_notes
        if (
            isinstance(
                note,
                dict,
            )
            and any(
                value not in (
                    None,
                    "",
                    [],
                    {},
                )
                for value in note.values()
            )
        )
    ]

    system_prompt = """
You are an academic feedback assistant evaluating one complete assessment
component.

Evaluate all supplied requirements together in one pass.

Use only the supplied student evidence, assessment requirements, relevant
guidance, and deterministic processing checks.

Accept valid alternative approaches. Do not impose a preferred proof,
formula, ontology, naming, code, or presentation form unless explicitly
required.

Do not invent student work, source references, missing work, tool results,
compilation results, syntax results, execution results, or logical
verification.

Use:

- met when evidence clearly demonstrates the requirement;
- partially_met when only part is demonstrated;
- not_met when relevant work is present but fails the requirement;
- missing when required work is demonstrably absent;
- not_assessable when evidence is insufficient or reliable verification is
  unavailable.

Never claim formal or tool verification unless a supplied deterministic check
establishes it.

Return exactly one JSON object matching the supplied structured schema.
Do not assign marks.
Do not generate record IDs.
""".strip()

    user_prompt = f"""
Evaluate this component in one baseline feedback pass.

COMPONENT

{_json_block(compact_component)}

PART ID

{_json_block(part_id)}

EVALUATION REQUIREMENTS

{_json_block(compact_requirements)}

VALID REQUIREMENT IDS

{_json_block(valid_requirement_ids)}

CANDIDATE STUDENT EVIDENCE

{_json_block(compact_units)}

VALID UNIT IDS

{_json_block(valid_unit_ids)}

VALID ARTIFACT IDS

{_json_block(valid_artifact_ids)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(compact_checks)}

RELEVANT GLOBAL REQUIREMENTS

{_json_block(relevant_global_requirements)}

FEEDBACK GUIDANCE

{_json_block(relevant_feedback_requirements)}

RELEVANT SPECIFICATION NOTES

{_json_block(compact_specification_notes)}

OUTPUT

criterion_assessments:

- include every VALID REQUIREMENT ID exactly once;
- use only VALID REQUIREMENT IDS;
- internal_finding must be one concise sentence, maximum 45 words;
- choose an appropriate verification_status and confidence;
- include at most one evidence item per criterion;
- use an empty evidence array when no valid direct reference is needed.

observations:

- return at most 2 observations for the whole component;
- combine related requirements into one observation where possible;
- each observation must be concise;
- internal_finding must be one sentence, maximum 45 words;
- student_feedback must be one or two sentences, maximum 60 words;
- use only VALID REQUIREMENT IDS;
- include at most one evidence item per observation;
- do not repeat criterion findings verbatim;
- include a suggestion only when genuinely useful;
- do not mention prompts, models, graders, audits, reflection, or hidden
  reasoning.

component_summary:

- maximum 70 words;
- synthesise the component's main strengths and areas for improvement;
- do not introduce new rubric requirements.

VERIFICATION STATUS

- verified_by_tool only when a deterministic check explicitly establishes
  the finding;
- supported_by_submission when directly supported by submitted work;
- inferred when cautious interpretation is required;
- not_verified when reliable verification is unavailable.

EVIDENCE SAFETY

For submission_reference evidence:

- artifact_id must come from VALID ARTIFACT IDS;
- unit_id must come from VALID UNIT IDS when referring to a unit;
- block_id, source_range, and excerpt may be used only when they appear in
  the supplied evidence;
- never invent an identifier, range, or excerpt.

Use absence evidence only when required work is demonstrably absent.

IMPORTANT

- Candidate evidence indicates relevance, not correctness.
- Do not infer missing work merely because one candidate unit does not show it.
- Deterministic structured evidence is stronger than model interpretation.
- A successful parse does not establish logical consistency.
- Do not claim compilation, execution, syntax validity, logical consistency,
  or formal verification unless a supplied deterministic check explicitly
  establishes that exact property.
- Do not use verified_by_tool unless a supplied deterministic check directly
  supports the same finding.
- Avoid repetitive feedback. Keep the entire response compact.

Keep the response compact. Criterion findings should be brief, and the
component should have no more than two student-facing observations.

Return only the JSON object.
""".strip()

    return (
        system_prompt,
        user_prompt,
    )