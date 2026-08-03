from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .feedback_models import (
    CriterionAssessment,
    FeedbackObservation,
    ReflectionDecision,
)


def _to_serializable(value: Any) -> Any:
    """Convert Pydantic models and nested values into JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    if isinstance(value, list):
        return [
            _to_serializable(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _to_serializable(item)
            for key, item in value.items()
        }

    return value


def _json_block(value: Any) -> str:
    return json.dumps(
        _to_serializable(value),
        ensure_ascii=False,
        indent=2,
    )


def build_requirement_reflection_prompt(
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
) -> tuple[str, str]:
    """
    Build a focused audit prompt for one existing requirement-level
    assessment.

    Reflection is treated as conservative quality assurance rather than
    fresh feedback generation.
    """

    component_id = component["component_id"]
    requirement_id = requirement["id"]

    requirement_text = requirement.get(
        "requirement",
        "",
    )

    valid_unit_ids = [
        unit.get("unit_id")
        for unit in candidate_units
        if (
            isinstance(unit, dict)
            and isinstance(
                unit.get("unit_id"),
                str,
            )
        )
    ]

    valid_artifact_ids = list(
        dict.fromkeys(
            unit.get("artifact_id")
            for unit in candidate_units
            if (
                isinstance(unit, dict)
                and isinstance(
                    unit.get("artifact_id"),
                    str,
                )
            )
        )
    )

    output_schema = (
        ReflectionDecision.model_json_schema()
    )

    system_prompt = """
You are a conservative second academic examiner.

Review one existing assessment against one exact rubric requirement and the
supplied student evidence.

The default decision is KEEP.

Set analysis.should_revise to true only when the original assessment contains
a clear, material defect supported by the exact requirement or supplied
student work.

Do not revise merely to improve wording, add detail, change tone, or propose
an optional improvement.

Do not introduce any concept that is absent from the exact requirement and
student evidence.

If revising:

1. analysis.requirement_basis must copy an exact verbatim phrase from the
   exact requirement;
2. the revision reason must identify the specific original claim that is
   wrong, unsupported, incomplete, or misleading;
3. every revised field must remain specific to the supplied requirement and
   student work;
4. preserve all correct parts of the original assessment;
5. use only supplied evidence references;
6. do not invent identifiers, excerpts, tests, tool results, or rubric
   expectations.

If you cannot identify a concrete and grounded defect, set
analysis.should_revise to false and omit every revised_* field.

Return only one JSON object conforming to the supplied schema.
""".strip()

    user_prompt = f"""
Audit the following existing assessment.

EXACT REQUIREMENT ID

{requirement_id}

EXACT REQUIREMENT TEXT

{requirement_text}

ORIGINAL CRITERION ASSESSMENT

{_json_block(initial_criterion_assessment)}

ORIGINAL STUDENT-FACING OBSERVATION

{_json_block(initial_observation)}

CANDIDATE STUDENT WORK

{_json_block(candidate_units)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(processing_checks)}

VALID UNIT IDS

{_json_block(valid_unit_ids)}

VALID ARTIFACT IDS

{_json_block(valid_artifact_ids)}

AUDIT TASK

Determine whether the original assessment is materially wrong, unsupported,
incomplete, or misleading for the exact requirement above.

Use the following rules:

- Review only requirement {requirement_id}.
- Treat the exact requirement text as the complete rubric scope.
- Do not add requirements that are not explicitly stated.
- Do not infer missing work from a single excerpt when other candidate units
  may contain relevant evidence.
- Use not_assessable when the supplied evidence is insufficient.
- Use missing only when required work is demonstrably absent.
- Accept alternative valid approaches unless the exact requirement mandates
  one method.
- Do not claim execution, compilation, proof correctness, or tool
  verification unless deterministic evidence supports it.
- Do not revise solely because confidence could be expressed differently.

KEEP DECISION

If the original assessment is adequately supported:

- set analysis.evidence_supported to true;
- set analysis.preferred_solution_bias to false unless clear bias exists;
- set analysis.confidence_assessment to appropriate unless clearly incorrect;
- set analysis.should_revise to false;
- explain briefly why the assessment is supported;
- use empty lists for unsupported_claims, overlooked_evidence, and
  missing_rubric_points;
- omit analysis.requirement_basis;
- omit every revised_* field.

REVISE DECISION

Set analysis.should_revise to true only when a concrete material defect is
demonstrated.

When revising:

- analysis.requirement_basis must be an exact verbatim phrase copied from:

  {requirement_text}

- do not paraphrase requirement_basis;
- identify the exact defective claim in revision_reason;
- provide all required revised_* fields;
- make the smallest necessary correction;
- use requirement ID {requirement_id} for absence evidence;
- use only valid unit and artifact IDs listed above.

A proposed revision is invalid if requirement_basis does not appear verbatim
in the exact requirement text.

OUTPUT JSON SCHEMA

{_json_block(output_schema)}

FINAL CHECK

Before returning the JSON object, confirm:

1. Every concept in the response comes from the exact requirement, original
   assessment, or supplied student work.
2. No new rubric requirement has been invented.
3. A revise decision contains an exact verbatim requirement_basis.
4. A revise decision contains a complete revision patch.
5. A keep decision contains no revised_* fields.
6. When uncertain, the decision is KEEP.

Return only the JSON object.
""".strip()

    return system_prompt, user_prompt