from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .feedback_models import RequirementFeedbackLLMOutput


def _to_serializable(value: Any) -> Any:
    """Convert Pydantic models and nested values into JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    if isinstance(value, list):
        return [_to_serializable(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _to_serializable(item)
            for key, item in value.items()
        }

    return value


def _json_block(value: Any) -> str:
    return json.dumps(
        _to_serializable(value),
        indent=2,
        ensure_ascii=False,
    )


def build_requirement_feedback_prompt(
    *,
    component: dict[str, Any],
    requirement: dict[str, Any],
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
    Build a narrow prompt for evaluating exactly one known assessment
    requirement.
    """

    component_id = component["component_id"]
    requirement_id = requirement["id"]

    valid_unit_ids = [
        unit.get("unit_id")
        for unit in candidate_units
        if (
            isinstance(unit, dict)
            and isinstance(unit.get("unit_id"), str)
        )
    ]

    output_schema = (
        RequirementFeedbackLLMOutput.model_json_schema()
    )

    system_prompt = """
You are an academic feedback assistant evaluating exactly one assessment
requirement for one component of a student submission.

The student may use any valid approach. Evaluate the semantic requirement,
not conformity to a preferred solution, naming convention, proof style,
ordering, code structure, ontology structure, or presentation style.

The supplied units were selected by an automatic alignment stage. Treat
them as candidate evidence, not as authoritative mappings.

Follow these rules:

1. Evaluate only the supplied requirement.
2. Do not evaluate any other requirement.
3. Base the judgement only on supplied student evidence and deterministic
   processing evidence.
4. Confirm that candidate units genuinely address the component before
   relying on them.
5. Accept alternative valid approaches that demonstrate the requirement.
6. Do not invent student content, assumptions, proof steps, source
   references, compilation results, or missing work.
7. Direct source content and deterministic checks are stronger evidence
   than semantic interpretation.
8. Use not_assessable when the supplied evidence cannot support a reliable
   judgement.
9. Use missing only when the required work is clearly expected and its
   absence is supported by the supplied evidence.
10. When compilation was not run, do not claim that the submission compiled
    or failed to compile.
11. Feedback must be specific, constructive, concise, and actionable.
12. Return only valid JSON matching the supplied output schema.

Every evidence object must contain evidence_type.

Use submission_reference for evidence from a supplied artifact, unit,
content block, excerpt, or source range.

Use absence only for required work that is demonstrably absent.

Do not return the requirement ID. Python already knows it.
Do not generate record IDs.
Do not assign marks.
Do not produce markdown outside the JSON object.
""".strip()

    user_prompt = f"""
Evaluate exactly one assessment requirement.

COMPONENT ID

{component_id}

PART ID

{_json_block(part_id)}

COMPONENT SPECIFICATION

{_json_block(component)}

REQUIREMENT ID CONTROLLED BY PYTHON

{requirement_id}

REQUIREMENT TO EVALUATE

{_json_block(requirement)}

AUTOMATIC ALIGNMENT RESULT

{_json_block(alignment)}

The alignment is tentative. Verify relevance from the actual unit content.

CANDIDATE PROCESSED UNITS

{_json_block(candidate_units)}

VALID CANDIDATE UNIT IDS

{_json_block(valid_unit_ids)}

SEMANTIC ANNOTATIONS

{_json_block(semantic_annotations)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(processing_checks)}

PROCESSING DIAGNOSTICS

{_json_block(processing_diagnostics)}

GLOBAL ASSESSMENT REQUIREMENTS

{_json_block(global_requirements)}

FEEDBACK REQUIREMENTS

{_json_block(feedback_requirements)}

RELEVANT SPECIFICATION NOTES

{_json_block(specification_notes)}

EVALUATION PROCEDURE

Determine whether the single supplied requirement is:

- met;
- partially_met;
- not_met;
- missing;
- not_assessable.

Use:

- met when the evidence clearly demonstrates the requirement;
- partially_met when some but not all of the requirement is demonstrated;
- not_met when relevant work is present but contradicts or fails the
  requirement;
- missing when the required work is demonstrably absent;
- not_assessable when the supplied evidence is irrelevant or insufficient.

Write:

- internal_finding as a precise evidence-grounded analysis;
- student_feedback as clear student-facing feedback;
- suggestion only when a practical improvement is appropriate;
- verification_status according to the strongest evidence;
- confidence according to the strength and completeness of evidence.

VERIFICATION STATUS

Use:

- verified_by_tool only for deterministic tool evidence;
- supported_by_submission for direct evidence in student work;
- inferred for cautious semantic interpretation;
- not_verified when reliable verification is unavailable.

ALIGNMENT SAFETY

- Ignore candidate units that do not genuinely address the component.
- Do not invent or search for replacement units.
- A high alignment confidence does not prove the requirement is met.
- A low alignment confidence does not prove the student is incorrect.
- If the supplied unit belongs to another component, use not_assessable.
- Do not force a positive or negative judgement from irrelevant evidence.

ALTERNATIVE APPROACHES

Accept different valid:

- proof strategies;
- theorem and predicate names;
- decomposition choices;
- code structures;
- ontology structures;
- argument structures;
- ordering and presentation styles.

Only require a specific form when the specification explicitly requires it.

EVIDENCE

For submission_reference evidence:

- artifact_id must identify a supplied candidate artifact;
- unit_id must identify a supplied candidate unit when referring to a unit;
- use a valid source range or block ID when available;
- never invent an artifact ID, unit ID, block ID, excerpt, or source range.

If no valid submission evidence supports the judgement, return an empty
evidence array or valid absence evidence. Do not fabricate a reference.

OUTPUT JSON SCHEMA

{_json_block(output_schema)}

FINAL CHECK

Before returning the response, verify that:

- only the single supplied requirement was evaluated;
- no requirement ID was generated;
- every source reference exists in the supplied candidate evidence;
- every evidence object contains evidence_type;
- no unsupported compilation claim was made;
- no preferred solution style was imposed;
- the response contains only one JSON object matching the schema.

Return only the JSON object.
""".strip()

    return system_prompt, user_prompt