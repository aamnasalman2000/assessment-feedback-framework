from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .alignment_models import AlignmentLLMOutput


SYSTEM_PROMPT = """
You are an assessment-alignment component in an academic feedback
pipeline.

Your task is to align the supplied student-submission units with the
assessment components they answer or support.

This is a document-level alignment task.

Return only structured data conforming exactly to the supplied JSON
schema.

Do not evaluate whether rubric requirements are met.
Do not assign marks.
Do not generate student-facing feedback.
Do not invent component IDs or unit IDs.
""".strip()


def _to_serializable(value: Any) -> Any:
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


def _json_text(value: Any) -> str:
    return json.dumps(
        _to_serializable(value),
        indent=2,
        ensure_ascii=False,
    )


def build_document_alignment_prompt(
    *,
    processed_units: list[Any],
    semantic_annotations: list[Any],
    component_candidates: list[dict[str, Any]],
) -> tuple[str, str]:
    valid_unit_ids = [
        unit.unit_id
        if hasattr(unit, "unit_id")
        else unit.get("unit_id")
        for unit in processed_units
        if (
            hasattr(unit, "unit_id")
            or isinstance(unit, dict)
        )
    ]

    valid_component_ids = [
        component["component_id"]
        for component in component_candidates
        if isinstance(component.get("component_id"), str)
    ]

    output_schema = AlignmentLLMOutput.model_json_schema()

    user_prompt = f"""
Align the complete processed submission to the supplied assessment
components.

PROCESSED SUBMISSION UNITS

{_json_text(processed_units)}

SEMANTIC ANNOTATIONS

{_json_text(semantic_annotations)}

ASSESSMENT COMPONENT CANDIDATES

{_json_text(component_candidates)}

VALID UNIT IDS

{_json_text(valid_unit_ids)}

VALID COMPONENT IDS

{_json_text(valid_component_ids)}

ALIGNMENT PROCEDURE

Produce exactly one component alignment for every component in VALID
COMPONENT IDS.

For each component:

- use primary_unit_ids for units that directly attempt to answer the
  component;
- use supporting_unit_ids for units that provide relevant supporting
  definitions, explanations, evidence, or implementation;
- use possibly_relevant_unit_ids only where the relationship is
  plausible but uncertain;
- place a unit in at most one category for the same component;
- allow one unit to support more than one component only where the
  supplied content genuinely supports multiple components;
- mark the component unresolved when no unit can be aligned reliably;
- provide an unresolved_reason when unresolved is true;
- set unresolved to false when at least one unit is aligned.

GROUNDING RULES

- Use only component IDs from VALID COMPONENT IDS.
- Use only unit IDs from VALID UNIT IDS.
- Do not invent, shorten, rename, or reconstruct identifiers.
- Base alignment on the unit content and semantic annotations.
- Use headings, statements, comments, propositions, entities, code,
  explanations, and explicit task references where available.
- Do not align solely from unit order, file order, numbering, or physical
  proximity.
- Do not import concepts from unrelated assessment components.
- Do not claim compilation success or failure.
- Do not assess criterion status.
- Do not generate marks or feedback.
- Keep confidence below 0.90 when the mapping is indirect or ambiguous.

DOCUMENT-LEVEL REASONING

Consider relationships across the complete submission.

A component may be answered by:

- one unit;
- several units;
- units from different artifacts;
- a primary unit plus supporting explanations;
- no identifiable unit.

A unit may support several components only when the source content
clearly justifies those links.

OUTPUT REQUIREMENTS

- component_alignments must contain every valid component exactly once;
- no unknown component IDs may appear;
- no unknown unit IDs may appear;
- no duplicate component IDs may appear;
- no unit may occur in more than one category for the same component;
- components without reliable evidence must still appear and be marked
  unresolved;
- summary must describe the overall alignment result;
- warnings should identify genuine ambiguity only.

OUTPUT JSON SCHEMA

{_json_text(output_schema)}

FINAL CHECK

Before responding, verify that:

- every valid component appears exactly once;
- all component IDs and unit IDs are copied exactly;
- every aligned unit is grounded in the supplied submission;
- unresolved components contain no aligned unit IDs;
- resolved components contain at least one aligned unit ID;
- no rubric judgement, marks, or student feedback are present;
- the response contains only one JSON object matching the schema.

Return only the structured JSON object.
""".strip()

    return SYSTEM_PROMPT, user_prompt