from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


SYSTEM_PROMPT = """
You are an assessment-alignment component in an academic feedback
pipeline.

Your task is to align supplied student-submission units with the
assessment components they answer or support.

This is an evidence-routing task, not an assessment task.

Return only structured data conforming exactly to the supplied JSON
schema.

Do not evaluate whether rubric requirements are met.
Do not assign marks.
Do not generate student-facing feedback.
Do not invent component IDs or unit IDs.
""".strip()


def _to_serializable(
    value: Any,
) -> Any:
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


def _json_text(
    value: Any,
) -> str:
    return json.dumps(
        _to_serializable(value),
        indent=2,
        ensure_ascii=False,
    )


def _compact_task_mapping(
    value: Any,
) -> dict[str, Any] | None:
    data = _to_serializable(value)

    if not isinstance(data, dict):
        return None

    compact: dict[str, Any] = {}

    for field_name in (
        "part_ids",
        "task_ids",
        "source",
        "confidence",
    ):
        field_value = data.get(
            field_name
        )

        if field_value not in (
            None,
            [],
            "",
        ):
            compact[
                field_name
            ] = field_value

    return compact or None


def _compact_attempt(
    value: Any,
) -> dict[str, Any] | None:
    data = _to_serializable(value)

    if not isinstance(data, dict):
        return None

    compact: dict[str, Any] = {}

    for field_name in (
        "attempt_index",
        "attempt_group_id",
        "declared_as_alternative",
    ):
        field_value = data.get(
            field_name
        )

        if field_value is not None:
            compact[
                field_name
            ] = field_value

    return compact or None


def _compact_lean_structured_data(
    data: dict[str, Any],
) -> dict[str, Any]:
    useful_fields = (
        "declaration_type",
        "declaration_name",
        "name",
        "statement",
        "target",
        "proof_style",
        "contains_sorry",
        "contains_admit",
    )

    return {
        field_name: data[field_name]
        for field_name in useful_fields
        if (
            field_name in data
            and data[field_name] is not None
        )
    }


def _compact_prolog_structured_data(
    data: dict[str, Any],
) -> dict[str, Any]:
    useful_fields = (
        "predicate_name",
        "formula",
        "answer_predicate",
        "task_id",
        "description",
    )

    return {
        field_name: data[field_name]
        for field_name in useful_fields
        if (
            field_name in data
            and data[field_name] is not None
        )
    }


def _compact_ontology_structured_data(
    data: dict[str, Any],
) -> dict[str, Any]:
    useful_fields = (
        "item_kind",
        "entity_type",
        "entity",
        "axiom_type",
        "subject",
        "parent",
        "property",
        "domain",
        "range",
        "subproperty",
        "superproperty",
        "filler",
        "cardinality",
        "characteristic",
        "members",
    )

    return {
        field_name: data[field_name]
        for field_name in useful_fields
        if (
            field_name in data
            and data[field_name] is not None
        )
    }


def _compact_report_structured_data(
    data: dict[str, Any],
) -> dict[str, Any]:
    useful_fields = (
        "section_title",
        "heading",
        "page",
        "page_number",
        "section_type",
    )

    return {
        field_name: data[field_name]
        for field_name in useful_fields
        if (
            field_name in data
            and data[field_name] is not None
        )
    }


def _compact_structured_data(
    *,
    unit_type: str | None,
    data: Any,
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    if (
        isinstance(unit_type, str)
        and unit_type.startswith("ontology_")
    ):
        compact = (
            _compact_ontology_structured_data(
                data
            )
        )

    elif unit_type in {
        "prolog_answer",
        "prolog_formula",
        "prolog_source",
    }:
        compact = (
            _compact_prolog_structured_data(
                data
            )
        )

    elif (
        isinstance(unit_type, str)
        and (
            unit_type.startswith("lean_")
            or unit_type
            in {
                "theorem",
                "lemma",
                "example",
                "definition",
                "axiom",
            }
        )
    ):
        compact = (
            _compact_lean_structured_data(
                data
            )
        )

    elif unit_type in {
        "report_section",
        "ontology_report_section",
        "prose_section",
    }:
        compact = (
            _compact_report_structured_data(
                data
            )
        )

    else:
        useful_fields = (
            "predicate_name",
            "formula",
            "declaration_type",
            "declaration_name",
            "statement",
            "target",
            "entity_type",
            "entity",
            "axiom_type",
            "subject",
            "property",
            "parent",
            "domain",
            "range",
            "filler",
            "cardinality",
            "section_title",
        )

        compact = {
            field_name: data[field_name]
            for field_name in useful_fields
            if (
                field_name in data
                and data[field_name] is not None
            )
        }

    return compact or None


def _compact_content_blocks(
    *,
    unit_type: str | None,
    blocks: Any,
) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    compact_blocks: list[
        dict[str, Any]
    ] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get(
            "block_type"
        )

        content = block.get(
            "content"
        )

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        # Formal artefacts usually already expose their important
        # structure through structured_data, so keep source text short.
        if (
            isinstance(unit_type, str)
            and (
                unit_type.startswith(
                    "ontology_"
                )
                or unit_type.startswith(
                    "lean_"
                )
                or unit_type.startswith(
                    "prolog_"
                )
            )
        ):
            limit = 350

        # Report prose carries semantic information that may be useful for
        # routing to the ontology component, so retain a little more.
        else:
            limit = 700

        compact_block: dict[
            str,
            Any,
        ] = {
            "block_type": block_type,
            "content": content[:limit],
        }

        block_id = block.get(
            "block_id"
        )

        if isinstance(block_id, str):
            compact_block[
                "block_id"
            ] = block_id

        compact_blocks.append(
            compact_block
        )

    # Alignment should not need many blocks from one unit.
    return compact_blocks[:2]


def _compact_processed_unit(
    unit: Any,
) -> dict[str, Any]:
    data = _to_serializable(unit)

    if not isinstance(data, dict):
        return {}

    unit_type = data.get(
        "unit_type"
    )

    compact: dict[str, Any] = {
        "unit_id": data.get(
            "unit_id"
        ),
        "unit_type": unit_type,
    }

    label = data.get(
        "label"
    )

    if isinstance(label, str):
        compact["label"] = label

    task_mapping = (
        _compact_task_mapping(
            data.get(
                "task_mapping"
            )
        )
    )

    if task_mapping is not None:
        compact[
            "task_mapping"
        ] = task_mapping

    attempt = _compact_attempt(
        data.get(
            "attempt"
        )
    )

    if attempt is not None:
        compact[
            "attempt"
        ] = attempt

    parent_unit_id = data.get(
        "parent_unit_id"
    )

    if isinstance(
        parent_unit_id,
        str,
    ):
        compact[
            "parent_unit_id"
        ] = parent_unit_id

    structured_data = (
        _compact_structured_data(
            unit_type=(
                unit_type
                if isinstance(
                    unit_type,
                    str,
                )
                else None
            ),
            data=data.get(
                "structured_data"
            ),
        )
    )

    if structured_data is not None:
        compact[
            "structured_data"
        ] = structured_data

    compact_blocks = (
        _compact_content_blocks(
            unit_type=(
                unit_type
                if isinstance(
                    unit_type,
                    str,
                )
                else None
            ),
            blocks=data.get(
                "content_blocks",
                [],
            ),
        )
    )

    if compact_blocks:
        compact[
            "content_blocks"
        ] = compact_blocks

    return {
        key: value
        for key, value
        in compact.items()
        if value is not None
    }


def _compact_semantic_annotation(
    annotation: Any,
) -> dict[str, Any]:
    data = _to_serializable(
        annotation
    )

    if not isinstance(data, dict):
        return {}

    compact: dict[str, Any] = {}

    unit_id = data.get(
        "unit_id"
    )

    if isinstance(unit_id, str):
        compact[
            "unit_id"
        ] = unit_id

    semantic_role = data.get(
        "semantic_role"
    )

    if isinstance(
        semantic_role,
        str,
    ):
        compact[
            "semantic_role"
        ] = semantic_role

    student_intent = data.get(
        "student_intent"
    )

    if isinstance(
        student_intent,
        dict,
    ):
        intent_type = (
            student_intent.get(
                "intent_type"
            )
        )

        description = (
            student_intent.get(
                "description"
            )
        )

        compact_intent: dict[
            str,
            Any,
        ] = {}

        if isinstance(
            intent_type,
            str,
        ):
            compact_intent[
                "intent_type"
            ] = intent_type

        if isinstance(
            description,
            str,
        ):
            compact_intent[
                "description"
            ] = description[:350]

        if compact_intent:
            compact[
                "student_intent"
            ] = compact_intent

    strategy = data.get(
        "strategy"
    )

    if isinstance(strategy, dict):
        compact_strategy: dict[
            str,
            Any,
        ] = {}

        category = strategy.get(
            "category"
        )

        name = strategy.get(
            "name"
        )

        if isinstance(
            category,
            str,
        ):
            compact_strategy[
                "category"
            ] = category

        if isinstance(
            name,
            str,
        ):
            compact_strategy[
                "name"
            ] = name

        if compact_strategy:
            compact[
                "strategy"
            ] = compact_strategy

    summary = data.get(
        "summary"
    )

    if isinstance(summary, str):
        compact[
            "summary"
        ] = summary[:500]

    return compact


def _compact_processed_units(
    processed_units: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact_unit
        for unit in processed_units
        if (
            compact_unit
            := _compact_processed_unit(
                unit
            )
        )
    ]


def _compact_semantic_annotations(
    semantic_annotations: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact_annotation
        for annotation
        in semantic_annotations
        if (
            compact_annotation
            := _compact_semantic_annotation(
                annotation
            )
        )
    ]


def build_document_alignment_prompt(
    *,
    processed_units: list[Any],
    semantic_annotations: list[Any],
    component_candidates: list[
        dict[str, Any]
    ],
) -> tuple[str, str]:
    compact_processed_units = (
        _compact_processed_units(
            processed_units
        )
    )

    compact_semantic_annotations = (
        _compact_semantic_annotations(
            semantic_annotations
        )
    )

    valid_unit_ids = [
        unit["unit_id"]
        for unit
        in compact_processed_units
        if isinstance(
            unit.get("unit_id"),
            str,
        )
    ]

    valid_component_ids = [
        component["component_id"]
        for component
        in component_candidates
        if (
            isinstance(
                component,
                dict,
            )
            and isinstance(
                component.get(
                    "component_id"
                ),
                str,
            )
        )
    ]

    user_prompt = f"""
Align the supplied student-submission units to the assessment components.

Alignment performs evidence routing only. It does not determine whether
assessment requirements are satisfied.

PROCESSED UNITS

{_json_text(compact_processed_units)}

SEMANTIC INTERPRETATIONS

{_json_text(compact_semantic_annotations)}

ASSESSMENT COMPONENTS

{_json_text(component_candidates)}

VALID UNIT IDS

{_json_text(valid_unit_ids)}

VALID COMPONENT IDS

{_json_text(valid_component_ids)}

ALIGNMENT RULES

Produce exactly one alignment for every component in VALID COMPONENT IDS.

Use:

- primary_unit_ids for units that directly answer or implement the
  component;
- supporting_unit_ids for relevant explanation, definitions, context, or
  supporting implementation;
- possibly_relevant_unit_ids only when relevance is plausible but
  uncertain.

A unit may appear in only one category for the same component.

The same unit may support different components only when its content
genuinely applies to each.

Mark a component unresolved only when no unit can be aligned reliably.
An unresolved component must have no aligned unit IDs and must provide an
unresolved_reason.

EVIDENCE PRIORITY

Prefer deterministic processed information when it gives a clear mapping.

Strong signals include:

- task_mapping;
- answer predicate names;
- theorem or declaration names;
- ontology unit types and structured axiom fields;
- report section labels;
- explicit task references.

Semantic interpretations are secondary evidence. Use them to clarify the
meaning of a unit, but do not allow an uncertain semantic interpretation
to override clear deterministic structure.

GROUNDING

- Use only IDs listed in VALID UNIT IDS and VALID COMPONENT IDS.
- Copy identifiers exactly.
- Do not invent or reconstruct identifiers.
- Do not infer assessment correctness.
- Do not assign marks.
- Do not generate feedback.
- Do not make compilation claims.
- Do not treat alignment confidence as criterion satisfaction.
- Keep confidence below 0.90 for indirect or ambiguous mappings.

DOCUMENT-LEVEL CONSIDERATIONS

A component may be supported by one unit, several units, or units from
different artefacts.

For heterogeneous submissions, formal artefacts may provide primary
implementation evidence while report sections provide supporting
explanation.

Do not align units merely because they occur near one another or appear in
a similar order. Prefer explicit structural and semantic evidence.

OUTPUT REQUIREMENTS

- return every valid component exactly once;
- use no unknown component or unit IDs;
- use no duplicate component IDs;
- place no unit in multiple categories for one component;
- resolved components must contain at least one aligned unit;
- unresolved components must contain no aligned units;
- summary should briefly describe the overall alignment;
- warnings should contain only genuine ambiguity.

Return only one structured JSON object matching the supplied schema.
""".strip()

    return (
        SYSTEM_PROMPT,
        user_prompt,
    )