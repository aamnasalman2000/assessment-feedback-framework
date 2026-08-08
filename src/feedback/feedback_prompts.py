from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def _to_serializable(
    value: Any,
) -> Any:
    if isinstance(
        value,
        BaseModel,
    ):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    if isinstance(
        value,
        list,
    ):
        return [
            _to_serializable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            key: _to_serializable(
                item
            )
            for key, item
            in value.items()
        }

    return value


def _json_block(
    value: Any,
) -> str:
    return json.dumps(
        _to_serializable(
            value
        ),
        indent=2,
        ensure_ascii=False,
    )


def _compact_task_mapping(
    value: Any,
) -> dict[str, Any] | None:
    data = _to_serializable(
        value
    )

    if not isinstance(
        data,
        dict,
    ):
        return None

    compact: dict[
        str,
        Any,
    ] = {}

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


def _compact_source_range(
    value: Any,
) -> dict[str, Any] | None:
    data = _to_serializable(
        value
    )

    if not isinstance(
        data,
        dict,
    ):
        return None

    compact = {
        field_name: data[
            field_name
        ]
        for field_name in (
            "start_line",
            "end_line",
            "start_column",
            "end_column",
        )
        if (
            field_name in data
            and data[
                field_name
            ] is not None
        )
    }

    return compact or None


def _compact_structured_data(
    *,
    unit_type: str | None,
    data: Any,
) -> dict[str, Any] | None:
    if not isinstance(
        data,
        dict,
    ):
        return None

    if (
        isinstance(
            unit_type,
            str,
        )
        and (
            unit_type.startswith(
                "prolog_"
            )
            or unit_type
            == "prolog_source"
        )
    ):
        useful_fields = (
            "predicate_name",
            "answer_predicate",
            "formula",
            "task_id",
            "description",
        )

    elif (
        isinstance(
            unit_type,
            str,
        )
        and unit_type.startswith(
            "ontology_"
        )
        and unit_type
        != "ontology_report_section"
    ):
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

    elif (
        isinstance(
            unit_type,
            str,
        )
        and (
            unit_type.startswith(
                "lean_"
            )
            or unit_type
            in {
                "theorem",
                "lemma",
                "example",
                "axiom",
                "definition",
                "abbreviation",
            }
        )
    ):
        useful_fields = (
            "declaration_type",
            "declaration_name",
            "name",
            "statement",
            "target",
            "proof_style",
            "contains_sorry",
            "contains_admit",
            "scope",
        )

    elif unit_type in {
        "ontology_report_section",
        "report_section",
        "prose_section",
    }:
        useful_fields = (
            "section_title",
            "heading",
            "section_type",
            "page",
            "page_number",
            "word_count",
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
            "word_count",
        )

    compact = {
        field_name: data[
            field_name
        ]
        for field_name
        in useful_fields
        if (
            field_name in data
            and data[
                field_name
            ] is not None
        )
    }

    return compact or None


def _content_limit_for_unit(
    unit_type: str | None,
) -> int:
    if unit_type in {
        "ontology_report_section",
        "report_section",
        "prose_section",
    }:
        return 900

    if (
        isinstance(
            unit_type,
            str,
        )
        and unit_type.startswith(
            "lean_"
        )
    ):
        return 1000

    if unit_type in {
        "theorem",
        "lemma",
        "example",
        "definition",
    }:
        return 1000

    if (
        isinstance(
            unit_type,
            str,
        )
        and unit_type.startswith(
            "prolog_"
        )
    ):
        return 500

    if (
        isinstance(
            unit_type,
            str,
        )
        and unit_type.startswith(
            "ontology_"
        )
    ):
        return 350

    return 700


def _compact_content_blocks(
    *,
    unit_type: str | None,
    blocks: Any,
) -> list[dict[str, Any]]:
    if not isinstance(
        blocks,
        list,
    ):
        return []

    compact_blocks: list[
        dict[str, Any]
    ] = []

    limit = (
        _content_limit_for_unit(
            unit_type
        )
    )

    for block in blocks:
        if not isinstance(
            block,
            dict,
        ):
            continue

        content = block.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):
            continue

        content = content.strip()

        if not content:
            continue

        compact_block: dict[
            str,
            Any,
        ] = {
            "block_type": (
                block.get(
                    "block_type"
                )
            ),
            "content": (
                content[
                    :limit
                ]
            ),
        }

        block_id = block.get(
            "block_id"
        )

        if isinstance(
            block_id,
            str,
        ):
            compact_block[
                "block_id"
            ] = block_id

        source_range = (
            _compact_source_range(
                block.get(
                    "source_range"
                )
            )
        )

        if (
            source_range
            is not None
        ):
            compact_block[
                "source_range"
            ] = source_range

        compact_blocks.append(
            compact_block
        )

    return compact_blocks[:2]


def _compact_candidate_unit(
    unit: Any,
) -> dict[str, Any]:
    data = _to_serializable(
        unit
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    unit_type = data.get(
        "unit_type"
    )

    compact: dict[
        str,
        Any,
    ] = {
        "artifact_id": (
            data.get(
                "artifact_id"
            )
        ),
        "unit_id": (
            data.get(
                "unit_id"
            )
        ),
        "unit_type": (
            unit_type
        ),
    }

    label = data.get(
        "label"
    )

    if isinstance(
        label,
        str,
    ):
        compact[
            "label"
        ] = label

    task_mapping = (
        _compact_task_mapping(
            data.get(
                "task_mapping"
            )
        )
    )

    if (
        task_mapping
        is not None
    ):
        compact[
            "task_mapping"
        ] = task_mapping

    attempt = data.get(
        "attempt"
    )

    if isinstance(
        attempt,
        dict,
    ):
        compact_attempt = {
            field_name: attempt[
                field_name
            ]
            for field_name in (
                "attempt_index",
                "attempt_group_id",
                "declared_as_alternative",
            )
            if (
                field_name
                in attempt
                and attempt[
                    field_name
                ] is not None
            )
        }

        if compact_attempt:
            compact[
                "attempt"
            ] = compact_attempt

    source_range = (
        _compact_source_range(
            data.get(
                "source_range"
            )
        )
    )

    if (
        source_range
        is not None
    ):
        compact[
            "source_range"
        ] = source_range

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

    if (
        structured_data
        is not None
    ):
        compact[
            "structured_data"
        ] = structured_data

    content_blocks = (
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

    if content_blocks:
        compact[
            "content_blocks"
        ] = content_blocks

    extraction_confidence = (
        data.get(
            "extraction_confidence"
        )
    )

    if isinstance(
        extraction_confidence,
        (
            int,
            float,
        ),
    ):
        compact[
            "extraction_confidence"
        ] = extraction_confidence

    return {
        key: value
        for key, value
        in compact.items()
        if value is not None
    }


def _compact_candidate_units(
    values: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact
        for value in values
        if (
            compact
            := _compact_candidate_unit(
                value
            )
        )
    ]


def _compact_semantic_annotation(
    value: Any,
) -> dict[str, Any]:
    data = _to_serializable(
        value
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    compact: dict[
        str,
        Any,
    ] = {}

    unit_id = data.get(
        "unit_id"
    )

    if isinstance(
        unit_id,
        str,
    ):
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
        compact_intent: dict[
            str,
            Any,
        ] = {}

        intent_type = (
            student_intent.get(
                "intent_type"
            )
        )

        if isinstance(
            intent_type,
            str,
        ):
            compact_intent[
                "intent_type"
            ] = intent_type

        description = (
            student_intent.get(
                "description"
            )
        )

        if isinstance(
            description,
            str,
        ):
            compact_intent[
                "description"
            ] = description[
                :300
            ]

        if compact_intent:
            compact[
                "student_intent"
            ] = compact_intent

    strategy = data.get(
        "strategy"
    )

    if isinstance(
        strategy,
        dict,
    ):
        compact_strategy: dict[
            str,
            Any,
        ] = {}

        for field_name in (
            "category",
            "name",
        ):
            field_value = (
                strategy.get(
                    field_name
                )
            )

            if isinstance(
                field_value,
                str,
            ):
                compact_strategy[
                    field_name
                ] = field_value

        if compact_strategy:
            compact[
                "strategy"
            ] = compact_strategy

    summary = data.get(
        "summary"
    )

    if isinstance(
        summary,
        str,
    ):
        compact[
            "summary"
        ] = summary[
            :400
        ]

    confidence = data.get(
        "confidence"
    )

    if isinstance(
        confidence,
        (
            int,
            float,
        ),
    ):
        compact[
            "confidence"
        ] = confidence

    return compact


def _compact_semantic_annotations(
    values: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact
        for value in values
        if (
            compact
            := _compact_semantic_annotation(
                value
            )
        )
    ]


def _compact_component(
    component: dict[str, Any],
) -> dict[str, Any]:
    compact: dict[
        str,
        Any,
    ] = {
        "component_id": (
            component.get(
                "component_id"
            )
        ),
        "title": (
            component.get(
                "title"
            )
        ),
        "component_type": (
            component.get(
                "component_type"
            )
        ),
        "artifact_type": (
            component.get(
                "artifact_type"
            )
        ),
    }

    artifact = component.get(
        "artifact"
    )

    if isinstance(
        artifact,
        dict,
    ):
        compact[
            "artifact"
        ] = {
            key: value
            for key, value
            in artifact.items()
            if key in {
                "artifact_type",
                "answer_location",
                "answer_locations",
            }
        }

    artifacts = component.get(
        "artifacts"
    )

    if isinstance(
        artifacts,
        list,
    ):
        compact[
            "artifacts"
        ] = [
            {
                key: value
                for key, value
                in item.items()
                if key in {
                    "artifact_type",
                    "required",
                }
            }
            for item in artifacts
            if isinstance(
                item,
                dict,
            )
        ]

    task = component.get(
        "task"
    )

    if isinstance(
        task,
        dict,
    ):
        compact_task: dict[
            str,
            Any,
        ] = {}

        for field_name in (
            "summary",
            "description",
            "target_statement",
            "semantic_goal",
            "expected_properties",
            "required_examples",
            "required_cases",
            "stated_conditions",
        ):
            field_value = task.get(
                field_name
            )

            if field_value not in (
                None,
                [],
                {},
                "",
            ):
                compact_task[
                    field_name
                ] = field_value

        if compact_task:
            compact[
                "task"
            ] = compact_task

    evaluation_guidance = (
        component.get(
            "evaluation_guidance"
        )
    )

    if isinstance(
        evaluation_guidance,
        dict,
    ):
        compact[
            "evaluation_guidance"
        ] = evaluation_guidance

    return {
        key: value
        for key, value
        in compact.items()
        if value not in (
            None,
            [],
            {},
            "",
        )
    }


def _compact_requirement(
    requirement: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: requirement[
            key
        ]
        for key in (
            "id",
            "criterion",
            "requirement",
        )
        if (
            key in requirement
            and requirement[
                key
            ] is not None
        )
    }


def _compact_alignment(
    alignment: dict[str, Any],
) -> dict[str, Any]:
    useful_fields = (
        "component_id",
        "primary_unit_ids",
        "supporting_unit_ids",
        "possibly_relevant_unit_ids",
        "confidence",
        "unresolved",
        "unresolved_reason",
    )

    return {
        field_name: alignment[
            field_name
        ]
        for field_name in useful_fields
        if (
            field_name in alignment
            and alignment[
                field_name
            ] not in (
                None,
                [],
                "",
            )
        )
    }


def _compact_processing_check(
    value: Any,
) -> dict[str, Any]:
    data = _to_serializable(
        value
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    useful_fields = (
        "check_type",
        "status",
        "tool",
        "summary",
    )

    return {
        field_name: data[
            field_name
        ]
        for field_name in useful_fields
        if (
            field_name in data
            and data[
                field_name
            ] not in (
                None,
                "",
                [],
            )
        )
    }


def _compact_processing_checks(
    values: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact
        for value in values
        if (
            compact
            := _compact_processing_check(
                value
            )
        )
    ]


def _compact_processing_diagnostic(
    value: Any,
) -> dict[str, Any]:
    data = _to_serializable(
        value
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    compact: dict[
        str,
        Any,
    ] = {}

    for field_name in (
        "diagnostic_id",
        "severity",
        "diagnostic_type",
        "message",
        "related_unit_ids",
    ):
        field_value = data.get(
            field_name
        )

        if field_value not in (
            None,
            "",
            [],
        ):
            compact[
                field_name
            ] = field_value

    source_range = (
        _compact_source_range(
            data.get(
                "source_range"
            )
        )
    )

    if (
        source_range
        is not None
    ):
        compact[
            "source_range"
        ] = source_range

    return compact


def _compact_processing_diagnostics(
    values: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact
        for value in values
        if (
            compact
            := _compact_processing_diagnostic(
                value
            )
        )
    ]


def _filter_global_requirements(
    *,
    global_requirements: list[dict[str, Any]],
    component_id: str,
    part_id: str | None,
) -> list[dict[str, Any]]:
    selected: list[
        dict[str, Any]
    ] = []

    for requirement in (
        global_requirements
    ):
        if not isinstance(
            requirement,
            dict,
        ):
            continue

        applies_to = (
            requirement.get(
                "applies_to"
            )
        )

        if not isinstance(
            applies_to,
            str,
        ):
            selected.append(
                requirement
            )
            continue

        scope = applies_to.lower()

        keep = False

        if scope in {
            "entire_submission",
            "all_components",
        }:
            keep = True

        elif (
            part_id
            and (
                scope == part_id.lower()
                or scope.startswith(
                    part_id.lower()
                )
            )
        ):
            keep = True

        elif (
            component_id.startswith(
                "part_1"
            )
            and "part_1"
            in scope
        ):
            keep = True

        elif (
            component_id.startswith(
                "part_2"
            )
            and "part_2"
            in scope
        ):
            keep = True

        elif (
            component_id
            in scope
        ):
            keep = True

        if keep:
            selected.append(
                {
                    key: value
                    for key, value
                    in requirement.items()
                    if key in {
                        "id",
                        "requirement",
                        "applies_to",
                        "evaluation_dimension",
                    }
                }
            )

    return selected


def _filter_feedback_requirements(
    *,
    component_id: str,
    feedback_requirements: list[str],
) -> list[str]:
    selected: list[str] = []

    for requirement in (
        feedback_requirements
    ):
        if not isinstance(
            requirement,
            str,
        ):
            continue

        text = requirement.strip()

        if not text:
            continue

        lower = text.lower()

        if (
            component_id
            == "part_1_ontology"
            and lower.startswith(
                "prolog feedback:"
            )
        ):
            continue

        if (
            component_id.startswith(
                "part_2_"
            )
            and lower.startswith(
                "ontology feedback:"
            )
        ):
            continue

        selected.append(
            text
        )

    return selected


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
    component_id = component[
        "component_id"
    ]

    requirement_id = requirement[
        "id"
    ]

    compact_component = (
        _compact_component(
            component
        )
    )

    compact_requirement = (
        _compact_requirement(
            requirement
        )
    )

    compact_units = (
        _compact_candidate_units(
            candidate_units
        )
    )

    compact_semantics = (
        _compact_semantic_annotations(
            semantic_annotations
        )
    )

    compact_alignment = (
        _compact_alignment(
            alignment
        )
    )

    compact_checks = (
        _compact_processing_checks(
            processing_checks
        )
    )

    compact_diagnostics = (
        _compact_processing_diagnostics(
            processing_diagnostics
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

    system_prompt = """
You are an academic feedback assistant evaluating exactly one assessment
requirement for one component of a student submission.

Evaluate the semantic requirement rather than conformity to a preferred
solution, naming convention, proof style, code structure, ontology
structure, formula syntax, ordering, or presentation style unless the
assessment explicitly requires that form.

The supplied units were selected through deterministic preprocessing and
component alignment. Treat them as candidate evidence, not as proof that
the requirement is satisfied.

Rules:

1. Evaluate only the supplied requirement.
2. Do not evaluate or score other requirements.
3. Use only supplied student evidence and deterministic processing evidence.
4. Accept valid alternative approaches.
5. Do not invent student content, proof steps, ontology axioms, formulas,
   compilation results, source references, or missing work.
6. Prefer direct structured source data and deterministic checks over
   model-generated semantic interpretation.
7. Use not_assessable when available evidence cannot support a reliable
   judgement.
8. Use missing only when required work is demonstrably absent.
9. Never claim compilation, consistency, syntax validity, or tool
   verification unless deterministic evidence establishes it.
10. Feedback must be concise, specific, constructive, and actionable.
11. Every evidence object must contain evidence_type.
12. Return only one valid JSON object matching the structured output schema
    supplied separately by the client.

Use submission_reference for supplied student evidence.
Use absence only when required work is demonstrably absent.

Do not return the requirement ID.
Do not generate record IDs.
Do not assign marks.
Do not produce Markdown outside the JSON object.
""".strip()

    user_prompt = f"""
Evaluate exactly one assessment requirement.

COMPONENT

{_json_block(compact_component)}

PART ID

{_json_block(part_id)}

REQUIREMENT ID CONTROLLED BY PYTHON

{requirement_id}

REQUIREMENT

{_json_block(compact_requirement)}

REQUIREMENT-LEVEL ALIGNMENT

{_json_block(compact_alignment)}

CANDIDATE STUDENT EVIDENCE

{_json_block(compact_units)}

VALID CANDIDATE UNIT IDS

{_json_block(valid_unit_ids)}

VALID CANDIDATE ARTIFACT IDS

{_json_block(valid_artifact_ids)}

SEMANTIC INTERPRETATIONS

{_json_block(compact_semantics)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(compact_checks)}

PROCESSING DIAGNOSTICS

{_json_block(compact_diagnostics)}

RELEVANT GLOBAL REQUIREMENTS

{_json_block(relevant_global_requirements)}

FEEDBACK GUIDANCE

{_json_block(relevant_feedback_requirements)}

RELEVANT SPECIFICATION NOTES

{_json_block(specification_notes)}

JUDGEMENT

Return one of:

- met
- partially_met
- not_met
- missing
- not_assessable

Use:

- met when supplied evidence clearly demonstrates the requirement;
- partially_met when only part of the requirement is demonstrated;
- not_met when relevant work is present but contradicts or fails the
  requirement;
- missing when expected work is demonstrably absent;
- not_assessable when evidence is irrelevant, insufficient, or reliable
  verification is unavailable.

FIELDS

internal_finding:
Give a precise evidence-grounded analysis.

student_feedback:
Give concise student-facing feedback describing what was demonstrated or
what needs improvement.

suggestion:
Include only when a practical improvement is appropriate.

verification_status:

- verified_by_tool only when deterministic tool evidence establishes the
  finding;
- supported_by_submission when directly supported by student work;
- inferred when cautious semantic interpretation is required;
- not_verified when reliable verification is unavailable.

confidence:
Reflect the strength and completeness of the supplied evidence.

EVIDENCE SAFETY

For submission_reference evidence:

- artifact_id must come from VALID CANDIDATE ARTIFACT IDS;
- unit_id must come from VALID CANDIDATE UNIT IDS when referring to a unit;
- use block_id or source_range only when supplied;
- never invent an artifact ID, unit ID, block ID, excerpt, or source range.

If no valid submission reference supports the judgement, return an empty
evidence array or valid absence evidence.

IMPORTANT ASSESSMENT RULES

- Alignment indicates relevance only; it does not prove correctness.
- Semantic annotations are interpretive and may be wrong.
- Deterministic structured data takes precedence where they conflict.
- Do not penalise a valid alternative solution.
- Do not infer omitted requirements from a preferred answer.
- Do not claim a formula, proof, ontology, or program has been formally
  verified unless deterministic evidence explicitly says so.

FINAL CHECK

Before returning:

- only this requirement has been evaluated;
- no requirement ID or record ID has been generated;
- every evidence reference exists in the supplied candidate evidence;
- every evidence object includes evidence_type;
- no unsupported tool, compilation, syntax, or consistency claim appears;
- no preferred solution form has been imposed;
- the response contains exactly one JSON object.

Return only the JSON object.
""".strip()

    return (
        system_prompt,
        user_prompt,
    )