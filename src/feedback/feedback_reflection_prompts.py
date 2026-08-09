from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .feedback_models import (
    CriterionAssessment,
    FeedbackObservation,
    ReflectionAnalysis,
)


# ============================================================
# Basic serialization
# ============================================================


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
            _to_serializable(item)
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            key: _to_serializable(item)
            for key, item in value.items()
        }

    return value


def _json_block(
    value: Any,
) -> str:
    return json.dumps(
        _to_serializable(value),
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Candidate-evidence compaction
# ============================================================


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
        field_name: data[field_name]
        for field_name in (
            "start_line",
            "end_line",
            "start_column",
            "end_column",
        )
        if (
            field_name in data
            and data[field_name]
            is not None
        )
    }

    return compact or None


def _is_ontology_structural_unit(
    unit_type: str | None,
) -> bool:
    if not isinstance(
        unit_type,
        str,
    ):
        return False

    if (
        unit_type
        == "ontology_report_section"
    ):
        return False

    return unit_type.startswith(
        "ontology_"
    )


def _is_task_5_prolog_unit(
    unit_type: str | None,
) -> bool:
    return (
        unit_type
        == "prolog_task_5_answer"
    )


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
        and unit_type.startswith(
            "prolog_"
        )
    ):
        useful_fields = (
            "predicate_name",
            "predicate_arity",
            "formula",
            "is_task_5_answer",
            "has_explanation",
        )

    elif _is_ontology_structural_unit(
        unit_type
    ):
        useful_fields = (
            "entity_type",
            "entity",
            "name",
            "axiom_type",
            "restriction_type",
            "subject",
            "subclass",
            "superclass",
            "property",
            "domain",
            "range",
            "subproperty",
            "superproperty",
            "filler",
            "cardinality",
        )

    else:
        useful_fields = (
            "predicate_name",
            "formula",
            "statement",
            "target",
            "word_count",
        )

    compact = {
        field_name: data[field_name]
        for field_name in useful_fields
        if (
            field_name in data
            and data[field_name]
            is not None
        )
    }

    return compact or None


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

    if not isinstance(
        unit_type,
        str,
    ):
        unit_type = None

    compact: dict[
        str,
        Any,
    ] = {
        "artifact_id": data.get(
            "artifact_id"
        ),
        "unit_id": data.get(
            "unit_id"
        ),
        "unit_type": unit_type,
        "label": data.get(
            "label"
        ),
    }

    source_range = (
        _compact_source_range(
            data.get(
                "source_range"
            )
        )
    )

    if source_range is not None:
        compact[
            "source_range"
        ] = source_range

    structured_data = (
        _compact_structured_data(
            unit_type=unit_type,
            data=data.get(
                "structured_data"
            ),
        )
    )

    if structured_data is not None:
        compact[
            "structured_data"
        ] = structured_data

    # Structural ontology units and Task 5 formulas are sufficiently
    # represented by their compact structured_data.
    #
    # For other units, retain a small amount of raw content so the model can
    # inspect the student's submitted text/formula directly.
    if (
        not _is_ontology_structural_unit(
            unit_type
        )
        and not _is_task_5_prolog_unit(
            unit_type
        )
    ):
        content_blocks = data.get(
            "content_blocks"
        )

        if isinstance(
            content_blocks,
            list,
        ):
            compact_blocks: list[
                dict[str, Any]
            ] = []

            for block in content_blocks[:2]:
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

                block_data: dict[
                    str,
                    Any,
                ] = {
                    "content": (
                        content
                        .strip()[:700]
                    ),
                }

                block_id = block.get(
                    "block_id"
                )

                if isinstance(
                    block_id,
                    str,
                ):
                    block_data[
                        "block_id"
                    ] = block_id

                block_type = block.get(
                    "block_type"
                )

                if isinstance(
                    block_type,
                    str,
                ):
                    block_data[
                        "block_type"
                    ] = block_type

                block_range = (
                    _compact_source_range(
                        block.get(
                            "source_range"
                        )
                    )
                )

                if (
                    block_range
                    is not None
                ):
                    block_data[
                        "source_range"
                    ] = block_range

                compact_blocks.append(
                    block_data
                )

            if compact_blocks:
                compact[
                    "content_blocks"
                ] = compact_blocks

    return {
        key: value
        for key, value
        in compact.items()
        if value not in (
            None,
            "",
            [],
            {},
        )
    }


def _compact_candidate_units(
    values: list[Any],
) -> list[dict[str, Any]]:
    return [
        compact
        for value in values
        if (
            compact := (
                _compact_candidate_unit(
                    value
                )
            )
        )
    ]


# ============================================================
# Processing-check compaction
# ============================================================


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

    return {
        field_name: data[field_name]
        for field_name in (
            "check_type",
            "status",
            "tool",
            "summary",
        )
        if (
            field_name in data
            and data[field_name]
            not in (
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
            compact := (
                _compact_processing_check(
                    value
                )
            )
        )
    ]


# ============================================================
# Existing-feedback compaction
# ============================================================


def _compact_original_criterion(
    value: CriterionAssessment,
) -> dict[str, Any]:
    """
    Keep only the original judgement that the reflector must audit.

    Evidence is intentionally omitted because candidate student work is
    supplied separately as the authoritative evidence pool.
    """
    return {
        "requirement_id": (
            value.requirement_id
        ),
        "status": (
            value.status
        ),
        "internal_finding": (
            value.internal_finding
        ),
        "verification_status": (
            value.verification_status
        ),
        "confidence": (
            value.confidence
        ),
    }


def _compact_original_observation(
    value: FeedbackObservation,
) -> dict[str, Any]:
    """
    Keep only student-facing claims that may need correction.

    Structural IDs and duplicate evidence are omitted.
    """
    compact: dict[
        str,
        Any,
    ] = {
        "feedback_type": (
            value.feedback_type
        ),
        "student_feedback": (
            value.student_feedback
        ),
        "verification_status": (
            value.verification_status
        ),
        "confidence": (
            value.confidence
        ),
    }

    if value.suggestion is not None:
        compact[
            "suggestion"
        ] = value.suggestion

    return compact


# ============================================================
# Shared reflection context
# ============================================================


def _build_compact_context(
    *,
    candidate_units: list[Any],
    processing_checks: list[Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    compact_units = (
        _compact_candidate_units(
            candidate_units
        )
    )

    compact_checks = (
        _compact_processing_checks(
            processing_checks
        )
    )

    valid_unit_ids = [
        unit["unit_id"]
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
            unit[
                "artifact_id"
            ]
            for unit in compact_units
            if isinstance(
                unit.get(
                    "artifact_id"
                ),
                str,
            )
        )
    )

    return (
        compact_units,
        compact_checks,
        valid_unit_ids,
        valid_artifact_ids,
    )


# ============================================================
# Stage 1: audit prompt
# ============================================================


def build_requirement_reflection_audit_prompt(
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
    Build the Stage-1 reflection audit prompt.

    Stage 1 decides only whether the original requirement-level assessment
    should be retained or revised.

    It must not generate revised feedback.
    """

    component_id = component[
        "component_id"
    ]

    requirement_id = requirement[
        "id"
    ]

    requirement_text = requirement.get(
        "requirement",
        "",
    )

    (
        compact_units,
        compact_checks,
        valid_unit_ids,
        valid_artifact_ids,
    ) = _build_compact_context(
        candidate_units=(
            candidate_units
        ),
        processing_checks=(
            processing_checks
        ),
    )

    system_prompt = """
You are a conservative second academic examiner.

Perform Stage 1 of a two-stage reflection process.

Your only task in Stage 1 is to AUDIT an existing requirement-level
assessment and decide whether it should be KEPT or REVISED.

Do not generate revised feedback in this stage.

The exact requirement supplied in the prompt defines the complete scope of
the audit.

DEFAULT DECISION

KEEP is the default only when the original assessment is materially correct,
within criterion scope, and supported by the supplied evidence.

REVISE when a clear material defect exists.

MATERIAL DEFECTS

A material defect includes:

- an incorrect requirement status;
- an unsupported material claim;
- overlooked supplied evidence;
- criterion leakage;
- imposing a condition absent from the exact requirement;
- preferred-solution bias;
- treating additional valid structure as a failure without an explicit
  exclusivity condition;
- conflating syntax with semantics;
- conflating semantic correctness with formatting or framework compliance;
- claiming supplied student work is absent when it is present;
- importing a neighbouring requirement;
- making a logical claim that does not follow from the supplied expression;
- materially misleading student feedback.

CRITERION SCOPE

The exact requirement is authoritative.

Do not evaluate neighbouring requirements.

Do not import another requirement simply because it belongs to the same task.

If the exact requirement concerns syntactic well-formedness, judge syntax
only. Semantic correctness must not change the status of a syntax-only
criterion.

If the exact requirement concerns one semantic property, do not require
additional semantic properties unless explicitly stated.

If the exact requirement says X is required, X together with Y still
satisfies the requirement unless the requirement explicitly says:

- only;
- exactly;
- exclusively;
- must not;
- no other.

Do not convert "requires X" into "requires X and nothing else."

Do not assume a preferred formula or operator is mandatory unless explicitly
required.

EVIDENCE DISCIPLINE

Use the supplied student evidence directly.

Structured_data is student evidence.

A formula, predicate, class, axiom, report section, or other item present in
the supplied evidence must not be described as absent.

Do not invent:

- rubric requirements;
- student content;
- evidence;
- identifiers;
- tool results;
- syntax checks;
- execution results;
- compilation results;
- ontology reasoning results;
- proof results.

Deterministic processing evidence takes precedence over unsupported
interpretation.

DECISION OUTPUT

Return only the Stage-1 audit.

If KEEP:

- should_revise = false;
- requirement_basis must be omitted;
- explain briefly in revision_reason why no material revision is required.

If REVISE:

- should_revise = true;
- requirement_basis must be an exact verbatim phrase copied from the exact
  requirement;
- revision_reason must identify the specific material defect.

Do not generate revised_status, revised feedback, revised evidence, or any
other revision patch in Stage 1.

Return exactly one JSON object matching the structured schema supplied
separately.
""".strip()

    user_prompt = f"""
STAGE 1 — AUDIT ONLY

COMPONENT ID

{component_id}

PART ID

{_json_block(part_id)}

EXACT REQUIREMENT ID

{requirement_id}

EXACT REQUIREMENT TEXT

{requirement_text}

IMPORTANT

The EXACT REQUIREMENT TEXT above defines the complete criterion scope.

ORIGINAL CRITERION ASSESSMENT

{_json_block(
    _compact_original_criterion(
        initial_criterion_assessment
    )
)}

ORIGINAL STUDENT-FACING OBSERVATION

{_json_block(
    _compact_original_observation(
        initial_observation
    )
)}

CANDIDATE STUDENT WORK

{_json_block(compact_units)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(compact_checks)}

VALID UNIT IDS

{_json_block(valid_unit_ids)}

VALID ARTIFACT IDS

{_json_block(valid_artifact_ids)}

AUDIT PROCEDURE

Perform these checks in order.

1. EXACT-SCOPE CHECK

Identify exactly what the requirement asks.

Do not add conditions absent from its wording.

Ask whether the original assessment evaluates that exact property.

If the original finding materially criticises a different property, this is
criterion leakage and requires revision.

2. EVIDENCE CHECK

Inspect the supplied student work directly rather than trusting the original
assessment's interpretation.

Evidence present in structured_data counts as present student evidence.

Do not describe supplied work as absent.

3. EXCLUSIVITY CHECK

Check whether the original assessment invented an exclusivity condition.

If the requirement requires X and the student provides X plus Y, the
requirement is not failed merely because Y is also present unless Y is
explicitly forbidden.

4. CRITERION-TYPE CHECK

Keep distinct dimensions separate, including:

- syntax and well-formedness;
- semantic correctness;
- task compliance;
- framework compliance;
- presence or absence;
- explanation quality.

A failure in one dimension does not automatically imply failure in another.

5. LOGICAL-CLAIM CHECK

Check whether each material statement in the original finding actually
follows from the supplied formula, axiom, report text, or deterministic
evidence.

Do not accept a confident explanation merely because it sounds plausible.

6. STATUS CHECK

After completing the earlier checks, decide whether the original status is
still justified for this exact requirement.

KEEP only if the original assessment survives every material check.

REVISE if a material defect is identified.

FINAL INTERNAL CHECK BEFORE KEEP

Before setting should_revise=false, verify:

- the original finding evaluates only this exact requirement;
- it does not import a neighbouring requirement;
- it does not impose unstated exclusivity;
- it does not confuse syntax and semantics;
- it does not ignore supplied structured evidence;
- its logical claims follow from the supplied evidence;
- the status would remain the same if every neighbouring criterion were
  hidden.

If any of these checks reveals a material defect, set should_revise=true.

OUTPUT

Return the Stage-1 audit only.

Do not generate a revision patch.

Return exactly one JSON object.
""".strip()

    return (
        system_prompt,
        user_prompt,
    )


# ============================================================
# Stage 2: revision prompt
# ============================================================


def build_requirement_reflection_revision_prompt(
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
    audit_analysis: ReflectionAnalysis,
    global_requirements: list[dict[str, Any]],
    feedback_requirements: list[str],
    specification_notes: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Build the Stage-2 revision prompt.

    This function is called only when Stage 1 has already established that
    the original assessment contains a material defect.

    Stage 2 generates the smallest complete correction.
    """

    component_id = component[
        "component_id"
    ]

    requirement_id = requirement[
        "id"
    ]

    requirement_text = requirement.get(
        "requirement",
        "",
    )

    (
        compact_units,
        compact_checks,
        valid_unit_ids,
        valid_artifact_ids,
    ) = _build_compact_context(
        candidate_units=(
            candidate_units
        ),
        processing_checks=(
            processing_checks
        ),
    )

    system_prompt = """
You are performing Stage 2 of a two-stage academic feedback reflection
process.

Stage 1 has already determined that the original requirement-level
assessment contains a material defect.

Your task is now to generate the SMALLEST COMPLETE REVISION necessary to
correct that defect.

Do not re-audit whether revision is needed.

The Stage-1 audit decision is authoritative for this call.

SCOPE

Revise only the exact requirement supplied in the prompt.

Do not introduce neighbouring rubric requirements.

Do not add optional improvements unless necessary to correct the identified
defect.

Preserve every correct part of the original assessment.

CRITERION DISCIPLINE

The revised status and feedback must evaluate only the exact requirement.

Do not:

- judge semantics when the exact criterion is syntax-only;
- judge syntax when the criterion is semantic;
- impose unstated exclusivity;
- import conditions from another criterion;
- require a preferred formulation unless mandated;
- claim supplied evidence is absent when it is present.

STUDENT-FACING FEEDBACK

The revised_student_feedback is shown directly to the student.

It must discuss only the student's submitted work and the exact current
criterion.

Never mention:

- the original assessment;
- the previous assessment;
- the previous feedback;
- the audit;
- reflection;
- Stage 1;
- Stage 2;
- a correction made by the system.

Do not explain why an earlier judgement was wrong.

For example, do not write:

"The original assessment incorrectly interpreted..."

Instead write only the corrected student-facing judgement, such as:

"Your submitted formula is well formed according to the supplied modal logic
syntax."

The revised student feedback must remain strictly within the current
criterion.

EVIDENCE

Use only valid supplied artifact and unit identifiers.

Do not invent source evidence.

Use absence evidence only when required work is demonstrably absent.

Do not claim tool verification unless deterministic processing evidence
supports it.

REVISION PATCH

Return:

- revised_status;
- revised_internal_finding;
- revised_student_feedback;
- revised_suggestion when genuinely useful;
- revised_verification_status;
- revised_confidence;
- revised_evidence.

The patch must be internally consistent.

Return exactly one JSON object matching the structured schema supplied
separately.
""".strip()

    user_prompt = f"""
STAGE 2 — GENERATE REVISION PATCH

COMPONENT ID

{component_id}

PART ID

{_json_block(part_id)}

EXACT REQUIREMENT ID

{requirement_id}

EXACT REQUIREMENT TEXT

{requirement_text}

STAGE-1 AUDIT

{_json_block(audit_analysis)}

The Stage-1 audit has already established that revision is required.

Its requirement_basis must correspond to the exact requirement above.

ORIGINAL CRITERION ASSESSMENT

{_json_block(
    _compact_original_criterion(
        initial_criterion_assessment
    )
)}

ORIGINAL STUDENT-FACING OBSERVATION

{_json_block(
    _compact_original_observation(
        initial_observation
    )
)}

CANDIDATE STUDENT WORK

{_json_block(compact_units)}

DETERMINISTIC PROCESSING CHECKS

{_json_block(compact_checks)}

VALID UNIT IDS

{_json_block(valid_unit_ids)}

VALID ARTIFACT IDS

{_json_block(valid_artifact_ids)}

REVISION TASK

Correct only the material defect identified by Stage 1.

Use the smallest change necessary.

The revised internal finding must:

- evaluate the exact requirement;
- state what the supplied evidence establishes;
- avoid neighbouring rubric criteria;
- avoid unsupported logical conclusions.

The revised student feedback must:

- be understandable to the student;
- be accurate;
- remain strictly within the current criterion;
- not append unrelated semantic, syntactic, structural, or stylistic
  criticism.

The revised suggestion:

- may be omitted when the requirement is fully met and no criterion-specific
  action is needed;
- must remain within the exact requirement when supplied;
- must not introduce another rubric condition.

The revised verification status must reflect the available evidence.

Use "supported_by_submission" only when the conclusion follows directly from
supplied student evidence.

Use "verified_by_tool" only when deterministic processing evidence actually
verifies the relevant property.

Use "inferred" where the judgement requires interpretation not directly
verified by deterministic evidence.

EVIDENCE RULES

Submission-reference evidence must use only the valid IDs listed above.

Do not invent evidence.

Prefer concise evidence.

You do not need to cite every candidate unit when a smaller set is sufficient
to support the revised judgement.

For count-based requirements, use enough evidence to support the count where
necessary.

FINAL CHECK

Before returning:

1. The patch addresses only {requirement_id}.
2. The patch directly corrects the Stage-1 defect.
3. No neighbouring criterion has been imported.
4. No unstated exclusivity has been invented.
5. Revised student feedback contains no out-of-scope caveat.
6. Evidence uses only valid supplied identifiers.
7. All required revised fields are present.
8. The response contains exactly one JSON object.
9. revised_student_feedback does not mention the original assessment,
   previous feedback, audit, reflection, or revision process.

Return only the revision patch.
""".strip()

    return (
        system_prompt,
        user_prompt,
    )