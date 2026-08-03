from __future__ import annotations

import json
from typing import Any

from .semantic_taxonomy import (
    LEAN_STRATEGY_CATEGORY_MAP,
    LEAN_STRATEGY_NAMES,
    STRATEGY_CATEGORIES,
)


SYSTEM_PROMPT = """
You are a semantic analysis component in an academic assessment
feedback pipeline.

Your task is to analyse a student's submission using:

1. the original raw source;
2. the deterministic processed representation;
3. optional assessment metadata;
4. an optional rubric.

Return only structured data that conforms exactly to the supplied JSON
schema.

Your role is semantic interpretation, not deterministic extraction.
The processed representation is the authoritative source for unit IDs,
content block IDs, context IDs, source ranges, and structural metadata.
"""


SEMANTIC_EXTRACTION_RULES = """
GENERAL REQUIREMENTS

- Produce exactly one semantic annotation for every unit listed in the
  processed submission.
- Do not omit any processed unit.
- Do not produce more than one annotation for the same unit.
- Do not create annotations for units that do not exist.
- Preserve unit IDs exactly as provided.
- Preserve context IDs and content block IDs exactly as provided.
- Never invent unit IDs, context IDs, or content block IDs.
- Region IDs may be created only when defining a corresponding entry in
  document_regions.
- Every region ID referenced by a unit annotation must exist in
  document_regions.
- Do not infer administrative metadata such as model names, extractor
  versions, or submission IDs. These values are controlled by the
  application.
- Copy every unit_id exactly as written in the VALID UNIT IDS section.
- Do not shorten, rewrite, normalize, or infer unit IDs.
- Copy the complete unit ID string verbatim, including every prefix and
  separator.

EVIDENCE AND TRACEABILITY

- Base every semantic claim on the raw source or processed
  representation.
- Use evidence references wherever the source supports the inference.
- Use the smallest relevant source range or structural reference.
- Evidence excerpts must accurately reproduce the relevant source.
- Do not invent quotations or source ranges.
- Distinguish explicit evidence from interpretation.
- Do not infer meaning solely from a unit's label, sequence number, or
  position in the file.

UNIT ANALYSIS

For every processed unit:

- identify what proposition, definition, implementation, explanation,
  result, or answer the unit contains;
- identify what the student is attempting to accomplish;
- identify the reasoning, proof, computational, modelling, analytical,
  or presentation strategies used;
- identify context declarations that are genuinely relevant;
- distinguish declarations that are merely in scope from declarations
  actually used by the unit;
- describe how the unit contributes to the assessment task;
- produce a specific summary grounded in the unit's actual content.

Do not produce generic summaries such as:

- "This unit explains the example."
- "This unit contains an assumption."
- "This unit interprets the problem."
- "This is the next part of the answer."

A useful summary must explain what the student actually states, proves,
implements, derives, or attempts.

STUDENT COMMENTS

Student-authored comments are important semantic evidence.

For every substantive comment:

- determine whether it is a heading, task description, explanation,
  strategy, step explanation, implementation note, interpretation,
  uncertainty statement, TODO, or another meaningful role;
- connect it to the relevant unit or units;
- preserve expressed uncertainty, confusion, justification, or
  self-evaluation;
- compare claims in comments with the corresponding formal content;
- identify inconsistencies between comments and code.

Do not classify a comment as decorative unless it is purely formatting
or a separator.

DOCUMENT REGIONS

Create document regions only when the raw source provides evidence of a
meaningful grouping, such as:

- an assessment part;
- a question;
- a scenario;
- a task;
- a section or subsection;
- supporting material.

Use section declarations, headings, task descriptions, and source
ranges as evidence.

A region may contain one or several units. Do not create unnecessary
regions merely to populate the field.

RELATIONSHIPS

Record relationships only when they are supported by the submission.

Examples include:

- one unit answering a task;
- one unit defining material used by another;
- one unit depending on or using another;
- an explanation supporting a formal solution;
- one unit extending, evaluating, contradicting, or providing an
  alternative to another.

Do not infer a relationship merely because two units appear next to
each other.

CONFIDENCE

Confidence must reflect both:

1. the clarity of the source evidence; and
2. the completeness of the interpretation.

- Use high confidence only for claims directly and clearly supported by
  the source.
- Do not assign confidence above 0.90 to vague, generic, incomplete, or
  weakly evidenced interpretations.
- Reduce confidence when source content is ambiguous or incomplete.
- A structurally valid response is not sufficient justification for
  high confidence.

UNRESOLVED OBSERVATIONS AND WARNINGS

Use unresolved_observations for meaningful semantic uncertainty that
cannot be safely resolved from the supplied evidence.

Use warnings for issues such as:

- incomplete-looking work;
- inconsistencies between comments and formal content;
- unclear intent;
- ambiguous relationships;
- insufficient evidence.

Do not mention nonexistent unit IDs, context IDs, content block IDs, or
region IDs in unresolved observations or warnings.
"""


LEAN_ANALYSIS_RULES = """
LEAN-SPECIFIC REQUIREMENTS

For each Lean declaration containing a formal statement and proof body:

- set student_intent unless no reasonable interpretation is supported;
- set strategy unless no proof or reasoning approach is visible;
- identify the proposition being proved;
- identify the structure of the assumptions and target;
- identify the proof techniques actually used;
- identify context declarations explicitly used by the proof;
- distinguish used declarations from declarations merely available in
  the surrounding section;
- analyse student comments associated with the proof;
- identify mismatches between inline comments and formal proposition
  types.

Relevant Lean proof techniques may include:

- implication introduction;
- direct implication application;
- conjunction decomposition;
- conjunction construction;
- disjunction introduction;
- disjunction elimination;
- case analysis;
- nested case analysis;
- pattern matching;
- contradiction;
- negation introduction;
- intermediate derivation;
- local binding with let;
- rewriting;
- induction;
- use of an existing theorem or hypothesis.

PROOF COMPLETENESS

Inspect whether the extracted proof body visibly produces or constructs
a value corresponding to the declared target.

Distinguish between:

- a proof that visibly reaches and constructs the target;
- a partial proof that only decomposes assumptions or derives
  intermediate facts;
- an explicitly unfinished proof;
- a proof whose completeness cannot be determined from textual evidence.

Do not claim that a Lean proof compiles or is formally correct unless
compiler results are explicitly supplied.

When compilation has not been run, describe only what is visible in the
source, for example:

- "The extracted proof visibly constructs the target."
- "The proof body stops after decomposing the assumptions."
- "The proof derives intermediate propositions but does not visibly
  construct the final conjunction."

COMMENTS AND FORMAL CONTENT

Check whether comments correctly describe formal expressions.

For example, if the formal type is:

    f → ¬r

but an inline comment says:

    f → r

record this as a comment-to-code inconsistency.

Do not treat an incorrect comment as changing the actual formal type.

STUDENT UNCERTAINTY

Comments such as:

- "I do not see how this can be proven."
- "This appears contradictory."
- "I am unsure about this step."

must be captured as student uncertainty or interpretation.

Do not convert the student's uncertainty into a factual conclusion.
Distinguish:

- what the student believes;
- what the formal statement contains;
- what the proof attempt actually does.
"""


def _json_text(value: Any) -> str:
    if value is None:
        return "Not provided."

    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _collect_valid_identifiers(
    processed_submission: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    unit_ids: list[str] = []
    context_ids: list[str] = []
    content_block_ids: list[str] = []

    for artifact in processed_submission.get("artifacts", []):
        for unit in artifact.get("units", []):
            unit_id = unit.get("unit_id")
            if isinstance(unit_id, str):
                unit_ids.append(unit_id)

            for block in unit.get("content_blocks", []):
                block_id = block.get("block_id")
                if isinstance(block_id, str):
                    content_block_ids.append(block_id)

            structured_data = unit.get("structured_data", {})

            for context in structured_data.get(
                "context_declarations",
                [],
            ):
                context_id = context.get("context_id")
                if isinstance(context_id, str):
                    context_ids.append(context_id)

    return (
        list(dict.fromkeys(unit_ids)),
        list(dict.fromkeys(context_ids)),
        list(dict.fromkeys(content_block_ids)),
    )
    
def _collect_assessment_identifiers(
    rubric: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """
    Collect valid assessment component IDs and evaluation requirement IDs
    from the assessment specification supplied through the rubric argument.
    """
    if not isinstance(rubric, dict):
        return [], []

    component_ids: list[str] = []
    requirement_ids: list[str] = []

    for part in rubric.get("parts", []):
        for component in part.get("components", []):
            component_id = component.get("component_id")

            if isinstance(component_id, str):
                component_ids.append(component_id)

            for requirement in component.get(
                "evaluation_requirements",
                [],
            ):
                requirement_id = requirement.get("id")

                if isinstance(requirement_id, str):
                    requirement_ids.append(requirement_id)

    return (
        list(dict.fromkeys(component_ids)),
        list(dict.fromkeys(requirement_ids)),
    )


def build_semantic_extraction_prompt(
    *,
    raw_source: str,
    processed_submission: dict[str, Any],
    assessment_metadata: dict[str, Any] | None = None,
    rubric: dict[str, Any] | None = None,
) -> str:
    unit_ids, context_ids, content_block_ids = (
        _collect_valid_identifiers(processed_submission)
    )

    return f"""

{SEMANTIC_EXTRACTION_RULES.strip()}

{LEAN_ANALYSIS_RULES.strip()}

VALID UNIT IDS — COPY EXACTLY

The following are the only valid processed unit IDs. Copy each complete
string exactly as shown:

{_json_text(unit_ids)}

The following are the only valid context IDs:

{_json_text(context_ids)}

The following are the only valid content block IDs:

{_json_text(content_block_ids)}

You must return exactly {len(unit_ids)} unit annotations.

The unit_annotations array must contain each valid unit ID exactly once
and must contain no other unit IDs.

ORIGINAL RAW SUBMISSION

```lean
{raw_source}
```

PROCESSED SUBMISSION

```json
{_json_text(processed_submission)}
```

ASSESSMENT METADATA

```json
{_json_text(assessment_metadata)}
```

RUBRIC

```json
{_json_text(rubric)}
```

FINAL CHECK BEFORE RESPONDING

Before returning the structured response, verify that:

- every processed unit appears exactly once in unit_annotations;
- every annotation.unit_id is copied verbatim from VALID UNIT IDS;
- no shortened, reconstructed, or unknown unit ID is present;
- every referenced context ID exists in the valid context ID list;
- every referenced content block ID exists in the valid content block
  ID list;
- every referenced region ID is defined in document_regions;
- summaries describe the actual statement or proof;
- Lean units with statements and proof bodies have an intent and
  strategy whenever these can reasonably be inferred;
- substantive student comments have been analysed;
- incomplete-looking proofs are described cautiously and are not
  claimed to compile;
- confidence values match the quality and completeness of the evidence.

Return only the structured response required by the JSON schema.
""".strip()

def build_semantic_unit_prompt(
    *,
    raw_source: str,
    unit: dict[str, Any],
    all_unit_ids: list[str],
    assessment_metadata: dict[str, Any] | None = None,
    rubric: dict[str, Any] | None = None,
) -> str:
    structured_data = unit.get("structured_data", {})

    valid_context_ids = [
        context["context_id"]
        for context in structured_data.get(
            "context_declarations",
            [],
        )
        if isinstance(context.get("context_id"), str)
    ]

    valid_content_block_ids = [
        block["block_id"]
        for block in unit.get("content_blocks", [])
        if isinstance(block.get("block_id"), str)
    ]
    
    valid_component_ids, valid_requirement_ids = (
        _collect_assessment_identifiers(rubric)
    )
    
    strategy_names = "\n".join(
        f"- {name}: category={LEAN_STRATEGY_CATEGORY_MAP[name]}"
        for name in LEAN_STRATEGY_NAMES
    )

    return f"""
{LEAN_ANALYSIS_RULES.strip()}

PER-UNIT EXTRACTION REQUIREMENTS

Analyse exactly one processed Lean unit.

The application controls the unit ID and document-level metadata.
Do not generate:

- unit_id;
- region_ids;
- source submission IDs;
- processed submission IDs;
- model names;
- extractor versions;
- document regions;
- document-level warnings.

Return only the semantic interpretation required by the supplied JSON
schema.

For this unit:

- determine the unit's semantic role;
- infer the student's intent when reasonably supported;
- identify the proof or reasoning strategy actually visible;
- identify context declarations that are genuinely used or relevant;
- analyse substantive student-authored comments;
- preserve student uncertainty;
- identify comment-to-code inconsistencies;
- describe visible proof completeness cautiously;
- assign evidence-sensitive confidence;
- do not claim that the proof compiles unless compiler evidence is
  supplied.

SEMANTIC ROLE GUIDANCE

A Lean theorem, example, or formal proof attempt that directly answers
an assessment task should normally use:

    main_answer

Do not use:

    explanation

merely because the unit contains comments or proof code.

Use another semantic role only when the unit clearly serves that
different function.

STUDENT INTENT GUIDANCE

For a Lean theorem or proof attempt, student_intent should normally be
present with:

    intent_type: prove_claim

unless the supplied evidence does not support that interpretation.

The intent description should state the proposition or logical goal the
student is attempting to establish.

STRATEGY GUIDANCE

Populate strategy whenever a visible proof approach exists.

Describe the actual structure used, such as:

- implication introduction;
- direct hypothesis application;
- conjunction decomposition;
- conjunction construction;
- disjunction introduction;
- disjunction elimination;
- case analysis;
- nested case analysis;
- contradiction;
- intermediate derivation;
- local binding.

Do not use generic labels such as "logical reasoning" when a more
specific proof strategy is visible.

STRATEGY VOCABULARY

Choose exactly one strategy name from the following controlled
vocabulary. Use the category paired with that strategy name.

{strategy_names}

Normalisation rules:

- Use "case_analysis" when the proof splits into explicit cases.
- Use "conjunction_elimination" when conjunction assumptions are
  unpacked.
- Use "conjunction_construction" only when the proof visibly constructs
  a conjunction.
- Use "decomposition_only" when the student only unpacks assumptions
  and does not proceed to a more substantive proof step.
- Use "direct_application" when an available hypothesis is directly
  applied to produce the target.
- Do not invent new strategy names.
- The strategy category must match the category paired with the selected
  strategy name.

EVIDENCE REQUIREMENTS

Both student_intent.evidence and strategy.evidence must contain at least
one evidence item.

Evidence must refer to visible material in the supplied unit, such as:

- a match expression;
- a let binding;
- a hypothesis application;
- a student-authored comment;
- a valid content block.

Use the smallest source range that supports the interpretation.

Do not use inferred or missing proof steps as evidence.
Evidence must come only from the supplied unit.

Use high confidence only when the interpretation is directly supported
by explicit evidence.

COMMENTS

Every substantive student-authored comment in this unit must be
considered.

For comments expressing uncertainty, use the uncertainty comment role.

For comments that incorrectly describe a formal expression:

- preserve the comment text faithfully;
- annotate its semantic role;
- describe the inconsistency in the summary or another appropriate
  structured description;
- do not treat the comment as changing the Lean proposition.

VALID RUBRIC REQUIREMENT IDS

Use only these requirement IDs:

{_json_text(valid_requirement_ids)}

If these lists are empty, return empty assessment_components and
rubric_requirements arrays. Never invent an assessment component ID or
requirement ID.

VALID CONTEXT IDS FOR THIS UNIT

Use only these context IDs:

{_json_text(valid_context_ids)}

If none are genuinely relevant, return an empty relevant_context array.

VALID CONTENT BLOCK IDS FOR THIS UNIT

Use only these content block IDs in evidence references:

{_json_text(valid_content_block_ids)}

VALID TARGET UNIT IDS

Relationships may refer only to these unit IDs:

{_json_text(all_unit_ids)}

Do not invent relationships merely because units appear near one
another.

ORIGINAL RAW SUBMISSION

```lean
{raw_source}
```

UNIT TO ANALYSE

```json
{_json_text(unit)}
```

ASSESSMENT METADATA

```json
{_json_text(assessment_metadata)}
```

RUBRIC

```json
{_json_text(rubric)}
```

FINAL CHECK BEFORE RESPONDING

Verify that:

semantic_role reflects the unit's function;
student_intent is populated when reasonably inferable;
strategy is populated when a proof approach is visible;
substantive comments have been analysed;
student uncertainty has not been converted into a factual claim;
comment-to-code inconsistencies are not ignored;
proof completeness is described only from visible evidence;
relevant_context contains only valid context IDs;
relationships contain only valid target unit IDs;
evidence references use valid IDs and accurate source ranges;
confidence is lower for incomplete, ambiguous, or contradictory work.

Return only the structured response required by the JSON schema.
""".strip()