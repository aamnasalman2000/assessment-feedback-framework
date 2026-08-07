from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ANSWER_TASK_MAPPING = {
    "answer1": "part_2_task_1",
    "answer2": "part_2_task_2",
    "answer3": "part_2_task_3",
    "answer4": "part_2_task_4",
    "answer51": "part_2_task_5",
    "answer52": "part_2_task_5",
    "answer53": "part_2_task_5",
}

TASK_5_ATTEMPT_INDEX = {
    "answer51": 1,
    "answer52": 2,
    "answer53": 3,
}

EXPECTED_ANSWER_PREDICATES = tuple(
    ANSWER_TASK_MAPPING
)

ANSWER_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<name>"
    r"answer1|answer2|answer3|answer4|"
    r"answer51|answer52|answer53"
    r")\s*\(",
    re.IGNORECASE | re.MULTILINE,
)

LINE_COMMENT_PATTERN = re.compile(
    r"^\s*%(?P<content>.*)$"
)

TASK_5_HEADING_PATTERN = re.compile(
    r"^\s*%+\s*ANSWER\s+5\.(?P<index>[123])\b",
    re.IGNORECASE,
)

PLACEHOLDER_COMMENT_PATTERNS = (
    re.compile(
        r"write\s+your\s+answer\s+here",
        re.IGNORECASE,
    ),
    re.compile(
        r"give\s+a\s+formula\s+below",
        re.IGNORECASE,
    ),
    re.compile(
        r"remove\s+the\s+%",
        re.IGNORECASE,
    ),
    re.compile(
        r"answers?\s+are\s+allowed",
        re.IGNORECASE,
    ),
    re.compile(
        r"insert\s+your\s+formula",
        re.IGNORECASE,
    ),
        re.compile(
        r"that\s+is:\s*if\s+your\s+formula",
        re.IGNORECASE,
    ),
    re.compile(
        r"if\s+your\s+formula\s+holds\s+everywhere",
        re.IGNORECASE,
    ),
    re.compile(
        r"this\s+kind\s+of\s+problematic\s+sandwich",
        re.IGNORECASE,
    ),
)


@dataclass
class SourceRange:
    start_position: int
    end_position: int
    start_line: int
    end_line: int


@dataclass
class PrologClause:
    predicate_name: str
    content: str
    start_position: int
    end_position: int
    source_range: SourceRange


def _line_number_at_position(
    content: str,
    position: int,
) -> int:
    return content.count(
        "\n",
        0,
        position,
    ) + 1


def _source_range(
    content: str,
    start_position: int,
    end_position: int,
) -> SourceRange:
    start_line = _line_number_at_position(
        content,
        start_position,
    )

    if end_position <= start_position:
        end_line = start_line
    else:
        end_line = _line_number_at_position(
            content,
            end_position - 1,
        )

    return SourceRange(
        start_position=start_position,
        end_position=end_position,
        start_line=start_line,
        end_line=end_line,
    )


def _range_dict(
    source_range: SourceRange,
) -> dict[str, int]:
    return {
        "start_line": source_range.start_line,
        "end_line": source_range.end_line,
    }


def _mask_comments_and_strings(
    content: str,
) -> str:
    """
    Create a same-length source mask.

    Prolog comments and string contents are replaced with spaces while
    preserving newlines and source positions.
    """
    masked = list(content)

    position = 0
    length = len(content)

    while position < length:
        if content[position] == "%":
            while (
                position < length
                and content[position] != "\n"
            ):
                masked[position] = " "
                position += 1

            continue

        if content.startswith("/*", position):
            masked[position] = " "

            if position + 1 < length:
                masked[position + 1] = " "

            position += 2

            while position < length:
                if content.startswith(
                    "*/",
                    position,
                ):
                    masked[position] = " "

                    if position + 1 < length:
                        masked[position + 1] = " "

                    position += 2
                    break

                if masked[position] != "\n":
                    masked[position] = " "

                position += 1

            continue

        if content[position] in {
            "'",
            '"',
        }:
            quote = content[position]
            position += 1

            while position < length:
                character = content[position]

                if character == "\\":
                    if masked[position] != "\n":
                        masked[position] = " "

                    position += 1

                    if position < length:
                        if masked[position] != "\n":
                            masked[position] = " "

                        position += 1

                    continue

                if character == quote:
                    position += 1
                    break

                if masked[position] != "\n":
                    masked[position] = " "

                position += 1

            continue

        position += 1

    return "".join(masked)


def _find_clause_end(
    *,
    masked_content: str,
    start_position: int,
) -> int:
    """
    Find the terminating full stop of a Prolog clause.

    Full stops nested inside parentheses, brackets, or braces are ignored.
    """
    round_depth = 0
    square_depth = 0
    curly_depth = 0

    position = start_position

    while position < len(masked_content):
        character = masked_content[position]

        if character == "(":
            round_depth += 1

        elif character == ")":
            round_depth = max(
                0,
                round_depth - 1,
            )

        elif character == "[":
            square_depth += 1

        elif character == "]":
            square_depth = max(
                0,
                square_depth - 1,
            )

        elif character == "{":
            curly_depth += 1

        elif character == "}":
            curly_depth = max(
                0,
                curly_depth - 1,
            )

        elif (
            character == "."
            and round_depth == 0
            and square_depth == 0
            and curly_depth == 0
        ):
            return position + 1

        position += 1

    return len(masked_content)


def _find_answer_clauses(
    content: str,
    masked_content: str,
) -> list[PrologClause]:
    clauses: list[PrologClause] = []

    for match in ANSWER_PATTERN.finditer(
        masked_content
    ):
        predicate_name = (
            match.group("name").lower()
        )

        start_position = match.start("name")
        end_position = _find_clause_end(
            masked_content=masked_content,
            start_position=start_position,
        )

        clause_content = content[
            start_position:end_position
        ].strip()

        if not clause_content:
            continue

        clauses.append(
            PrologClause(
                predicate_name=predicate_name,
                content=clause_content,
                start_position=start_position,
                end_position=end_position,
                source_range=_source_range(
                    content,
                    start_position,
                    end_position,
                ),
            )
        )

    return clauses


def _extract_formula(
    clause_content: str,
) -> str | None:
    """
    Extract the argument supplied to an answer predicate.

    This keeps nested Prolog terms intact.
    """
    open_parenthesis = clause_content.find("(")

    if open_parenthesis < 0:
        return None

    round_depth = 0
    quote: str | None = None
    escaped = False

    for position in range(
        open_parenthesis,
        len(clause_content),
    ):
        character = clause_content[position]

        if escaped:
            escaped = False
            continue

        if character == "\\":
            escaped = True
            continue

        if quote is not None:
            if character == quote:
                quote = None

            continue

        if character in {
            "'",
            '"',
        }:
            quote = character
            continue

        if character == "(":
            round_depth += 1
            continue

        if character == ")":
            round_depth -= 1

            if round_depth == 0:
                return clause_content[
                    open_parenthesis + 1:position
                ].strip()

    return None


def _normalise_comment_line(
    line: str,
) -> str:
    match = LINE_COMMENT_PATTERN.match(line)

    if match is None:
        return ""

    return match.group(
        "content"
    ).strip()


def _is_instructional_comment(
    text: str,
) -> bool:
    if not text:
        return True

    return any(
        pattern.search(text)
        for pattern in PLACEHOLDER_COMMENT_PATTERNS
    )


def _task_5_comment_region(
    *,
    content: str,
    attempt_index: int,
    clause_start_position: int,
) -> tuple[str | None, SourceRange | None]:
    """
    Extract student-authored comments between an ANSWER 5.x heading and
    the corresponding answer predicate.

    Template instructions and empty placeholders are excluded.
    """
    lines = content.splitlines(
        keepends=True
    )

    line_starts: list[int] = []
    offset = 0

    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    clause_start_line = (
        _line_number_at_position(
            content,
            clause_start_position,
        )
    )

    heading_line_index: int | None = None

    for index in range(
        clause_start_line - 2,
        -1,
        -1,
    ):
        line = lines[index]

        match = TASK_5_HEADING_PATTERN.match(
            line
        )

        if match is None:
            continue

        if int(match.group("index")) == attempt_index:
            heading_line_index = index

        break

    if heading_line_index is None:
        return None, None

    student_comment_lines: list[
        tuple[int, str]
    ] = []

    for index in range(
        heading_line_index + 1,
        clause_start_line - 1,
    ):
        line = lines[index]
        normalised = _normalise_comment_line(
            line
        )

        if not normalised:
            continue

        if _is_instructional_comment(
            normalised
        ):
            continue

        student_comment_lines.append(
            (
                index,
                normalised,
            )
        )

    if not student_comment_lines:
        return None, None

    start_line_index = (
        student_comment_lines[0][0]
    )

    end_line_index = (
        student_comment_lines[-1][0]
    )

    explanation = "\n".join(
        text
        for _, text in student_comment_lines
    ).strip()

    if not explanation:
        return None, None

    start_position = line_starts[
        start_line_index
    ]

    end_position = (
        line_starts[end_line_index]
        + len(lines[end_line_index])
    )

    return (
        explanation,
        _source_range(
            content,
            start_position,
            end_position,
        ),
    )


def _task_mapping(
    predicate_name: str,
) -> dict[str, Any]:
    component_id = ANSWER_TASK_MAPPING[
        predicate_name
    ]

    return {
        "part_ids": [
            "part_2_prolog_modal_logic",
        ],
        "task_ids": [
            component_id,
        ],
        "source": "declared",
        "confidence": 1.0,
        "evidence": [
            (
                f"Predicate {predicate_name}/1 "
                f"maps to {component_id}."
            ),
        ],
    }


def _attempt_record(
    predicate_name: str,
) -> dict[str, Any] | None:
    attempt_index = TASK_5_ATTEMPT_INDEX.get(
        predicate_name
    )

    if attempt_index is None:
        return None

    return {
        "attempt_index": attempt_index,
        "attempt_group_id": (
            "part_2_task_5_examples"
        ),
        "declared_as_alternative": True,
    }


def _build_answer_unit(
    *,
    clause: PrologClause,
    artifact_id: str,
    unit_index: int,
    full_content: str,
) -> tuple[
    dict[str, Any],
    bool,
]:
    unit_id = (
        f"{artifact_id}_unit_{unit_index:03d}"
    )

    formula = _extract_formula(
        clause.content
    )

    content_blocks: list[dict[str, Any]] = [
        {
            "block_id": (
                f"{unit_id}_clause"
            ),
            "block_type": "code",
            "content": clause.content,
            "language": "prolog",
            "source_range": _range_dict(
                clause.source_range
            ),
        }
    ]

    if formula:
        content_blocks.append(
            {
                "block_id": (
                    f"{unit_id}_formula"
                ),
                "block_type": "formula",
                "content": formula,
                "language": "prolog",
                "source_range": _range_dict(
                    clause.source_range
                ),
            }
        )

    task_5_explanation: str | None = None
    explanation_range: (
        SourceRange | None
    ) = None

    attempt_index = (
        TASK_5_ATTEMPT_INDEX.get(
            clause.predicate_name
        )
    )

    if attempt_index is not None:
        (
            task_5_explanation,
            explanation_range,
        ) = _task_5_comment_region(
            content=full_content,
            attempt_index=attempt_index,
            clause_start_position=(
                clause.start_position
            ),
        )

        if (
            task_5_explanation is not None
            and explanation_range is not None
        ):
            content_blocks.append(
                {
                    "block_id": (
                        f"{unit_id}_explanation"
                    ),
                    "block_type": (
                        "explanation"
                    ),
                    "content": (
                        task_5_explanation
                    ),
                    "language": "en",
                    "source_range": (
                        _range_dict(
                            explanation_range
                        )
                    ),
                }
            )

    unit: dict[str, Any] = {
        "unit_id": unit_id,
        "unit_type": (
            "prolog_task_5_answer"
            if attempt_index is not None
            else "prolog_answer"
        ),
        "label": clause.predicate_name,
        "task_mapping": _task_mapping(
            clause.predicate_name
        ),
        "source_range": _range_dict(
            clause.source_range
        ),
        "content_blocks": content_blocks,
        "structured_data": {
            "predicate_name": (
                clause.predicate_name
            ),
            "predicate_arity": 1,
            "formula": formula,
            "is_task_5_answer": (
                attempt_index is not None
            ),
            "has_explanation": (
                task_5_explanation is not None
            ),
        },
        "extraction_confidence": 0.98,
    }

    attempt = _attempt_record(
        clause.predicate_name
    )

    if attempt is not None:
        unit["attempt"] = attempt

    return (
        unit,
        task_5_explanation is not None,
    )


def _build_diagnostics(
    *,
    units: list[dict[str, Any]],
    found_predicates: set[str],
    task_5_explanations: dict[str, bool],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    diagnostic_index = 1

    for predicate_name in (
        EXPECTED_ANSWER_PREDICATES
    ):
        if predicate_name in found_predicates:
            continue

        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "warning",
                "diagnostic_type": (
                    "missing_answer_predicate"
                ),
                "message": (
                    "The expected Prolog predicate "
                    f"{predicate_name}/1 was not "
                    "identified."
                ),
                "related_unit_ids": [],
            }
        )

        diagnostic_index += 1

    unit_lookup = {
        unit["label"]: unit["unit_id"]
        for unit in units
        if isinstance(
            unit.get("label"),
            str,
        )
    }

    for predicate_name in (
        "answer51",
        "answer52",
        "answer53",
    ):
        if predicate_name not in found_predicates:
            continue

        if task_5_explanations.get(
            predicate_name,
            False,
        ):
            continue

        related_unit_id = unit_lookup.get(
            predicate_name
        )

        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "warning",
                "diagnostic_type": (
                    "missing_task_5_explanation"
                ),
                "message": (
                    "No student-authored Task 5 "
                    "explanation was identified for "
                    f"{predicate_name}/1."
                ),
                "related_unit_ids": (
                    [related_unit_id]
                    if related_unit_id
                    else []
                ),
            }
        )

        diagnostic_index += 1

    if not units:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "no_answers_extracted"
                ),
                "message": (
                    "No supported Assessment 2 "
                    "answer predicates were extracted "
                    "from the Prolog source."
                ),
                "related_unit_ids": [],
            }
        )

    return diagnostics


def process_prolog_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact_id = artifact["artifact_id"]
    artifact_type = artifact["artifact_type"]
    content = artifact["content"][
        "raw_content"
    ]

    masked_content = (
        _mask_comments_and_strings(
            content
        )
    )

    clauses = _find_answer_clauses(
        content,
        masked_content,
    )

    units: list[dict[str, Any]] = []
    task_5_explanations: dict[
        str,
        bool,
    ] = {}

    for unit_index, clause in enumerate(
        clauses,
        start=1,
    ):
        (
            unit,
            has_explanation,
        ) = _build_answer_unit(
            clause=clause,
            artifact_id=artifact_id,
            unit_index=unit_index,
            full_content=content,
        )

        units.append(unit)

        if (
            clause.predicate_name
            in TASK_5_ATTEMPT_INDEX
        ):
            task_5_explanations[
                clause.predicate_name
            ] = has_explanation

    found_predicates = {
        clause.predicate_name
        for clause in clauses
    }

    diagnostics = _build_diagnostics(
        units=units,
        found_predicates=found_predicates,
        task_5_explanations=(
            task_5_explanations
        ),
    )

    if not units:
        status = "failed"

    elif any(
        diagnostic["severity"] == "error"
        for diagnostic in diagnostics
    ):
        status = "partial"

    elif len(found_predicates) < len(
        EXPECTED_ANSWER_PREDICATES
    ):
        status = "partial"

    else:
        status = "success"

    checks = [
        {
            "check_type": (
                "prolog_textual_extraction"
            ),
            "status": (
                "passed"
                if units
                else "failed"
            ),
            "tool": (
                "assessment-feedback-"
                "preprocessor"
            ),
            "summary": (
                f"Extracted {len(units)} "
                "Assessment 2 answer "
                "predicate(s)."
            ),
        },
        {
            "check_type": "prolog_parse",
            "status": "not_run",
            "tool": None,
            "summary": (
                "A Prolog parser or interpreter "
                "was not executed during this "
                "preprocessing version."
            ),
        },
        {
            "check_type": "prolog_load",
            "status": "not_run",
            "tool": None,
            "summary": (
                "The submitted source was not "
                "loaded in SWI-Prolog during "
                "preprocessing."
            ),
        },
    ]

    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "processing": {
            "status": status,
            "extraction_mode": "textual",
            "detected_content_type": (
                "assessment_2_prolog_modal_logic"
            ),
            "tools": [
                {
                    "name": (
                        "assessment-feedback-"
                        "preprocessor"
                    ),
                    "version": "0.3.0",
                    "purpose": (
                        "Structural extraction of "
                        "Assessment 2 Prolog answer "
                        "predicates, modal formulas, "
                        "and Task 5 explanations."
                    ),
                }
            ],
            "checks": checks,
            "diagnostics": diagnostics,
        },
        "units": units,
    }