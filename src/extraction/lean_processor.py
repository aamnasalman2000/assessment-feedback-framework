from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_DECLARATIONS = {
    "theorem",
    "lemma",
    "example",
    "axiom",
    "opaque",
    "def",
    "abbrev",
}

DECLARATION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kind>theorem|lemma|example|axiom|opaque|def|abbrev)\b",
    re.MULTILINE,
)

NAMED_DECLARATION_PATTERN = re.compile(
    r"^\s*"
    r"(?:theorem|lemma|axiom|opaque|def|abbrev)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)",
)

STRUCTURAL_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kind>"
    r"theorem|lemma|example|axiom|opaque|def|abbrev|"
    r"variable|variables|"
    r"section|namespace|end|"
    r"set_option|import|open|include|omit"
    r")\b",
    re.MULTILINE,
)

SORRY_PATTERN = re.compile(
    r"\b(?:sorry|admit)\b",
)

EXPLICIT_EXPLANATION_PATTERN = re.compile(
    r"^\s*Explanation\b",
    re.IGNORECASE,
)

NAMED_EXPLANATION_PATTERN = re.compile(
    r"^\s*Explanation\s+for\s+"
    r"(?P<label>[A-Za-z_][A-Za-z0-9_']*)\s*:",
    re.IGNORECASE,
)

SECTION_NAME_PATTERN = re.compile(
    r"^\s*(?:section|namespace)"
    r"(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?",
)

END_NAME_PATTERN = re.compile(
    r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?",
)

MALFORMED_BLOCK_COMMENT_END_PATTERN = re.compile(
    r"-\*/"
)

EXPLANATION_HEADING_PATTERN = re.compile(
    r"(?im)^\s*Explanation\s+for\s+"
    r"(?P<label>[A-Za-z_][A-Za-z0-9_']*)\s*:"
)


@dataclass
class SourceRange:
    start_position: int
    end_position: int
    start_line: int
    end_line: int


@dataclass
class CommentSpan:
    comment_type: str
    raw_content: str
    normalized_content: str
    source_range: SourceRange
    is_closed: bool = True


@dataclass
class StructuralItem:
    kind: str
    start_position: int
    end_position: int
    indent: int
    content: str


@dataclass
class ContextDeclaration:
    context_id: str
    declaration_type: str
    content: str
    source_range: SourceRange
    scope_depth: int


@dataclass
class LeanDeclaration:
    declaration_type: str
    label: str | None
    start_position: int
    end_position: int
    start_line: int
    end_line: int
    content: str
    masked_content: str
    scope_path: list[str] = field(default_factory=list)
    context_declarations: list[ContextDeclaration] = field(
        default_factory=list
    )
    explanation: str | None = None
    explanation_range: SourceRange | None = None


def _line_number_at_position(
    content: str,
    position: int,
) -> int:
    return content.count("\n", 0, position) + 1


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


def _normalize_comment(
    raw_content: str,
    comment_type: str,
) -> str:
    if comment_type == "line":
        lines: list[str] = []

        for line in raw_content.splitlines():
            stripped = line.strip()

            if stripped.startswith("--"):
                stripped = stripped[2:].strip()

            lines.append(stripped)

        return "\n".join(lines).strip()

    normalized = raw_content

    if normalized.startswith("/-"):
        normalized = normalized[2:]

    if normalized.endswith("-/"):
        normalized = normalized[:-2]

    return normalized.strip()


def _scan_source(
    content: str,
) -> tuple[str, list[CommentSpan], int]:
    """
    Produce a same-length source mask.

    Code is preserved. Comments and string contents are replaced with spaces,
    while newlines are retained so positions and line numbers remain stable.

    Lean block comments may be nested.
    """
    masked = list(content)
    comments: list[CommentSpan] = []

    position = 0
    length = len(content)
    unclosed_block_comments = 0

    while position < length:
        if content.startswith("--", position):
            start = position

            while (
                position < length
                and content[position] != "\n"
            ):
                if masked[position] != "\n":
                    masked[position] = " "

                position += 1

            raw_content = content[start:position]

            comments.append(
                CommentSpan(
                    comment_type="line",
                    raw_content=raw_content,
                    normalized_content=_normalize_comment(
                        raw_content,
                        "line",
                    ),
                    source_range=_source_range(
                        content,
                        start,
                        position,
                    ),
                )
            )

            continue

        if content.startswith("/-", position):
            start = position
            depth = 0

            while position < length:
                if content.startswith("/-", position):
                    depth += 1

                    if masked[position] != "\n":
                        masked[position] = " "

                    if (
                        position + 1 < length
                        and masked[position + 1] != "\n"
                    ):
                        masked[position + 1] = " "

                    position += 2
                    continue

                if content.startswith("-/", position):
                    depth -= 1

                    if masked[position] != "\n":
                        masked[position] = " "

                    if (
                        position + 1 < length
                        and masked[position + 1] != "\n"
                    ):
                        masked[position + 1] = " "

                    position += 2

                    if depth == 0:
                        break

                    continue

                if masked[position] != "\n":
                    masked[position] = " "

                position += 1

            is_closed = depth == 0

            if not is_closed:
                unclosed_block_comments += depth

            raw_content = content[start:position]

            comments.append(
                CommentSpan(
                    comment_type="block",
                    raw_content=raw_content,
                    normalized_content=_normalize_comment(
                        raw_content,
                        "block",
                    ),
                    source_range=_source_range(
                        content,
                        start,
                        position,
                    ),
                    is_closed=is_closed,
                )
            )

            continue

        if content[position] == '"':
            # Preserve the quote itself but mask the string's contents.
            position += 1

            while position < length:
                current = content[position]

                if current == "\\":
                    if masked[position] != "\n":
                        masked[position] = " "

                    position += 1

                    if position < length:
                        if masked[position] != "\n":
                            masked[position] = " "

                        position += 1

                    continue

                if current == '"':
                    position += 1
                    break

                if masked[position] != "\n":
                    masked[position] = " "

                position += 1

            continue

        position += 1

    return (
        "".join(masked),
        comments,
        unclosed_block_comments,
    )


def _last_code_position(
    masked_content: str,
    start_position: int,
    end_position: int,
) -> int:
    position = end_position - 1

    while position >= start_position:
        if not masked_content[position].isspace():
            return position + 1

        position -= 1

    return start_position


def _find_structural_items(
    content: str,
    masked_content: str,
) -> list[StructuralItem]:
    matches = list(
        STRUCTURAL_PATTERN.finditer(masked_content)
    )

    items: list[StructuralItem] = []

    for index, match in enumerate(matches):
        start_position = match.start()

        if index + 1 < len(matches):
            candidate_end = matches[index + 1].start()
        else:
            candidate_end = len(content)

        code_end = _last_code_position(
            masked_content,
            start_position,
            candidate_end,
        )

        if code_end <= start_position:
            continue

        items.append(
            StructuralItem(
                kind=match.group("kind"),
                start_position=start_position,
                end_position=code_end,
                indent=len(match.group("indent")),
                content=content[
                    start_position:code_end
                ].rstrip(),
            )
        )

    return items

def _merge_consecutive_line_comments(
    comments: list[CommentSpan],
) -> list[CommentSpan]:
    """Merge directly adjacent `--` comments into one logical block."""

    if not comments:
        return []

    merged: list[CommentSpan] = []

    for comment in comments:
        if comment.comment_type != "line" or not merged:
            merged.append(comment)
            continue

        previous = merged[-1]

        if previous.comment_type != "line":
            merged.append(comment)
            continue

        is_adjacent = (
            comment.source_range.start_line
            == previous.source_range.end_line + 1
        )

        if not is_adjacent:
            merged.append(comment)
            continue

        merged[-1] = CommentSpan(
            comment_type="line",
            raw_content=(
                f"{previous.raw_content.rstrip()}\n"
                f"{comment.raw_content.lstrip()}"
            ),
            normalized_content=(
                f"{previous.normalized_content.rstrip()}\n"
                f"{comment.normalized_content.lstrip()}"
            ),
            source_range=SourceRange(
                start_position=(
                    previous.source_range.start_position
                ),
                end_position=(
                    comment.source_range.end_position
                ),
                start_line=(
                    previous.source_range.start_line
                ),
                end_line=comment.source_range.end_line,
            ),
            is_closed=(
                previous.is_closed and comment.is_closed
            ),
        )

    return merged


def _declaration_label(
    declaration_type: str,
    declaration_content: str,
    example_counter: int,
) -> str | None:
    if declaration_type == "example":
        return f"example_{example_counter}"

    match = NAMED_DECLARATION_PATTERN.match(
        declaration_content
    )

    if match is None:
        return None

    return match.group("name")


def _scope_name(
    item: StructuralItem,
) -> str:
    match = SECTION_NAME_PATTERN.match(item.content)

    if match is None or match.group(1) is None:
        return f"anonymous_{item.kind}"

    return match.group(1)


def _find_declarations(
    content: str,
    masked_content: str,
) -> list[LeanDeclaration]:
    structural_items = _find_structural_items(
        content,
        masked_content,
    )

    declarations: list[LeanDeclaration] = []

    scope_path: list[str] = []
    active_contexts: list[ContextDeclaration] = []

    example_counter = 0
    context_counter = 0

    for item in structural_items:
        if item.kind in {"section", "namespace"}:
            scope_path.append(_scope_name(item))
            continue

        if item.kind == "end":
            if scope_path:
                closing_depth = len(scope_path)
                scope_path.pop()

                active_contexts = [
                    context
                    for context in active_contexts
                    if context.scope_depth < closing_depth
                ]

            continue

        if item.kind in {"variable", "variables"}:
            context_counter += 1

            active_contexts.append(
                ContextDeclaration(
                    context_id=(
                        f"context_{context_counter:03d}"
                    ),
                    declaration_type=item.kind,
                    content=item.content,
                    source_range=_source_range(
                        content,
                        item.start_position,
                        item.end_position,
                    ),
                    scope_depth=len(scope_path),
                )
            )

            continue

        if item.kind not in SUPPORTED_DECLARATIONS:
            continue

        declaration_type = item.kind

        if declaration_type == "example":
            example_counter += 1

        declaration_content = item.content

        label = _declaration_label(
            declaration_type,
            declaration_content,
            example_counter,
        )

        source_range = _source_range(
            content,
            item.start_position,
            item.end_position,
        )

        declarations.append(
            LeanDeclaration(
                declaration_type=declaration_type,
                label=label,
                start_position=item.start_position,
                end_position=item.end_position,
                start_line=source_range.start_line,
                end_line=source_range.end_line,
                content=declaration_content,
                masked_content=masked_content[
                    item.start_position:item.end_position
                ],
                scope_path=list(scope_path),
                context_declarations=list(
                    active_contexts
                ),
            )
        )

    return declarations


def _find_top_level_assignment(
    masked_content: str,
) -> int | None:
    """
    Find a declaration's top-level :=.

    Occurrences inside comments and strings have already been masked.
    """
    round_depth = 0
    square_depth = 0
    curly_depth = 0

    position = 0

    while position < len(masked_content) - 1:
        character = masked_content[position]

        if character == "(":
            round_depth += 1
        elif character == ")":
            round_depth = max(0, round_depth - 1)
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth = max(0, square_depth - 1)
        elif character == "{":
            curly_depth += 1
        elif character == "}":
            curly_depth = max(0, curly_depth - 1)

        if (
            masked_content.startswith(":=", position)
            and round_depth == 0
            and square_depth == 0
            and curly_depth == 0
        ):
            return position

        position += 1

    return None


def _split_statement_and_body(
    declaration: LeanDeclaration,
    full_content: str,
) -> tuple[
    str,
    str | None,
    SourceRange,
    SourceRange | None,
]:
    assignment_position = _find_top_level_assignment(
        declaration.masked_content
    )

    if assignment_position is None:
        statement_range = _source_range(
            full_content,
            declaration.start_position,
            declaration.end_position,
        )

        return (
            declaration.content.strip(),
            None,
            statement_range,
            None,
        )

    absolute_assignment_position = (
        declaration.start_position
        + assignment_position
    )

    statement_end = absolute_assignment_position

    body_start = absolute_assignment_position + 2

    while (
        body_start < declaration.end_position
        and full_content[body_start].isspace()
    ):
        body_start += 1

    statement = full_content[
        declaration.start_position:statement_end
    ].strip()

    proof_body = full_content[
        body_start:declaration.end_position
    ].strip()

    statement_range = _source_range(
        full_content,
        declaration.start_position,
        statement_end,
    )

    proof_range = None

    if proof_body:
        proof_range = _source_range(
            full_content,
            body_start,
            declaration.end_position,
        )

    return (
        statement,
        proof_body or None,
        statement_range,
        proof_range,
    )


def _comments_between(
    comments: list[CommentSpan],
    start_position: int,
    end_position: int,
) -> list[CommentSpan]:
    return [
        comment
        for comment in comments
        if (
            comment.source_range.start_position
            >= start_position
            and comment.source_range.end_position
            <= end_position
        )
    ]


def _attach_explanations(
    declarations: list[LeanDeclaration],
    comments: list[CommentSpan],
    content_length: int,
) -> None:
    declarations_by_label: dict[
        str,
        list[LeanDeclaration],
    ] = {}

    for declaration in declarations:
        if declaration.label is None:
            continue

        declarations_by_label.setdefault(
            declaration.label,
            [],
        ).append(declaration)

    used_comment_positions: set[int] = set()

    # First attach explicitly named explanations, even if they appear
    # much later in the source.
    for comment in comments:
        normalized = comment.normalized_content
        match = NAMED_EXPLANATION_PATTERN.match(
            normalized
        )

        if match is None:
            continue

        label = match.group("label")
        candidates = declarations_by_label.get(
            label,
            [],
        )

        if not candidates:
            continue

        preceding_candidates = [
            declaration
            for declaration in candidates
            if (
                declaration.start_position
                < comment.source_range.start_position
            )
        ]

        if preceding_candidates:
            target = preceding_candidates[-1]
        else:
            target = candidates[0]

        target.explanation = normalized
        target.explanation_range = (
            comment.source_range
        )

        used_comment_positions.add(
            comment.source_range.start_position
        )

    # Then attach a generic "Explanation:" comment when it occurs
    # after a declaration and before the next declaration.
    for index, declaration in enumerate(declarations):
        if declaration.explanation is not None:
            continue

        if index + 1 < len(declarations):
            next_start = declarations[
                index + 1
            ].start_position
        else:
            next_start = content_length

        trailing_comments = _comments_between(
            comments,
            declaration.end_position,
            next_start,
        )

        for comment in trailing_comments:
            if (
                comment.source_range.start_position
                in used_comment_positions
            ):
                continue

            normalized = comment.normalized_content

            if not EXPLICIT_EXPLANATION_PATTERN.match(
                normalized
            ):
                continue

            declaration.explanation = normalized
            declaration.explanation_range = (
                comment.source_range
            )

            used_comment_positions.add(
                comment.source_range.start_position
            )

            break


def _unit_type_for_declaration(
    declaration_type: str,
) -> str:
    mapping = {
        "theorem": "lean_theorem",
        "lemma": "lean_lemma",
        "example": "lean_example",
        "axiom": "lean_axiom",
        "opaque": "lean_opaque_declaration",
        "def": "lean_definition",
        "abbrev": "lean_abbreviation",
    }

    return mapping.get(
        declaration_type,
        "lean_declaration",
    )


def _range_dict(
    source_range: SourceRange,
) -> dict[str, int]:
    return {
        "start_line": source_range.start_line,
        "end_line": source_range.end_line,
    }


def _build_unit(
    *,
    declaration: LeanDeclaration,
    artifact_id: str,
    unit_index: int,
    full_content: str,
) -> dict[str, Any]:
    unit_id = f"{artifact_id}_unit_{unit_index:03d}"

    (
        statement,
        proof_body,
        statement_range,
        proof_range,
    ) = _split_statement_and_body(
        declaration,
        full_content,
    )

    content_blocks: list[dict[str, Any]] = [
        {
            "block_id": f"{unit_id}_statement",
            "block_type": "formal_statement",
            "content": statement,
            "language": "lean",
            "source_range": _range_dict(
                statement_range
            ),
        }
    ]

    if proof_body is not None and proof_range is not None:
        content_blocks.append(
            {
                "block_id": f"{unit_id}_proof",
                "block_type": "proof_body",
                "content": proof_body,
                "language": "lean",
                "source_range": _range_dict(
                    proof_range
                ),
            }
        )

    if (
        declaration.explanation
        and declaration.explanation_range
    ):
        content_blocks.append(
            {
                "block_id": (
                    f"{unit_id}_explanation"
                ),
                "block_type": "explanation",
                "content": declaration.explanation,
                "language": "en",
                "source_range": _range_dict(
                    declaration.explanation_range
                ),
            }
        )

    contains_sorry = bool(
        SORRY_PATTERN.search(
            declaration.masked_content
        )
    )

    context_data = [
        {
            "context_id": context.context_id,
            "declaration_type": (
                context.declaration_type
            ),
            "content": context.content,
            "source_range": _range_dict(
                context.source_range
            ),
        }
        for context in declaration.context_declarations
    ]

    return {
        "unit_id": unit_id,
        "unit_type": _unit_type_for_declaration(
            declaration.declaration_type
        ),
        "label": declaration.label,
        "source_range": {
            "start_line": declaration.start_line,
            "end_line": declaration.end_line,
        },
        "content_blocks": content_blocks,
        "structured_data": {
            "declaration_type": (
                declaration.declaration_type
            ),
            "contains_sorry": contains_sorry,
            "scope_path": declaration.scope_path,
            "context_declarations": context_data,
            "proof_style": (
                "none"
                if proof_body is None
                else (
                    "tactic"
                    if proof_body.lstrip().startswith("by")
                    else "term"
                )
            ),
        },
        "extraction_confidence": 0.9,
    }
 
def _recover_named_explanations(
    *,
    declarations: list[LeanDeclaration],
    comments: list[CommentSpan],
    full_content: str,
) -> None:
    declarations_by_label = {
        declaration.label: declaration
        for declaration in declarations
        if declaration.label is not None
    }

    for comment in comments:
        if comment.is_closed:
            continue

        text = comment.normalized_content
        matches = list(
            EXPLANATION_HEADING_PATTERN.finditer(text)
        )

        for index, match in enumerate(matches):
            label = match.group("label")
            declaration = declarations_by_label.get(label)

            if declaration is None:
                continue

            if index + 1 < len(matches):
                explanation_end = matches[index + 1].start()
            else:
                explanation_end = len(text)

            explanation = text[
                match.start():explanation_end
            ].strip()

            # Remove a malformed terminator when present, but retain
            # a diagnostic elsewhere.
            explanation = re.sub(
                r"\s*-\*/\s*(?:/-\s*)?\Z",
                "",
                explanation,
            ).rstrip()
            
            # The final malformed comment in this submission ends with a
            # lone `-` rather than either `-/` or `-*/`.
            if (
                not comment.is_closed
                and explanation.endswith("\n-")
            ):
                explanation = explanation[:-2].rstrip()

            relative_start = (
                comment.raw_content.find(
                    text[match.start():match.end()]
                )
            )

            if relative_start < 0:
                relative_start = 0

            absolute_start = (
                comment.source_range.start_position
                + relative_start
            )

            declaration.explanation = explanation
            declaration.explanation_range = _source_range(
                full_content,
                absolute_start,
                min(
                    comment.source_range.end_position,
                    absolute_start + len(explanation),
                ),
            )    


def _build_diagnostics(
    *,
    units: list[dict[str, Any]],
    unclosed_block_comment_count: int,
    malformed_terminator_count: int,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    diagnostic_index = 1

    for unit in units:
        structured_data = unit.get(
            "structured_data",
            {},
        )

        if not structured_data.get("contains_sorry"):
            continue

        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "warning",
                "diagnostic_type": "lean_sorry",
                "message": (
                    "The declaration contains a `sorry` "
                    "or `admit` placeholder and is "
                    "therefore incomplete."
                ),
                "source_range": unit["source_range"],
                "related_unit_ids": [
                    unit["unit_id"]
                ],
            }
        )

        diagnostic_index += 1

    if unclosed_block_comment_count > 0:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "unclosed_block_comment"
                ),
                "message": (
                    "The Lean source contains "
                    f"{unclosed_block_comment_count} "
                    "unclosed block comment(s)."
                ),
                "related_unit_ids": [],
            }
        )

        diagnostic_index += 1
    if malformed_terminator_count > 0:
        diagnostics.append(
        {
            "diagnostic_id": (
                f"diagnostic_{diagnostic_index:03d}"
            ),
            "severity": "error",
            "diagnostic_type": (
                "malformed_block_comment_terminator"
            ),
            "message": (
                f"Found {malformed_terminator_count} "
                "occurrence(s) of `-*/`. Lean block "
                "comments must end with `-/`."
            ),
            "related_unit_ids": [],
        }
    )

    diagnostic_index += 1

    if not units:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "warning",
                "diagnostic_type": (
                    "no_declarations_extracted"
                ),
                "message": (
                    "No supported Lean declarations were "
                    "identified in the source file."
                ),
                "related_unit_ids": [],
            }
        )

    return diagnostics


def process_lean_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact_id = artifact["artifact_id"]
    artifact_type = artifact["artifact_type"]
    content = artifact["content"]["raw_content"]
    malformed_terminator_count = len(
        MALFORMED_BLOCK_COMMENT_END_PATTERN.findall(content)
    )

    (
        masked_content,
        comments,
        unclosed_block_comment_count,
    ) = _scan_source(content)
    
    comments = _merge_consecutive_line_comments(comments)

    declarations = _find_declarations(
        content,
        masked_content,
    )

    _attach_explanations(
        declarations,
        comments,
        len(content),
    )
    
    _recover_named_explanations(
    declarations=declarations,
    comments=comments,
    full_content=content,
)

    units = [
        _build_unit(
            declaration=declaration,
            artifact_id=artifact_id,
            unit_index=index,
            full_content=content,
        )
        for index, declaration in enumerate(
            declarations,
            start=1,
        )
    ]

    diagnostics = _build_diagnostics(
        units=units,
        unclosed_block_comment_count=(
            unclosed_block_comment_count
        ),
        malformed_terminator_count=malformed_terminator_count,
    )

    contains_error = any(
        diagnostic["severity"] == "error"
        for diagnostic in diagnostics
    )

    if contains_error:
        status = "partial"
    elif units:
        status = "success"
    else:
        status = "failed"

    checks = [
        {
            "check_type": "lean_textual_extraction",
            "status": (
                "passed"
                if units
                else "failed"
            ),
            "tool": (
                "assessment-feedback-preprocessor"
            ),
            "summary": (
                f"Extracted {len(units)} Lean "
                "declaration(s)."
            ),
        },
        {
            "check_type": "lean_compile",
            "status": "not_run",
            "tool": None,
            "summary": (
                "Lean compilation is not included in "
                "this preprocessing version."
            ),
        },
    ]

    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "processing": {
            "status": status,
            "extraction_mode": "textual",
            "detected_content_type": "lean_4_source",
            "tools": [
                {
                    "name": (
                        "assessment-feedback-preprocessor"
                    ),
                    "version": "0.2.0",
                    "purpose": (
                        "Structural textual extraction "
                        "of Lean declarations, context, "
                        "comments, and proof bodies."
                    ),
                }
            ],
            "checks": checks,
            "diagnostics": diagnostics,
        },
        "units": units,
    }