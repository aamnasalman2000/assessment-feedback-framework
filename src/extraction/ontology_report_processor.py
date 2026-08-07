from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PART_ID = "part_1_ontology"

PAGE_MARKER_PATTERN = re.compile(
    r"^\[PAGE\s+(?P<page_number>\d+)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_HEADING_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<number>\d+)\s*[\.\)]?\s*"
    r"(?P<title>"
    r"overview|"
    r"justi(?:fi|ﬁ)cation|"
    r"axiom|"
    r"advantages?\s+and\s+disadvantages?|"
    r"advantages?|"
    r"disadvantages?"
    r")"
    r"[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

COVERSHEET_MARKERS = (
    "digital education service",
    "coursework submission coversheet",
    "declaration of academic integrity",
    "module code:",
    "module title:",
    "assessment title:",
    "student id",
    "submission date:",
)

FOOTER_PATTERNS = (
    re.compile(
        r"^\s*©\s*University of Leeds.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*Digital Education Service\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
)


@dataclass
class SourceRange:
    start_line: int
    end_line: int


@dataclass
class ReportSection:
    title: str
    content: str
    source_range: SourceRange
    page_numbers: list[int]


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


def _normalise_heading(
    title: str,
) -> str:
    normalised = (
        title
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )

    return " ".join(
        normalised.strip().split()
    ).title()

def _normalise_ligatures(
    text: str,
) -> str:
    """
    Replace common PDF-extraction ligatures while preserving character
    positions and line structure.
    """
    return (
        text
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )


def _page_number_at_position(
    content: str,
    position: int,
) -> int | None:
    page_number: int | None = None

    for match in PAGE_MARKER_PATTERN.finditer(
        content,
        0,
        position,
    ):
        page_number = int(
            match.group("page_number")
        )

    return page_number


def _page_numbers_for_range(
    content: str,
    start_position: int,
    end_position: int,
) -> list[int]:
    page_numbers: list[int] = []

    starting_page = _page_number_at_position(
        content,
        start_position,
    )

    if starting_page is not None:
        page_numbers.append(starting_page)

    for match in PAGE_MARKER_PATTERN.finditer(
        content,
        start_position,
        end_position,
    ):
        page_number = int(
            match.group("page_number")
        )

        if page_number not in page_numbers:
            page_numbers.append(page_number)

    return page_numbers


def _remove_page_markers(
    text: str,
) -> str:
    return PAGE_MARKER_PATTERN.sub(
        "",
        text,
    )


def _remove_footer_boilerplate(
    text: str,
) -> str:
    cleaned = text

    for pattern in FOOTER_PATTERNS:
        cleaned = pattern.sub(
            "",
            cleaned,
        )

    return cleaned


def _looks_like_coversheet(
    text: str,
) -> bool:
    lowered = text.lower()

    marker_count = sum(
        marker in lowered
        for marker in COVERSHEET_MARKERS
    )

    return marker_count >= 3


def _clean_section_text(
    text: str,
) -> str:
    cleaned = _remove_page_markers(text)
    cleaned = _remove_footer_boilerplate(
        cleaned
    )

    lines = [
        line.rstrip()
        for line in cleaned.splitlines()
    ]

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    compacted: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank and previous_blank:
            continue

        compacted.append(line)
        previous_blank = is_blank

    return "\n".join(compacted).strip()


def _find_report_start(
    content: str,
) -> int:
    """
    Skip a coversheet when the first page contains submission metadata.

    The first recognised report-section heading is treated as the start of
    the substantive report.
    """
    first_heading = SECTION_HEADING_PATTERN.search(
        content
    )

    if first_heading is None:
        return 0

    prefix = content[:first_heading.start()]

    if _looks_like_coversheet(prefix):
        return first_heading.start()

    return 0


def _extract_sections(
    content: str,
) -> list[ReportSection]:
    report_start = _find_report_start(content)

    heading_matches = list(
        SECTION_HEADING_PATTERN.finditer(
            content,
            report_start,
        )
    )

    sections: list[ReportSection] = []

    if not heading_matches:
        fallback_text = _clean_section_text(
            content[report_start:]
        )

        if fallback_text:
            sections.append(
                ReportSection(
                    title="Ontology Report",
                    content=fallback_text,
                    source_range=_source_range(
                        content,
                        report_start,
                        len(content),
                    ),
                    page_numbers=(
                        _page_numbers_for_range(
                            content,
                            report_start,
                            len(content),
                        )
                    ),
                )
            )

        return sections

    for index, heading_match in enumerate(
        heading_matches
    ):
        section_start = heading_match.start()

        if index + 1 < len(heading_matches):
            section_end = heading_matches[
                index + 1
            ].start()
        else:
            section_end = len(content)

        raw_section = content[
            section_start:section_end
        ]

        cleaned_section = _clean_section_text(
            raw_section
        )

        if not cleaned_section:
            continue

        sections.append(
            ReportSection(
                title=_normalise_heading(
                    heading_match.group("title")
                ),
                content=cleaned_section,
                source_range=_source_range(
                    content,
                    section_start,
                    section_end,
                ),
                page_numbers=(
                    _page_numbers_for_range(
                        content,
                        section_start,
                        section_end,
                    )
                ),
            )
        )

    return sections


def _build_unit(
    *,
    section: ReportSection,
    artifact_id: str,
    unit_index: int,
) -> dict[str, Any]:
    unit_id = (
        f"{artifact_id}_unit_{unit_index:03d}"
    )

    return {
        "unit_id": unit_id,
        "unit_type": "ontology_report_section",
        "label": section.title,
        "task_mapping": {
            "part_ids": [
                PART_ID,
            ],
            "source": "inferred",
            "confidence": 0.95,
            "evidence": [
                (
                    "The section occurs in the "
                    "submitted Part 1 ontology report."
                ),
            ],
        },
        "source_range": _range_dict(
            section.source_range
        ),
        "content_blocks": [
            {
                "block_id": (
                    f"{unit_id}_heading"
                ),
                "block_type": "heading",
                "content": section.title,
                "language": "en",
                "source_range": _range_dict(
                    section.source_range
                ),
            },
            {
                "block_id": (
                    f"{unit_id}_prose"
                ),
                "block_type": "prose",
                "content": section.content,
                "language": "en",
                "source_range": _range_dict(
                    section.source_range
                ),
            },
        ],
        "structured_data": {
            "section_title": section.title,
            "page_numbers": (
                section.page_numbers
            ),
        },
        "extraction_confidence": 0.95,
    }


def _build_diagnostics(
    *,
    units: list[dict[str, Any]],
    original_content: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    diagnostic_index = 1

    if not original_content.strip():
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "empty_report_content"
                ),
                "message": (
                    "No textual content was available "
                    "for the ontology report."
                ),
                "related_unit_ids": [],
            }
        )

        return diagnostics

    if not units:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "no_report_sections_extracted"
                ),
                "message": (
                    "No substantive report sections "
                    "were extracted."
                ),
                "related_unit_ids": [],
            }
        )

        return diagnostics

    recognised_labels = {
        str(unit.get("label", "")).lower()
        for unit in units
    }

    expected_sections = {
        "overview",
        "justification",
        "axiom",
        "advantages and disadvantages",
    }

    missing_sections = (
        expected_sections - recognised_labels
    )

    if missing_sections:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_{diagnostic_index:03d}"
                ),
                "severity": "info",
                "diagnostic_type": (
                    "unrecognised_report_sections"
                ),
                "message": (
                    "The following common report "
                    "sections were not identified: "
                    + ", ".join(
                        sorted(missing_sections)
                    )
                    + "."
                ),
                "related_unit_ids": [],
            }
        )

    return diagnostics


def process_ontology_report_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact_id = artifact["artifact_id"]
    artifact_type = artifact["artifact_type"]
    content = artifact["content"][
        "raw_content"
    ]

    sections = _extract_sections(content)

    units = [
        _build_unit(
            section=section,
            artifact_id=artifact_id,
            unit_index=index,
        )
        for index, section in enumerate(
            sections,
            start=1,
        )
    ]

    diagnostics = _build_diagnostics(
        units=units,
        original_content=content,
    )

    contains_error = any(
        diagnostic["severity"] == "error"
        for diagnostic in diagnostics
    )

    if contains_error and units:
        status = "partial"
    elif contains_error:
        status = "failed"
    elif units:
        status = "success"
    else:
        status = "failed"

    checks = [
        {
            "check_type": (
                "report_text_extraction"
            ),
            "status": (
                "passed"
                if content.strip()
                else "failed"
            ),
            "tool": "pypdf",
            "summary": (
                "Text was extracted from the "
                "submitted ontology report."
                if content.strip()
                else (
                    "No report text was available."
                )
            ),
        },
        {
            "check_type": (
                "report_section_extraction"
            ),
            "status": (
                "passed"
                if units
                else "failed"
            ),
            "tool": (
                "assessment-feedback-preprocessor"
            ),
            "summary": (
                f"Extracted {len(units)} "
                "substantive report section(s)."
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
                "ontology_design_report"
            ),
            "tools": [
                {
                    "name": "pypdf",
                    "version": None,
                    "purpose": (
                        "PDF text extraction during "
                        "submission ingestion."
                    ),
                },
                {
                    "name": (
                        "assessment-feedback-preprocessor"
                    ),
                    "version": "0.3.0",
                    "purpose": (
                        "Removal of report boilerplate "
                        "and structural extraction of "
                        "ontology-report sections."
                    ),
                },
            ],
            "checks": checks,
            "diagnostics": diagnostics,
        },
        "units": units,
    }