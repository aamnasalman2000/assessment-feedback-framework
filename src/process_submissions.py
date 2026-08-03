from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from extraction.lean_processor import process_lean_artifact
from storage.json_store import read_json, write_json
from validation.schema_validator import (
    ArtefactValidationError,
    validate_artefact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "processed_submission.schema.json"
)

PROCESSING_VERSION = "0.1.0"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess student submissions into "
            "structured processed-submission artefacts."
        )
    )

    parser.add_argument(
        "--assessment-id",
        required=True,
        help="Assessment identifier, such as assessment_1.",
    )

    parser.add_argument(
        "--student-id",
        help="Process one student submission.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every metadata file for the assessment.",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help=(
            "Path to the processed-submission JSON schema."
        ),
    )

    return parser.parse_args()


def metadata_directory(
    assessment_id: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / assessment_id
        / "metadata"
    )


def processed_directory(
    assessment_id: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / assessment_id
        / "processed"
    )


def process_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact_type = artifact["artifact_type"]

    if artifact_type == "lean_source":
        return process_lean_artifact(artifact)

    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_type": artifact_type,
        "processing": {
            "status": "failed",
            "extraction_mode": "manual",
            "detected_content_type": None,
            "tools": [],
            "checks": [
                {
                    "check_type": (
                        "artifact_preprocessing"
                    ),
                    "status": "not_run",
                    "tool": None,
                    "summary": (
                        "No processor is currently "
                        f"available for {artifact_type}."
                    ),
                }
            ],
            "diagnostics": [
                {
                    "diagnostic_id": "diagnostic_001",
                    "severity": "warning",
                    "diagnostic_type": (
                        "unsupported_artifact_type"
                    ),
                    "message": (
                        "No preprocessing implementation "
                        f"exists for artifact type "
                        f"`{artifact_type}`."
                    ),
                    "related_unit_ids": [],
                }
            ],
        },
        "units": [],
    }


def build_processed_submission(
    submission: dict[str, Any],
) -> dict[str, Any]:
    source_submission_id = submission["submission_id"]

    processed_artifacts = [
        process_artifact(artifact)
        for artifact in submission["artifacts"]
    ]

    return {
        "schema_version": "1.0",
        "processed_submission_id": (
            f"processed_{source_submission_id}"
        ),
        "source_submission_id": source_submission_id,
        "student_id": submission["student_id"],
        "assessment_id": submission["assessment_id"],
        "processing_version": PROCESSING_VERSION,
        "artifacts": processed_artifacts,
        "cross_artifact_links": [],
    }


def process_student(
    *,
    assessment_id: str,
    student_id: str,
    schema_path: Path,
) -> None:
    input_path = (
        metadata_directory(assessment_id)
        / f"{student_id}.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {input_path}"
        )

    submission = read_json(input_path)

    if submission["assessment_id"] != assessment_id:
        raise ValueError(
            "Metadata assessment ID does not match "
            f"the requested assessment: {input_path}"
        )

    if submission["student_id"] != student_id:
        raise ValueError(
            "Metadata student ID does not match "
            f"the requested student: {input_path}"
        )

    processed_submission = build_processed_submission(
        submission
    )

    validate_artefact(
        artefact=processed_submission,
        schema_path=schema_path,
    )

    output_path = (
        processed_directory(assessment_id)
        / f"{student_id}.json"
    )

    write_json(
        path=output_path,
        data=processed_submission,
    )

    unit_count = sum(
        len(artifact["units"])
        for artifact in processed_submission["artifacts"]
    )

    diagnostic_count = sum(
        len(artifact["processing"]["diagnostics"])
        for artifact in processed_submission["artifacts"]
    )

    print(
        f"✓ {student_id}: "
        f"{unit_count} units, "
        f"{diagnostic_count} diagnostics"
    )


def process_all(
    *,
    assessment_id: str,
    schema_path: Path,
) -> int:
    root = metadata_directory(assessment_id)

    if not root.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {root}"
        )

    metadata_files = sorted(root.glob("*.json"))

    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata JSON files found in: {root}"
        )

    successes = 0
    failures = 0

    for metadata_file in metadata_files:
        student_id = metadata_file.stem

        try:
            process_student(
                assessment_id=assessment_id,
                student_id=student_id,
                schema_path=schema_path,
            )
            successes += 1

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            TypeError,
            ArtefactValidationError,
            OSError,
        ) as error:
            failures += 1
            print(f"✗ {student_id}: {error}")

    print()
    print(
        f"Completed: {successes} succeeded, "
        f"{failures} failed."
    )

    return failures


def main() -> None:
    args = parse_arguments()

    schema_path = args.schema.resolve()

    if not schema_path.exists():
        raise SystemExit(
            f"Schema file not found: {schema_path}"
        )

    if args.all and args.student_id:
        raise SystemExit(
            "Use either --student-id or --all, not both."
        )

    if args.all:
        failures = process_all(
            assessment_id=args.assessment_id,
            schema_path=schema_path,
        )

        if failures:
            raise SystemExit(1)

        return

    if args.student_id is None:
        raise SystemExit(
            "Either --student-id or --all must be provided."
        )

    process_student(
        assessment_id=args.assessment_id,
        student_id=args.student_id,
        schema_path=schema_path,
    )


if __name__ == "__main__":
    main()