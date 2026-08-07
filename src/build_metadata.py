from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.submission import build_submission
from src.storage.json_store import write_json
from src.validation.schema_validator import (
    ArtefactValidationError,
    validate_artefact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "student_submission.schema.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build student submission metadata."
    )

    parser.add_argument(
        "--assessment-id",
        required=True,
        help="Assessment identifier (e.g. assessment_1).",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate metadata for every student submission.",
    )

    parser.add_argument(
        "--student-id",
        help="Generate metadata for a single student.",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the student submission schema.",
    )

    return parser.parse_args()


def submissions_directory(assessment_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / assessment_id
        / "submissions"
    )


def metadata_directory(assessment_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / assessment_id
        / "metadata"
    )


def build_metadata(
    *,
    assessment_id: str,
    student_id: str,
    schema_path: Path,
) -> None:
    student_directory = (
        submissions_directory(assessment_id)
        / student_id
    )

    if not student_directory.exists():
        raise FileNotFoundError(
            f"Student directory not found: {student_directory}"
        )

    submission = build_submission(
        assessment_id=assessment_id,
        student_id=student_id,
        student_directory=student_directory,
    )

    validate_artefact(
        artefact=submission,
        schema_path=schema_path,
    )

    output_path = (
        metadata_directory(assessment_id)
        / f"{student_id}.json"
    )

    write_json(
        path=output_path,
        data=submission,
    )

    print(f"✓ {student_id}")


def build_all(
    assessment_id: str,
    schema_path: Path,
) -> None:
    root = submissions_directory(assessment_id)

    if not root.exists():
        raise FileNotFoundError(root)

    student_directories = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
    )

    successes = 0
    failures = 0

    for directory in student_directories:
        student_id = directory.name

        try:
            build_metadata(
                assessment_id=assessment_id,
                student_id=student_id,
                schema_path=schema_path,
            )
            successes += 1

        except (
            FileNotFoundError,
            ValueError,
            ArtefactValidationError,
            OSError,
        ) as error:
            failures += 1
            print(f"✗ {student_id}: {error}")

    print()
    print(f"Completed: {successes} succeeded, {failures} failed.")


def main() -> None:
    args = parse_arguments()

    schema_path = args.schema.resolve()

    if args.all:
        build_all(
            assessment_id=args.assessment_id,
            schema_path=schema_path,
        )
        return

    if args.student_id is None:
        raise SystemExit(
            "Either --student-id or --all must be provided."
        )

    build_metadata(
        assessment_id=args.assessment_id,
        student_id=args.student_id,
        schema_path=schema_path,
    )


if __name__ == "__main__":
    main()