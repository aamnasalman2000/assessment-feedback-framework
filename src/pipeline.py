from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

SEMANTIC_SCHEMA_VERSION = "v2"

from extraction.semantic_extractors import (
    SemanticExtractor,
    SemanticExtractionError,
    save_semantic_result,
)
from extraction.semantic_llm_client import (
    OpenAICompatibleStructuredClient,
)
from process_submissions import (
    DEFAULT_SCHEMA_PATH,
    metadata_directory,
    process_student,
    processed_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic and semantic extraction "
            "for one student submission."
        )
    )

    parser.add_argument(
        "--assessment-id",
        required=True,
        help="Assessment identifier, such as assessment_1.",
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Student identifier, such as student_3.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier used for semantic extraction.",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the processed-submission JSON schema.",
    )

    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="Optional path to the rubric JSON.",
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Optional OpenAI-compatible API base URL. "
            "Leave unset when using the official OpenAI API."
        ),
    )

    parser.add_argument(
        "--retain-raw-model-response",
        action="store_true",
        help="Store the raw model response in the semantic output.",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"but found {type(data).__name__}."
        )

    return data


def build_assessment_metadata(
    submission: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact metadata object without duplicating raw source content
    in the semantic extraction prompt.
    """
    artifacts: list[dict[str, Any]] = []

    for artifact in submission.get("artifacts", []):
        artifacts.append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_type": artifact.get("artifact_type"),
                "source": artifact.get("source"),
            }
        )

    return {
        "schema_version": submission.get("schema_version"),
        "submission_id": submission.get("submission_id"),
        "student_id": submission.get("student_id"),
        "assessment_id": submission.get("assessment_id"),
        "artifacts": artifacts,
    }


def semantic_directory(
    assessment_id: str,
    model: str,
    schema_version: str = SEMANTIC_SCHEMA_VERSION,
) -> Path:
    model_name = model.split(":")[0]

    return (
        PROJECT_ROOT
        / "data"
        / assessment_id
        / "semantic"
        / schema_version
        / model_name
    )


def create_sdk_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
        timeout=600.0,
        max_retries=0,
    )


def run_semantic_extraction(
    *,
    metadata: dict[str, Any],
    processed_submission: dict[str, Any],
    model: str,
    rubric: dict[str, Any] | None,
    base_url: str | None,
    retain_raw_model_response: bool,
):
    sdk_client = create_sdk_client()

    llm_client = OpenAICompatibleStructuredClient(
        client=sdk_client,
        model=model,
    )

    extractor = SemanticExtractor(
        llm_client=llm_client,
        retain_raw_model_response=retain_raw_model_response,
    )

    return extractor.extract(
        raw_submission=metadata,
        processed_submission=processed_submission,
        assessment_metadata=build_assessment_metadata(
            metadata
        ),
        rubric=rubric,
    )


def run_pipeline(
    *,
    assessment_id: str,
    student_id: str,
    model: str,
    schema_path: Path,
    rubric_path: Path | None,
    base_url: str | None,
    retain_raw_model_response: bool,
) -> Path:
    print(
        f"Running deterministic extraction for "
        f"{assessment_id}/{student_id}..."
    )

    process_student(
        assessment_id=assessment_id,
        student_id=student_id,
        schema_path=schema_path,
    )

    metadata_path = (
        metadata_directory(assessment_id)
        / f"{student_id}.json"
    )

    processed_path = (
        processed_directory(assessment_id)
        / f"{student_id}.json"
    )

    metadata = read_json(metadata_path)
    processed_submission = read_json(processed_path)

    rubric = (
        read_json(rubric_path)
        if rubric_path is not None
        else None
    )

    print(
        f"Running semantic extraction with model "
        f"{model}..."
    )

    semantic_result = run_semantic_extraction(
        metadata=metadata,
        processed_submission=processed_submission,
        model=model,
        rubric=rubric,
        base_url=base_url,
        retain_raw_model_response=(
            retain_raw_model_response
        ),
    )

    output_path = (
        semantic_directory(
            assessment_id=assessment_id,
            model=model,
        )
        / f"{student_id}.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_semantic_result(
        result=semantic_result,
        output_path=str(output_path),
    )

    return output_path


def main() -> None:
    args = parse_arguments()

    schema_path = args.schema.resolve()

    if not schema_path.exists():
        raise SystemExit(
            f"Schema file not found: {schema_path}"
        )

    rubric_path: Path | None = None

    if args.rubric is not None:
        rubric_path = args.rubric.resolve()

        if not rubric_path.exists():
            raise SystemExit(
                f"Rubric file not found: {rubric_path}"
            )

    try:
        output_path = run_pipeline(
            assessment_id=args.assessment_id,
            student_id=args.student_id,
            model=args.model,
            schema_path=schema_path,
            rubric_path=rubric_path,
            base_url=args.base_url,
            retain_raw_model_response=(
                args.retain_raw_model_response
            ),
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        SemanticExtractionError,
    ) as error:
        raise SystemExit(
            f"Pipeline failed: {error}"
        ) from error

    print()
    print("Pipeline completed successfully.")
    print(f"Semantic output: {output_path}")


if __name__ == "__main__":
    main()