from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.alignment import DocumentAlignmentService
from src.extraction.semantic_llm_client import (
    OpenAICompatibleStructuredClient,
)
from src.extraction.semantic_models import (
    SemanticExtractionResult,
)
from src.feedback.feedback_input_models import (
    ProcessedSubmission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_ID = "assessment_1"
STUDENT_ID = "student_3"
MODEL = "qwen3:8b"

SEMANTIC_VERSION = "v1"
ALIGNMENT_VERSION = "v1"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"but found {type(data).__name__}."
        )

    return data


def create_sdk_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
        timeout=1200.0,
        max_retries=0,
    )


def main() -> None:
    model_directory = MODEL.split(":")[0]

    processed_path = (
        PROJECT_ROOT
        / "data"
        / ASSESSMENT_ID
        / "processed"
        / f"{STUDENT_ID}.json"
    )

    semantic_path = (
        PROJECT_ROOT
        / "data"
        / ASSESSMENT_ID
        / "semantic"
        / SEMANTIC_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    specification_path = (
        PROJECT_ROOT
        / "specs"
        / f"{ASSESSMENT_ID}_spec.json"
    )

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "alignment"
        / ALIGNMENT_VERSION
        / model_directory
        / f"{STUDENT_ID}.json"
    )

    print("Loading processed submission...")

    processed_submission = ProcessedSubmission.model_validate(
        load_json(processed_path)
    )

    print("Loading semantic extraction...")

    semantic_extraction = SemanticExtractionResult.model_validate(
        load_json(semantic_path)
    )

    print("Loading assessment specification...")

    assessment_specification = load_json(
        specification_path
    )

    sdk_client = create_sdk_client()

    llm_client = OpenAICompatibleStructuredClient(
        client=sdk_client,
        model=MODEL,
    )

    alignment_service = DocumentAlignmentService(
        llm_client=llm_client,
        aligner_version="1.0",
    )

    print("Running document-level component alignment...")

    alignment_result = alignment_service.align_submission(
        processed_submission=processed_submission,
        semantic_extraction=semantic_extraction,
        assessment_specification=assessment_specification,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            alignment_result.model_dump(
                mode="json",
                exclude_none=True,
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Alignment completed successfully.")
    print(f"Saved alignment to: {output_path}")


if __name__ == "__main__":
    main()