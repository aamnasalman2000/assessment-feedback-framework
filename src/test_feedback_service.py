from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.alignment.alignment_models import AlignmentResult
from src.extraction.semantic_models import SemanticExtractionResult
from src.feedback.feedback_input_models import ProcessedSubmission
from src.feedback.feedback_llm_client import FeedbackStructuredClient
from src.feedback.feedback_service import FeedbackService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_ID = "assessment_1"
STUDENT_ID = "student_3"
MODEL = "qwen3:8b"

SEMANTIC_VERSION = "v1"
ALIGNMENT_VERSION = "v1"
FEEDBACK_VERSION = "v2"


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

    alignment_path = (
        PROJECT_ROOT
        / "artifacts"
        / ASSESSMENT_ID
        / "alignment"
        / ALIGNMENT_VERSION
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
        / "llm_feedback"
        / FEEDBACK_VERSION
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

    print("Loading alignment result...")

    alignment_result = AlignmentResult.model_validate(
        load_json(alignment_path)
    )

    print("Loading assessment specification...")

    assessment_specification = load_json(
        specification_path
    )

    sdk_client = create_sdk_client()

    feedback_client = FeedbackStructuredClient(
        client=sdk_client,
        model=MODEL,
    )

    feedback_service = FeedbackService(
        llm_client=feedback_client,
    )

    print(
        "Generating component-level feedback "
        f"for {ASSESSMENT_ID}/{STUDENT_ID}..."
    )

    feedback = feedback_service.generate_feedback(
        processed_submission=processed_submission,
        semantic_extraction=semantic_extraction,
        alignment_result=alignment_result,
        assessment_specification=assessment_specification,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            feedback.model_dump(
                mode="json",
                exclude_none=True,
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Feedback generation completed successfully.")
    print(f"Saved feedback to: {output_path}")


if __name__ == "__main__":
    main()