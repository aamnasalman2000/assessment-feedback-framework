from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.feedback.feedback_generator import (
    build_prompt_for_assessment,
)
from src.feedback.feedback_input_models import (
    FeedbackGenerationInput,
)
from src.feedback.feedback_llm_client import (
    FeedbackStructuredClient,
)
from src.feedback.feedback_validation import (
    validate_feedback_against_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_sdk_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
        timeout=1200.0,
        max_retries=0,
    )


def main() -> None:
    generation_input = FeedbackGenerationInput(
        processed_submission=load_json(
            PROJECT_ROOT
            / "data"
            / "assessment_1"
            / "processed"
            / "student_3.json"
        ),
        semantic_extraction=load_json(
            PROJECT_ROOT
            / "data"
            / "assessment_1"
            / "semantic"
            / "v1"
            / "qwen3"
            / "student_3.json"
        ),
        assessment_specification=load_json(
            PROJECT_ROOT
            / "specs"
            / "assessment_1_spec.json"
        ),
    )

    print("Building whole-assessment feedback prompt...")

    system_prompt, user_prompt = build_prompt_for_assessment(
        generation_input
    )

    print("Sending feedback prompt to qwen3:8b...")

    feedback_client = FeedbackStructuredClient(
        client=create_sdk_client(),
        model="qwen3:8b",
    )

    feedback = feedback_client.generate_feedback(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("Validating feedback references...")

    validate_feedback_against_input(
        feedback=feedback,
        generation_input=generation_input,
    )

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "assessment_1"
        / "llm_feedback"
        / "v1"
        / "qwen3"
        / "student_3.json"
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

    print("Feedback generation completed successfully.")
    print(f"Saved feedback to: {output_path}")


if __name__ == "__main__":
    main()