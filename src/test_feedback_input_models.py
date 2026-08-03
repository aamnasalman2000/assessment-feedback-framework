from __future__ import annotations

import json
from pathlib import Path

from src.feedback import FeedbackGenerationInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    processed_submission = load_json(
        PROJECT_ROOT
        / "data"
        / "assessment_1"
        / "processed"
        / "student_3.json"
    )

    semantic_extraction = load_json(
        PROJECT_ROOT
        / "data"
        / "assessment_1"
        / "semantic"
        / "v1"
        / "qwen3"
        / "student_3.json"
    )

    assessment_specification = load_json(
        PROJECT_ROOT
        / "specs"
        / "assessment_1_spec.json"
    )

    generation_input = FeedbackGenerationInput(
        processed_submission=processed_submission,
        semantic_extraction=semantic_extraction,
        assessment_specification=assessment_specification,
    )

    print("FeedbackGenerationInput validated successfully.")
    print(
        f"Processed submission: "
        f"{generation_input.processed_submission.processed_submission_id}"
    )
    print(
        f"Assessment: "
        f"{generation_input.assessment_specification.assessment_id}"
    )
    print(
        f"Semantic units: "
        f"{len(generation_input.semantic_extraction.units)}"
    )


    if __name__ == "__main__":
        main()