from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extraction.semantic_extractors import (
    SemanticExtractor,
    save_semantic_result,
)
from extraction.semantic_llm_client import (
    OpenAICompatibleStructuredClient,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semantic annotations for a processed submission."
    )

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Path to the raw submission JSON.",
    )

    parser.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Path to the deterministic processed submission JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the semantic JSON should be written.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Name of the LLM used for semantic extraction.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_submission = load_json(args.raw)
    processed_submission = load_json(args.processed)

    # Replace this with the actual client configuration you use.
    raise NotImplementedError(
        "Configure the LLM client before running semantic extraction."
    )

    llm_client = OpenAICompatibleStructuredClient(
        client=client,
        model=args.model,
    )

    extractor = SemanticExtractor(
        llm_client=llm_client,
        retain_raw_model_response=False,
    )

    result = extractor.extract(
        raw_submission=raw_submission,
        processed_submission=processed_submission,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    save_semantic_result(
        result=result,
        output_path=str(args.output),
    )

    print(f"Semantic extraction written to: {args.output}")


if __name__ == "__main__":
    main()