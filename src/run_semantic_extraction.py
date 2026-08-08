from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extraction.huggingface_structured_client import (
    HuggingFaceStructuredClient,
)
from extraction.semantic_extractors import (
    SemanticExtractor,
    save_semantic_result,
)
from extraction.semantic_models import (
    SemanticExtractionResult,
    SemanticUnitAnnotation,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable semantic annotations "
            "for a processed submission."
        )
    )

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help=(
            "Path to the raw submission JSON."
        ),
    )

    parser.add_argument(
        "--processed",
        type=Path,
        required=True,
        help=(
            "Path to the deterministic processed "
            "submission JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Path where the semantic JSON should "
            "be written."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model name used for "
            "semantic extraction."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help=(
            "Maximum number of tokens generated "
            "for each semantic unit."
        ),
    )

    parser.add_argument(
        "--retain-raw-model-response",
        action="store_true",
        help=(
            "Retain raw model responses in the "
            "saved semantic output."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore an existing output file and "
            "restart semantic extraction."
        ),
    )

    return parser.parse_args()


def collect_units(
    processed_submission: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Collect every valid unit together with its parent
    artifact metadata.
    """
    records: list[dict[str, Any]] = []

    for artifact in processed_submission.get(
        "artifacts",
        [],
    ):
        artifact_id = artifact.get(
            "artifact_id"
        )

        artifact_type = artifact.get(
            "artifact_type"
        )

        if not isinstance(
            artifact_id,
            str,
        ):
            continue

        if not isinstance(
            artifact_type,
            str,
        ):
            continue

        for unit in artifact.get(
            "units",
            [],
        ):
            if not isinstance(
                unit,
                dict,
            ):
                continue

            unit_id = unit.get(
                "unit_id"
            )

            if not isinstance(
                unit_id,
                str,
            ):
                continue

            records.append(
                {
                    "artifact_id": (
                        artifact_id
                    ),
                    "artifact_type": (
                        artifact_type
                    ),
                    "artifact": artifact,
                    "unit": unit,
                }
            )

    return records


def build_single_unit_submission(
    *,
    processed_submission: dict[str, Any],
    artifact: dict[str, Any],
    unit: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a temporary processed submission containing
    exactly one semantic unit.

    This keeps each model request independent and small.
    """
    single_artifact = {
        **artifact,
        "units": [
            unit
        ],
    }

    return {
        **processed_submission,
        "artifacts": [
            single_artifact
        ],
    }


def load_existing_annotations(
    output_path: Path,
) -> tuple[
    dict[str, SemanticUnitAnnotation],
    dict[str, Any],
]:
    """
    Load previously completed annotations from the output
    file so an interrupted run can continue.

    Returns:
        annotations_by_id
        raw_unit_responses
    """
    if not output_path.exists():
        return {}, {}

    existing_data = load_json(
        output_path
    )

    annotations_by_id: dict[
        str,
        SemanticUnitAnnotation,
    ] = {}

    for annotation_data in (
        existing_data.get(
            "unit_annotations",
            [],
        )
    ):
        try:
            annotation = (
                SemanticUnitAnnotation
                .model_validate(
                    annotation_data
                )
            )
        except Exception as exc:
            print(
                "Warning: ignoring invalid "
                "checkpoint annotation: "
                f"{exc}"
            )
            continue

        annotations_by_id[
            annotation.unit_id
        ] = annotation

    raw_unit_responses: dict[
        str,
        Any,
    ] = {}

    raw_model_response = (
        existing_data.get(
            "raw_model_response"
        )
    )

    if isinstance(
        raw_model_response,
        dict,
    ):
        unit_responses = (
            raw_model_response.get(
                "unit_responses"
            )
        )

        if isinstance(
            unit_responses,
            dict,
        ):
            raw_unit_responses.update(
                unit_responses
            )

    return (
        annotations_by_id,
        raw_unit_responses,
    )


def build_result(
    *,
    processed_submission: dict[str, Any],
    model_name: str,
    annotations: list[
        SemanticUnitAnnotation
    ],
    raw_unit_responses: dict[
        str,
        Any,
    ],
    retain_raw_model_response: bool,
    complete: bool,
    completed_count: int,
    total_count: int,
) -> SemanticExtractionResult:
    warnings: list[str] = []

    if not complete:
        warnings.append(
            "Semantic extraction is incomplete: "
            f"{completed_count}/{total_count} "
            "units have been processed."
        )

    return SemanticExtractionResult(
        schema_version="2.0",
        source_submission_id=str(
            processed_submission.get(
                "source_submission_id",
                "",
            )
        ),
        processed_submission_id=str(
            processed_submission.get(
                "processed_submission_id",
                "",
            )
        ),
        extractor_version="2.0",
        model_name=model_name,
        document_regions=[],
        unit_annotations=annotations,
        unresolved_observations=[],
        warnings=warnings,
        raw_model_response=(
            {
                "unit_responses": (
                    raw_unit_responses
                )
            }
            if retain_raw_model_response
            else None
        ),
    )


def save_checkpoint(
    *,
    output_path: Path,
    processed_submission: dict[str, Any],
    model_name: str,
    annotations_by_id: dict[
        str,
        SemanticUnitAnnotation
    ],
    ordered_unit_ids: list[str],
    raw_unit_responses: dict[
        str,
        Any,
    ],
    retain_raw_model_response: bool,
) -> None:
    """
    Save all completed units in deterministic submission
    order after every successful model call.
    """
    ordered_annotations = [
        annotations_by_id[unit_id]
        for unit_id in ordered_unit_ids
        if unit_id in annotations_by_id
    ]

    completed_count = len(
        ordered_annotations
    )

    total_count = len(
        ordered_unit_ids
    )

    complete = (
        completed_count
        == total_count
    )

    result = build_result(
        processed_submission=(
            processed_submission
        ),
        model_name=model_name,
        annotations=ordered_annotations,
        raw_unit_responses=(
            raw_unit_responses
        ),
        retain_raw_model_response=(
            retain_raw_model_response
        ),
        complete=complete,
        completed_count=(
            completed_count
        ),
        total_count=total_count,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_semantic_result(
        result=result,
        output_path=str(
            output_path
        ),
    )


def main() -> None:
    args = parse_args()

    raw_submission = load_json(
        args.raw
    )

    processed_submission = load_json(
        args.processed
    )

    unit_records = collect_units(
        processed_submission
    )

    if not unit_records:
        raise RuntimeError(
            "No semantic units were found "
            "in the processed submission."
        )

    ordered_unit_ids = [
        record["unit"]["unit_id"]
        for record in unit_records
    ]

    print(
        f"Found {len(unit_records)} "
        "semantic unit(s)."
    )

    if (
        args.output.exists()
        and not args.no_resume
    ):
        (
            annotations_by_id,
            raw_unit_responses,
        ) = load_existing_annotations(
            args.output
        )

        print(
            "Resume enabled: "
            f"{len(annotations_by_id)} "
            "completed unit(s) found."
        )

    else:
        annotations_by_id = {}
        raw_unit_responses = {}

        if (
            args.output.exists()
            and args.no_resume
        ):
            print(
                "Existing semantic output "
                "will be replaced."
            )

    print(
        f"Loading semantic model: "
        f"{args.model}"
    )

    llm_client = (
        HuggingFaceStructuredClient(
            model=args.model,
            max_new_tokens=(
                args.max_new_tokens
            ),
        )
    )

    extractor = SemanticExtractor(
        llm_client=llm_client,
        retain_raw_model_response=(
            args.retain_raw_model_response
        ),
    )

    total_units = len(
        unit_records
    )

    for index, record in enumerate(
        unit_records,
        start=1,
    ):
        artifact = record[
            "artifact"
        ]

        artifact_type = record[
            "artifact_type"
        ]

        unit = record[
            "unit"
        ]

        unit_id = unit[
            "unit_id"
        ]

        if unit_id in annotations_by_id:
            print(
                f"[{index}/{total_units}] "
                f"Skipping {unit_id} "
                "(already completed)"
            )
            continue

        print()
        print(
            f"[{index}/{total_units}] "
            f"Processing {unit_id} "
            f"({artifact_type})"
        )

        single_processed_submission = (
            build_single_unit_submission(
                processed_submission=(
                    processed_submission
                ),
                artifact=artifact,
                unit=unit,
            )
        )

        try:
            unit_result = (
                extractor.extract(
                    raw_submission=(
                        raw_submission
                    ),
                    processed_submission=(
                        single_processed_submission
                    ),
                )
            )

        except Exception:
            print()
            print(
                "Semantic extraction failed "
                f"for {unit_id}."
            )

            print(
                "Previously completed units "
                "remain saved."
            )

            raise

        if (
            len(
                unit_result
                .unit_annotations
            )
            != 1
        ):
            raise RuntimeError(
                "Expected exactly one "
                "semantic annotation for "
                f"{unit_id}, but received "
                f"{len(unit_result.unit_annotations)}."
            )

        annotation = (
            unit_result
            .unit_annotations[0]
        )

        annotations_by_id[
            unit_id
        ] = annotation

        if (
            args.retain_raw_model_response
            and isinstance(
                unit_result
                .raw_model_response,
                dict,
            )
        ):
            unit_responses = (
                unit_result
                .raw_model_response
                .get(
                    "unit_responses"
                )
            )

            if isinstance(
                unit_responses,
                dict,
            ):
                raw_unit_responses.update(
                    unit_responses
                )

        save_checkpoint(
            output_path=args.output,
            processed_submission=(
                processed_submission
            ),
            model_name=(
                llm_client.model_name
            ),
            annotations_by_id=(
                annotations_by_id
            ),
            ordered_unit_ids=(
                ordered_unit_ids
            ),
            raw_unit_responses=(
                raw_unit_responses
            ),
            retain_raw_model_response=(
                args
                .retain_raw_model_response
            ),
        )

        print(
            "✓ Checkpoint saved: "
            f"{len(annotations_by_id)}/"
            f"{total_units} units"
        )

    print()
    print(
        "✓ Semantic extraction complete."
    )

    print(
        f"✓ Output written to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()