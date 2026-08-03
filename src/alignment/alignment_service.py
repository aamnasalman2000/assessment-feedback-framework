from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from src.extraction.semantic_extractors import (
    StructuredLLMClient,
)
from src.extraction.semantic_models import (
    SemanticExtractionResult,
)
from src.feedback.feedback_input_models import (
    ProcessedSubmission,
)

from .alignment_models import (
    AlignmentLLMOutput,
    AlignmentResult,
)
from .alignment_prompts import (
    build_document_alignment_prompt,
)


class AlignmentError(RuntimeError):
    """Raised when document-level assessment alignment fails."""


def build_component_candidates(
    assessment_specification: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a compact representation of assessment components.

    Marks, detailed rubric requirements, and unrelated metadata are
    intentionally excluded from the alignment prompt.
    """

    candidates: list[dict[str, Any]] = []

    for part in assessment_specification.get("parts", []):
        part_id = part.get("part_id")

        for component in part.get("components", []):
            component_id = component.get("component_id")

            if not isinstance(component_id, str):
                continue

            task = component.get("task", {})

            candidate: dict[str, Any] = {
                "part_id": part_id,
                "component_id": component_id,
                "title": component.get("title"),
                "artifact_type": component.get(
                    "artifact_type"
                ),
            }

            if isinstance(task, dict):
                candidate["task_summary"] = task.get(
                    "summary"
                )

                target = task.get("target")

                if isinstance(target, dict):
                    candidate["target"] = {
                        "proposition": target.get(
                            "proposition"
                        ),
                        "description": target.get(
                            "description"
                        ),
                    }

                stated_conditions = task.get(
                    "stated_conditions"
                )

                if isinstance(stated_conditions, list):
                    candidate["stated_conditions"] = [
                        {
                            "id": item.get("id"),
                            "description": item.get(
                                "description"
                            ),
                            "logical_form": item.get(
                                "logical_form"
                            ),
                        }
                        for item in stated_conditions
                        if isinstance(item, dict)
                    ]

                required_cases = task.get(
                    "required_cases"
                )

                if isinstance(required_cases, list):
                    candidate["required_cases"] = (
                        required_cases
                    )

            candidates.append(candidate)

    return candidates


def _build_alignment_output_schema(
    *,
    valid_component_ids: list[str],
    valid_unit_ids: list[str],
) -> dict[str, Any]:
    schema = deepcopy(
        AlignmentLLMOutput.model_json_schema()
    )

    unit_id_fields = {
        "primary_unit_ids",
        "supporting_unit_ids",
        "possibly_relevant_unit_ids",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")

            if isinstance(properties, dict):
                component_id_schema = properties.get(
                    "component_id"
                )

                if isinstance(component_id_schema, dict):
                    component_id_schema.clear()
                    component_id_schema.update(
                        {
                            "type": "string",
                            "enum": valid_component_ids,
                        }
                    )

                for field_name in unit_id_fields:
                    unit_ids_schema = properties.get(
                        field_name
                    )

                    if isinstance(unit_ids_schema, dict):
                        unit_ids_schema.clear()
                        unit_ids_schema.update(
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": valid_unit_ids,
                                },
                                "uniqueItems": True,
                            }
                        )

            for nested_value in value.values():
                visit(nested_value)

        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)

    return schema


class DocumentAlignmentService:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        aligner_version: str = "1.0",
    ) -> None:
        self.llm_client = llm_client
        self.aligner_version = aligner_version

    def align_submission(
        self,
        *,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        assessment_specification: dict[str, Any],
    ) -> AlignmentResult:
        processed_units = [
            unit
            for artifact in processed_submission.artifacts
            for unit in artifact.units
        ]

        semantic_annotations = list(
            semantic_extraction.unit_annotations
        )

        component_candidates = build_component_candidates(
            assessment_specification
        )

        valid_component_ids = [
            candidate["component_id"]
            for candidate in component_candidates
        ]

        valid_unit_ids = [
            unit.unit_id
            for unit in processed_units
        ]

        if not valid_component_ids:
            raise AlignmentError(
                "No valid assessment components were found."
            )

        if not valid_unit_ids:
            raise AlignmentError(
                "No processed submission units were found."
            )

        system_prompt, user_prompt = (
            build_document_alignment_prompt(
                processed_units=processed_units,
                semantic_annotations=semantic_annotations,
                component_candidates=component_candidates,
            )
        )

        output_schema = _build_alignment_output_schema(
            valid_component_ids=valid_component_ids,
            valid_unit_ids=valid_unit_ids,
        )

        response = self.llm_client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )

        try:
            llm_output = AlignmentLLMOutput.model_validate(
                response
            )
        except ValidationError as exc:
            raise AlignmentError(
                "The model returned invalid document alignment."
            ) from exc

        self._validate_output_references(
            output=llm_output,
            valid_component_ids=set(
                valid_component_ids
            ),
            valid_unit_ids=set(valid_unit_ids),
        )

        result = AlignmentResult(
            schema_version="1.0",
            assessment_id=processed_submission.assessment_id,
            source_submission_id=(
                processed_submission.source_submission_id
            ),
            processed_submission_id=(
                processed_submission.processed_submission_id
            ),
            model_name=self.llm_client.model_name,
            aligner_version=self.aligner_version,
            **llm_output.model_dump(),
        )

        return result

    @staticmethod
    def _validate_output_references(
        *,
        output: AlignmentLLMOutput,
        valid_component_ids: set[str],
        valid_unit_ids: set[str],
    ) -> None:
        errors: list[str] = []

        returned_component_ids = {
            alignment.component_id
            for alignment in output.component_alignments
        }

        missing_component_ids = (
            valid_component_ids - returned_component_ids
        )

        unknown_component_ids = (
            returned_component_ids - valid_component_ids
        )

        if missing_component_ids:
            errors.append(
                "Missing component alignments for: "
                + ", ".join(
                    sorted(missing_component_ids)
                )
            )

        if unknown_component_ids:
            errors.append(
                "Unknown component IDs: "
                + ", ".join(
                    sorted(unknown_component_ids)
                )
            )

        for alignment in output.component_alignments:
            aligned_unit_ids = (
                set(alignment.primary_unit_ids)
                | set(alignment.supporting_unit_ids)
                | set(
                    alignment.possibly_relevant_unit_ids
                )
            )

            unknown_unit_ids = (
                aligned_unit_ids - valid_unit_ids
            )

            if unknown_unit_ids:
                errors.append(
                    f"Component {alignment.component_id!r} "
                    f"references unknown unit IDs: "
                    + ", ".join(
                        sorted(unknown_unit_ids)
                    )
                )

        if errors:
            raise AlignmentError(
                "Alignment output contains invalid references:\n- "
                + "\n- ".join(errors)
            )