from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from openai import APITimeoutError
from pydantic import ValidationError

from .semantic_models import (
    SemanticExtractionResult,
    SemanticUnitAnnotation,
    SemanticUnitLLMOutput,
)
from .semantic_prompts import (
    SYSTEM_PROMPT,
    build_semantic_unit_prompt,
)
from .semantic_taxonomy import (
    LEAN_STRATEGY_CATEGORY_MAP,
    LEAN_STRATEGY_NAMES,
    STRATEGY_CATEGORIES,
)


class SemanticExtractionError(RuntimeError):
    """Raised when semantic extraction cannot produce valid output."""


class StructuredLLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

def _normalise_strategy_category(
    response: dict[str, Any],
) -> dict[str, Any]:
    """
    Small models occasionally return an incorrect strategy category even
    though the strategy name is correct. Since the mapping is
    deterministic, overwrite the category using the controlled taxonomy.
    """

    strategy = response.get("strategy")

    if not isinstance(strategy, dict):
        return response

    strategy_name = strategy.get("name")

    if not isinstance(strategy_name, str):
        return response

    expected_category = LEAN_STRATEGY_CATEGORY_MAP.get(
        strategy_name
    )

    if expected_category is not None:
        strategy["category"] = expected_category

    return response


class SemanticExtractor:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        retain_raw_model_response: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.retain_raw_model_response = retain_raw_model_response

    def extract(
        self,
        *,
        raw_submission: dict[str, Any],
        processed_submission: dict[str, Any],
        assessment_metadata: dict[str, Any] | None = None,
        rubric: dict[str, Any] | None = None,
    ) -> SemanticExtractionResult:
        raw_source = self._get_raw_source(raw_submission)

        units = [
            unit
            for artifact in processed_submission.get("artifacts", [])
            for unit in artifact.get("units", [])
            if isinstance(unit.get("unit_id"), str)
        ]

        if not units:
            raise SemanticExtractionError(
                "No valid units were found in the processed submission."
            )

        all_unit_ids = [unit["unit_id"] for unit in units]

        unit_annotations: list[SemanticUnitAnnotation] = []
        raw_unit_responses: dict[str, Any] = {}

        for index, unit in enumerate(units, start=1):
            unit_id = unit["unit_id"]

            print(
                f"Semantic extraction for unit "
                f"{index}/{len(units)}: {unit_id}"
            )

            annotation, raw_response = self._extract_single_unit(
                raw_source=raw_source,
                unit=unit,
                all_unit_ids=all_unit_ids,
                assessment_metadata=assessment_metadata,
                rubric=rubric,
            )

            unit_annotations.append(annotation)
            raw_unit_responses[unit_id] = raw_response

            print(
                f"✓ Completed semantic extraction for "
                f"{unit_id}"
            )

        result = SemanticExtractionResult(
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
            model_name=self.llm_client.model_name,
            document_regions=[],
            unit_annotations=unit_annotations,
            unresolved_observations=[],
            warnings=[],
            raw_model_response=(
                {"unit_responses": raw_unit_responses}
                if self.retain_raw_model_response
                else None
            ),
        )

        self._validate_result_references(
            result=result,
            processed_submission=processed_submission,
        )

        return result

    def _extract_single_unit(
        self,
        *,
        raw_source: str,
        unit: dict[str, Any],
        all_unit_ids: list[str],
        assessment_metadata: dict[str, Any] | None,
        rubric: dict[str, Any] | None,
    ) -> tuple[SemanticUnitAnnotation, dict[str, Any]]:
        unit_id = unit.get("unit_id")

        if not isinstance(unit_id, str):
            raise SemanticExtractionError(
                "Cannot semantically extract a unit without a valid unit_id."
            )

        compact_unit = self._build_compact_unit_input(unit)

        unit_source = self._get_unit_source_excerpt(
            raw_source,
            unit,
            context_lines=0,
        )

        prompt = build_semantic_unit_prompt(
            raw_source=unit_source,
            unit=compact_unit,
            all_unit_ids=all_unit_ids,
            assessment_metadata=assessment_metadata,
            rubric=rubric,
        )

        output_schema = self._build_unit_output_schema(
            unit=unit,
            all_unit_ids=all_unit_ids,
        )

        # Remove this debug output after testing.
        print(
            f"Raw source excerpt for {unit_id}:\n"
            f"{unit_source}\n"
        )

        max_attempts = 2
        response: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.llm_client.generate_structured(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    output_schema=output_schema,
                )
                break

            except APITimeoutError as exc:
                if attempt == max_attempts:
                    raise SemanticExtractionError(
                        f"Semantic extraction timed out for {unit_id} "
                        f"after {max_attempts} attempts."
                    ) from exc

                print(
                    f"Timeout for {unit_id}; retrying "
                    f"({attempt + 1}/{max_attempts})..."
                )

                time.sleep(2)

        if response is None:
            raise SemanticExtractionError(
                f"No semantic response was returned for {unit_id}."
            )

        response = _normalise_strategy_category(response)

        try:
            llm_output = SemanticUnitLLMOutput.model_validate(
                response
            )
        except ValidationError as exc:
            raise SemanticExtractionError(
                f"LLM returned invalid semantic output "
                f"for {unit_id}: {exc}"
            ) from exc

        annotation = SemanticUnitAnnotation(
            unit_id=unit_id,
            region_ids=[],
            **llm_output.model_dump(),
        )

        self._validate_strategy_taxonomy(annotation)

        return annotation, response

    @staticmethod
    def _validate_strategy_taxonomy(
        annotation: SemanticUnitAnnotation,
    ) -> None:
        if annotation.strategy is None:
            return

        strategy_name = annotation.strategy.name
        strategy_category = annotation.strategy.category

        expected_category = LEAN_STRATEGY_CATEGORY_MAP.get(
            strategy_name
        )

        if expected_category is None:
            raise SemanticExtractionError(
                f"Unknown strategy name for "
                f"{annotation.unit_id}: {strategy_name}"
            )

        if strategy_category != expected_category:
            raise SemanticExtractionError(
                f"Invalid strategy category for "
                f"{annotation.unit_id}: "
                f"{strategy_name!r} must use "
                f"{expected_category!r}, not "
                f"{strategy_category!r}."
            )

    @staticmethod
    def _get_raw_source(
        raw_submission: dict[str, Any],
    ) -> str:
        artifacts = raw_submission.get("artifacts", [])

        for artifact in artifacts:
            content = artifact.get("content", {})
            raw_content = content.get("raw_content")

            if isinstance(raw_content, str):
                return raw_content

        raise SemanticExtractionError(
            "No raw source content was found in the submission."
        )

    @staticmethod
    def _get_unit_source_excerpt(
        raw_source: str,
        unit: dict[str, Any],
        *,
        context_lines: int = 0,
    ) -> str:
        source_range = unit.get("source_range") or {}

        start_line = source_range.get("start_line")
        end_line = source_range.get("end_line")

        if (
            not isinstance(start_line, int)
            or not isinstance(end_line, int)
        ):
            return raw_source

        lines = raw_source.splitlines()

        # source_range line numbers are assumed to be 1-based.
        start_index = max(
            0,
            start_line - 1 - context_lines,
        )

        # Python slicing excludes the end index.
        end_index = min(
            len(lines),
            end_line + context_lines,
        )

        excerpt_lines = lines[start_index:end_index]

        return "\n".join(excerpt_lines)

    @staticmethod
    def _build_compact_unit_input(
        unit: dict[str, Any],
    ) -> dict[str, Any]:
        structured_data = unit.get("structured_data") or {}

        return {
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "label": unit.get("label"),
            "source_range": unit.get("source_range"),
            "content_blocks": unit.get(
                "content_blocks",
                [],
            ),
            "structured_data": {
                "context_declarations": structured_data.get(
                    "context_declarations",
                    [],
                ),
                "scope_path": structured_data.get(
                    "scope_path",
                    [],
                ),
                "contains_sorry": structured_data.get(
                    "contains_sorry",
                    False,
                ),
                "proof_style": structured_data.get(
                    "proof_style"
                ),
            },
        }

    @staticmethod
    def _build_unit_output_schema(
        *,
        unit: dict[str, Any],
        all_unit_ids: list[str],
    ) -> dict[str, Any]:
        schema = deepcopy(
            SemanticUnitLLMOutput.model_json_schema()
        )

        structured_data = unit.get("structured_data") or {}

        valid_context_ids = [
            context["context_id"]
            for context in structured_data.get(
                "context_declarations",
                [],
            )
            if isinstance(context.get("context_id"), str)
        ]

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")

                if isinstance(properties, dict):
                    context_id_schema = properties.get(
                        "context_id"
                    )

                    if (
                        valid_context_ids
                        and isinstance(
                            context_id_schema,
                            dict,
                        )
                    ):
                        context_id_schema.clear()
                        context_id_schema.update(
                            {
                                "type": "string",
                                "enum": valid_context_ids,
                            }
                        )

                    target_unit_schema = properties.get(
                        "target_unit_id"
                    )

                    if isinstance(
                        target_unit_schema,
                        dict,
                    ):
                        target_unit_schema.clear()
                        target_unit_schema.update(
                            {
                                "type": "string",
                                "enum": all_unit_ids,
                            }
                        )

                    related_unit_ids_schema = properties.get(
                        "related_unit_ids"
                    )

                    if isinstance(
                        related_unit_ids_schema,
                        dict,
                    ):
                        related_unit_ids_schema.clear()
                        related_unit_ids_schema.update(
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": all_unit_ids,
                                },
                            }
                        )

                    category_schema = properties.get(
                        "category"
                    )

                    name_schema = properties.get(
                        "name"
                    )

                    if (
                        isinstance(category_schema, dict)
                        and isinstance(name_schema, dict)
                    ):
                        category_schema.clear()
                        category_schema.update(
                            {
                                "type": "string",
                                "enum": list(
                                    STRATEGY_CATEGORIES
                                ),
                            }
                        )

                        name_schema.clear()
                        name_schema.update(
                            {
                                "type": "string",
                                "enum": list(
                                    LEAN_STRATEGY_NAMES
                                ),
                            }
                        )

                    evidence_schema = properties.get(
                        "evidence"
                    )

                    if isinstance(
                        evidence_schema,
                        dict,
                    ):
                        evidence_schema["minItems"] = 1

                for nested_value in value.values():
                    visit(nested_value)

            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(schema)

        return schema

    @staticmethod
    def _validate_result_references(
        *,
        result: SemanticExtractionResult,
        processed_submission: dict[str, Any],
    ) -> None:
        valid_unit_ids = {
            unit["unit_id"]
            for artifact in processed_submission.get(
                "artifacts",
                [],
            )
            for unit in artifact.get("units", [])
            if isinstance(unit.get("unit_id"), str)
        }

        valid_context_ids_by_unit: dict[str, set[str]] = {}

        for artifact in processed_submission.get(
            "artifacts",
            [],
        ):
            for unit in artifact.get("units", []):
                unit_id = unit.get("unit_id")

                if not isinstance(unit_id, str):
                    continue

                contexts = unit.get(
                    "structured_data",
                    {},
                ).get(
                    "context_declarations",
                    [],
                )

                valid_context_ids_by_unit[unit_id] = {
                    context["context_id"]
                    for context in contexts
                    if isinstance(
                        context.get("context_id"),
                        str,
                    )
                }

        errors: list[str] = []
        seen_annotation_unit_ids: set[str] = set()

        for annotation in result.unit_annotations:
            if annotation.unit_id in seen_annotation_unit_ids:
                errors.append(
                    "Duplicate unit annotation for unit ID: "
                    f"{annotation.unit_id}"
                )

            seen_annotation_unit_ids.add(
                annotation.unit_id
            )

            if annotation.unit_id not in valid_unit_ids:
                errors.append(
                    "Unknown unit ID in semantic output: "
                    f"{annotation.unit_id}"
                )
                continue

            valid_context_ids = (
                valid_context_ids_by_unit.get(
                    annotation.unit_id,
                    set(),
                )
            )

            for selection in annotation.relevant_context:
                if (
                    selection.context_id
                    not in valid_context_ids
                ):
                    errors.append(
                        f"Unknown context ID "
                        f"{selection.context_id!r} "
                        f"for unit "
                        f"{annotation.unit_id!r}"
                    )

            for relationship in annotation.relationships:
                if (
                    relationship.target_unit_id
                    not in valid_unit_ids
                ):
                    errors.append(
                        "Unknown relationship target unit ID: "
                        f"{relationship.target_unit_id}"
                    )

        missing_unit_ids = (
            valid_unit_ids - seen_annotation_unit_ids
        )

        for unit_id in sorted(missing_unit_ids):
            errors.append(
                f"Missing semantic annotation for "
                f"unit ID: {unit_id}"
            )

        valid_region_ids = {
            region.region_id
            for region in result.document_regions
        }

        if (
            len(valid_region_ids)
            != len(result.document_regions)
        ):
            errors.append(
                "Duplicate region IDs found in "
                "document_regions."
            )

        for region in result.document_regions:
            for unit_id in region.unit_ids:
                if unit_id not in valid_unit_ids:
                    errors.append(
                        f"Unknown unit ID {unit_id!r} "
                        f"in region {region.region_id!r}"
                    )

        for annotation in result.unit_annotations:
            for region_id in annotation.region_ids:
                if region_id not in valid_region_ids:
                    errors.append(
                        f"Unknown region ID {region_id!r} "
                        f"referenced by unit "
                        f"{annotation.unit_id!r}"
                    )

        if errors:
            raise SemanticExtractionError(
                "Semantic output contains invalid "
                "references:\n- "
                + "\n- ".join(errors)
            )


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_semantic_result(
    result: SemanticExtractionResult,
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result.model_dump(mode="json"),
            file,
            ensure_ascii=False,
            indent=2,
        )