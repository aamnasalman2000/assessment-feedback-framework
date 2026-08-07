from __future__ import annotations

import json
import re
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
    get_strategy_category_map,
    get_strategy_names,
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
    *,
    artifact_type: str,
) -> dict[str, Any]:
    """
    Correct the strategy category using the controlled taxonomy.

    The model chooses the strategy name, while Python deterministically
    supplies the category associated with that strategy.
    """
    strategy = response.get("strategy")

    if not isinstance(strategy, dict):
        return response

    strategy_name = strategy.get("name")

    if not isinstance(strategy_name, str):
        return response

    category_map = get_strategy_category_map(
        artifact_type
    )

    expected_category = category_map.get(
        strategy_name
    )

    if expected_category is not None:
        strategy["category"] = (
            expected_category
        )

    return response


class SemanticExtractor:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        retain_raw_model_response: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.retain_raw_model_response = (
            retain_raw_model_response
        )

    def extract(
        self,
        *,
        raw_submission: dict[str, Any],
        processed_submission: dict[str, Any],
        assessment_metadata: dict[str, Any] | None = None,
        rubric: dict[str, Any] | None = None,
    ) -> SemanticExtractionResult:
        raw_artifacts = (
            self._index_raw_artifacts(
                raw_submission
            )
        )

        unit_records = (
            self._collect_unit_records(
                processed_submission
            )
        )

        if not unit_records:
            raise SemanticExtractionError(
                "No valid units were found in the "
                "processed submission."
            )

        all_unit_ids = [
            record["unit"]["unit_id"]
            for record in unit_records
        ]

        unit_annotations: list[
            SemanticUnitAnnotation
        ] = []

        raw_unit_responses: dict[
            str,
            Any,
        ] = {}

        for index, record in enumerate(
            unit_records,
            start=1,
        ):
            unit = record["unit"]
            artifact_id = record["artifact_id"]
            artifact_type = record[
                "artifact_type"
            ]

            unit_id = unit["unit_id"]

            raw_source = raw_artifacts.get(
                artifact_id,
                "",
            )

            if not raw_source:
                raise SemanticExtractionError(
                    "No raw source content was found "
                    f"for artifact {artifact_id!r}, "
                    f"required by unit {unit_id!r}."
                )

            print(
                "Semantic extraction for unit "
                f"{index}/{len(unit_records)}: "
                f"{unit_id} "
                f"({artifact_type})"
            )

            annotation, raw_response = (
                self._extract_single_unit(
                    raw_source=raw_source,
                    unit=unit,
                    artifact_type=artifact_type,
                    all_unit_ids=all_unit_ids,
                    assessment_metadata=(
                        assessment_metadata
                    ),
                    rubric=rubric,
                )
            )

            unit_annotations.append(
                annotation
            )

            raw_unit_responses[unit_id] = (
                raw_response
            )

            print(
                "✓ Completed semantic extraction "
                f"for {unit_id}"
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
            model_name=(
                self.llm_client.model_name
            ),
            document_regions=[],
            unit_annotations=unit_annotations,
            unresolved_observations=[],
            warnings=[],
            raw_model_response=(
                {
                    "unit_responses": (
                        raw_unit_responses
                    )
                }
                if self.retain_raw_model_response
                else None
            ),
        )

        self._validate_result_references(
            result=result,
            processed_submission=(
                processed_submission
            ),
        )

        return result

    def _extract_single_unit(
        self,
        *,
        raw_source: str,
        unit: dict[str, Any],
        artifact_type: str,
        all_unit_ids: list[str],
        assessment_metadata: (
            dict[str, Any] | None
        ),
        rubric: dict[str, Any] | None,
    ) -> tuple[
        SemanticUnitAnnotation,
        dict[str, Any],
    ]:
        unit_id = unit.get("unit_id")

        if not isinstance(unit_id, str):
            raise SemanticExtractionError(
                "Cannot semantically extract a "
                "unit without a valid unit_id."
            )

        compact_unit = (
            self._build_compact_unit_input(
                unit
            )
        )

        unit_source = (
            self._get_unit_source_excerpt(
                raw_source,
                unit,
                context_lines=0,
            )
        )

        prompt = build_semantic_unit_prompt(
            raw_source=unit_source,
            unit=compact_unit,
            artifact_type=artifact_type,
            all_unit_ids=all_unit_ids,
            assessment_metadata=(
                assessment_metadata
            ),
            rubric=rubric,
        )

        output_schema = (
            self._build_unit_output_schema(
                unit=unit,
                artifact_type=artifact_type,
                all_unit_ids=all_unit_ids,
            )
        )

        max_attempts = 2
        response: dict[str, Any] | None = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                response = (
                    self.llm_client
                    .generate_structured(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt,
                        output_schema=(
                            output_schema
                        ),
                    )
                )

                break

            except APITimeoutError as exc:
                if attempt == max_attempts:
                    raise SemanticExtractionError(
                        "Semantic extraction timed "
                        f"out for {unit_id} after "
                        f"{max_attempts} attempts."
                    ) from exc

                print(
                    f"Timeout for {unit_id}; "
                    "retrying "
                    f"({attempt + 1}/"
                    f"{max_attempts})..."
                )

                time.sleep(2)

        if response is None:
            raise SemanticExtractionError(
                "No semantic response was "
                f"returned for {unit_id}."
            )

        response = (
            _normalise_strategy_category(
                response,
                artifact_type=artifact_type,
            )
        )

        try:
            llm_output = (
                SemanticUnitLLMOutput
                .model_validate(response)
            )

        except ValidationError as exc:
            raise SemanticExtractionError(
                "LLM returned invalid semantic "
                f"output for {unit_id}: {exc}"
            ) from exc

        annotation_data = (
            llm_output.model_dump()
        )

        # Domain annotations are owned by Python.
        # Any LLM-provided value is overwritten.
        annotation_data[
            "domain_annotations"
        ] = self._build_domain_annotations(
            unit=unit,
            artifact_type=artifact_type,
        )

        try:
            annotation = (
                SemanticUnitAnnotation(
                    unit_id=unit_id,
                    region_ids=[],
                    **annotation_data,
                )
            )

        except ValidationError as exc:
            raise SemanticExtractionError(
                "Could not construct the final "
                f"semantic annotation for "
                f"{unit_id}: {exc}"
            ) from exc

        self._validate_strategy_taxonomy(
            annotation=annotation,
            artifact_type=artifact_type,
        )

        return annotation, response

    @staticmethod
    def _index_raw_artifacts(
        raw_submission: dict[str, Any],
    ) -> dict[str, str]:
        raw_artifacts: dict[str, str] = {}

        for artifact in raw_submission.get(
            "artifacts",
            [],
        ):
            artifact_id = artifact.get(
                "artifact_id"
            )

            content = artifact.get(
                "content",
                {},
            )

            raw_content = content.get(
                "raw_content"
            )

            if (
                isinstance(artifact_id, str)
                and isinstance(
                    raw_content,
                    str,
                )
            ):
                raw_artifacts[
                    artifact_id
                ] = raw_content

        return raw_artifacts

    @staticmethod
    def _collect_unit_records(
        processed_submission: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for artifact in (
            processed_submission.get(
                "artifacts",
                [],
            )
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

                if not isinstance(
                    unit.get("unit_id"),
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
                        "unit": unit,
                    }
                )

        return records

    @staticmethod
    def _validate_strategy_taxonomy(
        *,
        annotation: SemanticUnitAnnotation,
        artifact_type: str,
    ) -> None:
        if annotation.strategy is None:
            return

        strategy_name = (
            annotation.strategy.name
        )

        strategy_category = (
            annotation.strategy.category
        )

        category_map = (
            get_strategy_category_map(
                artifact_type
            )
        )

        expected_category = (
            category_map.get(
                strategy_name
            )
        )

        if expected_category is None:
            raise SemanticExtractionError(
                "Unknown strategy name for "
                f"{annotation.unit_id} "
                f"({artifact_type}): "
                f"{strategy_name}"
            )

        if (
            strategy_category
            != expected_category
        ):
            raise SemanticExtractionError(
                "Invalid strategy category for "
                f"{annotation.unit_id}: "
                f"{strategy_name!r} must use "
                f"{expected_category!r}, not "
                f"{strategy_category!r}."
            )

    @staticmethod
    def _get_unit_source_excerpt(
        raw_source: str,
        unit: dict[str, Any],
        *,
        context_lines: int = 0,
    ) -> str:
        source_range = (
            unit.get("source_range")
            or {}
        )

        start_line = source_range.get(
            "start_line"
        )

        end_line = source_range.get(
            "end_line"
        )

        if (
            not isinstance(start_line, int)
            or not isinstance(
                end_line,
                int,
            )
        ):
            return SemanticExtractor._content_blocks_text(
                unit
            )

        lines = raw_source.splitlines()

        start_index = max(
            0,
            start_line
            - 1
            - context_lines,
        )

        end_index = min(
            len(lines),
            end_line + context_lines,
        )

        excerpt_lines = lines[
            start_index:end_index
        ]

        excerpt = "\n".join(
            excerpt_lines
        ).strip()

        if excerpt:
            return excerpt

        return (
            SemanticExtractor
            ._content_blocks_text(unit)
        )

    @staticmethod
    def _content_blocks_text(
        unit: dict[str, Any],
    ) -> str:
        content_values: list[str] = []

        for block in unit.get(
            "content_blocks",
            [],
        ):
            content = block.get(
                "content"
            )

            if isinstance(content, str):
                if content.strip():
                    content_values.append(
                        content.strip()
                    )

        return "\n\n".join(
            content_values
        )

    @staticmethod
    def _build_compact_unit_input(
        unit: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Retain deterministic unit data needed for interpretation.

        Unlike the original Lean-only implementation, this preserves
        ontology, Prolog, report, and Lean structured fields.
        """
        return {
            "unit_id": unit.get(
                "unit_id"
            ),
            "unit_type": unit.get(
                "unit_type"
            ),
            "label": unit.get(
                "label"
            ),
            "task_mapping": deepcopy(
                unit.get(
                    "task_mapping"
                )
            ),
            "attempt": deepcopy(
                unit.get("attempt")
            ),
            "source_range": deepcopy(
                unit.get(
                    "source_range"
                )
            ),
            "content_blocks": deepcopy(
                unit.get(
                    "content_blocks",
                    [],
                )
            ),
            "structured_data": deepcopy(
                unit.get(
                    "structured_data",
                    {},
                )
            ),
            "extraction_confidence": (
                unit.get(
                    "extraction_confidence"
                )
            ),
        }

    @staticmethod
    def _build_domain_annotations(
        *,
        unit: dict[str, Any],
        artifact_type: str,
    ) -> list[dict[str, Any]]:
        structured_data = (
            unit.get("structured_data")
            or {}
        )

        if artifact_type == "lean_source":
            return [
                {
                    "annotation_type": "lean",
                    "declaration_type": (
                        structured_data.get(
                            "declaration_type"
                        )
                    ),
                    "proof_style": (
                        structured_data.get(
                            "proof_style"
                        )
                    ),
                    "contains_sorry": (
                        structured_data.get(
                            "contains_sorry"
                        )
                    ),
                    "concepts": [],
                }
            ]

        if artifact_type == "prolog_source":
            formula = structured_data.get(
                "formula"
            )

            return [
                {
                    "annotation_type": (
                        "prolog"
                    ),
                    "predicate_name": (
                        structured_data.get(
                            "predicate_name"
                        )
                    ),
                    "formula_summary": (
                        formula
                        if isinstance(
                            formula,
                            str,
                        )
                        else None
                    ),
                    "operators": (
                        SemanticExtractor
                        ._extract_prolog_operators(
                            formula
                        )
                    ),
                    "modal_relations": (
                        SemanticExtractor
                        ._extract_modal_relations(
                            formula
                        )
                    ),
                    "constraint_type": (
                        str(
                            unit.get(
                                "unit_type",
                                "",
                            )
                        )
                    ),
                }
            ]

        if artifact_type == "owl_ontology":
            entity_names = (
                SemanticExtractor
                ._collect_ontology_entities(
                    structured_data
                )
            )

            return [
                {
                    "annotation_type": (
                        "ontology"
                    ),
                    "entity_type": (
                        structured_data.get(
                            "entity_type"
                        )
                    ),
                    "entity_names": (
                        entity_names
                    ),
                    "axiom_type": (
                        structured_data.get(
                            "axiom_type"
                        )
                    ),
                    "subject": (
                        structured_data.get(
                            "subject"
                        )
                    ),
                    "property_name": (
                        structured_data.get(
                            "property"
                        )
                    ),
                    "target": (
                        SemanticExtractor
                        ._ontology_target(
                            structured_data
                        )
                    ),
                    "filler": (
                        structured_data.get(
                            "filler"
                        )
                    ),
                    "cardinality": (
                        structured_data.get(
                            "cardinality"
                        )
                    ),
                    "modelling_concepts": [],
                }
            ]

        if artifact_type == "ontology_report":
            return [
                {
                    "annotation_type": (
                        "report"
                    ),
                    "section_type": (
                        structured_data.get(
                            "section_title"
                        )
                        or unit.get("label")
                    ),
                    "discussed_entities": [],
                    "claims": [],
                    "justification_targets": [],
                }
            ]

        return []

    @staticmethod
    def _extract_prolog_operators(
        formula: Any,
    ) -> list[str]:
        if not isinstance(formula, str):
            return []

        supported_operators = (
            "and",
            "or",
            "not",
            "implies",
            "dia",
            "box",
        )

        operators: list[str] = []

        for operator in (
            supported_operators
        ):
            if re.search(
                rf"\b{re.escape(operator)}\s*\(",
                formula,
            ):
                operators.append(
                    operator
                )

        return operators

    @staticmethod
    def _extract_modal_relations(
        formula: Any,
    ) -> list[str]:
        if not isinstance(formula, str):
            return []

        relations: list[str] = []

        pattern = re.compile(
            r"\b(?:dia|box)\s*\(\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)"
        )

        for match in pattern.finditer(
            formula
        ):
            relation = match.group(1)

            if relation not in relations:
                relations.append(
                    relation
                )

        return relations

    @staticmethod
    def _collect_ontology_entities(
        structured_data: dict[str, Any],
    ) -> list[str]:
        entity_names: list[str] = []

        singular_fields = (
            "entity",
            "subject",
            "parent",
            "property",
            "domain",
            "range",
            "subproperty",
            "superproperty",
            "filler",
            "characteristic",
        )

        for field_name in singular_fields:
            value = structured_data.get(
                field_name
            )

            if (
                isinstance(value, str)
                and value
                and value
                not in entity_names
            ):
                entity_names.append(
                    value
                )

        members = structured_data.get(
            "members"
        )

        if isinstance(members, list):
            for member in members:
                if (
                    isinstance(member, str)
                    and member
                    and member
                    not in entity_names
                ):
                    entity_names.append(
                        member
                    )

        return entity_names

    @staticmethod
    def _ontology_target(
        structured_data: dict[str, Any],
    ) -> str | None:
        for field_name in (
            "parent",
            "domain",
            "range",
            "superproperty",
            "characteristic",
        ):
            value = structured_data.get(
                field_name
            )

            if isinstance(value, str):
                return value

        return None

    @staticmethod
    def _build_unit_output_schema(
        *,
        unit: dict[str, Any],
        artifact_type: str,
        all_unit_ids: list[str],
    ) -> dict[str, Any]:
        schema = deepcopy(
            SemanticUnitLLMOutput
            .model_json_schema()
        )

        # The LLM does not own deterministic domain annotations.
        properties = schema.get(
            "properties"
        )

        if isinstance(properties, dict):
            properties.pop(
                "domain_annotations",
                None,
            )

        required = schema.get(
            "required"
        )

        if isinstance(required, list):
            schema["required"] = [
                field_name
                for field_name in required
                if field_name
                != "domain_annotations"
            ]

        structured_data = (
            unit.get("structured_data")
            or {}
        )

        valid_context_ids = [
            context["context_id"]
            for context in (
                structured_data.get(
                    "context_declarations",
                    [],
                )
            )
            if isinstance(
                context,
                dict,
            )
            and isinstance(
                context.get(
                    "context_id"
                ),
                str,
            )
        ]

        strategy_names = list(
            get_strategy_names(
                artifact_type
            )
        )

        strategy_category_map = (
            get_strategy_category_map(
                artifact_type
            )
        )

        strategy_categories = list(
            dict.fromkeys(
                strategy_category_map.values()
            )
        )

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                nested_properties = (
                    value.get(
                        "properties"
                    )
                )

                if isinstance(
                    nested_properties,
                    dict,
                ):
                    context_id_schema = (
                        nested_properties.get(
                            "context_id"
                        )
                    )

                    if isinstance(
                        context_id_schema,
                        dict,
                    ):
                        context_id_schema.clear()

                        if valid_context_ids:
                            context_id_schema.update(
                                {
                                    "type": (
                                        "string"
                                    ),
                                    "enum": (
                                        valid_context_ids
                                    ),
                                }
                            )
                        else:
                            context_id_schema.update(
                                {
                                    "type": (
                                        "string"
                                    )
                                }
                            )

                    target_unit_schema = (
                        nested_properties.get(
                            "target_unit_id"
                        )
                    )

                    if isinstance(
                        target_unit_schema,
                        dict,
                    ):
                        target_unit_schema.clear()
                        target_unit_schema.update(
                            {
                                "type": "string",
                                "enum": (
                                    all_unit_ids
                                ),
                            }
                        )

                    related_unit_ids_schema = (
                        nested_properties.get(
                            "related_unit_ids"
                        )
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
                                    "type": (
                                        "string"
                                    ),
                                    "enum": (
                                        all_unit_ids
                                    ),
                                },
                            }
                        )

                    category_schema = (
                        nested_properties.get(
                            "category"
                        )
                    )

                    name_schema = (
                        nested_properties.get(
                            "name"
                        )
                    )

                    if (
                        isinstance(
                            category_schema,
                            dict,
                        )
                        and isinstance(
                            name_schema,
                            dict,
                        )
                    ):
                        category_schema.clear()
                        category_schema.update(
                            {
                                "type": "string",
                                "enum": (
                                    strategy_categories
                                ),
                            }
                        )

                        name_schema.clear()
                        name_schema.update(
                            {
                                "type": "string",
                                "enum": (
                                    strategy_names
                                ),
                            }
                        )

                    evidence_schema = (
                        nested_properties.get(
                            "evidence"
                        )
                    )

                    if isinstance(
                        evidence_schema,
                        dict,
                    ):
                        evidence_schema[
                            "minItems"
                        ] = 1

                for nested_value in (
                    value.values()
                ):
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
            for artifact in (
                processed_submission.get(
                    "artifacts",
                    [],
                )
            )
            for unit in artifact.get(
                "units",
                [],
            )
            if isinstance(
                unit.get("unit_id"),
                str,
            )
        }

        valid_context_ids_by_unit: dict[
            str,
            set[str],
        ] = {}

        for artifact in (
            processed_submission.get(
                "artifacts",
                [],
            )
        ):
            for unit in artifact.get(
                "units",
                [],
            ):
                unit_id = unit.get(
                    "unit_id"
                )

                if not isinstance(
                    unit_id,
                    str,
                ):
                    continue

                structured_data = (
                    unit.get(
                        "structured_data"
                    )
                    or {}
                )

                contexts = (
                    structured_data.get(
                        "context_declarations",
                        [],
                    )
                )

                valid_context_ids_by_unit[
                    unit_id
                ] = {
                    context["context_id"]
                    for context in contexts
                    if isinstance(
                        context,
                        dict,
                    )
                    and isinstance(
                        context.get(
                            "context_id"
                        ),
                        str,
                    )
                }

        errors: list[str] = []

        seen_annotation_unit_ids: set[
            str
        ] = set()

        for annotation in (
            result.unit_annotations
        ):
            if (
                annotation.unit_id
                in seen_annotation_unit_ids
            ):
                errors.append(
                    "Duplicate unit annotation "
                    "for unit ID: "
                    f"{annotation.unit_id}"
                )

            seen_annotation_unit_ids.add(
                annotation.unit_id
            )

            if (
                annotation.unit_id
                not in valid_unit_ids
            ):
                errors.append(
                    "Unknown unit ID in semantic "
                    "output: "
                    f"{annotation.unit_id}"
                )

                continue

            valid_context_ids = (
                valid_context_ids_by_unit.get(
                    annotation.unit_id,
                    set(),
                )
            )

            for selection in (
                annotation.relevant_context
            ):
                if (
                    selection.context_id
                    not in valid_context_ids
                ):
                    errors.append(
                        "Unknown context ID "
                        f"{selection.context_id!r} "
                        "for unit "
                        f"{annotation.unit_id!r}"
                    )

            for relationship in (
                annotation.relationships
            ):
                if (
                    relationship.target_unit_id
                    not in valid_unit_ids
                ):
                    errors.append(
                        "Unknown relationship "
                        "target unit ID: "
                        f"{relationship.target_unit_id}"
                    )

        missing_unit_ids = (
            valid_unit_ids
            - seen_annotation_unit_ids
        )

        for unit_id in sorted(
            missing_unit_ids
        ):
            errors.append(
                "Missing semantic annotation "
                f"for unit ID: {unit_id}"
            )

        valid_region_ids = {
            region.region_id
            for region in (
                result.document_regions
            )
        }

        if (
            len(valid_region_ids)
            != len(
                result.document_regions
            )
        ):
            errors.append(
                "Duplicate region IDs found "
                "in document_regions."
            )

        for region in (
            result.document_regions
        ):
            for unit_id in region.unit_ids:
                if (
                    unit_id
                    not in valid_unit_ids
                ):
                    errors.append(
                        f"Unknown unit ID "
                        f"{unit_id!r} in region "
                        f"{region.region_id!r}"
                    )

        for annotation in (
            result.unit_annotations
        ):
            for region_id in (
                annotation.region_ids
            ):
                if (
                    region_id
                    not in valid_region_ids
                ):
                    errors.append(
                        f"Unknown region ID "
                        f"{region_id!r} "
                        "referenced by unit "
                        f"{annotation.unit_id!r}"
                    )

        if errors:
            raise SemanticExtractionError(
                "Semantic output contains "
                "invalid references:\n- "
                + "\n- ".join(errors)
            )


def load_json(
    path: str,
) -> dict[str, Any]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_semantic_result(
    result: SemanticExtractionResult,
    output_path: str,
) -> None:
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.model_dump(
                mode="json"
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )