from __future__ import annotations

import gc
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
    ComponentAlignment,
)
from .alignment_prompts import (
    build_document_alignment_prompt,
)


class AlignmentError(RuntimeError):
    """Raised when assessment alignment fails."""


def _build_component_candidate(
    *,
    component: dict[str, Any],
    part_id: str | None,
) -> dict[str, Any] | None:
    """
    Build the compact component representation supplied to the aligner.

    Only information useful for evidence routing is retained. Detailed
    rubric requirements and marks remain the responsibility of feedback
    generation.
    """
    component_id = component.get(
        "component_id"
    )

    if not isinstance(
        component_id,
        str,
    ):
        return None

    candidate: dict[str, Any] = {
        "part_id": part_id,
        "component_id": component_id,
        "title": component.get(
            "title"
        ),
    }

    artifact_type = component.get(
        "artifact_type"
    )

    if isinstance(
        artifact_type,
        str,
    ):
        candidate[
            "artifact_type"
        ] = artifact_type

    artifact = component.get(
        "artifact"
    )

    if isinstance(
        artifact,
        dict,
    ):
        candidate["artifact"] = {
            key: value
            for key, value
            in artifact.items()
            if key
            in {
                "artifact_type",
                "answer_location",
                "answer_locations",
            }
        }

    artifacts = component.get(
        "artifacts"
    )

    if isinstance(
        artifacts,
        list,
    ):
        candidate["artifacts"] = [
            {
                key: value
                for key, value
                in artifact_spec.items()
                if key
                in {
                    "artifact_type",
                    "required",
                }
            }
            for artifact_spec
            in artifacts
            if isinstance(
                artifact_spec,
                dict,
            )
        ]

    task = component.get(
        "task",
        {},
    )

    if isinstance(
        task,
        dict,
    ):
        task_summary = task.get(
            "summary"
        )

        if not isinstance(
            task_summary,
            str,
        ):
            task_summary = task.get(
                "description"
            )

        if isinstance(
            task_summary,
            str,
        ):
            candidate[
                "task_summary"
            ] = task_summary

        target = task.get(
            "target"
        )

        if isinstance(
            target,
            dict,
        ):
            candidate[
                "target"
            ] = {
                key: value
                for key, value
                in target.items()
                if key
                in {
                    "proposition",
                    "concept",
                    "description",
                    "exact_student_identifier_required",
                }
            }

        target_statement = task.get(
            "target_statement"
        )

        if isinstance(
            target_statement,
            str,
        ):
            candidate[
                "target_statement"
            ] = target_statement

        stated_conditions = task.get(
            "stated_conditions"
        )

        if isinstance(
            stated_conditions,
            list,
        ):
            candidate[
                "stated_conditions"
            ] = [
                {
                    key: value
                    for key, value
                    in item.items()
                    if key
                    in {
                        "id",
                        "description",
                        "logical_form",
                        "semantic_structure",
                        "formalisation_note",
                    }
                }
                for item in stated_conditions
                if isinstance(
                    item,
                    dict,
                )
            ]

        required_cases = task.get(
            "required_cases"
        )

        if isinstance(
            required_cases,
            list,
        ):
            candidate[
                "required_cases"
            ] = required_cases

        semantic_goal = task.get(
            "semantic_goal"
        )

        if isinstance(
            semantic_goal,
            dict,
        ):
            candidate[
                "semantic_goal"
            ] = semantic_goal

        expected_properties = task.get(
            "expected_properties"
        )

        if isinstance(
            expected_properties,
            list,
        ):
            candidate[
                "expected_properties"
            ] = expected_properties

        required_examples = task.get(
            "required_examples"
        )

        if isinstance(
            required_examples,
            int,
        ):
            candidate[
                "required_examples"
            ] = required_examples

    return candidate


def _infer_assessment_2_part_id(
    *,
    component: dict[str, Any],
    assessment_specification: dict[str, Any],
) -> str | None:
    """
    Infer the parent part for Assessment 2 top-level components.
    """
    component_id = component.get(
        "component_id"
    )

    if not isinstance(
        component_id,
        str,
    ):
        return None

    assessment_structure = (
        assessment_specification.get(
            "assessment_structure",
            {},
        )
    )

    if not isinstance(
        assessment_structure,
        dict,
    ):
        return None

    part_ids = [
        part.get("id")
        for part
        in assessment_structure.values()
        if (
            isinstance(
                part,
                dict,
            )
            and isinstance(
                part.get("id"),
                str,
            )
        )
    ]

    for part_id in part_ids:
        if component_id == part_id:
            return part_id

    if component_id.startswith(
        "part_1"
    ):
        for part_id in part_ids:
            if part_id.startswith(
                "part_1"
            ):
                return part_id

    if component_id.startswith(
        "part_2"
    ):
        for part_id in part_ids:
            if part_id.startswith(
                "part_2"
            ):
                return part_id

    return None


def build_component_candidates(
    assessment_specification: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build compact assessment-component representations.

    Assessment 1:
        parts -> components

    Assessment 2:
        top-level components
    """
    candidates: list[
        dict[str, Any]
    ] = []

    for part in (
        assessment_specification.get(
            "parts",
            [],
        )
    ):
        if not isinstance(
            part,
            dict,
        ):
            continue

        part_id = part.get(
            "part_id"
        )

        if not isinstance(
            part_id,
            str,
        ):
            part_id = None

        for component in part.get(
            "components",
            [],
        ):
            if not isinstance(
                component,
                dict,
            ):
                continue

            candidate = (
                _build_component_candidate(
                    component=component,
                    part_id=part_id,
                )
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

    for component in (
        assessment_specification.get(
            "components",
            [],
        )
    ):
        if not isinstance(
            component,
            dict,
        ):
            continue

        part_id = (
            _infer_assessment_2_part_id(
                component=component,
                assessment_specification=(
                    assessment_specification
                ),
            )
        )

        candidate = (
            _build_component_candidate(
                component=component,
                part_id=part_id,
            )
        )

        if candidate is not None:
            candidates.append(
                candidate
            )

    return candidates


def _build_alignment_output_schema(
    *,
    valid_component_ids: list[str],
    valid_unit_ids: list[str],
) -> dict[str, Any]:
    schema = deepcopy(
        AlignmentLLMOutput
        .model_json_schema()
    )

    unit_id_fields = {
        "primary_unit_ids",
        "supporting_unit_ids",
        "possibly_relevant_unit_ids",
    }

    def visit(
        value: Any,
    ) -> None:
        if isinstance(
            value,
            dict,
        ):
            properties = value.get(
                "properties"
            )

            if isinstance(
                properties,
                dict,
            ):
                component_id_schema = (
                    properties.get(
                        "component_id"
                    )
                )

                if isinstance(
                    component_id_schema,
                    dict,
                ):
                    component_id_schema.clear()
                    component_id_schema.update(
                        {
                            "type": "string",
                            "enum": (
                                valid_component_ids
                            ),
                        }
                    )

                for field_name in (
                    unit_id_fields
                ):
                    unit_ids_schema = (
                        properties.get(
                            field_name
                        )
                    )

                    if isinstance(
                        unit_ids_schema,
                        dict,
                    ):
                        unit_ids_schema.clear()
                        unit_ids_schema.update(
                            {
                                "type": "array",
                                "items": {
                                    "type": (
                                        "string"
                                    ),
                                    "enum": (
                                        valid_unit_ids
                                    ),
                                },
                                "uniqueItems": True,
                            }
                        )

            for nested_value in (
                value.values()
            ):
                visit(
                    nested_value
                )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                visit(item)

    visit(schema)

    return schema


def _predicate_from_answer_location(
    value: Any,
) -> str | None:
    """
    Convert answer1/1 into answer1.
    """
    if not isinstance(
        value,
        str,
    ):
        return None

    value = value.strip()

    if not value:
        return None

    return value.split(
        "/",
        1,
    )[0]


def _get_unit_structured_data(
    unit: Any,
) -> dict[str, Any]:
    value = getattr(
        unit,
        "structured_data",
        None,
    )

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


def _get_unit_task_ids(
    unit: Any,
) -> list[str]:
    task_mapping = getattr(
        unit,
        "task_mapping",
        None,
    )

    if task_mapping is None:
        return []

    task_ids = getattr(
        task_mapping,
        "task_ids",
        None,
    )

    if not isinstance(
        task_ids,
        list,
    ):
        return []

    return [
        task_id
        for task_id in task_ids
        if isinstance(
            task_id,
            str,
        )
    ]


def _artifact_type_matches(
    *,
    actual: str,
    expected: str,
) -> bool:
    """
    Account for specification/processor naming differences.
    """
    if actual == expected:
        return True

    aliases = {
        "report": {
            "report",
            "ontology_report",
        },
        "ontology_report": {
            "report",
            "ontology_report",
        },
    }

    return (
        actual
        in aliases.get(
            expected,
            set(),
        )
    )


def _select_assessment_2_units(
    *,
    processed_submission: ProcessedSubmission,
    component: dict[str, Any],
) -> list[Any]:
    """
    Deterministically narrow the evidence pool for one Assessment 2
    component before asking the LLM to perform alignment.

    This is candidate selection only.
    """
    component_id = component[
        "component_id"
    ]

    unit_records: list[
        tuple[Any, str]
    ] = [
        (
            unit,
            artifact.artifact_type,
        )
        for artifact
        in processed_submission.artifacts
        for unit in artifact.units
    ]

    expected_artifact_types: set[
        str
    ] = set()

    artifacts = component.get(
        "artifacts"
    )

    if isinstance(
        artifacts,
        list,
    ):
        for artifact_spec in artifacts:
            if not isinstance(
                artifact_spec,
                dict,
            ):
                continue

            artifact_type = (
                artifact_spec.get(
                    "artifact_type"
                )
            )

            if isinstance(
                artifact_type,
                str,
            ):
                expected_artifact_types.add(
                    artifact_type
                )

    if expected_artifact_types:
        selected = [
            unit
            for unit, actual_type
            in unit_records
            if any(
                _artifact_type_matches(
                    actual=actual_type,
                    expected=expected_type,
                )
                for expected_type
                in expected_artifact_types
            )
        ]

        if selected:
            return selected

    artifact_spec = component.get(
        "artifact"
    )

    expected_predicates: set[
        str
    ] = set()

    expected_artifact_type: (
        str | None
    ) = None

    if isinstance(
        artifact_spec,
        dict,
    ):
        artifact_type = (
            artifact_spec.get(
                "artifact_type"
            )
        )

        if isinstance(
            artifact_type,
            str,
        ):
            expected_artifact_type = (
                artifact_type
            )

        single_location = (
            _predicate_from_answer_location(
                artifact_spec.get(
                    "answer_location"
                )
            )
        )

        if single_location:
            expected_predicates.add(
                single_location
            )

        multiple_locations = (
            artifact_spec.get(
                "answer_locations"
            )
        )

        if isinstance(
            multiple_locations,
            list,
        ):
            for location in (
                multiple_locations
            ):
                predicate = (
                    _predicate_from_answer_location(
                        location
                    )
                )

                if predicate:
                    expected_predicates.add(
                        predicate
                    )

    if expected_predicates:
        selected: list[
            Any
        ] = []

        for unit, actual_type in (
            unit_records
        ):
            if (
                expected_artifact_type
                is not None
                and not _artifact_type_matches(
                    actual=actual_type,
                    expected=(
                        expected_artifact_type
                    ),
                )
            ):
                continue

            structured_data = (
                _get_unit_structured_data(
                    unit
                )
            )

            predicate_name = (
                structured_data.get(
                    "predicate_name"
                )
            )

            if (
                isinstance(
                    predicate_name,
                    str,
                )
                and predicate_name
                in expected_predicates
            ):
                selected.append(
                    unit
                )
                continue

            if (
                component_id
                in _get_unit_task_ids(
                    unit
                )
            ):
                selected.append(
                    unit
                )

        if selected:
            return selected

    if (
        expected_artifact_type
        is not None
    ):
        selected = [
            unit
            for unit, actual_type
            in unit_records
            if _artifact_type_matches(
                actual=actual_type,
                expected=(
                    expected_artifact_type
                ),
            )
        ]

        if selected:
            return selected

    return [
        unit
        for unit, _ in unit_records
        if (
            component_id
            in _get_unit_task_ids(
                unit
            )
        )
    ]


def _chunk_list(
    values: list[Any],
    *,
    chunk_size: int,
) -> list[list[Any]]:
    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be at least 1."
        )

    return [
        values[
            index:
            index + chunk_size
        ]
        for index in range(
            0,
            len(values),
            chunk_size,
        )
    ]


def _release_inference_memory() -> None:
    """
    Release Python references and unused CUDA cache between independent
    model calls.

    This does not unload the model itself.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        pass


class DocumentAlignmentService:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        aligner_version: str = "1.0",
        assessment_2_chunk_size: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.aligner_version = (
            aligner_version
        )
        self.assessment_2_chunk_size = (
            assessment_2_chunk_size
        )

    def align_submission(
        self,
        *,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        assessment_specification: dict[str, Any],
    ) -> AlignmentResult:
        component_candidates = (
            build_component_candidates(
                assessment_specification
            )
        )

        if not component_candidates:
            raise AlignmentError(
                "No valid assessment "
                "components were found."
            )

        all_units = [
            unit
            for artifact
            in processed_submission.artifacts
            for unit in artifact.units
        ]

        if not all_units:
            raise AlignmentError(
                "No processed submission "
                "units were found."
            )

        is_assessment_2_layout = (
            isinstance(
                assessment_specification.get(
                    "components"
                ),
                list,
            )
            and bool(
                assessment_specification.get(
                    "components"
                )
            )
        )

        if is_assessment_2_layout:
            return (
                self._align_assessment_2(
                    processed_submission=(
                        processed_submission
                    ),
                    semantic_extraction=(
                        semantic_extraction
                    ),
                    component_candidates=(
                        component_candidates
                    ),
                )
            )

        return self._align_whole_submission(
            processed_submission=(
                processed_submission
            ),
            semantic_extraction=(
                semantic_extraction
            ),
            component_candidates=(
                component_candidates
            ),
        )

    def _align_whole_submission(
        self,
        *,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        component_candidates: list[
            dict[str, Any]
        ],
    ) -> AlignmentResult:
        """
        Preserve the existing Assessment 1 behaviour.
        """
        processed_units = [
            unit
            for artifact
            in processed_submission.artifacts
            for unit in artifact.units
        ]

        semantic_annotations = list(
            semantic_extraction
            .unit_annotations
        )

        valid_component_ids = [
            candidate[
                "component_id"
            ]
            for candidate
            in component_candidates
        ]

        valid_unit_ids = [
            unit.unit_id
            for unit in processed_units
        ]

        system_prompt, user_prompt = (
            build_document_alignment_prompt(
                processed_units=(
                    processed_units
                ),
                semantic_annotations=(
                    semantic_annotations
                ),
                component_candidates=(
                    component_candidates
                ),
            )
        )

        output_schema = (
            _build_alignment_output_schema(
                valid_component_ids=(
                    valid_component_ids
                ),
                valid_unit_ids=(
                    valid_unit_ids
                ),
            )
        )

        response = (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                output_schema=(
                    output_schema
                ),
            )
        )

        try:
            llm_output = (
                AlignmentLLMOutput
                .model_validate(
                    response
                )
            )

        except ValidationError as exc:
            raise AlignmentError(
                "The model returned invalid "
                "document alignment."
            ) from exc

        self._validate_output_references(
            output=llm_output,
            valid_component_ids=set(
                valid_component_ids
            ),
            valid_unit_ids=set(
                valid_unit_ids
            ),
        )

        return AlignmentResult(
            schema_version="1.0",
            assessment_id=(
                processed_submission
                .assessment_id
            ),
            source_submission_id=(
                processed_submission
                .source_submission_id
            ),
            processed_submission_id=(
                processed_submission
                .processed_submission_id
            ),
            model_name=(
                self.llm_client
                .model_name
            ),
            aligner_version=(
                self.aligner_version
            ),
            **llm_output.model_dump(),
        )

    def _align_assessment_2(
        self,
        *,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        component_candidates: list[
            dict[str, Any]
        ],
    ) -> AlignmentResult:
        """
        Align Assessment 2 component-by-component.

        Large components are divided into bounded chunks to keep inference
        memory usage manageable.
        """
        semantic_by_unit_id = {
            annotation.unit_id: annotation
            for annotation
            in semantic_extraction
            .unit_annotations
        }

        final_alignments: list[
            ComponentAlignment
        ] = []

        warnings: list[str] = []

        component_count = len(
            component_candidates
        )

        for component_index, component in (
            enumerate(
                component_candidates,
                start=1,
            )
        ):
            component_id = component[
                "component_id"
            ]

            candidate_units = (
                _select_assessment_2_units(
                    processed_submission=(
                        processed_submission
                    ),
                    component=component,
                )
            )

            print()
            print(
                "Alignment for component "
                f"{component_index}/"
                f"{component_count}: "
                f"{component_id}"
            )

            print(
                "Candidate units:",
                len(candidate_units),
            )

            if not candidate_units:
                final_alignments.append(
                    ComponentAlignment(
                        component_id=(
                            component_id
                        ),
                        primary_unit_ids=[],
                        supporting_unit_ids=[],
                        possibly_relevant_unit_ids=[],
                        reason=(
                            "No processed units were "
                            "deterministically identified "
                            "as candidate evidence for "
                            "this component."
                        ),
                        confidence=1.0,
                        unresolved=True,
                        unresolved_reason=(
                            "No candidate submission "
                            "units were identified."
                        ),
                    )
                )

                warnings.append(
                    f"{component_id}: no candidate "
                    "units were identified."
                )

                continue

            chunks = _chunk_list(
                candidate_units,
                chunk_size=(
                    self
                    .assessment_2_chunk_size
                ),
            )

            chunk_alignments: list[
                ComponentAlignment
            ] = []

            for chunk_index, chunk in (
                enumerate(
                    chunks,
                    start=1,
                )
            ):
                print(
                    "  Chunk "
                    f"{chunk_index}/"
                    f"{len(chunks)} "
                    f"({len(chunk)} units)"
                )

                chunk_unit_ids = {
                    unit.unit_id
                    for unit in chunk
                }

                chunk_semantics = [
                    semantic_by_unit_id[
                        unit_id
                    ]
                    for unit_id
                    in chunk_unit_ids
                    if (
                        unit_id
                        in semantic_by_unit_id
                    )
                ]

                try:
                    chunk_output = (
                        self
                        ._align_component_chunk(
                            component=component,
                            processed_units=chunk,
                            semantic_annotations=(
                                chunk_semantics
                            ),
                        )
                    )

                    chunk_alignments.append(
                        chunk_output
                    )

                finally:
                    _release_inference_memory()

            merged = (
                self
                ._merge_component_chunks(
                    component_id=(
                        component_id
                    ),
                    alignments=(
                        chunk_alignments
                    ),
                )
            )

            final_alignments.append(
                merged
            )

            _release_inference_memory()

            print(
                "✓ Completed alignment for "
                f"{component_id}"
            )

        llm_output = AlignmentLLMOutput(
            component_alignments=(
                final_alignments
            ),
            summary=(
                "Assessment components were "
                "aligned using deterministic "
                "candidate selection followed by "
                "component-level model-assisted "
                "evidence routing."
            ),
            warnings=warnings,
        )

        valid_component_ids = {
            candidate[
                "component_id"
            ]
            for candidate
            in component_candidates
        }

        valid_unit_ids = {
            unit.unit_id
            for artifact
            in processed_submission.artifacts
            for unit in artifact.units
        }

        self._validate_output_references(
            output=llm_output,
            valid_component_ids=(
                valid_component_ids
            ),
            valid_unit_ids=(
                valid_unit_ids
            ),
        )

        return AlignmentResult(
            schema_version="1.0",
            assessment_id=(
                processed_submission
                .assessment_id
            ),
            source_submission_id=(
                processed_submission
                .source_submission_id
            ),
            processed_submission_id=(
                processed_submission
                .processed_submission_id
            ),
            model_name=(
                self.llm_client
                .model_name
            ),
            aligner_version=(
                self.aligner_version
            ),
            **llm_output.model_dump(),
        )

    def _align_component_chunk(
        self,
        *,
        component: dict[str, Any],
        processed_units: list[Any],
        semantic_annotations: list[Any],
    ) -> ComponentAlignment:
        """
        Run one bounded model request for one component and one unit chunk.
        """
        component_id = component[
            "component_id"
        ]

        valid_unit_ids = [
            unit.unit_id
            for unit in processed_units
        ]

        system_prompt, user_prompt = (
            build_document_alignment_prompt(
                processed_units=(
                    processed_units
                ),
                semantic_annotations=(
                    semantic_annotations
                ),
                component_candidates=[
                    component
                ],
            )
        )

        output_schema = (
            _build_alignment_output_schema(
                valid_component_ids=[
                    component_id
                ],
                valid_unit_ids=(
                    valid_unit_ids
                ),
            )
        )

        response = (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                output_schema=(
                    output_schema
                ),
            )
        )

        try:
            output = (
                AlignmentLLMOutput
                .model_validate(
                    response
                )
            )

        except ValidationError as exc:
            raise AlignmentError(
                "The model returned invalid "
                "component alignment for "
                f"{component_id!r}."
            ) from exc

        self._validate_output_references(
            output=output,
            valid_component_ids={
                component_id
            },
            valid_unit_ids=set(
                valid_unit_ids
            ),
        )

        if (
            len(
                output.component_alignments
            )
            != 1
        ):
            raise AlignmentError(
                "Expected exactly one component "
                "alignment for "
                f"{component_id!r}, but received "
                f"{len(output.component_alignments)}."
            )

        return (
            output.component_alignments[
                0
            ]
        )

    @staticmethod
    def _merge_component_chunks(
        *,
        component_id: str,
        alignments: list[
            ComponentAlignment
        ],
    ) -> ComponentAlignment:
        """
        Merge chunk-level results for one component.

        Category precedence:
        primary > supporting > possibly relevant
        """
        primary_ids: list[str] = []
        supporting_ids: list[str] = []
        possible_ids: list[str] = []

        reasons: list[str] = []
        confidences: list[
            float
        ] = []

        for alignment in alignments:
            reasons.append(
                alignment.reason
            )

            confidences.append(
                alignment.confidence
            )

            for unit_id in (
                alignment
                .primary_unit_ids
            ):
                if (
                    unit_id
                    not in primary_ids
                ):
                    primary_ids.append(
                        unit_id
                    )

            for unit_id in (
                alignment
                .supporting_unit_ids
            ):
                if (
                    unit_id
                    not in supporting_ids
                ):
                    supporting_ids.append(
                        unit_id
                    )

            for unit_id in (
                alignment
                .possibly_relevant_unit_ids
            ):
                if (
                    unit_id
                    not in possible_ids
                ):
                    possible_ids.append(
                        unit_id
                    )

        supporting_ids = [
            unit_id
            for unit_id
            in supporting_ids
            if unit_id
            not in primary_ids
        ]

        possible_ids = [
            unit_id
            for unit_id
            in possible_ids
            if (
                unit_id
                not in primary_ids
                and unit_id
                not in supporting_ids
            )
        ]

        has_units = bool(
            primary_ids
            or supporting_ids
            or possible_ids
        )

        if confidences:
            confidence = (
                sum(confidences)
                / len(confidences)
            )
        else:
            confidence = 0.0

        unique_reasons = list(
            dict.fromkeys(
                reason
                for reason in reasons
                if reason
            )
        )

        reason = " ".join(
            unique_reasons
        )

        if len(reason) > 1200:
            reason = (
                reason[:1197]
                + "..."
            )

        if not reason:
            reason = (
                "Chunk-level alignment "
                "results were merged."
            )

        if has_units:
            return ComponentAlignment(
                component_id=(
                    component_id
                ),
                primary_unit_ids=(
                    primary_ids
                ),
                supporting_unit_ids=(
                    supporting_ids
                ),
                possibly_relevant_unit_ids=(
                    possible_ids
                ),
                reason=reason,
                confidence=confidence,
                unresolved=False,
                unresolved_reason=None,
            )

        return ComponentAlignment(
            component_id=(
                component_id
            ),
            primary_unit_ids=[],
            supporting_unit_ids=[],
            possibly_relevant_unit_ids=[],
            reason=reason,
            confidence=confidence,
            unresolved=True,
            unresolved_reason=(
                "No candidate unit was aligned "
                "reliably across the component "
                "chunks."
            ),
        )

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
            for alignment
            in output.component_alignments
        }

        missing_component_ids = (
            valid_component_ids
            - returned_component_ids
        )

        unknown_component_ids = (
            returned_component_ids
            - valid_component_ids
        )

        if missing_component_ids:
            errors.append(
                "Missing component alignments "
                "for: "
                + ", ".join(
                    sorted(
                        missing_component_ids
                    )
                )
            )

        if unknown_component_ids:
            errors.append(
                "Unknown component IDs: "
                + ", ".join(
                    sorted(
                        unknown_component_ids
                    )
                )
            )

        for alignment in (
            output.component_alignments
        ):
            aligned_unit_ids = (
                set(
                    alignment
                    .primary_unit_ids
                )
                | set(
                    alignment
                    .supporting_unit_ids
                )
                | set(
                    alignment
                    .possibly_relevant_unit_ids
                )
            )

            unknown_unit_ids = (
                aligned_unit_ids
                - valid_unit_ids
            )

            if unknown_unit_ids:
                errors.append(
                    f"Component "
                    f"{alignment.component_id!r} "
                    "references unknown unit IDs: "
                    + ", ".join(
                        sorted(
                            unknown_unit_ids
                        )
                    )
                )

        if errors:
            raise AlignmentError(
                "Alignment output contains "
                "invalid references:\n- "
                + "\n- ".join(
                    errors
                )
            )