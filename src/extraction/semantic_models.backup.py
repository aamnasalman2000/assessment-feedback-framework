from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SemanticRole(str, Enum):
    MAIN_ANSWER = "main_answer"
    SUPPORTING_DEFINITION = "supporting_definition"
    ASSUMPTION = "assumption"
    METHOD = "method"
    IMPLEMENTATION = "implementation"
    REASONING = "reasoning"
    EVIDENCE = "evidence"
    RESULT = "result"
    INTERPRETATION = "interpretation"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"
    EXPLANATION = "explanation"
    OTHER = "other"


class IntentType(str, Enum):
    PROVE_CLAIM = "prove_claim"
    DEFINE_CONCEPT = "define_concept"
    IMPLEMENT_SOLUTION = "implement_solution"
    EXPLAIN_REASONING = "explain_reasoning"
    ANALYSE_RESULT = "analyse_result"
    COMPARE_APPROACHES = "compare_approaches"
    JUSTIFY_DECISION = "justify_decision"
    PRESENT_RESULT = "present_result"
    ANSWER_QUESTION = "answer_question"
    OTHER = "other"


class CommentRole(str, Enum):
    HEADING = "heading"
    TASK_DESCRIPTION = "task_description"
    STUDENT_EXPLANATION = "student_explanation"
    STRATEGY = "strategy"
    STEP_EXPLANATION = "step_explanation"
    IMPLEMENTATION_NOTE = "implementation_note"
    RESULT_INTERPRETATION = "result_interpretation"
    UNCERTAINTY = "uncertainty"
    TODO = "todo"
    DECORATIVE_SEPARATOR = "decorative_separator"
    OTHER = "other"


class RelationshipType(str, Enum):
    ANSWERS = "answers"
    DEFINES = "defines"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"
    EXPLAINS = "explains"
    IMPLEMENTS = "implements"
    PRODUCES = "produces"
    EVALUATES = "evaluates"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    SUMMARISES = "summarises"
    ALTERNATIVE_TO = "alternative_to"
    OTHER = "other"


class SourceRange(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> SourceRange:
        if self.end_line < self.start_line:
            raise ValueError(
                "end_line must be greater than or equal to start_line"
            )

        return self


class EvidenceReference(BaseModel):
    reference_type: Literal[
        "raw_source",
        "unit",
        "content_block",
        "context",
        "comment",
        "assessment_metadata",
        "rubric",
    ]

    reference_id: str | None = None
    source_range: SourceRange | None = None
    excerpt: str | None = None


class InferredValue(BaseModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class DocumentRegion(BaseModel):
    region_id: str
    region_type: Literal[
        "assessment_part",
        "question",
        "scenario",
        "task",
        "section",
        "subsection",
        "supporting_material",
        "unclassified",
    ]

    label: str | None = None
    source_range: SourceRange
    unit_ids: list[str] = Field(default_factory=list)
    parent_region_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class StudentIntent(BaseModel):
    intent_type: IntentType
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class StrategyAnnotation(BaseModel):
    category: Literal[
        "reasoning_strategy",
        "proof_construction",
        "computational_method",
        "modelling_method",
        "analytical_method",
        "presentation_method",
        "other",
    ]

    name: str
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ContextSelection(BaseModel):
    context_id: str
    relevance: Literal[
        "required",
        "supporting",
        "possibly_relevant",
    ]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class CommentAnnotation(BaseModel):
    source_range: SourceRange
    role: CommentRole
    content: str
    related_unit_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class UnitRelationship(BaseModel):
    target_unit_id: str
    relationship_type: RelationshipType
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticUnitLLMOutput(BaseModel):
    semantic_role: SemanticRole
    student_intent: StudentIntent | None = None
    strategy: StrategyAnnotation | None = None

    relevant_context: list[ContextSelection] = Field(default_factory=list)
    comment_annotations: list[CommentAnnotation] = Field(
        default_factory=list
    )
    relationships: list[UnitRelationship] = Field(default_factory=list)
    summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticUnitAnnotation(BaseModel):
    unit_id: str
    region_ids: list[str] = Field(default_factory=list)

    semantic_role: SemanticRole
    student_intent: StudentIntent | None = None
    strategy: StrategyAnnotation | None = None

    relevant_context: list[ContextSelection] = Field(default_factory=list)
    comment_annotations: list[CommentAnnotation] = Field(
        default_factory=list
    )
    relationships: list[UnitRelationship] = Field(default_factory=list)
    
    summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    # Domain-specific annotations can be added later if needed.
    # domain_annotations: dict[str, Any] = Field(default_factory=dict)


class SemanticExtractionResult(BaseModel):
    schema_version: str = "2.1"

    source_submission_id: str
    processed_submission_id: str

    extractor_version: str
    model_name: str

    document_regions: list[DocumentRegion] = Field(default_factory=list)
    unit_annotations: list[SemanticUnitAnnotation] = Field(
        default_factory=list
    )

    unresolved_observations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    raw_model_response: dict[str, Any] | None = None