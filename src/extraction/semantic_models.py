from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

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
    FORMAL_CONSTRAINT = "formal_constraint"
    ONTOLOGY_ENTITY = "ontology_entity"
    ONTOLOGY_AXIOM = "ontology_axiom"
    DESIGN_JUSTIFICATION = "design_justification"
    CRITICAL_REFLECTION = "critical_reflection"
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
    MODEL_DOMAIN = "model_domain"
    FORMALISE_CONSTRAINT = "formalise_constraint"
    JUSTIFY_MODELLING_CHOICE = "justify_modelling_choice"
    EVALUATE_DESIGN = "evaluate_design"
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
    SPECIALISES = "specialises"
    RESTRICTS = "restricts"
    HAS_DOMAIN = "has_domain"
    HAS_RANGE = "has_range"
    JUSTIFIES = "justifies"
    DESCRIBES = "describes"
    CORRESPONDS_TO = "corresponds_to"
    OTHER = "other"


class SourceRange(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "SourceRange":
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
    
class LeanDomainAnnotation(BaseModel):
    annotation_type: Literal["lean"] = "lean"

    declaration_type: str | None = None
    proof_style: str | None = None
    contains_sorry: bool | None = None
    concepts: list[str] = Field(default_factory=list)


class PrologDomainAnnotation(BaseModel):
    annotation_type: Literal["prolog"] = "prolog"

    predicate_name: str | None = None
    formula_summary: str | None = None
    operators: list[str] = Field(default_factory=list)
    modal_relations: list[str] = Field(default_factory=list)
    constraint_type: str | None = None


class OntologyDomainAnnotation(BaseModel):
    annotation_type: Literal["ontology"] = "ontology"

    entity_type: str | None = None
    entity_names: list[str] = Field(default_factory=list)
    axiom_type: str | None = None

    subject: str | None = None
    property_name: str | None = None
    target: str | None = None
    filler: str | None = None
    cardinality: int | None = Field(default=None, ge=0)

    modelling_concepts: list[str] = Field(
        default_factory=list
    )


class ReportDomainAnnotation(BaseModel):
    annotation_type: Literal["report"] = "report"

    section_type: str | None = None
    discussed_entities: list[str] = Field(
        default_factory=list
    )
    claims: list[str] = Field(default_factory=list)
    justification_targets: list[str] = Field(
        default_factory=list
    )


DomainAnnotation = Annotated[
    LeanDomainAnnotation
    | PrologDomainAnnotation
    | OntologyDomainAnnotation
    | ReportDomainAnnotation,
    Field(discriminator="annotation_type"),
]
    
class SemanticUnitLLMOutput(BaseModel):
    semantic_role: SemanticRole
    student_intent: StudentIntent | None = None
    strategy: StrategyAnnotation | None = None

    relevant_context: list[ContextSelection] = Field(
        default_factory=list
    )
    comment_annotations: list[CommentAnnotation] = Field(
        default_factory=list
    )
    relationships: list[UnitRelationship] = Field(
        default_factory=list
    )

    domain_annotations: list[DomainAnnotation] = Field(
        default_factory=list
    )

    summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticUnitAnnotation(BaseModel):
    unit_id: str
    region_ids: list[str] = Field(default_factory=list)

    semantic_role: SemanticRole
    student_intent: StudentIntent | None = None
    strategy: StrategyAnnotation | None = None

    relevant_context: list[ContextSelection] = Field(
        default_factory=list
    )
    comment_annotations: list[CommentAnnotation] = Field(
        default_factory=list
    )
    relationships: list[UnitRelationship] = Field(
        default_factory=list
    )

    domain_annotations: list[DomainAnnotation] = Field(
        default_factory=list
    )

    summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticExtractionResult(BaseModel):
    schema_version: str = "1.0"

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