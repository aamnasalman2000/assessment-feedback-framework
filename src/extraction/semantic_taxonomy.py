from typing import Final


STRATEGY_CATEGORIES: Final[tuple[str, ...]] = (
    "reasoning_strategy",
    "reasoning_strategy",
    "computational_method",
    "modelling_strategy",
    "other",
)


LEAN_STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "case_analysis",
    "conjunction_elimination",
    "conjunction_construction",
    "direct_application",
    "implication_elimination",
    "contradiction",
    "proof_by_contradiction",
    "negation_introduction",
    "disjunction_elimination",
    "disjunction_introduction",
    "rewriting",
    "simplification",
    "induction",
    "decomposition_only",
    "other",
)


LEAN_STRATEGY_CATEGORY_MAP: Final[dict[str, str]] = {
    "case_analysis": "reasoning_strategy",
    "conjunction_elimination": "reasoning_strategy",
    "conjunction_construction": "reasoning_strategy",
    "direct_application": "reasoning_strategy",
    "implication_elimination": "reasoning_strategy",
    "contradiction": "reasoning_strategy",
    "proof_by_contradiction": "reasoning_strategy",
    "negation_introduction": "reasoning_strategy",
    "disjunction_elimination": "reasoning_strategy",
    "disjunction_introduction": "reasoning_strategy",
    "rewriting": "reasoning_strategy",
    "simplification": "reasoning_strategy",
    "induction": "reasoning_strategy",
    "decomposition_only": "reasoning_strategy",
    "other": "other",
}

PROLOG_STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "predicate_definition",
    "modal_formulation",
    "constraint_encoding",
    "implication_modelling",
    "quantified_constraint",
    "alternative_solution",
    "logical_composition",
    "other",
)

PROLOG_STRATEGY_CATEGORY_MAP = {
    "predicate_definition": "computational_method",
    "modal_formulation": "modelling_method",
    "constraint_encoding": "modelling_method",
    "implication_modelling": "reasoning_strategy",
    "quantified_constraint": "reasoning_strategy",
    "alternative_solution": "reasoning_strategy",
    "logical_composition": "reasoning_strategy",
    "other": "other",
}

ONTOLOGY_STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "class_hierarchy_design",
    "property_modelling",
    "restriction_modelling",
    "cardinality_modelling",
    "domain_range_specification",
    "subproperty_design",
    "property_characterisation",
    "ontology_pattern",
    "other",
)

ONTOLOGY_STRATEGY_CATEGORY_MAP = {
    "class_hierarchy_design": "modelling_method",
    "property_modelling": "modelling_method",
    "restriction_modelling": "modelling_method",
    "cardinality_modelling": "modelling_method",
    "domain_range_specification": "modelling_method",
    "subproperty_design": "modelling_method",
    "property_characterisation": "modelling_method",
    "ontology_pattern": "analytical_method",
    "other": "other",
}

REPORT_STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "design_justification",
    "critical_evaluation",
    "comparison",
    "advantages_disadvantages",
    "explanation",
    "other",
)

REPORT_STRATEGY_CATEGORY_MAP = {
    "design_justification": "presentation_method",
    "critical_evaluation": "analytical_method",
    "comparison": "analytical_method",
    "advantages_disadvantages": "analytical_method",
    "explanation": "presentation_method",
    "other": "other",
}

UNIT_TYPES = {
    "lean_code",
    "lean_comment",

    "prolog_answer",
    "prolog_task_5_answer",

    "ontology_class",
    "ontology_object_property",
    "ontology_datatype_property",
    "ontology_subclass_axiom",
    "ontology_restriction",
    "ontology_property_domain",
    "ontology_property_range",
    "ontology_subproperty_axiom",
    "ontology_property_characteristic",
    "ontology_intersection_axiom",

    "ontology_report_section",
}

from typing import Final


ARTIFACT_STRATEGY_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "lean_source": LEAN_STRATEGY_NAMES,
    "prolog_source": PROLOG_STRATEGY_NAMES,
    "owl_ontology": ONTOLOGY_STRATEGY_NAMES,
    "ontology_report": REPORT_STRATEGY_NAMES,
}


ARTIFACT_STRATEGY_CATEGORY_MAP: Final[dict[str, dict[str, str]]] = {
    "lean_source": LEAN_STRATEGY_CATEGORY_MAP,
    "prolog_source": PROLOG_STRATEGY_CATEGORY_MAP,
    "owl_ontology": ONTOLOGY_STRATEGY_CATEGORY_MAP,
    "ontology_report": REPORT_STRATEGY_CATEGORY_MAP,
}


def get_strategy_names(artifact_type: str) -> tuple[str, ...]:
    """
    Return the valid strategy names for a given artifact type.
    Falls back to the Lean taxonomy for unknown artifact types.
    """
    return ARTIFACT_STRATEGY_NAMES.get(
        artifact_type,
        LEAN_STRATEGY_NAMES,
    )


def get_strategy_category_map(artifact_type: str) -> dict[str, str]:
    """
    Return the strategy→category mapping for a given artifact type.
    Falls back to the Lean taxonomy for unknown artifact types.
    """
    return ARTIFACT_STRATEGY_CATEGORY_MAP.get(
        artifact_type,
        LEAN_STRATEGY_CATEGORY_MAP,
    )


def get_strategy_category(
    artifact_type: str,
    strategy_name: str,
) -> str:
    """
    Return the category for a specific strategy.
    Unknown strategies are classified as 'other'.
    """
    category_map = get_strategy_category_map(artifact_type)
    return category_map.get(strategy_name, "other")