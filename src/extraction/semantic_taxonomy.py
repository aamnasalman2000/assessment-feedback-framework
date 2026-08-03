from typing import Final


STRATEGY_CATEGORIES: Final[tuple[str, ...]] = (
    "reasoning_strategy",
    "proof_construction",
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
    "conjunction_construction": "proof_construction",
    "direct_application": "proof_construction",
    "implication_elimination": "reasoning_strategy",
    "contradiction": "reasoning_strategy",
    "proof_by_contradiction": "reasoning_strategy",
    "negation_introduction": "proof_construction",
    "disjunction_elimination": "reasoning_strategy",
    "disjunction_introduction": "proof_construction",
    "rewriting": "reasoning_strategy",
    "simplification": "reasoning_strategy",
    "induction": "reasoning_strategy",
    "decomposition_only": "reasoning_strategy",
    "other": "other",
}