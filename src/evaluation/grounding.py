from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .matching import (
    GeneratedObservationMatchState,
)
from .models import (
    GeneratedObservationResult,
    GroundingConfiguration,
    GroundingEvaluation,
)


@dataclass(frozen=True)
class GroundingResult:
    generated_observation_results: tuple[
        GeneratedObservationResult,
        ...
    ]


@dataclass(frozen=True)
class SubmissionEvidenceAssessment:
    """
    Deterministic assessment of the evidence references
    attached to one generated observation.

    This class assesses traceability only. A valid
    submission reference means that the cited source
    exists; it does not independently prove that the
    generated semantic claim is correct.
    """

    concrete_total: int
    concrete_resolved: int

    has_absence_evidence: bool

    tool_checks_total: int
    tool_checks_passed: int

    artifact_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]

    basis_codes: tuple[str, ...]


class ObservationGrounder:
    """
    Deterministic grounding of generated feedback after
    human-feedback matching.

    Human-aligned observations do not require grounding.

    Generated observations with no human mapping are
    assessed using evidence that can be independently
    checked against the processed submission.

    Grounding principles
    --------------------

    1. A generator verification_status is descriptive
       metadata, not independent proof.

    2. A submission_reference is considered traceable
       when its artifact/unit or artifact/block reference
       resolves against the processed submission.

    3. Resolved submission references provide partial
       support only. They establish traceability but do
       not independently verify the generated semantic
       interpretation.

    4. Structured absence evidence provides partial
       support only.

    5. Full support is reserved for an explicitly cited
       tool check that can independently be found in the
       processed submission with status='passed'.

    6. Claims of tool verification without such an
       explicit corroborating check are not assessable.

    7. Missing or broken evidence remains unresolved
       rather than being automatically labelled false.

    No LLM judge is used.
    """

    def __init__(
        self,
        *,
        configuration: GroundingConfiguration,
    ) -> None:
        self.configuration = configuration

    def ground(
        self,
        *,
        generated_feedback: dict[str, Any],
        processed_submission: dict[str, Any],
        generated_match_states: tuple[
            GeneratedObservationMatchState,
            ...
        ],
    ) -> GroundingResult:

        observations = generated_feedback.get(
            "observations",
            [],
        )

        if not isinstance(observations, list):
            raise ValueError(
                "generated_feedback.observations "
                "must be a list."
            )

        observation_lookup = {
            self._required_string(
                observation,
                "observation_id",
            ): observation
            for observation in observations
        }

        criterion_lookup = (
            self._build_criterion_lookup(
                generated_feedback
            )
        )

        submission_index = (
            self._build_submission_index(
                processed_submission
            )
        )

        results = []

        for state in generated_match_states:

            observation_id = (
                state.generated_observation_id
            )

            observation = observation_lookup.get(
                observation_id
            )

            if observation is None:
                raise ValueError(
                    "Generated match state references "
                    "unknown observation "
                    f"{observation_id!r}."
                )

            feedback_type = self._required_string(
                observation,
                "feedback_type",
            )

            # ------------------------------------------------
            # Human-aligned generated feedback
            # ------------------------------------------------

            if state.mapping_status in {
                "matched",
                "partially_matched",
            }:
                results.append(
                    GeneratedObservationResult(
                        generated_observation_id=(
                            observation_id
                        ),
                        feedback_type=feedback_type,
                        mapping_status=(
                            state.mapping_status
                        ),
                        match_ids=list(
                            state.match_ids
                        ),
                        classification=(
                            "human_aligned"
                        ),
                        grounding_evaluation=None,
                    )
                )

                continue

            # ------------------------------------------------
            # Grounding disabled
            # ------------------------------------------------

            if (
                not self.configuration.enabled
                or not self.configuration
                .ground_unmatched_generated_observations
            ):
                evaluation = GroundingEvaluation(
                    status="not_assessable",
                    evidence_coverage=(
                        "not_applicable"
                    ),
                    rubric_alignment=(
                        "not_applicable"
                    ),
                    requirement_ids=(
                        self._string_list(
                            observation,
                            "requirement_ids",
                        )
                    ),
                    artifact_ids=[],
                    unit_ids=[],
                    basis_codes=[
                        "grounding_disabled"
                    ],
                    rationale=(
                        "Grounding was disabled for "
                        "unmatched generated "
                        "observations."
                    ),
                )

                results.append(
                    self._build_unmatched_result(
                        observation=observation,
                        state=state,
                        evaluation=evaluation,
                    )
                )

                continue

            evaluation = self._ground_observation(
                observation=observation,
                criterion_lookup=(
                    criterion_lookup
                ),
                submission_index=(
                    submission_index
                ),
            )

            results.append(
                self._build_unmatched_result(
                    observation=observation,
                    state=state,
                    evaluation=evaluation,
                )
            )

        return GroundingResult(
            generated_observation_results=tuple(
                results
            )
        )

    def _ground_observation(
        self,
        *,
        observation: dict[str, Any],
        criterion_lookup: dict[
            str,
            dict[str, Any],
        ],
        submission_index: dict[str, Any],
    ) -> GroundingEvaluation:

        verification_status = (
            observation.get(
                "verification_status"
            )
        )

        requirement_ids = self._string_list(
            observation,
            "requirement_ids",
        )

        rubric_alignment = (
            self._determine_rubric_alignment(
                observation=observation,
                criterion_lookup=(
                    criterion_lookup
                ),
            )
        )

        evidence_assessment = (
            self._assess_evidence(
                observation=observation,
                submission_index=(
                    submission_index
                ),
            )
        )

        artifact_ids = list(
            evidence_assessment.artifact_ids
        )

        unit_ids = list(
            evidence_assessment.unit_ids
        )

        # ----------------------------------------------------
        # Deterministic rubric contradiction
        # ----------------------------------------------------

        if rubric_alignment == "not_aligned":
            return GroundingEvaluation(
                status="contradicted",
                evidence_coverage=(
                    self._evidence_coverage(
                        evidence_assessment
                    )
                ),
                rubric_alignment=(
                    "not_aligned"
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "rubric_status_contradiction"
                ],
                rationale=(
                    "The generated observation's "
                    "feedback type conflicts with "
                    "the status of its linked "
                    "criterion assessment."
                ),
            )

        # ----------------------------------------------------
        # Independently corroborated tool evidence
        # ----------------------------------------------------

        if (
            evidence_assessment.tool_checks_total
            > 0
        ):
            if (
                evidence_assessment
                .tool_checks_passed
                == evidence_assessment
                .tool_checks_total
            ):
                return GroundingEvaluation(
                    status="supported",
                    evidence_coverage=(
                        "sufficient"
                    ),
                    rubric_alignment=(
                        rubric_alignment
                    ),
                    requirement_ids=(
                        requirement_ids
                    ),
                    artifact_ids=artifact_ids,
                    unit_ids=unit_ids,
                    basis_codes=list(
                        evidence_assessment
                        .basis_codes
                    ),
                    rationale=(
                        "The generated observation "
                        "explicitly cites one or more "
                        "tool checks, and every cited "
                        "check is independently "
                        "recorded as passed in the "
                        "processed submission."
                    ),
                )

            return GroundingEvaluation(
                status="not_assessable",
                evidence_coverage=(
                    "insufficient"
                ),
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=list(
                    evidence_assessment
                    .basis_codes
                ),
                rationale=(
                    "The observation cites a tool "
                    "check, but the processed "
                    "submission does not independently "
                    "confirm every cited check as "
                    "passed."
                ),
            )

        # ----------------------------------------------------
        # Concrete submission references
        #
        # Resolved references demonstrate traceability,
        # not independent semantic correctness.
        # ----------------------------------------------------

        if (
            evidence_assessment
            .concrete_resolved
            > 0
        ):
            if (
                evidence_assessment
                .concrete_resolved
                == evidence_assessment
                .concrete_total
            ):
                coverage = "sufficient"
                basis = list(
                    evidence_assessment
                    .basis_codes
                )

                if (
                    "all_submission_references_resolved"
                    not in basis
                ):
                    basis.append(
                        "all_submission_references_resolved"
                    )
            else:
                coverage = "partial"
                basis = list(
                    evidence_assessment
                    .basis_codes
                )

                if (
                    "partial_submission_reference_resolution"
                    not in basis
                ):
                    basis.append(
                        "partial_submission_reference_resolution"
                    )

            return GroundingEvaluation(
                status="partially_supported",
                evidence_coverage=coverage,
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=basis,
                rationale=(
                    "The observation is traceable to "
                    "real submission evidence through "
                    "resolvable artifact/unit or "
                    "artifact/block references. "
                    "Reference resolution does not, "
                    "however, independently verify "
                    "the semantic claim made by the "
                    "generated feedback."
                ),
            )

        # ----------------------------------------------------
        # Structured absence evidence
        # ----------------------------------------------------

        if (
            evidence_assessment
            .has_absence_evidence
        ):
            basis = list(
                evidence_assessment.basis_codes
            )

            if (
                "structured_absence_evidence"
                not in basis
            ):
                basis.append(
                    "structured_absence_evidence"
                )

            return GroundingEvaluation(
                status="partially_supported",
                evidence_coverage="partial",
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=basis,
                rationale=(
                    "The observation is grounded in "
                    "a structured absence assertion. "
                    "This provides traceable evidence "
                    "for an inferred omission or "
                    "missing element, but does not "
                    "independently prove the semantic "
                    "claim."
                ),
            )

        # ----------------------------------------------------
        # Generator claims tool verification, but no
        # independently inspectable tool evidence exists.
        # ----------------------------------------------------

        if (
            verification_status
            == "verified_by_tool"
        ):
            return GroundingEvaluation(
                status="not_assessable",
                evidence_coverage="none",
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "uncorroborated_verified_by_tool"
                ],
                rationale=(
                    "The generator labels this "
                    "observation as verified_by_tool, "
                    "but it does not provide an "
                    "explicit tool-check reference "
                    "that can be corroborated against "
                    "the processed submission."
                ),
            )

        # ----------------------------------------------------
        # Explicitly not verified
        # ----------------------------------------------------

        if verification_status == "not_verified":
            return GroundingEvaluation(
                status="not_assessable",
                evidence_coverage="none",
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "not_verified"
                ],
                rationale=(
                    "The observation is explicitly "
                    "marked as not verified and no "
                    "independent grounding evidence "
                    "is available."
                ),
            )

        # ----------------------------------------------------
        # Evidence was supplied but nothing could be
        # independently resolved.
        # ----------------------------------------------------

        if (
            evidence_assessment.concrete_total
            > 0
        ):
            return GroundingEvaluation(
                status="unresolved",
                evidence_coverage=(
                    "insufficient"
                ),
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "no_resolved_submission_references"
                ],
                rationale=(
                    "Submission evidence references "
                    "were supplied, but none could be "
                    "resolved against the processed "
                    "submission."
                ),
            )

        # ----------------------------------------------------
        # Inference with no independently inspectable
        # grounding evidence.
        # ----------------------------------------------------

        if verification_status == "inferred":
            return GroundingEvaluation(
                status="unresolved",
                evidence_coverage="none",
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "inference_without_grounding_evidence"
                ],
                rationale=(
                    "The observation is marked as "
                    "inferred but contains no "
                    "structured absence evidence or "
                    "resolvable submission reference."
                ),
            )

        # ----------------------------------------------------
        # Claimed submission support with no evidence
        # ----------------------------------------------------

        if (
            verification_status
            == "supported_by_submission"
        ):
            return GroundingEvaluation(
                status="unresolved",
                evidence_coverage="none",
                rubric_alignment=(
                    rubric_alignment
                ),
                requirement_ids=(
                    requirement_ids
                ),
                artifact_ids=artifact_ids,
                unit_ids=unit_ids,
                basis_codes=[
                    "claimed_submission_support_without_evidence"
                ],
                rationale=(
                    "The generator claims that the "
                    "observation is supported by the "
                    "submission, but no independently "
                    "inspectable evidence is supplied."
                ),
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return GroundingEvaluation(
            status="unresolved",
            evidence_coverage="unresolved",
            rubric_alignment=(
                rubric_alignment
            ),
            requirement_ids=(
                requirement_ids
            ),
            artifact_ids=artifact_ids,
            unit_ids=unit_ids,
            basis_codes=[
                "unknown_grounding_state"
            ],
            rationale=(
                "The observation could not be "
                "deterministically grounded using "
                "the available evidence."
            ),
        )

    def _assess_evidence(
        self,
        *,
        observation: dict[str, Any],
        submission_index: dict[str, Any],
    ) -> SubmissionEvidenceAssessment:

        evidence = observation.get(
            "evidence",
            [],
        )

        if not isinstance(evidence, list):
            evidence = []

        concrete_total = 0
        concrete_resolved = 0

        has_absence_evidence = False

        tool_checks_total = 0
        tool_checks_passed = 0

        artifact_ids: set[str] = set()
        unit_ids: set[str] = set()

        basis_codes: set[str] = set()

        for item in evidence:

            if not isinstance(item, dict):
                continue

            evidence_type = item.get(
                "evidence_type"
            )

            # --------------------------------------------
            # Structured absence
            # --------------------------------------------

            if evidence_type == "absence":
                has_absence_evidence = True

                basis_codes.add(
                    "structured_absence_evidence"
                )

                continue

            # --------------------------------------------
            # Explicit tool-check evidence
            #
            # This schema is supported for future /
            # improved generated feedback even though
            # the current outputs generally do not yet
            # contain it.
            # --------------------------------------------

            if evidence_type in {
                "tool_check",
                "processing_check",
            }:
                tool_checks_total += 1

                artifact_id = (
                    self._optional_string(
                        item.get("artifact_id")
                    )
                )

                check_type = (
                    self._optional_string(
                        item.get("check_type")
                    )
                )

                if artifact_id:
                    artifact_ids.add(
                        artifact_id
                    )

                if (
                    artifact_id
                    and check_type
                    and self._tool_check_passed(
                        submission_index=(
                            submission_index
                        ),
                        artifact_id=(
                            artifact_id
                        ),
                        check_type=(
                            check_type
                        ),
                    )
                ):
                    tool_checks_passed += 1

                    basis_codes.add(
                        "corroborated_tool_check"
                    )
                else:
                    basis_codes.add(
                        "uncorroborated_tool_check"
                    )

                continue

            # --------------------------------------------
            # Submission reference
            # --------------------------------------------

            if (
                evidence_type
                != "submission_reference"
            ):
                continue

            concrete_total += 1

            artifact_id = (
                self._optional_string(
                    item.get("artifact_id")
                )
            )

            unit_id = (
                self._optional_string(
                    item.get("unit_id")
                )
            )

            block_id = (
                self._optional_string(
                    item.get("block_id")
                )
            )

            if artifact_id:
                artifact_ids.add(
                    artifact_id
                )

            resolved = False

            # Unit-level reference
            if (
                artifact_id
                and unit_id
                and self._unit_exists(
                    submission_index=(
                        submission_index
                    ),
                    artifact_id=(
                        artifact_id
                    ),
                    unit_id=unit_id,
                )
            ):
                resolved = True

                unit_ids.add(
                    unit_id
                )

                basis_codes.add(
                    "resolved_unit_reference"
                )

            # Block-level reference
            if (
                artifact_id
                and block_id
            ):
                parent_unit_id = (
                    self._block_parent_unit(
                        submission_index=(
                            submission_index
                        ),
                        artifact_id=(
                            artifact_id
                        ),
                        block_id=block_id,
                    )
                )

                if parent_unit_id is not None:
                    resolved = True

                    unit_ids.add(
                        parent_unit_id
                    )

                    basis_codes.add(
                        "resolved_block_reference"
                    )

            if resolved:
                concrete_resolved += 1
            else:
                basis_codes.add(
                    "unresolved_submission_reference"
                )

        return SubmissionEvidenceAssessment(
            concrete_total=concrete_total,
            concrete_resolved=(
                concrete_resolved
            ),
            has_absence_evidence=(
                has_absence_evidence
            ),
            tool_checks_total=(
                tool_checks_total
            ),
            tool_checks_passed=(
                tool_checks_passed
            ),
            artifact_ids=tuple(
                sorted(artifact_ids)
            ),
            unit_ids=tuple(
                sorted(unit_ids)
            ),
            basis_codes=tuple(
                sorted(basis_codes)
            ),
        )

    def _determine_rubric_alignment(
        self,
        *,
        observation: dict[str, Any],
        criterion_lookup: dict[
            str,
            dict[str, Any],
        ],
    ) -> str:

        criterion_ids = self._string_list(
            observation,
            "criterion_assessment_ids",
        )

        if not criterion_ids:
            return "not_applicable"

        statuses = []

        for criterion_id in criterion_ids:

            criterion = criterion_lookup.get(
                criterion_id
            )

            if criterion is None:
                continue

            status = criterion.get(
                "status"
            )

            if isinstance(status, str):
                statuses.append(
                    status
                )

        if not statuses:
            return "not_applicable"

        feedback_type = self._required_string(
            observation,
            "feedback_type",
        )

        aligned_statuses = {
            "strength": {
                "met",
            },
            "error": {
                "not_met",
                "missing",
                "partially_met",
            },
            "omission": {
                "missing",
            },
            "suggestion": {
                "met",
                "not_met",
                "partially_met",
                "not_assessable",
            },
        }

        expected = aligned_statuses[
            feedback_type
        ]

        aligned_count = sum(
            1
            for status in statuses
            if status in expected
        )

        if aligned_count == len(statuses):
            return "aligned"

        if aligned_count > 0:
            return "partially_aligned"

        return "not_aligned"

    @staticmethod
    def _evidence_coverage(
        assessment: (
            SubmissionEvidenceAssessment
        ),
    ) -> str:

        if assessment.tool_checks_total > 0:
            if (
                assessment.tool_checks_passed
                == assessment.tool_checks_total
            ):
                return "sufficient"

            return "insufficient"

        if assessment.concrete_total > 0:
            if (
                assessment.concrete_resolved
                == assessment.concrete_total
            ):
                return "sufficient"

            if (
                assessment.concrete_resolved
                > 0
            ):
                return "partial"

            return "insufficient"

        if assessment.has_absence_evidence:
            return "partial"

        return "none"

    @staticmethod
    def _build_criterion_lookup(
        generated_feedback: dict[
            str,
            Any,
        ],
    ) -> dict[str, dict[str, Any]]:

        result = {}

        for criterion in generated_feedback.get(
            "criterion_assessments",
            [],
        ):

            if not isinstance(
                criterion,
                dict,
            ):
                continue

            criterion_id = criterion.get(
                "criterion_assessment_id"
            )

            if (
                isinstance(
                    criterion_id,
                    str,
                )
                and criterion_id.strip()
            ):
                result[
                    criterion_id.strip()
                ] = criterion

        return result

    @staticmethod
    def _build_submission_index(
        processed_submission: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        """
        Build an index containing:

            artifacts[artifact_id].units
            artifacts[artifact_id].blocks
            artifacts[artifact_id].checks

        blocks maps each block_id to its parent unit_id.
        """

        artifacts_index = {}

        for artifact in processed_submission.get(
            "artifacts",
            [],
        ):

            if not isinstance(
                artifact,
                dict,
            ):
                continue

            artifact_id = (
                artifact.get(
                    "artifact_id"
                )
            )

            if (
                not isinstance(
                    artifact_id,
                    str,
                )
                or not artifact_id.strip()
            ):
                continue

            artifact_id = (
                artifact_id.strip()
            )

            units: set[str] = set()

            blocks: dict[
                str,
                str,
            ] = {}

            for unit in artifact.get(
                "units",
                [],
            ):

                if not isinstance(
                    unit,
                    dict,
                ):
                    continue

                unit_id = (
                    unit.get(
                        "unit_id"
                    )
                )

                if (
                    not isinstance(
                        unit_id,
                        str,
                    )
                    or not unit_id.strip()
                ):
                    continue

                unit_id = (
                    unit_id.strip()
                )

                units.add(
                    unit_id
                )

                for block in unit.get(
                    "content_blocks",
                    [],
                ):

                    if not isinstance(
                        block,
                        dict,
                    ):
                        continue

                    block_id = block.get(
                        "block_id"
                    )

                    if (
                        isinstance(
                            block_id,
                            str,
                        )
                        and block_id.strip()
                    ):
                        blocks[
                            block_id.strip()
                        ] = unit_id

            processing = artifact.get(
                "processing",
                {},
            )

            if not isinstance(
                processing,
                dict,
            ):
                processing = {}

            checks = {}

            for check in processing.get(
                "checks",
                [],
            ):

                if not isinstance(
                    check,
                    dict,
                ):
                    continue

                check_type = (
                    check.get(
                        "check_type"
                    )
                )

                if (
                    isinstance(
                        check_type,
                        str,
                    )
                    and check_type.strip()
                ):
                    checks.setdefault(
                        check_type.strip(),
                        [],
                    ).append(
                        check
                    )

            artifacts_index[
                artifact_id
            ] = {
                "units": units,
                "blocks": blocks,
                "checks": checks,
            }

        return {
            "artifacts": artifacts_index
        }

    @staticmethod
    def _unit_exists(
        *,
        submission_index: dict[str, Any],
        artifact_id: str,
        unit_id: str,
    ) -> bool:

        artifact = (
            submission_index
            .get(
                "artifacts",
                {},
            )
            .get(
                artifact_id
            )
        )

        if artifact is None:
            return False

        return (
            unit_id
            in artifact.get(
                "units",
                set(),
            )
        )

    @staticmethod
    def _block_parent_unit(
        *,
        submission_index: dict[str, Any],
        artifact_id: str,
        block_id: str,
    ) -> str | None:

        artifact = (
            submission_index
            .get(
                "artifacts",
                {},
            )
            .get(
                artifact_id
            )
        )

        if artifact is None:
            return None

        return (
            artifact
            .get(
                "blocks",
                {},
            )
            .get(
                block_id
            )
        )

    @staticmethod
    def _tool_check_passed(
        *,
        submission_index: dict[str, Any],
        artifact_id: str,
        check_type: str,
    ) -> bool:

        artifact = (
            submission_index
            .get(
                "artifacts",
                {},
            )
            .get(
                artifact_id
            )
        )

        if artifact is None:
            return False

        checks = (
            artifact
            .get(
                "checks",
                {},
            )
            .get(
                check_type,
                [],
            )
        )

        return any(
            isinstance(check, dict)
            and check.get("status")
            == "passed"
            for check in checks
        )

    def _build_unmatched_result(
        self,
        *,
        observation: dict[str, Any],
        state: (
            GeneratedObservationMatchState
        ),
        evaluation: GroundingEvaluation,
    ) -> GeneratedObservationResult:

        classification = {
            "supported": (
                "additional_valid"
            ),
            "partially_supported": (
                "not_assessable"
            ),
            "unsupported": (
                "unsupported"
            ),
            "contradicted": (
                "contradicted"
            ),
            "not_assessable": (
                "not_assessable"
            ),
            "unresolved": (
                "unresolved"
            ),
        }[
            evaluation.status
        ]

        return GeneratedObservationResult(
            generated_observation_id=(
                state
                .generated_observation_id
            ),
            feedback_type=(
                self._required_string(
                    observation,
                    "feedback_type",
                )
            ),
            mapping_status="no_mapping",
            match_ids=[],
            classification=classification,
            grounding_evaluation=evaluation,
        )

    @staticmethod
    def _required_string(
        payload: dict[str, Any],
        field: str,
    ) -> str:

        value = payload.get(
            field
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{field} must be a "
                "non-empty string."
            )

        return value.strip()

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            return value.strip()

        return None

    @staticmethod
    def _string_list(
        payload: dict[str, Any],
        field: str,
    ) -> list[str]:

        value = payload.get(
            field,
            [],
        )

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            return []

        return [
            item.strip()
            for item in value
            if (
                isinstance(
                    item,
                    str,
                )
                and item.strip()
            )
        ]
