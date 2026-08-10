from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.alignment import (
    AlignmentResult,
    DocumentAlignmentService,
)
from src.extraction.semantic_models import (
    SemanticExtractionResult,
)
from src.feedback.baseline_feedback_service import (
    BaselineFeedbackService,
)
from src.feedback.baseline_reflection_service import (
    BaselineReflectionService,
)
from src.feedback.feedback_input_models import (
    FeedbackGenerationInput,
    ProcessedSubmission,
)
from src.feedback.feedback_models import (
    FeedbackLLMOutput,
)
from src.feedback.feedback_service import (
    FeedbackService,
)
from src.feedback.feedback_llm_client import (
    FeedbackStructuredClient,
)


class ExperimentRunner:
    """
    Run feedback-generation ablations:

        A = baseline prompting
        B = baseline prompting + reflection
        C = criterion-decomposed prompting
        D = criterion-decomposed prompting + reflection

    Expensive intermediate results are persisted so runs can
    safely resume after interruption.
    """

    def __init__(
        self,
        *,
        feedback_client: FeedbackStructuredClient,
        alignment_client: Any,
        artifacts_root: str | Path = (
            "artifacts/experiments"
        ),
        resume: bool = True,
    ) -> None:
        self.feedback_client = feedback_client

        self.alignment_client = (
            alignment_client
        )

        self.artifacts_root = Path(
            artifacts_root
        )

        self.resume = resume

        self.artifacts_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Public API
    # ========================================================

    def run(
        self,
        *,
        assessment_id: str,
        student_id: str,
        pipelines: list[str],
        processed_path: str | Path,
        semantic_path: str | Path,
        specification_path: str | Path,
    ) -> dict[str, Any]:

        requested = {
            pipeline.upper()
            for pipeline in pipelines
        }

        invalid = (
            requested
            - {
                "A",
                "B",
                "C",
                "D",
            }
        )

        if invalid:
            raise ValueError(
                "Unknown pipeline(s): "
                + ", ".join(
                    sorted(invalid)
                )
            )

        (
            processed_submission,
            semantic_extraction,
            assessment_specification,
        ) = self._load_inputs(
            processed_path=(
                processed_path
            ),
            semantic_path=(
                semantic_path
            ),
            specification_path=(
                specification_path
            ),
        )

        generation_input = (
            FeedbackGenerationInput(
                processed_submission=(
                    processed_submission
                ),
                semantic_extraction=(
                    semantic_extraction
                ),
                assessment_specification=(
                    assessment_specification
                ),
            )
        )

        run_summary: dict[
            str,
            Any,
        ] = {
            "assessment_id": (
                assessment_id
            ),
            "student_id": (
                student_id
            ),
            "pipelines": {},
        }

        # ----------------------------------------------------
        # Pipeline A
        # ----------------------------------------------------

        a_result: (
            FeedbackLLMOutput
            | None
        ) = None

        if (
            "A" in requested
            or "B" in requested
        ):
            (
                a_result,
                a_runtime,
                a_source,
            ) = self._run_pipeline_a(
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
                generation_input=(
                    generation_input
                ),
            )

            run_summary[
                "pipelines"
            ]["A"] = {
                "runtime_seconds": (
                    a_runtime
                ),
                "source": a_source,
            }

        # ----------------------------------------------------
        # Pipeline B
        # ----------------------------------------------------

        if "B" in requested:
            if a_result is None:
                raise RuntimeError(
                    "Pipeline B requires "
                    "Pipeline A output."
                )

            (
                _b_result,
                b_runtime,
                b_source,
            ) = self._run_pipeline_b(
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
                generation_input=(
                    generation_input
                ),
                baseline_result=(
                    a_result
                ),
            )

            run_summary[
                "pipelines"
            ]["B"] = {
                "runtime_seconds": (
                    b_runtime
                ),
                "source": b_source,
            }

        # ----------------------------------------------------
        # Alignment for C/D
        # ----------------------------------------------------

        alignment_result: (
            AlignmentResult
            | None
        ) = None

        if (
            "C" in requested
            or "D" in requested
        ):
            (
                alignment_result,
                alignment_runtime,
                alignment_source,
            ) = self._run_alignment(
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
                processed_submission=(
                    processed_submission
                ),
                semantic_extraction=(
                    semantic_extraction
                ),
                assessment_specification=(
                    assessment_specification
                ),
            )

            run_summary[
                "alignment"
            ] = {
                "runtime_seconds": (
                    alignment_runtime
                ),
                "source": (
                    alignment_source
                ),
            }

        # ----------------------------------------------------
        # Pipeline C
        # ----------------------------------------------------

        if "C" in requested:
            if alignment_result is None:
                raise RuntimeError(
                    "Pipeline C requires "
                    "alignment."
                )

            (
                _c_result,
                c_runtime,
                c_source,
            ) = self._run_pipeline_c_or_d(
                pipeline="C",
                self_reflective=False,
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
                processed_submission=(
                    processed_submission
                ),
                semantic_extraction=(
                    semantic_extraction
                ),
                alignment_result=(
                    alignment_result
                ),
                assessment_specification=(
                    assessment_specification
                ),
            )

            run_summary[
                "pipelines"
            ]["C"] = {
                "runtime_seconds": (
                    c_runtime
                ),
                "source": c_source,
            }

        # ----------------------------------------------------
        # Pipeline D
        # ----------------------------------------------------

        if "D" in requested:
            if alignment_result is None:
                raise RuntimeError(
                    "Pipeline D requires "
                    "alignment."
                )

            (
                _d_result,
                d_runtime,
                d_source,
            ) = self._run_pipeline_c_or_d(
                pipeline="D",
                self_reflective=True,
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
                processed_submission=(
                    processed_submission
                ),
                semantic_extraction=(
                    semantic_extraction
                ),
                alignment_result=(
                    alignment_result
                ),
                assessment_specification=(
                    assessment_specification
                ),
            )

            run_summary[
                "pipelines"
            ]["D"] = {
                "runtime_seconds": (
                    d_runtime
                ),
                "source": d_source,
            }

        self._save_run_summary(
            assessment_id=(
                assessment_id
            ),
            student_id=(
                student_id
            ),
            summary=(
                run_summary
            ),
        )

        return run_summary

    # ========================================================
    # Input loading
    # ========================================================

    @staticmethod
    def _load_inputs(
        *,
        processed_path: str | Path,
        semantic_path: str | Path,
        specification_path: str | Path,
    ) -> tuple[
        ProcessedSubmission,
        SemanticExtractionResult,
        dict[str, Any],
    ]:

        with Path(
            processed_path
        ).open(
            "r",
            encoding="utf-8",
        ) as file:
            processed_submission = (
                ProcessedSubmission
                .model_validate(
                    json.load(file)
                )
            )

        with Path(
            semantic_path
        ).open(
            "r",
            encoding="utf-8",
        ) as file:
            semantic_extraction = (
                SemanticExtractionResult
                .model_validate(
                    json.load(file)
                )
            )

        with Path(
            specification_path
        ).open(
            "r",
            encoding="utf-8",
        ) as file:
            assessment_specification = (
                json.load(file)
            )

        return (
            processed_submission,
            semantic_extraction,
            assessment_specification,
        )

    # ========================================================
    # Paths
    # ========================================================

    def _result_path(
        self,
        *,
        pipeline: str,
        assessment_id: str,
        student_id: str,
    ) -> Path:

        return (
            self.artifacts_root
            / "results"
            / pipeline
            / assessment_id
            / f"{student_id}.json"
        )

    def _alignment_path(
        self,
        *,
        assessment_id: str,
        student_id: str,
    ) -> Path:

        return (
            self.artifacts_root
            / "alignment"
            / assessment_id
            / f"{student_id}.json"
        )

    # ========================================================
    # Generic result loading/saving
    # ========================================================

    @staticmethod
    def _load_feedback_result(
        path: Path,
    ) -> FeedbackLLMOutput:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return (
                FeedbackLLMOutput
                .model_validate(
                    json.load(file)
                )
            )

    @staticmethod
    def _save_feedback_result(
        *,
        result: FeedbackLLMOutput,
        path: Path,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                file,
                indent=2,
                ensure_ascii=False,
            )

    # ========================================================
    # Pipeline A
    # ========================================================

    def _run_pipeline_a(
        self,
        *,
        assessment_id: str,
        student_id: str,
        generation_input: FeedbackGenerationInput,
    ) -> tuple[
        FeedbackLLMOutput,
        float,
        str,
    ]:

        result_path = (
            self._result_path(
                pipeline="A",
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
            )
        )

        if (
            self.resume
            and result_path.exists()
        ):
            return (
                self._load_feedback_result(
                    result_path
                ),
                0.0,
                "cached_result",
            )

        checkpoint_dir = (
            self.artifacts_root
            / "checkpoints"
            / "A"
            / assessment_id
            / student_id
        )

        service = (
            BaselineFeedbackService(
                llm_client=(
                    self.feedback_client
                ),
                checkpoint_dir=(
                    checkpoint_dir
                ),
                resume_from_checkpoints=(
                    self.resume
                ),
            )
        )

        start = time.perf_counter()

        result = service.generate_feedback(
            generation_input=(
                generation_input
            )
        )

        runtime = (
            time.perf_counter()
            - start
        )

        self._save_feedback_result(
            result=result,
            path=result_path,
        )

        return (
            result,
            runtime,
            "generated",
        )

    # ========================================================
    # Pipeline B
    # ========================================================

    def _run_pipeline_b(
        self,
        *,
        assessment_id: str,
        student_id: str,
        generation_input: FeedbackGenerationInput,
        baseline_result: FeedbackLLMOutput,
    ) -> tuple[
        FeedbackLLMOutput,
        float,
        str,
    ]:

        result_path = (
            self._result_path(
                pipeline="B",
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
            )
        )

        if (
            self.resume
            and result_path.exists()
        ):
            return (
                self._load_feedback_result(
                    result_path
                ),
                0.0,
                "cached_result",
            )

        checkpoint_dir = (
            self.artifacts_root
            / "checkpoints"
            / "B"
            / assessment_id
            / student_id
        )

        service = (
            BaselineReflectionService(
                llm_client=(
                    self.feedback_client
                ),
                checkpoint_dir=(
                    checkpoint_dir
                ),
                resume_from_checkpoints=(
                    self.resume
                ),
            )
        )

        start = time.perf_counter()

        result = service.reflect_feedback(
            generation_input=(
                generation_input
            ),
            baseline_result=(
                baseline_result
            ),
            log_prefix=(
                f"{assessment_id}_"
                f"{student_id}_pipeline_b"
            ),
        )

        runtime = (
            time.perf_counter()
            - start
        )

        self._save_feedback_result(
            result=result,
            path=result_path,
        )

        return (
            result,
            runtime,
            "generated",
        )

    # ========================================================
    # Alignment
    # ========================================================

    def _run_alignment(
        self,
        *,
        assessment_id: str,
        student_id: str,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        assessment_specification: dict[str, Any],
    ) -> tuple[
        AlignmentResult,
        float,
        str,
    ]:

        path = self._alignment_path(
            assessment_id=(
                assessment_id
            ),
            student_id=(
                student_id
            ),
        )

        if (
            self.resume
            and path.exists()
        ):
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return (
                    AlignmentResult
                    .model_validate(
                        json.load(file)
                    ),
                    0.0,
                    "cached_result",
                )

        service = (
            DocumentAlignmentService(
                self.alignment_client
            )
        )

        start = time.perf_counter()

        result = service.align_submission(
            processed_submission=(
                processed_submission
            ),
            semantic_extraction=(
                semantic_extraction
            ),
            assessment_specification=(
                assessment_specification
            ),
        )

        runtime = (
            time.perf_counter()
            - start
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return (
            result,
            runtime,
            "generated",
        )

    # ========================================================
    # Pipelines C / D
    # ========================================================

    def _run_pipeline_c_or_d(
        self,
        *,
        pipeline: str,
        self_reflective: bool,
        assessment_id: str,
        student_id: str,
        processed_submission: ProcessedSubmission,
        semantic_extraction: SemanticExtractionResult,
        alignment_result: AlignmentResult,
        assessment_specification: dict[str, Any],
    ) -> tuple[
        FeedbackLLMOutput,
        float,
        str,
    ]:

        result_path = (
            self._result_path(
                pipeline=pipeline,
                assessment_id=(
                    assessment_id
                ),
                student_id=(
                    student_id
                ),
            )
        )

        if (
            self.resume
            and result_path.exists()
        ):
            return (
                self._load_feedback_result(
                    result_path
                ),
                0.0,
                "cached_result",
            )

        checkpoint_dir = (
            self.artifacts_root
            / "checkpoints"
            / pipeline
            / assessment_id
            / student_id
        )

        reflection_service = None

        if self_reflective:
            from src.feedback.feedback_reflection_service import (
                FeedbackReflectionService,
            )

            reflection_service = (
                FeedbackReflectionService(
                    llm_client=(
                        self.feedback_client
                    ),
                    checkpoint_dir=(
                        checkpoint_dir
                        / "reflection"
                    ),
                    resume_from_checkpoints=(
                        self.resume
                    ),
                )
            )

        service = FeedbackService(
            llm_client=(
                self.feedback_client
            ),
            reflection_service=(
                reflection_service
            ),
            checkpoint_dir=(
                checkpoint_dir
            ),
            resume_from_checkpoints=(
                self.resume
            ),
        )

        start = time.perf_counter()

        result = service.generate_feedback(
            processed_submission=(
                processed_submission
            ),
            semantic_extraction=(
                semantic_extraction
            ),
            alignment_result=(
                alignment_result
            ),
            assessment_specification=(
                assessment_specification
            ),
            self_reflective=(
                self_reflective
            ),
        )

        runtime = (
            time.perf_counter()
            - start
        )

        self._save_feedback_result(
            result=result,
            path=result_path,
        )

        return (
            result,
            runtime,
            "generated",
        )

    # ========================================================
    # Run metadata
    # ========================================================

    def _save_run_summary(
        self,
        *,
        assessment_id: str,
        student_id: str,
        summary: dict[str, Any],
    ) -> None:

        path = (
            self.artifacts_root
            / "metadata"
            / assessment_id
            / f"{student_id}.json"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )