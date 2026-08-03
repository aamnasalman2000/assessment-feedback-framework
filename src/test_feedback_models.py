from src.feedback import GeneratedFeedback

example = GeneratedFeedback.model_validate(
    {
        "schema_version": "1.0",
        "feedback_id": "feedback_student_3_v1",
        "processed_submission_id": "processed_student_3",
        "student_id": "student_3",
        "assessment_id": "assessment_1",
        "generation": {
            "method": "baseline",
            "provider": "ollama",
            "model": "mistral",
            "prompt_template_id": "feedback_baseline_v1",
            "pipeline_version": "1.0",
        },
        "criterion_assessments": [
            {
                "criterion_assessment_id": "ca_001",
                "requirement_id": "requirement_001",
                "status": "met",
                "internal_finding": (
                    "The submission constructs the required conjunction."
                ),
                "verification_status": "supported_by_submission",
                "confidence": 0.95,
                "evidence": [
                    {
                        "evidence_type": "submission_reference",
                        "artifact_id": "artifact_001",
                        "unit_id": "unit_001",
                        "source_range": {
                            "start_line": 4,
                            "end_line": 8,
                        },
                    }
                ],
            }
        ],
        "observations": [
            {
                "observation_id": "observation_001",
                "feedback_type": "strength",
                "scope": "task",
                "task_ids": ["task_001"],
                "requirement_ids": ["requirement_001"],
                "criterion_assessment_ids": ["ca_001"],
                "internal_finding": (
                    "The conjunction was constructed correctly."
                ),
                "student_feedback": (
                    "You correctly constructed both parts of the "
                    "required conjunction."
                ),
                "verification_status": "supported_by_submission",
                "confidence": 0.95,
            }
        ],
        "feedback_sections": [
            {
                "section_id": "section_001",
                "label": "Task 1",
                "scope": "task",
                "task_ids": ["task_001"],
                "summary": (
                    "Your solution satisfies the main requirement for "
                    "this task."
                ),
                "observation_ids": ["observation_001"],
            }
        ],
        "overall_feedback": {
            "summary": (
                "The submission demonstrates a sound solution to the "
                "assessed task."
            ),
            "strengths_summary": (
                "The required logical structure was constructed correctly."
            ),
            "improvement_summary": (
                "Continue making each proof step explicit and easy to follow."
            ),
            "observation_ids": ["observation_001"],
        },
    }
)

print(
    example.model_dump_json(
        indent=2,
        exclude_none=True,
    )
)