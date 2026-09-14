# Assessment Feedback

Research code for processing student submissions and generating, aligning, and evaluating automated assessment feedback.

The pipeline combines deterministic extraction with optional semantic extraction through an OpenAI-compatible language-model service. Feedback generation can be run with four ablation variants:

- **A**: baseline prompting
- **B**: baseline prompting plus reflection
- **C**: criterion-decomposed prompting
- **D**: criterion-decomposed prompting plus reflection

## Project Layout

```text
assessment_feedback/
├── assessment_materials/   Source assessment files and reference materials
├── data/                    Submissions and generated intermediate data
│   └── assessment_N/
│       ├── submissions/     Raw student files
│       ├── metadata/        Normalised submission metadata
│       ├── processed/       Deterministically processed submissions
│       ├── semantic/        Semantic extraction results by model/version
│       └── human_feedback/  Human feedback used for comparison/evaluation
├── artifacts/               Feedback, alignment, and checkpoint outputs
├── results/                 Evaluation results and metrics
├── schemas/                 JSON Schemas for project data contracts
├── specs/                   Assessment specifications and rubrics
├── scripts/                 Experiment orchestration code
└── src/                     Processing, extraction, feedback, and evaluation code
```

## Requirements

- Python 3.10 or newer
- Pydantic 2
- An OpenAI-compatible Python client (`openai`)
- `jsonschema` for schema validation
- `numpy` for evaluation metrics

The repository currently does not include a dependency lockfile or package manifest. Create and activate a virtual environment, then install the dependencies available in your environment. For example:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install openai pydantic jsonschema numpy pytest
```

Some extraction clients may require additional local-model packages such as `torch` or `transformers`; install those only when using the corresponding client in `src/extraction/`.

## Data Flow

1. **Build metadata** from raw submission files.
2. **Process submissions** into schema-validated structured artefacts.
3. **Run semantic extraction** with a selected model.
4. **Align the submission** with the assessment specification when running pipelines C or D.
5. **Generate feedback** and persist resumable checkpoints and outputs.
6. **Evaluate** generated feedback against human feedback and evaluation schemas.

Generated files are written under the relevant `data/`, `artifacts/`, and `results/` directories. Most model-backed operations can resume from existing outputs and checkpoints.

## Usage

Run commands from the repository root so the `src` package imports resolve correctly.

### Build submission metadata

```powershell
python -m src.build_metadata --help
```

Use the help output for the available assessment and submission arguments. Metadata is written to `data/<assessment_id>/metadata/`.

### Process submissions

Process one student:

```powershell
python -m src.process_submissions `
  --assessment-id assessment_1 `
  --student-id student_3
```

Process every metadata file for an assessment:

```powershell
python -m src.process_submissions `
  --assessment-id assessment_1 `
  --all
```

Processed output is written to `data/<assessment_id>/processed/` and validated against `schemas/processed_submission.schema.json` by default.

### Run semantic extraction

The semantic pipeline first processes the selected submission and then calls an OpenAI-compatible model service:

```powershell
python -m src.pipeline `
  --assessment-id assessment_1 `
  --student-id student_3 `
  --model qwen3:8b
```

Optional arguments include `--rubric`, `--schema`, `--base-url`, and `--retain-raw-model-response`. Results are stored under `data/<assessment_id>/semantic/<version>/<model>/`.

The current default client is configured for a local Ollama-compatible endpoint at `http://localhost:11434/v1/` with the placeholder API key `ollama`. Start the model service and make the requested model available before running model-backed commands. For another OpenAI-compatible endpoint, pass `--base-url` and review the client configuration before use.

### Run feedback experiments

`scripts/run_feedback_experiments.py` exposes the `ExperimentRunner` class for programmatic runs. It accepts processed submission, semantic extraction, and assessment specification paths, plus a list containing any of `A`, `B`, `C`, and `D`.

Example:

```python
from scripts.run_feedback_experiments import ExperimentRunner

runner = ExperimentRunner(
    feedback_client=feedback_client,
    alignment_client=alignment_client,
    artifacts_root="artifacts/experiments",
    resume=True,
)

summary = runner.run(
    assessment_id="assessment_1",
    student_id="student_3",
    pipelines=["A", "B", "C", "D"],
    processed_path="data/assessment_1/processed/student_3.json",
    semantic_path="data/assessment_1/semantic/v1/qwen3/student_3.json",
    specification_path="specs/assessment_1_spec.json",
)
```

Pipeline B depends on A, and pipelines C and D require alignment. Results and resumable checkpoints are written below `artifacts/experiments/`.

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest
```

Tests that instantiate model clients or call model-backed services may require a running compatible model endpoint and the relevant local model.

## Schemas and Specifications

- `schemas/` defines the JSON contracts for submissions, processed artefacts, human feedback, generated feedback, and evaluation results.
- `specs/assessment_1_spec.json` and `specs/assessment_2_spec.json` define assessment-specific criteria used by extraction, alignment, and feedback generation.

## Reproducibility Notes

- Keep input data, model name, semantic schema version, and assessment specification recorded with each experiment.
- Do not commit environment files, virtual environments, caches, or raw model debug logs; these are excluded by `.gitignore`.
- Model-backed results can vary with model version and endpoint configuration. Preserve generated artifacts and run metadata when reporting results.