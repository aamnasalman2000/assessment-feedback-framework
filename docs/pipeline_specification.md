# Pipeline Specification

## 1. Purpose and Scope

This document specifies the end-to-end processing pipeline for a
rubric-guided framework that generates formative feedback for heterogeneous
academic submissions using large language models.

The pipeline transforms assessment specifications and student submissions
into structured, evidence-grounded feedback. It also supports the
normalisation of human marker feedback and the automatic evaluation of
generated feedback against human feedback, rubric requirements and submission
evidence.

The pipeline is designed to support heterogeneous academic artefacts,
including source code, formal proofs, ontologies, logic programs, reports and
other structured or unstructured submission formats.

The framework focuses on formative feedback generation. It does not generate
or predict student marks as part of the primary feedback-generation process.
Human marks may be retained as contextual metadata where they are available,
but they are not used as the principal evaluation target.

## 2. Pipeline Overview

The framework consists of eight main stages:

1. Assessment specification
2. Submission ingestion
3. Submission preprocessing
4. Criterion-level assessment
5. Feedback synthesis
6. Self-reflection and revision
7. Human feedback normalisation
8. Evaluation


Assessment brief and rubric
          │
          ▼
Assessment Specification
          │
          ├─────────────────────┐
          │                     │
Student files                  Human marker feedback
          │                     │
          ▼                     ▼
Submission                Human Feedback Normalisation
          │                     │
          ▼                     ▼
Processed Submission      Structured Human Feedback
          │                     │
          ▼                     │
Criterion-Level Assessment      │
          │                     │
          ▼                     │
Feedback Synthesis              │
          │                     │
          ▼                     │
Self-Reflection and Revision    │
          │                     │
          ▼                     │
Generated Feedback ─────────────┘
          │
          ▼
Evaluation Results


Each stage produces a structured data artefact that conforms to a dedicated
JSON Schema. Stages communicate through these artefacts rather than through
unstructured internal state.

## 3. Data Artefacts

The pipeline uses the following primary data artefacts.

| Artefact | Schema | Purpose |
|---|---|---|
| Assessment  specification | `assessment_specification.json` | Represents assessment parts, tasks, artefact expectations and rubric requirements. |
| Submission | `student_submission.schema.json` | Records the original student submission files and metadata. |
| Processed submission | `processed_submission.schema.json` | Stores extracted, segmented and task-mapped submission content. |
| Generated feedback | `generated_feedback.schema.json` | Stores criterion assessments, consolidated observations and student-facing feedback. |
| Human feedback | `human_feedback.schema.json` | Stores normalised marker feedback as structured observations and sections. |
| Evaluation results | `evaluation_results.schema.json` | Stores observation matching, evidence checking, rubric coverage and aggregate metrics. |

The schemas define the structural contracts of the framework. This document
defines the processing behaviour that produces and consumes those structures.


## 4. Stage 1: Assessment Specification

### Inputs

- Assessment brief
- Marking rubric
- Task instructions
- Supporting templates or starter files
- Expected submission formats

### Processing

The assessment ingestion stage transforms an assessment specification into a
structured representation that can be consumed by downstream pipeline stages.

The process identifies:

- assessment metadata;
- assessment parts;
- individual tasks;
- relationships between parts and tasks;
- expected submission artefact types;
- rubric criteria and requirements;
- mappings between tasks and rubric requirements;
- expected evidence associated with each requirement;
- assessment-level constraints and instructions.

Rubric requirements must be represented as independently assessable
statements. Where a rubric statement contains multiple distinct expectations,
it should be decomposed into separate requirements to enable independent
criterion-level assessment.

### Outputs

- `assessment_specification.json`

### Validation Rules

- Every task must have a unique identifier.
- Every rubric requirement must have a unique identifier.
- Every rubric requirement must be associated with at least one assessment
  scope, such as a part or task.
- References between parts, tasks and requirements must resolve.
- Expected artefact types must use supported values.

### Failure Conditions

The stage fails when:

- rubric requirements are missing;
- task boundaries cannot be determined;
- cross-references are invalid.


## 5. Stage 2: Submission Ingestion

### Inputs

- Student submission files
- Submission metadata
- Assessment specification

### Processing

The ingestion stage records the submitted files without altering their
contents. It identifies file formats, assigns stable artefact identifiers and
records relationships between files where known.

Multiple submitted files must remain distinguishable.

### Outputs

- `student_submission.json`

### Validation Rules

- Every submitted file must have a unique artefact identifier.
- File paths or storage references must resolve.
- File types must be recorded.
- The submission must reference a valid assessment identifier.
- Student identifiers must use the project pseudonymisation scheme.

### Failure Conditions

The stage may fail or produce a warning when:

- a file is missing;
- a file cannot be opened;
- the file format is unsupported;
- the submission is empty;
- expected task files are absent;
- duplicate files cannot be distinguished.

## 6. Stage 3: Submission Preprocessing

### Inputs

- `student_submission.json`
- `assessment_specification.json`
- Original submission files

### Processing

The preprocessing stage converts heterogeneous submission artefacts into a
common structured representation.

Processing is artefact-specific. Examples include:

- extracting source code from programming files;
- separating Lean declarations, proofs and explanations;
- collecting compiler or interpreter diagnostics;
- extracting ontology entities, axioms and relationships;
- segmenting reports into headings, paragraphs and tables;
- extracting Prolog predicates, rules and queries;
- preserving image or poster references where textual extraction is limited.

The resulting content is divided into processing units and blocks. Units are
mapped to assessment parts and tasks where possible.

Task mapping may use:

- file names;
- headings;
- code declarations;
- explicit task labels;
- assessment-specific parsing rules;
- model-assisted mapping.

Raw diagnostic output must remain separate from student-facing feedback.

### Outputs

- `processed_submission.json`

### Validation Rules

- Every processed unit must reference a source artefact.
- Extracted blocks must retain provenance.
- Task mappings must reference valid task identifiers.
- Compiler and tool diagnostics must remain traceable to the relevant unit.
- Original submission content must not be overwritten.

### Failure Conditions

Potential failures include:

- parser failure;
- unsupported encoding;
- malformed source files;
- compiler or interpreter unavailability;
- unsuccessful task mapping;
- incomplete extraction;
- corrupted artefacts.

Partial processing should be permitted where possible. Processing warnings
must be retained in the output.

## 7. Stage 4: Criterion-Level Assessment

### Inputs

- `assessment_specification.json`
- `processed_submission.json`

### Processing

Each applicable rubric requirement is assessed independently.

For every requirement, the assessment component determines:

- whether the requirement is applicable;
- whether it is met, partially met, not met, missing or not assessable;
- which submission evidence supports the judgement;
- how the result was verified;
- an internal finding describing the judgement.

The model must not generate student-facing prose at this stage. The output of
this stage is an internal criterion-level representation.

Every applicable rubric requirement must appear exactly once in the criterion
assessment output.

### Outputs

An intermediate set of criterion assessments later stored within
`generated_feedback.json`.

### Validation Rules

- Every applicable requirement must have one criterion assessment.
- Requirement identifiers must resolve to the assessment specification.
- Evidence references must resolve to processed submission units or blocks.
- Status values must use the defined enumeration.
- An assessment marked `not_assessable` must include a reason.

### Failure Conditions

The stage fails or flags a warning when:

- a rubric requirement is skipped;
- referenced evidence cannot be found;
- the model produces conflicting assessments;
- an assessment is unsupported by submission evidence;
- the relevant submission content could not be processed.

## 8. Stage 5: Feedback Synthesis

### Inputs

- Criterion assessments
- Assessment structure
- Processed submission evidence

### Processing

Criterion-level findings are consolidated into student-facing observations.

One observation may represent:

- one rubric requirement;
- several related rubric requirements;
- one task;
- several related tasks;
- one or more submission attempts;
- an overall assessment-level issue.

Observations are classified as:

- `strength`
- `error`
- `omission`
- `suggestion`

Each observation stores both:

- an internal finding;
- student-facing feedback.

Suggestions are optional. Evidence excerpts may be included where they are
useful and concise.

The synthesis stage should avoid producing one isolated comment for every
rubric requirement. Related findings should be consolidated into coherent,
non-repetitive feedback.

### Outputs

Initial structured generated feedback containing:

- observations;
- feedback sections;
- overall feedback.

### Validation Rules

- Every observation must reference at least one relevant requirement.
- Evidence references must resolve.
- Student-facing wording must be consistent with the internal finding.
- Missing required work must be represented explicitly as an omission.
- Generated feedback must not include marks.
- Compiler diagnostics must be paraphrased rather than copied as raw output.

### Failure Conditions

Potential failures include:

- unsupported claims;
- duplicated observations;
- contradictory feedback;
- excessive criterion-by-criterion fragmentation;
- missing major issues;
- feedback that does not correspond to the cited evidence.

## 9. Stage 6: Self-Reflection and Revision

### Inputs

- Assessment specification
- Processed submission
- Initial criterion assessments
- Initial generated feedback

### Processing

A second evaluation pass reviews the initial output.

The verification stage checks:

1. Rubric completeness  
   Whether every applicable requirement was assessed.

2. Evidence support  
   Whether each finding is supported by the cited submission content.

3. Student-facing coverage  
   Whether significant errors, omissions and weaknesses identified internally
   are represented in the final feedback.

4. Internal consistency  
   Whether observations contradict one another or conflict with criterion
   assessments.

5. Relevance  
   Whether feedback remains within the scope of the assessment and rubric.

6. Presentation quality  
   Whether the feedback is clear, concise, non-repetitive and appropriately
   consolidated.

The verifier may request revisions to criterion assessments, observations,
feedback sections or overall feedback.

Only the final revised feedback is stored as the primary
`generated_feedback.json` output. For experiments comparing generation
strategies, the initial and revised outputs should also be retained separately.

### Outputs

- Final `generated_feedback.json`
- Optional experimental copy of the initial generated feedback
- Optional verification log

### Validation Rules

- All applicable rubric requirements must remain represented.
- Revised observations must retain valid evidence references.
- Revision must not introduce unsupported claims.
- The final output must conform to the generated feedback schema.

### Failure Conditions

The stage fails or produces an unresolved result when:

- verification repeatedly produces invalid output;
- evidence is insufficient;
- criterion assessments remain contradictory;
- required rubric coverage cannot be achieved;
- the verifier cannot determine whether a claim is supported.

## 10. Stage 7: Human Feedback Normalisation

### Inputs

- Original marker comments
- Optional marks and mark breakdown
- Assessment specification

### Processing

Human feedback is segmented into distinct evaluative observations.

Normalisation may:

- separate multiple claims contained in one paragraph;
- resolve references such as “this proof” or “the previous task”;
- assign feedback type;
- assign part, task or overall scope;
- map observations to rubric requirements where reliable;
- preserve the original wording and source location.

Normalisation must not introduce new judgements that were not expressed by the
marker.

Marks may be retained as contextual metadata. Marks must not automatically be
converted into feedback observations.

### Outputs

- `human_feedback.json`

### Validation Rules

- Every normalised observation must trace back to the original feedback.
- Original text must be preserved.
- Observation types must use the supported categories.
- Requirement mappings are optional and must not be invented.
- Ambiguous mappings must include a confidence value or annotation note.

### Failure Conditions

Potential failures include:

- missing original feedback;
- inability to distinguish separate claims;
- ambiguous task references;
- conflicting marker comments;
- incomplete source location information.

## 12. Validation and Error Handling

Every pipeline artefact must be validated against its JSON Schema before being
passed to the next stage.

Validation occurs:

- after initial creation;
- after any model-generated revision;
- before evaluation;
- before persistence as a final experimental output.

Errors are divided into:

- structural errors;
- reference errors;
- processing errors;
- model-output errors;
- unresolved academic judgements.

Structural errors prevent progression to the next stage.

Recoverable processing errors should be retained as diagnostics and should not
discard successfully processed content.

Model outputs that fail schema validation may be regenerated or repaired up to
a configured retry limit. The original invalid output should be retained in
experiment logs where appropriate.

The system must not silently substitute missing content, unsupported evidence
or invalid identifiers.

## 13. Reproducibility and Experiment Tracking

Every model-driven stage must record sufficient metadata to reproduce the run.

Metadata should include:

- model provider;
- model identifier;
- model version where available;
- prompt template identifier;
- pipeline version;
- generation method;
- sampling parameters;
- execution timestamp;
- schema version;
- preprocessing version;
- evaluation thresholds;
- embedding model where applicable.

The experimental framework will compare the following generation approaches:

1. Baseline generation
2. Criterion-decomposed generation
3. Self-reflective generation
4. Criterion-decomposed generation with self-reflection

Outputs from each experimental condition must be stored separately and linked
to the same assessment and submission identifiers.

## 15. Open Design Decisions

The following decisions remain to be finalised:

- exact preprocessing tools for each artefact type;
- whether criterion assessment and synthesis use the same model;
- whether verification produces a separate persisted record;
- matching-score weights and thresholds;
- handling of partially matched observations in recall calculations;
- handling of partially supported observations in precision calculations;
- definition of important rubric issues for student-facing issue coverage;
- retry and repair limits for invalid model output;
- storage format for experiment runs;
- final model providers and model versions.