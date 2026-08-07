# Agent Instructions for Mark 1 Part 2

These instructions apply to every file and notebook created under `mark 1 (part 2)/`.

## Mission

Continue the LiTS liver-tumour segmentation study from the verified Mark 4E state. First improve dataset understanding using training and validation data only. Then confirm or revise the frozen fusion policy through predeclared validation gates. Preserve reproducibility, provenance and the locked test split.

Operate as an autonomous technical agent. For each request, understand the intended outcome, inspect current evidence, reason about the smallest justified action, plan the steps, implement the requested notebook or repair, validate it, analyse available outputs, and report the next decision. Do not make the user repeatedly ask you to inspect obvious files, diagnose routine errors or convert planned work into a runnable notebook.

## Exclusive working directory

- New work must be created only under `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1 (part 2)`.
- Treat `mark 1\`, `Practice\`, `understanding the project\` and the dataset build as read-only inputs unless the user explicitly authorizes a change there.
- Never save new notebooks, figures, CSV files, JSON files, checkpoints, logs or temporary results in an older phase folder.
- Follow `06_WORKING_DIRECTORY_AND_DELIVERY_RULES.md` for every task.

## Mandatory startup procedure

1. Read every Markdown file in this folder before proposing or creating a notebook.
2. Read the latest controlling JSON artifacts:
   - `..\mark 1\mark_4e_outputs\mark_4e_gate_result.json`
   - `..\mark 1\mark_4e_outputs\mark_4e_provenance.json`
   - `..\mark 1\mark_4d_outputs\mark_4d_gate_result.json`
3. Verify the authoritative manifest SHA-256 before analysis.
4. Inspect existing artifacts before recomputing them.
5. State whether the task is diagnostic, training, validation confirmation or final evaluation.
6. Read `07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md` and declare the applicable success level and required artifacts before implementation.

## Source-of-truth hierarchy

1. Corrected manifest and source NIfTI files.
2. Latest machine-readable gate/provenance JSON and saved CSV outputs.
3. Executed notebooks and checkpoints.
4. This handoff documentation.
5. Older `understanding the project/` notes, which are historical and predate Mark 4E.

If sources disagree, do not average or silently choose. Report the conflict and use the latest verified machine-readable artifact. The Mark 4C raw gate has a known mean-patient population inconsistency; Mark 4D corrected it.

## Non-negotiable safeguards

- Never open or instantiate the test loader without a later explicit final authorization gate.
- Do not use test data for EDA, thresholds, fusion weights, sampling, training or model choice.
- Do not use ground-truth liver masks for deployable validation/test normalization or prediction gating.
- Do not change the manifest, split, label meaning, ROI rule, preprocessing, checkpoint or metric population silently.
- Do not pair files by directory order; use manifest `sample_id`, `volume_id` and paths.
- Resize masks only with nearest-neighbour interpolation.
- Record manifest and input-artifact hashes in every result.
- Report patient-level and lesion-size performance; aggregate Dice alone is insufficient.
- Keep volumes 104 and 116 visible in every model decision.
- Treat the current fusion result as temporary validation evidence. It is not permission to evaluate test data.

## Notebook creation standard

Create one new phase folder for each activity:

`step_NN_short_descriptive_name\`

Inside it create:

- `step_NN_short_descriptive_name.ipynb`;
- `README.md` describing the question, inputs, parameters, execution and decision gate;
- `outputs\` for every generated result;
- `error_fixes.md` when an execution error occurs;
- optional generator/patch scripts needed to maintain the notebook.

All notebook path constants must point to that phase's `outputs\` directory. Use `nbformat`, validate notebook JSON, parse every code cell and run a lightweight preflight. Leave expensive full execution for the user unless requested.

Every analytical notebook should contain:

1. `## tl;dr`
2. `## Context & Methods`
3. `### Key Assumptions`
4. `## Data`
5. `## Results`
6. `## Takeaways`

Every phase must save:

- provenance/configuration JSON;
- full result CSVs, not only displayed tables;
- bounded labelled visualizations;
- expected-versus-actual gate table;
- machine-readable gate/decision JSON;
- explicit `test_images_accessed: false` until final authorization.

Every result must be classified using the decision vocabulary in `07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md`. Never call a temporary continuation pass a final success.

## Autonomous request workflow

For every user request, perform this cycle without waiting for separate prompts when the action is safe and in scope:

1. **Understand** — restate internally the requested outcome and identify the controlling evidence.
2. **Inspect** — read the relevant notebook, outputs, gates, schemas and errors.
3. **Analyse** — identify the verified cause, uncertainty and scientific risk.
4. **Plan** — define a bounded sequence, required inputs, parameters, outputs and pass/fail gate.
5. **Act** — create or edit the requested `.ipynb`, Markdown, code or configuration in the correct new phase folder.
6. **Validate** — check structure, syntax, paths, hashes, imports, representative data and a safe preflight.
7. **React** — if an error occurs, diagnose the real cause, patch both the runnable notebook and its generator when present, preserve completed outputs and retest the failing path.
8. **Solve** — provide the corrected runnable artifact, not only an explanation or code snippet.
9. **Track** — update the phase README and `PROJECT_STATUS.md` with completed work, results, blockers and next step.
10. **Handoff** — tell the user exactly which notebook to run, where outputs will appear and which cells to rerun after a repair.

Ask the user only when a missing choice would materially change the scientific result, requires new authority, risks destructive changes or would unlock test data. Routine file discovery, schema inspection, error diagnosis, notebook creation and safe validation should be handled autonomously.

## Error-repair contract

- Read the full traceback and inspect the real file/schema before editing.
- Preserve completed training, caches and outputs whenever possible.
- Patch the notebook in place and patch its generator or template so regeneration does not reintroduce the error.
- Clear only stale outputs from the modified failing cell when appropriate.
- Rerun the smallest safe dependency chain that proves the fix.
- Record the error, root cause, files changed and verified rerun instruction in the phase's `error_fixes.md`.
- Do not tell the user to rerun an expensive training phase if saved checkpoints or caches can be reused.

## Immediate next task

Create and run a train/validation-only notebook named approximately:

`step_01_pretraining_dataset_characterization\step_01_pretraining_dataset_characterization.ipynb`

It must implement the full contract in `03_PRETRAINING_DATASET_AUDIT_CONTRACT.md`. Do not train a model in that notebook.

## Decision discipline

- Use only training data to derive sampling bins or training policies.
- Validation data may be used to measure distribution shift and identify whether predefined training strata cover validation phenotypes.
- Do not create per-validation-patient preprocessing or thresholds.
- Predeclare any later training ablation before running it.
- If the dataset audit finds a critical integrity, leakage, affine or label-alignment problem, stop training plans and fix/version the dataset first.
- If no critical issue exists, freeze the dataset data card and sampling policy, then proceed to the bounded fusion confirmation in the roadmap.

## User handoff style

Lead with the outcome. State verified numbers, the gate status, files created, execution status and the single next action. Do not make the user reconstruct prior context or rerun completed expensive work unnecessarily.

When the user asks to plan a next step, normally deliver both the plan and the runnable `.ipynb` unless the user explicitly asks for planning only. When the user reports an error, normally inspect and patch the actual notebook rather than returning a replacement snippet for manual copying.
