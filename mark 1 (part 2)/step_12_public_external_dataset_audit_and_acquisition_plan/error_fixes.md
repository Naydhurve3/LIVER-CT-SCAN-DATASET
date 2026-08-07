# Error fixes

## 5 August 2026 — Step 01 validation split label

- Error: the input-boundary assertion reported observed splits `['train', 'val']` because the notebook initially allowed `train` and `validation` only.
- Root cause: Step 01's machine-readable geometry schema uses the project abbreviation `val`.
- Scientific interpretation: this was a schema-label mismatch, not test leakage. No `test` split was present and no test path was opened.
- Repair: updated both the generator and notebook logic to accept the documented `val` abbreviation while still rejecting any split outside `train`, `val`, or `validation`.
- Preserved outputs: the initial `input_verification.csv` was diagnostic only; no downloads or expensive computations had started.
- Resume: the repaired notebook was regenerated and rerun from Cell 1. A manual replay may use **Restart Kernel and Run All**.

## 5 August 2026 — Safeguard gate-label clarification

- Issue: target keys `downloads performed` and `test source files reopened` had values `true`, where `true` meant their expected zero counts passed.
- Risk: a reader could misinterpret the keys as evidence that downloading or test reopening occurred.
- Repair: renamed them `zero external dataset downloads performed` and `local test source files remained sealed` in the generator and notebook, then regenerated all outputs and hashes.
- Resume: no manual repair is required; the final notebook is fully executed.

## 5 August 2026 — Notebook could not be opened from the long Windows path

- User-visible error: `Unable to open 'step_12_public_external_dataset_audit_and_acquisition_plan.ipynb'`.
- Diagnosis: the file exists and is readable; JSON parsing and `nbformat` validation both pass. It contains 14 cells, including six code cells, and is approximately 29 KB. The full path is 184 characters and contains multiple spaces and parentheses, making a client-side Windows/Jupyter path-handling issue the likely cause.
- Repair: created the identical short-name alias `step_12.ipynb` in the same phase directory and updated the generator to write both filenames.
- Outputs preserved: all previously executed Step 12 outputs and signed hashes remain unchanged.
- Resume: open `step_12.ipynb`. It is already executed; use **Restart Kernel and Run All** only if a fresh replay is desired.
