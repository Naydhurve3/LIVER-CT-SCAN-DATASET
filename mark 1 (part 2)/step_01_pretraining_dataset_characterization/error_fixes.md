# Step 01 Error Fixes

## 3 August 2026 — generator quoting failure

- Error: `SyntaxError: unterminated triple-quoted string literal` while first generating the notebook.
- Root cause: the generated code cell used an inner triple-double-quoted data-card string inside the generator's triple-double-quoted cell source.
- Repair: changed the inner data-card string to triple single quotes and retained the fix in `create_step_01_notebook.py`.
- Impact: no analysis executed and no completed output was lost or overwritten.
- Verification: regenerate the notebook, validate it with `nbformat`, parse every code cell, and rerun the safe preflight.
- Resume: manual execution should start from the notebook's first cell with Restart Kernel and Run All.

## 3 August 2026 — optional Markdown-table dependency

- Error: `ModuleNotFoundError: No module named 'tabulate'` during the safe environment preflight.
- Root cause: pandas `DataFrame.to_markdown()` requires the optional `tabulate` package, which is not installed in the project `.venv`.
- Repair: the generator now embeds bounded CSV blocks in the Markdown data card using built-in `DataFrame.to_csv()`; no dependency installation is required.
- Impact: the manifest/test-lock preflight had already passed; full analysis was not executed and no analytical output was lost.
- Verification: regenerate, parse every code cell, and rerun the first three code cells in a clean namespace.

## 3 August 2026 — Matplotlib boxplot keyword incompatibility

- Error: `TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'` in Cell 6, the geometry comparison/dashboard cell.
- Root cause: the active `ds_gpu` environment uses a Matplotlib version whose `Axes.boxplot` API requires `tick_labels`; the older `labels` alias is no longer accepted.
- Repair: changed `labels=ALLOWED_SPLITS` to `tick_labels=ALLOWED_SPLITS` in both `step_01_pretraining_dataset_characterization.ipynb` and `create_step_01_notebook.py`.
- Preserved work: the expensive source-NIfTI profiling cell and all outputs already written before Cell 6 remain intact. No output was deleted.
- Verification: validate notebook structure and syntax, then execute the repaired boxplot call with representative geometry-like arrays in the active `ds_gpu` environment.
- Resume: rerun Cell 6 (`Geometry comparisons, train-derived outliers, and dashboard`), then continue with Cell 7 onward. Do not restart or rerun the expensive source-NIfTI profiling cell unless its in-memory variables were lost.

## 3 August 2026 — false audit hold and missing approved source transform

- Observed result: the executed gate returned `HOLD_FIX_DATA_OR_LABEL_ISSUE`, reporting affine and label-containment failures.
- Root cause 1: tumour-empty volumes used `max(tumour.sum(), 1)` as the containment denominator, producing `0.0` and falsely treating not-applicable containment as a critical failure.
- Root cause 2: the source segmentation array was used for HU/image-label analysis without applying the manifest's approved `rot180` transform. This invalidated HU/contrast results for affected volumes and made their raw affines appear mismatched.
- Root cause 3: affine gating compared raw headers only. After composing the approved `rot180` voxel transform, all 33 rotated train/validation volumes numerically match the image affine. Five identity volumes (48–52) retain known header mismatch but have approved manual spatial status and matching shapes, so they are recorded as resolved header inconsistencies rather than unresolved critical failures.
- Repair: apply `approved_transform` to source segmentation before spatial/HU analysis; compare the image affine with the transform-composed effective segmentation affine; record raw, effective, and resolved affine states; assign `NaN` containment to tumour-empty volumes; and gate only finite containment values.
- Files patched: `step_01_pretraining_dataset_characterization.ipynb` and `create_step_01_notebook.py`.
- Preserved work: existing outputs remain on disk, but the HU/appearance, label-alignment, difficulty, data card, and gate artifacts are superseded until the repaired notebook reruns.
- Verification: notebook and generator parse; `nbformat` validation passes; all 33 `rot180` volumes match numerically after affine composition; all 117 train/validation volumes have resolved alignment; test data was not accessed.
- Resume: Restart Kernel and Run All is required because the source-NIfTI profiling cell must recompute corrected HU and spatial outputs. Do not advance to Step 02 from the current failed gate.

## 3 August 2026 — difficulty association population mismatch

- Error: patient-Dice associations used all 13 validation patients, including four tumour-empty patients, while the controlling Mark 4D/4E patient-Dice population is nine tumour-positive patients.
- Impact: associations involving lesion count or burden could be inflated by the structurally tumour-empty group and were not comparable with the controlling model metric.
- Repair: filter `mark4e_patient_dice` associations to `mark4e_has_tumour == True`; keep positive-slice outcomes on their existing tumour-positive patient coverage; save the analysis population in every association row.
- Files patched: notebook and generator.
- Resume: this correction is included in the same required Restart Kernel and Run All repair pass.
