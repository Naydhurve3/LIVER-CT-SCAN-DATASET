# Step 18 — Source-label concordance and failure adjudication

## Question

Do the sealed Step 16 predictions in the five tumour-negative controls and worst positive case overlap original 3D-IRCADb-01 source annotations that were intentionally excluded from the frozen hepatic-tumour truth?

## Scope

This is a read-only diagnostic audit. It loads existing Step 16 probability caches and original Step 13 DICOM masks. It performs no model loading, inference, tuning, threshold sweep, case exclusion, truth revision, or local LiTS test access.

## Frozen parameters

- Cases: `ircadb_05`, `ircadb_07`, `ircadb_11`, `ircadb_14`, `ircadb_18`, `ircadb_20`
- Prediction threshold: `0.70`
- Accepted truth: Step 14 `^livertumors?\d*$`
- Mask resize: nearest neighbour to 256 × 256
- Alignment: CT and mask `InstanceNumber`, with position-error verification

## Run

Open `step_18.ipynb`, choose the `ds_gpu` kernel, restart the kernel and Run All. Outputs are written only to `outputs/`.

## Decision gate

Success means all sealed inputs verify, every source mask aligns, all six cases are profiled, frozen metrics remain unchanged, and no inference/test access occurs. The expected result is `SOURCE_LABEL_CONCORDANCE_COMPLETE_EXPERT_REVIEW_REQUIRED`; source-mask overlap is not expert biological adjudication.

## Next action

After execution, use the signed report and panels for an optional blinded expert review. Do not rerun or tune Step 16.

## Completed result — 5 August 2026

- Result level: `SOURCE_LABEL_CONCORDANCE_COMPLETE_EXPERT_REVIEW_REQUIRED`; all six machine-readable requirements passed.
- Profiled 72 original source-label folders across the five negative controls and worst positive case; all DICOM masks aligned with zero missing instances and passed frozen-grid shape/position checks.
- `ircadb_07`: 5,334/6,174 prediction pixels (86.39%) overlap the excluded generic `tumor` mask. This is strong annotation-semantic concordance, not permission to revise frozen truth.
- `ircadb_14`: 6,694/7,527 prediction pixels (88.93%) overlap `metastasectomie`; source-label Dice is 0.8519. Biological interpretation still requires expert/source review.
- `ircadb_18`: only 16/1,328 prediction pixels intersect the accepted `livertumor` mask; accepted-label recall is 1.41% and Dice reconciles to 0.012992.
- `ircadb_20`: 445/621 prediction pixels (71.66%) overlap the source gallbladder mask, another failure mode requiring cautious interpretation.
- Case 5 predictions do not overlap either adrenal/surrenal tumour annotation; case 11 has only one predicted pixel.
- No excluded annotation explained more than 5% of predictions except cases 7 and 14.
- Step 16 results and metrics remain unchanged. No model loading, inference, tuning, threshold sweep, case exclusion, or local LiTS test access occurred.
- Combined Step 18 signature: `1f1f222d300e689a5e36787b4b7da60b0b27ddb594bab8d8aad736fe3ed35b6f`.

The notebook was executed successfully after repairing a stale Jupyter launcher, removing an unnecessary `tabulate` dependency, correcting the DICOM `(y,x)` to frozen NIfTI `(x,y)` transpose, and selecting specific rather than broad container masks in the figures. See `error_fixes.md`.
