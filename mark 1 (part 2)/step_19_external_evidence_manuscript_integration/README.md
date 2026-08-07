# Step 19 — External-evidence manuscript integration

## Purpose

Integrate the sealed LiTS held-out and 3D-IRCADb-01 external evaluations into a new paper-ready evidence package without modifying earlier phases.

## Inputs

Only sealed summary artifacts from Steps 5, 7, 11, and 16–18 are read. No source test image, mask, probability cache, model, loader, or inference path is opened.

## Outputs

The notebook writes comparative evidence tables, patient-distribution data, failure and claim matrices, two paper-ready figures, an updated manuscript draft, technical report, limitations document, provenance, gate, and signature under `outputs/`.

## Gate

Expected result: `MANUSCRIPT_EXTERNAL_EVIDENCE_UPDATED_OWNER_INPUT_REQUIRED`. The external pass is added to the evidence base, but the failed LiTS minimum-patient gate remains unchanged and submission remains blocked until owner declarations and the exact external-dataset citation are completed.

## Run

Open `step_19.ipynb`, select the `ds_gpu` kernel, restart, and Run All. This is a bounded read-only analysis and performs no inference.

## Completed result — 5 August 2026

- Result: `MANUSCRIPT_EXTERNAL_EVIDENCE_UPDATED_OWNER_INPUT_REQUIRED`; all six gate requirements passed.
- Reconciled the LiTS and 3D-IRCADb-01 headline metrics from sealed summary tables and preserved their different cohorts/label semantics as a mandatory comparison caveat.
- LiTS remains a failed formal model-acceptance result: global Dice 0.7677, mean positive-patient Dice 0.5073, and minimum Dice 0.
- The separate external contract passed: global Dice 0.8480, mean positive-patient Dice 0.7112 (95% CI 0.5678–0.8278), median 0.8355, and minimum 0.0130.
- Both datasets preserve a smallest-lesion weakness: LiTS detection 60.3%; external detection 63.64%.
- External negative-control and source-label caveats remain explicit; the manuscript cannot claim clinical readiness, universal superiority, or biological reinterpretation.
- Created the revised paper draft, technical report, limitations document, claim matrix, exact comparison tables, and two visually inspected manuscript figures.
- Submission remains blocked: owner declarations are incomplete and the exact scholarly 3D-IRCADb-01 citation still requires verification.
- No inference, tuning, test-source reopening, external rerun, submission, or payment action occurred.
- Combined Step 19 signature: `d6075d9fc6d5d802316f2e3e7d9f7ee3f379ae9fc6919053e292fb496b3e0fb3`.
