# Step 15 — Frozen external-evaluation contract

Open `step_15.ipynb` and use **Restart Kernel and Run All** from this directory.

This phase performs an artifact-only contract freeze. It verifies the signed Step 14 external cohort, all three checkpoint hashes, the frozen model policy and train-derived strata; then writes the complete external cohort, preprocessing, metrics, acceptance thresholds, one-time authorization rule, provenance, dashboard, gate and signature under `outputs/`.

It does not load a model, run inference or access the local LiTS test split. A pass must end at `EXTERNAL_EVALUATION_CONTRACT_FROZEN_AWAITING_AUTHORIZATION`.

## Verified execution — 5 August 2026

- Result level: `EXTERNAL_EVALUATION_CONTRACT_FROZEN_AWAITING_AUTHORIZATION`; all 10/10 freeze-readiness requirements passed.
- Frozen cohort: all 20 cases, 15 tumour-positive, 5 tumour-negative, zero exclusions and 24 source lesion-folder proxies.
- Train-derived patient burden coverage: Q1 = 1, Q2 = 7, Q3 = 0, Q4 = 7; the five tumour-negative patients are a separate negative-control stratum.
- Train-derived lesion-proxy coverage: Q1 = 0, Q2 = 2, Q3 = 4 and Q4 = 18. Folder-based lesion counts are explicitly labeled proxies.
- External median slice count is 127 versus 263 in training. All external geometry features remain inside observed training ranges; one case is a training-IQR outlier for through-plane spacing.
- Verified the three immutable checkpoint hashes, model-source identity, Step 03 policy hash, Step 14 signed package and all 60 normalized input paths.
- Frozen policy: broad HU window `[-160,240]`, original uint8 rounding, 256 × 256 grid, frozen predicted-liver ROI, maximum probability fusion, global threshold 0.70 and no post-processing.
- Current Step 15 signature: `aaca93be9d1d64e63807fb10f642b98cd0371504667f55ad69a14d8f1246f77d`; it supersedes the earlier time-dependent signature after the user's final rerun.
- Inference performed: false. Local LiTS test accessed: false.
- Next action: explicit authorization is required before creating/running the one-time Step 16 evaluation. The exact sentence is recorded in `outputs/gate_result.json`.
