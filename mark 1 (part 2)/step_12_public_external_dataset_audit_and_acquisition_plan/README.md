# Step 12 — Public External-Dataset Audit and Acquisition Plan

This data-only phase compares authoritative online liver CT sources against the completed LiTS train/validation audit and frozen pipeline requirements.

## Scope

The notebook audits online metadata and existing train/validation-derived profiles only. It performs no download, training or inference and never accesses the sealed local test split.

## Candidate sources

- 3D-IRCADb-01: small independent tumour-labelled feasibility candidate.
- HCC-TACE-Seg: larger independent HCC domain-shift candidate with DICOM-SEG complexity.
- MSD Task03 Liver: LiTS-overlapping; prohibited as independent external validation.
- CHAOS CT: healthy liver organ data; useful for liver-ROI shift only.

## Run

Prefer opening the short-path alias `step_12.ipynb` in `ds_gpu` and use **Restart Kernel and Run All**. It is identical to `step_12_public_external_dataset_audit_and_acquisition_plan.ipynb` but avoids Windows/Jupyter path-handling problems. Every output is written under `outputs/`.

## Decision boundary

This phase may recommend a source but does not accept online terms or download data on the user's behalf. A future ingestion phase must record terms, hashes, patient/series identities, geometry, label semantics and conversion parity before any model evaluation.

## Verified completion — 5 August 2026

- Result level: `PUBLIC_EXTERNAL_DATA_AUDIT_COMPLETE`; all 7/7 phase requirements passed.
- Audited four authoritative online sources and identified two independent tumour-labelled candidates.
- Priority scores: 3D-IRCADb-01 `4.25/5`, HCC-TACE-Seg `3.95/5`, MSD Task03 Liver `3.45/5`, and CHAOS CT `2.60/5`.
- MSD Task03 Liver is explicitly prohibited as independent external validation because it overlaps LiTS.
- CHAOS CT is restricted to liver-ROI/domain-shift work because it contains healthy CT livers without tumours.
- Produced a 12-check ingestion-QC contract, empty SHA-256 download-manifest template, source registry, leakage audit, compatibility matrix, acquisition plan and signed gate.
- Repaired the Step 01 `val` split-label schema mismatch and clarified zero-download/test-lock gate labels; both repairs are recorded in `error_fixes.md`.
- Validation: `nbformat` valid, all six code cells parse and executed with zero errors, 16 outputs exist, all 15 signed hashes pass, and the dashboard was visually inspected.
- Downloads performed: false. Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Next action: confirm the official 3D-IRCADb-01 reuse terms, then create Step 13 to ingest and QC that source only. No inference should run until all 12 ingestion checks pass.
