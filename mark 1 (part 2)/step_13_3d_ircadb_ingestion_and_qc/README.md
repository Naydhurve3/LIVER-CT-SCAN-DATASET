# Step 13 — 3D-IRCADb-01 Ingestion and Source QC

This notebook implements guarded acquisition, archive validation, safe extraction, DICOM header profiling, label discovery and a source-ingestion gate for the official 3D-IRCADb-01 dataset.

## Open

Use the short notebook name `step_13.ipynb` to avoid Windows path-handling issues.

## Default run

The default configuration performs no network request. It completes successfully with `EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED` and writes a signed preflight package.

## Enabling acquisition

Review the official IRCAD page and CC BY-NC-ND 4.0 conditions. If acceptable for this non-commercial project, set these three values in code Cell 1:

```python
USER_ACCEPTS_CC_BY_NC_ND_4_0 = True
DOWNLOAD_ENABLED = True
EXTRACT_ENABLED = True
```

Then use **Restart Kernel and Run All**. The combined archive is approximately 782 MB. Do not redistribute downloaded or derived dataset files.

## Boundary

This phase performs ingestion QC only. It does not convert the dataset to the LiTS schema, run inference, tune parameters, train models, or access the sealed local test split.

## Verified safe preflight — 5 August 2026

- Result level: `EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED`; this is the expected successful preflight state.
- Verified all 6/6 Step 12 prerequisites and all 5/5 phase-integrity safeguards.
- Recorded the official 782 MB combined archive, CC BY-NC-ND 4.0 terms, attribution, download URL and non-redistribution boundary.
- Created guarded streaming download, archive-size/ZIP validation, path-traversal-safe extraction, archive/file inventories, DICOM header profiling, SOP UID checks, and liver/tumour label discovery.
- Generated 17 output files plus empty `raw_downloads/` and `extracted/` staging directories; all 16 signed artifacts match their hashes.
- Validation: both notebook filenames pass `nbformat` and AST parsing; the short notebook's six code cells executed with zero errors; the dashboard was visually inspected.
- Download performed: false. Inference performed: false. Local test accessed: false.
- Next action: review the official license and, if acceptable, set the three authorization switches in code Cell 1 to `True`, then use **Restart Kernel and Run All**.

## Download authorization — 5 August 2026

- The user explicitly accepted the 3D-IRCADb-01 CC BY-NC-ND 4.0 terms and authorized the Step 13 download and extraction.
- The generator and both notebook names now contain the recorded authorization text and enabled acquisition switches.
- Authorization covers only the official combined archive, safe extraction and source QC. It does not authorize redistribution, model inference, tuning or local LiTS test access.

## Verified authorized ingestion — 5 August 2026

- Result level: `EXTERNAL_SOURCE_INGESTION_QC_PASS`.
- The official patient-level route produced 20 valid ZIP archives totaling 820,269,486 bytes; archive-set SHA-256: `55cd6722da6c02f8365ab566603db116128e0d854782d2a174b85056f4a5ba96`.
- Safe nested extraction found exactly 20 patients, one readable CT series per patient, 74–260 slices per case, 512 × 512 matrices, and zero duplicate SOP Instance UIDs.
- All 20 cases contain a liver label. Actual hepatic-tumour pixel semantics are resolved in Step 14 because source folder names also include non-hepatic tumours.
- A rerun initially double-counted nested `PATIENT_DICOM/PATIENT_DICOM` directories as 40 cases. The generator and notebook were repaired to keep only the deepest data-bearing directory, and the signed 20-patient pass was restored.
- Conversion performed: false. Inference performed: false. Local LiTS test accessed: false.
- Next action: use the signed Step 14 normalized conversion and parity-QC notebook.
