# Step 14 — 3D-IRCADb-01 normalized conversion and parity QC

Run `step_14.ipynb` with **Run All** from this folder. The notebook reads the signed Step 13 extraction, creates internal NIfTI derivatives under `outputs/normalized/`, and writes geometry, label-content, HU, parity, provenance, dashboard, gate, and signature artifacts under `outputs/`.

No model inference is performed. No local LiTS test image, mask, loader, or statistic is accessed. Normalized 3D-IRCADb derivatives are internal and must not be redistributed.

The pass condition is `EXTERNAL_NORMALIZED_CONVERSION_QC_PASS`. Only after that pass should Step 15 freeze an external-evaluation contract.

## Verified execution — 5 August 2026

- Result level: `EXTERNAL_NORMALIZED_CONVERSION_QC_PASS`; all 12/12 conversion-QC checks passed.
- Converted 20/20 cases to internal `ct_hu.nii.gz`, `liver_mask.nii.gz`, and `tumour_mask.nii.gz` files. Exact CT, liver-mask, and tumour-mask serialization parity passed for every case.
- Every label slice matched a CT instance and spatial position exactly. All CT arrays were finite and all liver masks were nonempty.
- Pixel inspection reconciled the official 15/20 hepatic-tumour prevalence. The accepted label family is `^livertumors?\d*$`; adrenal labels in case 5 and the generic non-liver `tumor` label in case 7 are excluded.
- Geometry spans 74–260 slices, 0.561–0.873 mm in-plane spacing, and 1–4 mm through-plane spacing. CT values span -2048 to 3247 after the source rescale transform.
- Tumour containment within the source liver mask is at least 98.545%; 223 tumour voxels lie outside the source liver masks in total. This is preserved as source-label QC evidence, not silently clipped.
- Final Step 14 signature SHA-256: `a023b54f4b93ca935162824b6e51d3e70a405383ae38ba5b89c675a24839dc90`. The earlier signature was superseded when the repaired notebook was rerun and regenerated time-dependent provenance.
- Model inference performed: false. Formal external evaluation performed: false. Local LiTS test accessed: false.
- Next action: create Step 15 to freeze the external cohort, preprocessing, checkpoint/model identity, metrics, failure policy, and one-time evaluation gate before any inference.

## Storage note — normalized volumes archived to keep this phase lean (14 August 2026)

- The normalized NIfTI tree (`outputs/normalized/`, 60 files / ~923 MB) was **removed** after the external evaluation (Step 16) completed. Downstream evidence (Steps 17–21) reads only the csv/json/png/signature artifacts that remain in `outputs/`, so nothing later depends on `outputs/normalized/`.
- The exact per-file SHA-256 for every normalized volume is preserved in `outputs/conversion_manifest.csv` (`ct_sha256`, `liver_sha256`, `tumour_sha256`), so any regenerated file can be verified byte-for-byte.
- **To regenerate if ever needed**: the source is the Step 13 archive held in `mark 1 (part 2)/step_13_3d_ircadb_ingestion_and_qc/outputs/raw_downloads/` (20 verified patient ZIPs, sha-256 in `download_manifest.csv`). Run `step_14.ipynb` with **Run All** from this folder (it reads the Step 13 extraction, no download needed). Regeneration is deterministic and its output must match the hashes in `conversion_manifest.csv`.
- Steps 15–16 (external evaluation) would also need to be re-run after regeneration if the actual evaluation results are ever required again.
