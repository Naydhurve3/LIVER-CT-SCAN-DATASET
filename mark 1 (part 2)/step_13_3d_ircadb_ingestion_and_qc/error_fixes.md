# Error fixes

## 5 August 2026 — Zero-event safeguard labels

- Issue: the initial preflight gate used target keys `local LiTS test sources accessed` and `model inference runs` with value `true`, where `true` meant their expected zero counts passed.
- Risk: the combined key/value could be misread as test access or inference having occurred.
- Repair: renamed the targets `zero local LiTS test source accesses` and `zero model inference runs` in the generator and both notebooks, then regenerated the preflight outputs and signature.
- Preserved work: no archive had been downloaded and no extracted data existed.
- Resume: open `step_13.ipynb`; the final preflight is fully executed.

## 5 August 2026 — Preflight visualization semantics

- Issue: the initial dashboard rendered unexecuted acquisition checks in red with `PENDING / FAIL`, which could imply observed data failures.
- Repair: when authorization is absent, incomplete acquisition/source checks now render amber as `PENDING`; red `FAIL` is reserved for an authorized ingestion that fails QC.
- Scientific impact: none. Gate values and source evidence are unchanged.
- Resume: no action required; the short notebook was regenerated and the preflight rerun.

## 5 August 2026 — Combined-archive endpoint stalled at zero bytes

- Error: after explicit authorization, the official combined archive endpoint remained active for several minutes while `3Dircadb1.zip.part` stayed at 0 bytes.
- Root cause: the combined link redirects to an on-demand Nextcloud WebDAV folder ZIP that did not begin streaming. A header probe confirmed the redirect; an official patient-level endpoint immediately transferred its 21.86 MB archive.
- Repair: stopped the stalled kernel, removed only the empty temporary `.part` file, and replaced the combined transfer with the 20 official patient-level archives exposed by the IRCAD page. Each archive is independently size-checked, ZIP-validated and SHA-256 hashed; a deterministic archive-set hash covers all 20.
- Resume: rerun `step_13.ipynb` from Cell 1. Existing valid patient archives are reused, so partial progress is preserved.

## 5 August 2026 — Missing pydicom and nested component archives

- Error: after all 20 patient archives downloaded and extracted, the inventory cell stopped because `pydicom` was not installed.
- Additional source finding: each patient archive expands to four nested ZIP files rather than directly to `PATIENT_DICOM/`, `MASKS_DICOM/`, `LABELLED_DICOM/` and `MESHES_VTK/` directories.
- Repair: installed `pydicom 3.0.2` in `ds_gpu`; versioned the extraction marker; added path-traversal-safe extraction of all four nested component archives for every patient.
- Preserved work: all 20 patient ZIPs and their computed hashes are reused; no network redownload is required.
- Resume: rerun `step_13.ipynb` from Cell 1. The notebook resumes at validation/extraction using the existing archives.

## 5 August 2026 — Repeated inner component directory

- Finding: the nested component archives expand as `PATIENT_DICOM/PATIENT_DICOM` and `MASKS_DICOM/MASKS_DICOM`, so a shallow directory match can select an empty wrapper instead of the data-bearing directory.
- Repair: patient discovery now selects the deepest `PATIENT_DICOM` directory containing files, resolves the patient root by the `3Dircadb1.*` ancestor, and selects the `MASKS_DICOM` directory with the greatest number of immediate label subdirectories.
- Preserved work: all 20 patient archives, nested extraction and the extraction marker are unchanged.
- Resume: regenerate and rerun `step_13.ipynb`; it reuses all existing data and starts the corrected inventory.
# Repair 7 - nested patient-directory double counting on rerun

- Symptom: a later authorized rerun reported 40 extracted patients and overwrote the previously passing Step 13 gate with `EXTERNAL_SOURCE_INGESTION_QC_FAIL`.
- Root cause: both levels of the source layout `PATIENT_DICOM/PATIENT_DICOM` satisfied the discovery predicate.
- Repair: patient discovery now retains only the deepest data-bearing `PATIENT_DICOM` directory for each case.
- Resume: regenerate `step_13.ipynb`, then Run All from the first cell. Existing downloads and extracted files are reused; no re-download is required.
