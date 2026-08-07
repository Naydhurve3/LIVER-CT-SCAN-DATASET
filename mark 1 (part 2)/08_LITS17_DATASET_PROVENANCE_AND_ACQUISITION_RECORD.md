# LiTS17 Dataset Provenance and Acquisition Record

Last verified: 6 August 2026.

## Technical summary

The local modelling cohort contains the complete `131` annotated CT/segmentation pairs conventionally numbered `0` through `130` from the LiTS training collection. The files were assembled from two Kaggle mirrors and the Hugging Face CADS `0004_lits` mirror, then standardized into the verified corrected build `build_corrected_20260713_214847_v2`.

The most specific retained acquisition mapping is:

| Acquisition source | Volume IDs in the local cohort | Count | Acquired form |
|---|---|---:|---|
| Kaggle `andrewmvd/liver-tumor-segmentation` | `0-50` | 51 | NIfTI image/segmentation pairs |
| Kaggle `andrewmvd/liver-tumor-segmentation-part-2` | `51-69`, `100` | 20 | NIfTI image/segmentation pairs; full archive used after per-file API failures |
| Hugging Face `huggingface/CADS-dataset`, configuration/subdirectory `0004_lits` | `70-99`, `101-130` | 60 | `.nii.gz`, gzip-decompressed to `.nii` |
| **Total** | **`0-130`** | **131** | **131 CT volumes plus 131 segmentations** |

This volume-level mapping was supplied by the dataset owner and is corroborated at aggregate level by the retained 13 July 2026 staging record, which states that `0-69,100` came from the AndrewMVD Kaggle mirror and `70-99,101-130` came from Hugging Face CADS. The retained machine-readable evidence does not independently distinguish Kaggle Part 1 from Part 2 for every file, so that finer split remains **owner-recorded provenance**, not a remotely checksummed reconstruction.

## Dataset identity and terminology

- Upstream dataset: **Liver Tumor Segmentation Benchmark (LiTS)**.
- Challenge origin: LiTS challenge associated with ISBI 2017 and MICCAI 2017/2018.
- Upstream challenge page: [LiTS CodaLab competition](https://competitions.codalab.org/competitions/17094).
- Benchmark reference: Bilic P, Christ P, Li HB, et al. *The Liver Tumor Segmentation Benchmark (LiTS).* Medical Image Analysis. 2023;84:102680. [doi:10.1016/j.media.2022.102680](https://doi.org/10.1016/j.media.2022.102680).
- Original source format used locally: paired three-dimensional NIfTI CT and segmentation files.
- Source label convention: `0 = background`, `1 = liver`, `2 = tumour/lesion`.
- Local modelling target: tumour-only foreground; liver support is stored or derived separately.

The LiTS benchmark describes `131` annotated training CT volumes and `70` separate challenge-test volumes. Therefore, the project's internal split named `test` is a patient-disjoint holdout carved from the 131 annotated training volumes; it is **not** the official 70-volume LiTS challenge test set. This distinction should be preserved in every future notebook and report.

## Download sources actually used

### Kaggle Part 1

- Dataset slug: `andrewmvd/liver-tumor-segmentation`
- URL: [https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
- Local coverage: volumes `0-50` (`51` pairs).
- Acquisition method: Kaggle API using the standard user-level Kaggle credential file.
- Account recorded by the owner: `nayankeshavraodhurve`.
- Security rule: the credential file contents, API token and any copied secrets are not part of project provenance and must never be committed or reproduced in logs.

### Kaggle Part 2

- Dataset slug: `andrewmvd/liver-tumor-segmentation-part-2`
- URL: [https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)
- Local coverage: volumes `51-69` and `100` (`20` pairs).
- Acquisition note: per-file Kaggle API requests returned HTTP 404 for multiple files; the complete Part 2 archive was downloaded instead.
- The retained `download_log.json` is a partial per-file-attempt log, not the final acquisition inventory. It records volume `100` as successful and many later IDs as failed/missing, while the final raw authoritative directories contain all IDs. It must not be used alone to infer final cohort coverage.

### Hugging Face CADS mirror

- Dataset: `huggingface/CADS-dataset`
- Dataset page: [https://huggingface.co/datasets/huggingface/CADS-dataset](https://huggingface.co/datasets/huggingface/CADS-dataset)
- LiTS configuration/subdirectory: [`0004_lits`](https://huggingface.co/datasets/huggingface/CADS-dataset/tree/main/0004_lits)
- Local coverage: volumes `70-99` and `101-130` (`60` pairs).
- Acquired form: `.nii.gz`; each file was gzip-decompressed to `.nii` before assembly into the raw authoritative directories.
- The decompression changes the wrapper representation but should not change NIfTI voxel content. Exact remote-compressed-to-local provenance cannot be reconstructed because the original compressed files, remote commit identifier and per-file remote checksums were not preserved in the current evidence set.

### Upstream LiTS source

- Challenge page: [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094)
- The CADS LiTS README points to the historical LiTS Google Drive source and states that official challenge-test ground truth remains private.
- All locally used volumes are treated as copies of the original annotated LiTS training cohort obtained through mirrors, not as new patients or independent datasets.

## Local authoritative storage

```text
D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\
  01_raw_authoritative\
    volumes\volume-{ID}.nii
    segmentations\segmentation-{ID}.nii
    nifti_downloads\download_log.json
  02_staging\
    build_20260713_112738\dataset_version.json
    build_corrected_20260713_214847_v2\
      dataset_version.json
      dataset_readiness.json
      manifests\slice_manifest.csv
      audits\strict_validation_summary.json
```

Verified filename inventory on 6 August 2026:

| Check | Result |
|---|---:|
| `volume-*.nii` files | 131 |
| Unique volume IDs | 131 |
| Volume ID range | `0-130` |
| Missing volume IDs | 0 |
| `segmentation-*.nii` files | 131 |
| Unique segmentation IDs | 131 |
| Segmentation ID range | `0-130` |
| Missing segmentation IDs | 0 |

This inventory verifies local completeness and naming. It does not by itself prove which remote mirror supplied each byte.

## Corrected modelling build

- Canonical build: `build_corrected_20260713_214847_v2`.
- Status: `verified_eda_ready` in the build record; subsequent project gates separately verified loader and model compatibility.
- Manifest: `manifests\slice_manifest.csv`.
- Manifest grain: one axial slice per row.
- Current manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
- Manifest rows: `58,638`.
- Patients/volumes: `131`.
- Derived spatial size: `256 x 256`.
- CT preprocessing: HU window `[-160,240]`, bilinear image resize, nearest-neighbour mask resize.
- Approved orientation handling: `84` identity volumes and `47` segmentations corrected by an in-plane 180-degree rotation.
- Strict validation failures: `0`.
- Spatially approved volumes: `131/131`.
- Source hashes complete: `true` in the corrected-build strict-validation summary.

The canonical internal split is patient-disjoint:

| Internal split | Volume IDs | Volumes | Slices | Permitted role |
|---|---|---:|---:|---|
| Train | `0-103` | 104 | 40,667 | Training and train-derived policies |
| Validation | `104-116` | 13 | 10,685 | Model selection and frozen validation gates |
| Locked internal test | `117-130` | 14 | 7,286 | Already evaluated once under the sealed project contract; never reuse for tuning |

The counts above come from previously sealed metadata. This provenance consolidation did not open or inspect locked-test images, masks or statistics.

## EDA and quality evidence

The detailed train/validation-only EDA is already preserved in:

- `step_01_pretraining_dataset_characterization\outputs\DATASET_DATA_CARD.md`
- `step_01_pretraining_dataset_characterization\outputs\volume_geometry_profile.csv`
- `step_01_pretraining_dataset_characterization\outputs\lesion_component_profile.csv`
- `step_01_pretraining_dataset_characterization\outputs\hu_contrast_per_volume.csv`
- `step_01_pretraining_dataset_characterization\outputs\label_qc_per_volume.csv`
- `step_01_pretraining_dataset_characterization\outputs\pretraining_dataset_gate.json`

That audit passed all 12 mandatory requirements, found zero critical integrity/leakage/geometry/label/ROI failures, documented geometry and HU domain shift, retained volumes 104 and 116 as focus cases, and froze a train-derived sampling policy. Future work should cite those files instead of recomputing EDA.

## Evidence hierarchy and confidence

| Claim | Evidence | Confidence | Limitation |
|---|---|---|---|
| Local cohort has 131 paired IDs `0-130` | Current filesystem filename inventory | High | Proves local inventory, not remote origin |
| Aggregate mirror mapping | `build_20260713_112738\dataset_version.json` source note | High | Does not split the Kaggle group into Part 1 versus Part 2 |
| Exact Part 1/Part 2 mapping | Owner's acquisition record in this document | Medium-high | Retained API log is incomplete and no archive manifest covers every original download |
| Corrected dataset identity | Current manifest hash plus corrected build JSON/audits | High | Applies to the local corrected build |
| Kaggle API per-file failures | `download_log.json` plus retained ZIPs | Medium | Log is partial and internally reflects an intermediate state |
| Hugging Face files were `.nii.gz` and decompressed | Owner acquisition record | Medium-high | Original compressed files and remote hashes are absent |
| All mirrors derive from LiTS | Provider descriptions, CodaLab page and LiTS benchmark paper | High at dataset-family level | Exact byte equivalence to an upstream release is not proven |

## Licence and redistribution boundary

The source pages expose different mirror-level licence notices:

- The current AndrewMVD Kaggle LiTS page displays **CC BY-NC-ND 4.0**.
- The current CADS `0004_lits` README displays **CC BY-NC-SA 4.0**.
- The CodaLab challenge page links terms and conditions, but the retained local evidence does not contain a dated copy of the exact terms accepted for the original LiTS distribution.

These notices should not be collapsed into a single invented licence for the mixed-source local copy. For this independent project, preserve attribution, keep use non-commercial, and do not redistribute raw CT volumes, masks, converted copies or packaged subsets without rechecking the applicable source and upstream terms. This record is a provenance statement, not legal advice.

Code, aggregate EDA tables and model outputs should be assessed separately from redistribution of the medical image files themselves.

## Known documentation conflicts resolved by this record

1. The historical `Practice\dataset_pipeline.ipynb` provenance cell records only `andrewmvd/liver-tumor-segmentation + part-2`; it omits the 60 Hugging Face-supplied volumes. Use this record plus the 13 July staging `dataset_version.json` note for acquisition provenance.
2. `Dataset\Liver\00_source_registry\source_registry.json` describes legacy PNG image/mask collections. Those are overlapping historical representations and are not the authoritative NIfTI acquisition record.
3. The project-root README and older archived documentation identify `andrewmvd/lits-png` as the dataset source. That is historical implementation context, not the source of the canonical NIfTI cohort used by the current Part 2 study.
4. Step 07's `DATASET_TERMS_VERIFICATION.md` correctly marked the exact acquired-copy terms unresolved at that time. The acquisition portals and their current licence notices are now documented, but conflicting mirror notices mean a single upstream redistribution licence is still not asserted.
5. The current Kaggle Part 1 page may show a broader inventory than the files originally obtained in July 2026. Current page contents must not be used to rewrite the recorded historical volume mapping.

## Retained evidence fingerprints

These hashes identify the exact local provenance records inspected for this document:

| Evidence file | SHA-256 |
|---|---|
| `build_20260713_112738\dataset_version.json` | `84ef32b860e881d9c1cb9289286d04724232b9bd7962758ce94d6c6184583591` |
| `01_raw_authoritative\nifti_downloads\download_log.json` | `a2c23465e1bb9cd7d0ad23958fc56ab3069c9b47cb4f4ed58fb3b871133aa4c7` |
| `Practice\dataset_pipeline.ipynb` | `9c41a68b242b041c5c78790ca499eb99eaccf9f1b3a4f6504fadc1b40cf5fac1` |
| `00_source_registry\source_registry.json` | `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173` |
| Corrected `dataset_version.json` | `fb39183db4d59b8010dbb77426e4ba2166a1316c6b2da6925996a41739fd22bf` |
| Corrected `dataset_readiness.json` | `2dad7f66e37edc57be4e27f1e68707f2d48a5bd432f962ba703734db7bf92bc8` |
| Corrected `strict_validation_summary.json` | `647637c6fb115ab74eb14367c0681273b3a7ae963ba4b5b8a5be3dec4a960602` |
| Corrected `slice_manifest.csv` | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |

If any of these files change, reverify the affected claims before treating this document as current.

## Minimum provenance procedure for any future re-download

To remove the remaining uncertainty in a future acquisition, save the following without saving credentials:

1. Source URL, dataset slug/configuration, access date and provider version or commit ID.
2. Exact command or API method, excluding tokens and credential contents.
3. Remote file listing before download.
4. SHA-256 of every downloaded archive or compressed NIfTI.
5. Archive-member listing and extraction log.
6. Source-to-local filename mapping by volume ID.
7. SHA-256 before and after decompression where applicable.
8. A 131-row source map containing `volume_id`, source portal, remote path, local CT path, local segmentation path and hashes.
9. A dated copy or URL snapshot of the applicable terms and required attribution.
10. A no-overlap check before treating any newly acquired cohort as independent evaluation data.

## Canonical reuse statement for this project

For future notebooks and internal technical documentation, use:

> The project uses the 131 annotated LiTS CT volumes numbered 0-130. Volumes 0-69 and 100 were obtained from the AndrewMVD Kaggle LiTS mirrors, while volumes 70-99 and 101-130 were obtained from the Hugging Face CADS `0004_lits` mirror and gzip-decompressed from `.nii.gz` to `.nii`. The cohort was standardized into corrected build `build_corrected_20260713_214847_v2`; its authoritative slice manifest has SHA-256 `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`. The project's internal test split is a sealed holdout from these 131 annotated volumes, not the official LiTS challenge test set.

Do not shorten this to “downloaded from Kaggle” because that loses the Hugging Face provenance for 60 volumes.
