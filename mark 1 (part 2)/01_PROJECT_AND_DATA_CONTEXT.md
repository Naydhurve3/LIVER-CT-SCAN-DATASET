# Project and Data Context

## Objective

Build a reproducible liver-tumour segmentation pipeline on the LiTS CT dataset that generalizes across patients, detects small lesions, avoids empty predictions on tumour-positive slices and controls false positives. The research uses staged validation gates rather than optimizing aggregate Dice alone.

## Workspace map

- Project root: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
- Authoritative historical implementation/evidence: `Practice\`
- Completed Mark 1 work: `mark 1\`
- New continuation workspace: `mark 1 (part 2)\`
- Historical documentation: `understanding the project\`
- Corrected dataset build: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- Manifest: `manifests\slice_manifest.csv`

## Authoritative dataset identity

- Dataset: LiTS liver CT collection.
- Corrected build ID: `build_corrected_20260713_214847_v2`.
- Manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
- Manifest grain: one axial slice per row.
- Total rows/slices: `58,638`.
- Total patients/volumes: `131`.
- Stored image and mask size: `256 x 256`.
- Preprocessing profile: HU window `[-160,240]`, bilinear image resize and nearest-neighbour mask resize.
- Target: tumour-only foreground; liver mask is stored separately.
- All `sample_id` values are unique.
- Verification status is verified and manual spatial status is approved in the corrected build.
- Canonical acquisition and mirror provenance: `08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md`.

## Locked split

| Split | Slices | Volumes | Tumour-positive slices |
|---|---:|---:|---:|
| Train | 40,667 | 104 | 4,930 |
| Validation | 10,685 | 13 | 1,042 |
| Test | 7,286 | 14 | 1,197 |
| Total | 58,638 | 131 | 7,169 |

Splits are patient-disjoint. The test split must remain locked through dataset characterization, training and validation selection.

## Manifest fields to use

- `sample_id`, `volume_id`, `slice_index`.
- `image_path`, `organ_mask_path`, `tumor_mask_path`.
- `source_volume_path`, `source_segmentation_path`.
- Source SHA-256 fields.
- `transform_applied`.
- `organ_pixels`, `tumor_pixels`, `organ_present`, `tumor_present`.
- `split`, integrity, verification, exclusion, build and spatial-approval fields.

## Historical data issues already resolved

- Earlier data contained 47 critical in-plane orientation failures.
- Volumes 83-99 and 101-130 required a 180-degree correction; 0-82 and 100 used identity.
- Slice-offset search selected offset zero, so the issue was orientation rather than slice numbering.
- A legacy tumour-burden calculation used a 512x512 denominator for 256x256 masks and under-reported burden fourfold.
- The corrected v2 build incorporates the approved transformations and is the only build allowed for current work.

## Current data evidence and remaining knowledge gap

Already verified:

- manifest provenance and patient-disjoint splits;
- source/derived alignment checks;
- binary mask and pixel-count checks;
- tumour containment and ROI coverage;
- round-trip ROI geometry;
- V116 tumour ROI coverage is 100% (`152,763/152,763` pixels);
- loader/model can overfit a deterministic subset.

Still required before further training:

- complete source geometry/acquisition profile;
- 3D lesion morphology and burden profile;
- HU/contrast and train-validation domain-shift analysis;
- training analog search for difficult validation phenotypes without using validation labels to define training weights;
- duplicate, near-duplicate, affine, slice-order and label-alignment QC summary;
- frozen train-derived sampling strata and data card.
