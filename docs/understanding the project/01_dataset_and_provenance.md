# Dataset and Provenance

## Authoritative build

- Dataset: LiTS liver CT collection.
- Corrected build ID: `build_corrected_20260713_214847_v2`.
- Build directory: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`.
- Manifest: `manifests\slice_manifest.csv`.
- Manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
- Manifest rows: `58,638`.
- Patient/volume count: `131`.
- Manifest columns: `27`.
- Every row has `verification_status=verified`.
- Every row has `manual_spatial_status=approved`.
- Every `sample_id` is unique.
- All rows are marked ready for non-spatial and spatial EDA.

## Split

Splits are patient-disjoint.

| Split | Slices | Volumes | Tumor-positive slices |
|---|---:|---:|---:|
| Train | 40,667 | 104 | 4,930 |
| Validation | 10,685 | 13 | 1,042 |
| Test | 7,286 | 14 | 1,197 |
| Total | 58,638 | 131 | 7,169 |

The older legacy CSV recorded `7,114` tumor-positive rows. The corrected manifest is authoritative for modeling and records `7,169`.

## Image and mask data

- Stored image size: `256 × 256`.
- Stored organ-mask size: `256 × 256`.
- Stored tumor-mask size: `256 × 256`.
- Image interpolation: bilinear.
- Mask interpolation: nearest neighbor.
- Manifest preprocessing profile: `HU[-160,240]_bilinear_image_nearest_mask_256`.
- Mask operation order: `derive_resize_nearest_then_transform_256`.
- Model target: tumor-only foreground.
- Liver/organ masks are stored separately.
- Tumor containment inside the organ mask was verified in the corrected build.

Important: the PNG inputs are already derived/windowed images. Do not apply a second HU transform to those PNGs unless a new source-NIfTI preprocessing build is intentionally created and versioned.

## Manifest grain and important fields

The grain is one axial slice per row.

Key fields:

- `sample_id`: stable slice key such as `v000_s0000`.
- `volume_id`: patient/scan identifier.
- `slice_index`: within-volume axial position.
- `image_path`: derived CT PNG.
- `organ_mask_path`: binary liver/organ mask PNG.
- `tumor_mask_path`: binary tumor mask PNG.
- `source_volume_path` and `source_segmentation_path`: authoritative NIfTI sources.
- Source SHA-256 columns for both NIfTI files.
- `transform_applied`: identity or orientation correction.
- `organ_pixels` and `tumor_pixels`.
- `organ_present` and `tumor_present`.
- `split`.
- Integrity, verification, exclusion, build, preprocessing, and spatial-approval fields.

## Orientation and pairing investigation

The initial forensic build contained 47 critical orientation failures.

- Source-to-derived regeneration passed for all 156 representative/screenshot rows.
- Local slice offset search selected offset `0` for all 156 rows.
- The problem was not slice numbering.
- The problem was incompatible in-plane orientation between source CT and segmentation for a block of volumes.
- Rotation by 180 degrees was recommended for volumes `83–99` and `101–130`.
- Identity was recommended for volumes `0–82` and `100`.
- Three-dimensional continuity could appear smooth even when in-plane orientation was wrong.
- Tiny boundary components were traced to the source segmentations, not PNG conversion.

The corrected v2 build incorporates the approved spatial decisions and is the only build used in the recorded modeling gates.

## Historical build facts

Before correction:

- 131 complete NIfTI volume/segmentation pairs.
- 131 shape matches.
- Only 79 affine matches.
- 30,105 legacy instance values wrapped beyond 255, but `instance mod 256` matched all 58,638 rows.
- Legacy CSV liver-present rows: `19,143`.
- Legacy CSV tumor-present rows: `7,114`.

These facts explain why file count and shape checks alone were insufficient.

## Data-integrity rules that remain mandatory

- Use `VerifiedManifestDataset`.
- Treat manifest paths and sample IDs as authoritative.
- Never pair files using directory order.
- Resize masks only with nearest-neighbor interpolation.
- Keep image, organ mask, and tumor mask transforms synchronized.
- Check patient-disjoint splits before every experiment.
- Keep the test loader locked until a validation gate explicitly authorizes one final evaluation.
- Record manifest and checkpoint hashes in every result.

