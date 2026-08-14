# data/ — Dataset Configuration & External Data Home

## Contents

| Path | Purpose |
|---|---|
| `file_list.json` | Index of the raw LiTS-17 dataset files |
| `splits/` | Dataset split manifests |
| `splits_stratified/` | Stratified patient-disjoint splits (Train 104 / Val 13 / Test 14) |
| `metadata/` | Dataset metadata / provenance records |
| `external/` | **External datasets** (see below) |

## External datasets — `data/external/`

- **3D-IRCADb-01** (external evaluation set): the raw download archives, extracted DICOM,
  normalized `.nii.gz` conversions, and `.vtk` meshes are **not duplicated here** — they remain
  in `mark 1 (part 2)/step_13_*` outputs (re-downloadable external data; recorded in
  `backup/08_ircadb_REFERENCE.md`). This folder is the designated home if you ever relocate them.

## Canonical dataset build (the actual source of truth)

The training/validation/test volumes live in the external staging build
`build_corrected_20260713_214847_v2` (manifest SHA-256
`575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`), referenced by the
framework's `manifest_dataset.py`. This `data/` folder holds the manifests and split
definitions, not the raw NIfTI/PNG volumes.
